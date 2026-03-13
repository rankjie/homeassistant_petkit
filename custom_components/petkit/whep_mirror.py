"""Mirror WHEP endpoint using an internal aiortc relay."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import secrets
from typing import TYPE_CHECKING, Any

from aiohttp import web

from homeassistant.components.http import HomeAssistantView

from .agora_api import SERVICE_IDS, AgoraAPIClient
from .agora_rtm import AgoraRTMSignaling
from .agora_websocket import AgoraWebSocketHandler
from .const import AGORA_APP_ID, DOMAIN, LOGGER
from .webrtc_common import (
    _add_offer_candidates,
    _get_live_feed_for_webrtc,
    _resolve_agora_user_id,
)

try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer as AiortcIceServer,
        RTCPeerConnection,
        RTCRtpSender,
        RTCSessionDescription,
    )
    from aiortc.contrib.media import MediaRelay
    from aiortc.sdp import candidate_from_sdp
except Exception as err:  # noqa: BLE001
    RTCConfiguration = None
    AiortcIceServer = None
    RTCPeerConnection = None
    RTCRtpSender = None
    RTCSessionDescription = None
    MediaRelay = None
    candidate_from_sdp = None
    AIORTC_IMPORT_ERROR = err
else:
    AIORTC_IMPORT_ERROR = None

if TYPE_CHECKING:
    from .camera import PetkitWebRTCCamera

TOKEN_REFRESH_INTERVAL_SECONDS = 20 * 60


@dataclass
class MirrorUpstreamSession:
    """One internal WebRTC ingest from Agora."""

    camera: PetkitWebRTCCamera
    peer_connection: Any
    agora_handler: AgoraWebSocketHandler
    agora_rtm: AgoraRTMSignaling
    relay: Any
    video_ready: asyncio.Event = field(default_factory=asyncio.Event)
    video_track: Any | None = None
    refresh_task: asyncio.Task[None] | None = None
    last_error: str | None = None

    @property
    def device_id(self) -> str:
        """Return device identifier."""
        return str(self.camera.device.id)


@dataclass
class MirrorDownstreamSession:
    """One downstream consumer served by the relay."""

    session_id: str
    kind: str
    peer_connection: Any


class PetkitMirrorRelayManager:
    """Manage internal upstream and downstream relay peers."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._upstreams: dict[str, MirrorUpstreamSession] = {}
        self._downstreams: dict[str, dict[str, MirrorDownstreamSession]] = {}

    async def create_downstream_offer(
        self,
        camera: PetkitWebRTCCamera,
        offer_sdp: str,
        *,
        session_id: str | None = None,
        kind: str = "whep",
    ) -> tuple[str, str]:
        """Create or reuse an upstream ingest, then answer one downstream offer."""
        device_id = str(camera.device.id)
        if kind == "whep":
            await self.close_downstreams_by_kind(device_id, kind)
        elif session_id is not None:
            await self.close_downstream(device_id, session_id)
        upstream = await self._ensure_upstream(camera)

        peer_connection = RTCPeerConnection()
        if session_id is None:
            session_id = secrets.token_hex(16)
        downstream = MirrorDownstreamSession(
            session_id=session_id,
            kind=kind,
            peer_connection=peer_connection,
        )

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = peer_connection.connectionState
            LOGGER.debug(
                "WHEP mirror downstream %s state=%s",
                device_id,
                state,
            )
            if state in {"failed", "closed"}:
                self.hass.async_create_task(
                    self._handle_downstream_closed(device_id, session_id)
                )

        sender = peer_connection.addTrack(upstream.relay.subscribe(upstream.video_track))
        self._prefer_h264(peer_connection, sender)

        await peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type="offer")
        )
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)
        await self._wait_for_ice_complete(peer_connection)

        async with self._lock:
            self._downstreams.setdefault(device_id, {})[session_id] = downstream

        return session_id, str(peer_connection.localDescription.sdp)

    async def close_device(self, device_id: str) -> bool:
        """Close downstream and upstream relay state for one camera."""
        downstreams: list[MirrorDownstreamSession] = []
        upstream = None
        async with self._lock:
            downstreams = list(self._downstreams.pop(device_id, {}).values())
            upstream = self._upstreams.pop(device_id, None)

        if not downstreams and upstream is None:
            return False

        for downstream in downstreams:
            await self._shutdown_peer(downstream.peer_connection)

        if upstream is not None:
            if upstream.refresh_task is not None:
                upstream.refresh_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await upstream.refresh_task
            await asyncio.gather(
                upstream.agora_handler.disconnect(),
                upstream.agora_rtm.stop_live(send_stop=True),
                self._shutdown_peer(upstream.peer_connection),
                return_exceptions=True,
            )

        return True

    async def close_all(self) -> None:
        """Close all relay sessions."""
        async with self._lock:
            device_ids = set(self._upstreams) | set(self._downstreams)
        for device_id in device_ids:
            await self.close_device(device_id)

    async def close_downstream(self, device_id: str, session_id: str) -> bool:
        """Close one downstream consumer and upstream if it was the last one."""
        async with self._lock:
            sessions = self._downstreams.get(device_id)
            downstream = sessions.pop(session_id, None) if sessions else None
            has_remaining = bool(sessions)
            if sessions is not None and not sessions:
                self._downstreams.pop(device_id, None)

        if downstream is None:
            return False

        await self._shutdown_peer(downstream.peer_connection)
        if not has_remaining:
            await self._close_upstream_if_unused(device_id)
        return True

    async def close_downstreams_by_kind(self, device_id: str, kind: str) -> bool:
        """Close downstream sessions matching kind and upstream if none remain."""
        async with self._lock:
            sessions = self._downstreams.get(device_id, {})
            matching_ids = [
                session_id
                for session_id, session in sessions.items()
                if session.kind == kind
            ]

        closed_any = False
        for session_id in matching_ids:
            closed_any = await self.close_downstream(device_id, session_id) or closed_any
        return closed_any

    async def add_downstream_candidate(
        self,
        device_id: str,
        session_id: str,
        candidate,
    ) -> bool:
        """Add trickled ICE candidate to a relay downstream peer."""
        if candidate_from_sdp is None:
            return False

        async with self._lock:
            downstream = self._downstreams.get(device_id, {}).get(session_id)

        if downstream is None:
            return False

        candidate_line = candidate.candidate or ""
        if not candidate_line:
            return True

        if candidate_line.startswith("candidate:"):
            candidate_line = candidate_line.removeprefix("candidate:")

        ice_candidate = candidate_from_sdp(candidate_line)
        ice_candidate.sdpMid = candidate.sdp_mid
        ice_candidate.sdpMLineIndex = candidate.sdp_m_line_index
        await downstream.peer_connection.addIceCandidate(ice_candidate)
        return True

    async def has_upstream(self, device_id: str) -> bool:
        """Return whether an active relay upstream already exists."""
        async with self._lock:
            upstream = self._upstreams.get(device_id)
        return (
            upstream is not None
            and upstream.video_ready.is_set()
            and upstream.peer_connection.connectionState not in {"failed", "closed"}
        )

    async def get_upstream_rtm(self, device_id: str) -> AgoraRTMSignaling | None:
        """Return the active upstream RTM session for one device."""
        async with self._lock:
            upstream = self._upstreams.get(device_id)
        if (
            upstream is None
            or not upstream.video_ready.is_set()
            or upstream.peer_connection.connectionState in {"failed", "closed"}
        ):
            return None
        return upstream.agora_rtm

    async def _ensure_upstream(
        self,
        camera: PetkitWebRTCCamera,
    ) -> MirrorUpstreamSession:
        """Ensure an internal ingest peer is running."""
        device_id = str(camera.device.id)
        async with self._lock:
            existing = self._upstreams.get(device_id)

        if (
            existing is not None
            and existing.video_ready.is_set()
            and existing.peer_connection.connectionState not in {"failed", "closed"}
        ):
            return existing

        if existing is not None:
            await self.close_device(device_id)

        live_feed = await _get_live_feed_for_webrtc(camera)
        if live_feed is None:
            raise RuntimeError("Live feed unavailable or missing RTM credentials")

        agora_user_id = _resolve_agora_user_id(camera, live_feed)
        async with AgoraAPIClient() as agora_client:
            agora_response = await agora_client.choose_server(
                app_id=AGORA_APP_ID,
                token=live_feed.rtc_token,
                channel_name=live_feed.channel_id,
                user_id=agora_user_id,
                service_flags=[
                    SERVICE_IDS["CHOOSE_SERVER"],
                    SERVICE_IDS["CLOUD_PROXY_FALLBACK"],
                ],
            )

        ice_servers = []
        for server in agora_response.get_ice_servers(use_all_turn_servers=False):
            ice_servers.append(
                AiortcIceServer(
                    urls=server.urls,
                    username=server.username,
                    credential=server.credential,
                )
            )

        peer_connection = RTCPeerConnection(
            RTCConfiguration(iceServers=ice_servers)
        )
        relay = MediaRelay()
        upstream = MirrorUpstreamSession(
            camera=camera,
            peer_connection=peer_connection,
            agora_handler=AgoraWebSocketHandler(
                rtc_token_provider=camera._refresh_rtc_token,
                prefer_instant_video=True,
                subscribe_retry_delay=1.0,
                subscribe_retry_attempts=3,
            ),
            agora_rtm=AgoraRTMSignaling(AGORA_APP_ID),
            relay=relay,
        )

        @peer_connection.on("track")
        def on_track(track: Any) -> None:
            LOGGER.debug(
                "WHEP mirror upstream %s track kind=%s",
                device_id,
                track.kind,
            )
            if track.kind == "video" and upstream.video_track is None:
                upstream.video_track = track
                upstream.video_ready.set()

            @track.on("ended")
            async def on_ended() -> None:
                LOGGER.debug(
                    "WHEP mirror upstream %s track ended kind=%s",
                    device_id,
                    track.kind,
                )
                self.hass.async_create_task(self.close_device(device_id))

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = peer_connection.connectionState
            LOGGER.debug("WHEP mirror upstream %s state=%s", device_id, state)
            if state in {"failed", "closed"}:
                upstream.last_error = f"upstream connection state={state}"
                self.hass.async_create_task(self.close_device(device_id))

        transceiver = peer_connection.addTransceiver("video", direction="recvonly")
        self._prefer_h264_transceiver(transceiver)

        offer = await peer_connection.createOffer()
        await peer_connection.setLocalDescription(offer)
        await self._wait_for_ice_complete(peer_connection)

        parsed_candidates = _add_offer_candidates(
            upstream.agora_handler,
            str(peer_connection.localDescription.sdp),
        )
        upstream.agora_handler.candidates = camera._filter_candidates(
            upstream.agora_handler.candidates,
            agora_response,
        )
        LOGGER.debug(
            "WHEP mirror upstream %s candidates=%d filtered=%d",
            device_id,
            parsed_candidates,
            len(upstream.agora_handler.candidates),
        )

        rtm_started = await upstream.agora_rtm.start_live(live_feed)
        if not rtm_started:
            LOGGER.warning(
                "WHEP mirror upstream %s RTM start_live not acknowledged",
                device_id,
            )

        answer_sdp = await upstream.agora_handler.connect_and_join(
            live_feed=live_feed,
            offer_sdp=str(peer_connection.localDescription.sdp),
            session_id=secrets.token_hex(16),
            app_id=AGORA_APP_ID,
            agora_response=agora_response,
        )
        if not answer_sdp:
            await asyncio.gather(
                upstream.agora_handler.disconnect(),
                upstream.agora_rtm.stop_live(send_stop=True),
                self._shutdown_peer(peer_connection),
                return_exceptions=True,
            )
            raise RuntimeError("Agora upstream negotiation failed")

        await peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type="answer")
        )
        await asyncio.wait_for(upstream.video_ready.wait(), timeout=20)
        upstream.refresh_task = self.hass.async_create_task(
            self._refresh_tokens(upstream)
        )

        async with self._lock:
            self._upstreams[device_id] = upstream

        return upstream

    async def _refresh_tokens(self, upstream: MirrorUpstreamSession) -> None:
        """Refresh RTM tokens while the upstream session is alive."""
        try:
            while True:
                await asyncio.sleep(TOKEN_REFRESH_INTERVAL_SECONDS)
                live_feed = await _get_live_feed_for_webrtc(upstream.camera)
                if live_feed is None:
                    continue
                await upstream.agora_rtm.update_tokens(live_feed)
        except asyncio.CancelledError:
            return

    async def _handle_downstream_closed(
        self,
        device_id: str,
        session_id: str,
    ) -> None:
        """Cleanup after downstream closure."""
        await self.close_downstream(device_id, session_id)

    async def _close_upstream_if_unused(self, device_id: str) -> None:
        """Close upstream relay only when no downstream consumers remain."""
        async with self._lock:
            if self._downstreams.get(device_id):
                return
            upstream = self._upstreams.pop(device_id, None)

        if upstream is None:
            return

        if upstream.refresh_task is not None:
            upstream.refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await upstream.refresh_task

        await asyncio.gather(
            upstream.agora_handler.disconnect(),
            upstream.agora_rtm.stop_live(send_stop=True),
            self._shutdown_peer(upstream.peer_connection),
            return_exceptions=True,
        )

    @staticmethod
    async def _shutdown_peer(peer_connection: Any) -> None:
        """Close one aiortc peer connection."""
        with contextlib.suppress(Exception):  # noqa: BLE001
            await peer_connection.close()

    @staticmethod
    async def _wait_for_ice_complete(peer_connection: Any) -> None:
        """Wait briefly for ICE gathering to finish."""
        if peer_connection.iceGatheringState == "complete":
            return

        ice_complete = asyncio.Event()

        @peer_connection.on("icegatheringstatechange")
        async def on_icegatheringstatechange() -> None:
            if peer_connection.iceGatheringState == "complete":
                ice_complete.set()

        await asyncio.wait_for(ice_complete.wait(), timeout=5)

    @staticmethod
    def _prefer_h264(peer_connection: Any, sender: Any) -> None:
        """Prefer H264 for downstream consumers when available."""
        try:
            transceiver = next(
                transceiver
                for transceiver in peer_connection.getTransceivers()
                if transceiver.sender == sender
            )
        except StopIteration:
            return

        PetkitMirrorRelayManager._prefer_h264_transceiver(transceiver)

    @staticmethod
    def _prefer_h264_transceiver(transceiver: Any) -> None:
        """Restrict codec preferences to H264 when supported."""
        if RTCRtpSender is None:
            return
        try:
            h264_codecs = [
                codec
                for codec in RTCRtpSender.getCapabilities("video").codecs
                if codec.mimeType == "video/H264"
            ]
            if h264_codecs:
                transceiver.setCodecPreferences(h264_codecs)
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Failed to set H264 codec preference: %s", err)


def _get_manager(hass) -> PetkitMirrorRelayManager:
    """Return the shared mirror relay manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("whep_mirror_manager")
    if manager is None:
        manager = PetkitMirrorRelayManager(hass)
        domain_data["whep_mirror_manager"] = manager
    return manager


async def async_cleanup_whep_mirror_sessions(hass) -> None:
    """Close all active mirror relay sessions."""
    manager = hass.data.get(DOMAIN, {}).pop("whep_mirror_manager", None)
    if manager is not None:
        await manager.close_all()


class PetkitWhepMirrorView(HomeAssistantView):
    """WHEP endpoint that relays one internal aiortc ingest."""

    url = "/api/petkit/whep_mirror/{device_id}"
    name = "api:petkit:whep_mirror"
    requires_auth = False

    async def post(self, request: web.Request, device_id: str) -> web.Response:
        """Receive SDP offer, relay via an internal aiortc peer, return answer."""
        hass = request.app["hass"]

        if not request.get("hass_user"):
            token = request.query.get("token")
            if token:
                if hass.auth.async_validate_access_token(token) is None:
                    return web.Response(status=401, text="Invalid token")
            else:
                return web.Response(status=401, text="Authentication required")

        cameras = hass.data.get(DOMAIN, {}).get("cameras", {})
        camera = cameras.get(device_id)
        if camera is None:
            return web.Response(status=404, text="Camera not found")

        offer_sdp = await request.text()
        if not offer_sdp or not offer_sdp.strip():
            return web.Response(status=400, text="Empty SDP offer")

        if AIORTC_IMPORT_ERROR is not None:
            return web.Response(
                status=503,
                text=f"aiortc relay unavailable: {AIORTC_IMPORT_ERROR}",
            )

        manager = _get_manager(hass)

        try:
            _, answer_sdp = await manager.create_downstream_offer(
                camera,
                offer_sdp,
                kind="whep",
            )
        except asyncio.TimeoutError:
            LOGGER.error("WHEP mirror timed out for %s", device_id)
            return web.Response(status=504, text="Timed out waiting for upstream video")
        except (OSError, RuntimeError, ValueError) as err:
            LOGGER.error("WHEP mirror failed for %s: %s", device_id, err)
            return web.Response(status=502, text=str(err))

        response = web.StreamResponse(
            status=201,
            headers={
                "Content-Type": "application/sdp",
                "Location": f"/api/petkit/whep_mirror/{device_id}",
            },
        )
        await response.prepare(request)
        await response.write(answer_sdp.encode())
        await response.write_eof()
        return response

    async def delete(self, request: web.Request, device_id: str) -> web.Response:
        """Tear down an active mirror session."""
        hass = request.app["hass"]

        if not request.get("hass_user"):
            token = request.query.get("token")
            if token:
                if hass.auth.async_validate_access_token(token) is None:
                    return web.Response(status=401, text="Invalid token")
            else:
                return web.Response(status=401, text="Authentication required")

        manager = _get_manager(hass)
        if not await manager.close_downstreams_by_kind(device_id, "whep"):
            return web.Response(status=404, text="No active mirror session")

        return web.Response(status=200, text="Session closed")
