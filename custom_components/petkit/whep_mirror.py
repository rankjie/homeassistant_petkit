"""Mirror WHEP endpoint using an internal aiortc relay."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import secrets
from typing import TYPE_CHECKING, Any

from .agora_api import SERVICE_IDS, AgoraAPIClient
from .agora_rtm import AgoraRTMSignaling
from .agora_websocket import AgoraWebSocketHandler
from .const import AGORA_APP_ID, DOMAIN, LOGGER
from .encoded_passthrough import (
    ensure_aiortc_decoder_passthrough_patch,
    replace_track_with_encoded_passthrough,
)
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

    @property
    def is_alive(self) -> bool:
        """Return whether the upstream session is ready and connected."""
        return (
            self.video_ready.is_set()
            and self.peer_connection.connectionState not in {"failed", "closed"}
        )


@dataclass
class MirrorDownstreamSession:
    """One downstream consumer served by the relay."""

    session_id: str
    peer_connection: Any


class PetkitMirrorRelayManager:
    """Manage internal upstream and downstream relay peers."""

    def __init__(self, hass) -> None:
        """Initialize browser relay session bookkeeping."""
        self.hass = hass
        self._lock = asyncio.Lock()
        self._upstreams: dict[str, MirrorUpstreamSession] = {}
        self._upstream_tasks: dict[str, asyncio.Task[MirrorUpstreamSession]] = {}
        self._downstreams: dict[str, dict[str, MirrorDownstreamSession]] = {}

    async def create_downstream_offer(
        self,
        camera: PetkitWebRTCCamera,
        offer_sdp: str,
        *,
        session_id: str | None = None,
    ) -> tuple[str, str]:
        """Create or reuse an upstream ingest, then answer one downstream offer."""
        device_id = str(camera.device.id)
        if session_id is not None:
            await self.close_downstream(device_id, session_id)
        upstream = await self._ensure_upstream(camera)

        peer_connection = RTCPeerConnection()
        if session_id is None:
            session_id = secrets.token_hex(16)
        downstream = MirrorDownstreamSession(
            session_id=session_id,
            peer_connection=peer_connection,
        )

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = peer_connection.connectionState
            LOGGER.debug(
                "Browser relay downstream %s state=%s",
                device_id,
                state,
            )
            if state in {"failed", "closed"}:
                self.hass.async_create_background_task(
                    self._handle_downstream_closed(device_id, session_id),
                    f"petkit browser relay close downstream {device_id}",
                )

        sender = peer_connection.addTrack(
            # Avoid unbounded per-subscriber frame queues when the downstream
            # encoder or consumer runs slower than the incoming stream.
            upstream.relay.subscribe(upstream.video_track, buffered=False)
        )
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
        ensure_task = None
        async with self._lock:
            downstreams = list(self._downstreams.pop(device_id, {}).values())
            upstream = self._upstreams.pop(device_id, None)
            ensure_task = self._upstream_tasks.pop(device_id, None)

        if ensure_task is not None and ensure_task is not asyncio.current_task():
            ensure_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ensure_task

        if not downstreams and upstream is None and ensure_task is None:
            return False

        if downstreams:
            await asyncio.gather(
                *(self._shutdown_peer(d.peer_connection) for d in downstreams),
                return_exceptions=True,
            )

        if upstream is not None:
            await self._shutdown_upstream(upstream)

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
        return upstream is not None and upstream.is_alive

    async def get_upstream_rtm(self, device_id: str) -> AgoraRTMSignaling | None:
        """Return the active upstream RTM session for one device."""
        async with self._lock:
            upstream = self._upstreams.get(device_id)
        if upstream is None or not upstream.is_alive:
            return None
        return upstream.agora_rtm

    async def maybe_close_idle_upstream(self, device_id: str) -> bool:
        """Close the upstream when no tracked downstreams remain."""
        had_upstream = await self.has_upstream(device_id)
        await self._close_upstream_if_unused(device_id)
        return had_upstream and not await self.has_upstream(device_id)

    async def _ensure_upstream(
        self,
        camera: PetkitWebRTCCamera,
    ) -> MirrorUpstreamSession:
        """Ensure an internal ingest peer is running."""
        device_id = str(camera.device.id)
        async with self._lock:
            existing = self._upstreams.get(device_id)
            if existing is not None and existing.is_alive:
                return existing
            ensure_task = self._upstream_tasks.get(device_id)
            if ensure_task is None:
                ensure_task = self.hass.async_create_task(self._create_upstream(camera))
                self._upstream_tasks[device_id] = ensure_task

        return await ensure_task

    async def _create_upstream(
        self,
        camera: PetkitWebRTCCamera,
    ) -> MirrorUpstreamSession:
        """Create one internal ingest peer connected to Agora."""
        device_id = str(camera.device.id)
        async with self._lock:
            existing = self._upstreams.get(device_id)
        if existing is not None:
            await self.close_device(device_id)

        try:
            return await self._build_upstream(camera)
        finally:
            async with self._lock:
                current_task = self._upstream_tasks.get(device_id)
                if current_task is asyncio.current_task():
                    self._upstream_tasks.pop(device_id, None)

    async def _build_upstream(
        self,
        camera: PetkitWebRTCCamera,
    ) -> MirrorUpstreamSession:
        """Negotiate the upstream aiortc ingest peer against Agora."""
        device_id = str(camera.device.id)
        ensure_aiortc_decoder_passthrough_patch()

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

        ice_servers = [
            AiortcIceServer(
                urls=server.urls,
                username=server.username,
                credential=server.credential,
            )
            for server in agora_response.get_ice_servers(use_all_turn_servers=False)
        ]

        peer_connection = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        relay = MediaRelay()
        upstream = MirrorUpstreamSession(
            camera=camera,
            peer_connection=peer_connection,
            agora_handler=AgoraWebSocketHandler(
                rtc_token_provider=camera.async_refresh_rtc_token,
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
                "Browser relay upstream %s track kind=%s",
                device_id,
                track.kind,
            )
            if track.kind == "video" and upstream.video_track is None:
                packet_track = replace_track_with_encoded_passthrough(
                    peer_connection,
                    track,
                    on_started=upstream.video_ready.set,
                )
                upstream.video_track = packet_track or track
                if packet_track is None:
                    upstream.video_ready.set()

            @track.on("ended")
            async def on_ended() -> None:
                LOGGER.debug(
                    "Browser relay upstream %s track ended kind=%s",
                    device_id,
                    track.kind,
                )
                self.hass.async_create_background_task(
                    self.close_device(device_id),
                    f"petkit browser relay close device {device_id}",
                )

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = peer_connection.connectionState
            LOGGER.debug("Browser relay upstream %s state=%s", device_id, state)
            if state in {"failed", "closed"}:
                upstream.last_error = f"upstream connection state={state}"
                self.hass.async_create_background_task(
                    self.close_device(device_id),
                    f"petkit browser relay close device {device_id}",
                )

        transceiver = peer_connection.addTransceiver("video", direction="recvonly")
        self._prefer_h264_transceiver(transceiver)

        offer = await peer_connection.createOffer()
        await peer_connection.setLocalDescription(offer)
        await self._wait_for_ice_complete(peer_connection)

        parsed_candidates = _add_offer_candidates(
            upstream.agora_handler,
            str(peer_connection.localDescription.sdp),
        )
        upstream.agora_handler.candidates = camera.filter_agora_candidates(
            upstream.agora_handler.candidates,
            agora_response,
        )
        LOGGER.debug(
            "Browser relay upstream %s candidates=%d filtered=%d",
            device_id,
            parsed_candidates,
            len(upstream.agora_handler.candidates),
        )

        rtm_started = await upstream.agora_rtm.start_live(live_feed)
        if not rtm_started:
            LOGGER.debug(
                "Browser relay upstream %s RTM start_live not acknowledged",
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
        upstream.refresh_task = self.hass.async_create_background_task(
            self._refresh_tokens(upstream),
            f"petkit browser relay refresh tokens {device_id}",
        )

        async with self._lock:
            self._upstreams[device_id] = upstream

        return upstream

    async def _refresh_tokens(self, upstream: MirrorUpstreamSession) -> None:
        """Refresh RTM tokens while the upstream session is alive."""
        while True:
            await asyncio.sleep(TOKEN_REFRESH_INTERVAL_SECONDS)
            live_feed = await _get_live_feed_for_webrtc(upstream.camera)
            if live_feed is None:
                continue
            await upstream.agora_rtm.update_tokens(live_feed)

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

        await self._shutdown_upstream(upstream)

    async def _shutdown_upstream(self, upstream: MirrorUpstreamSession) -> None:
        """Cancel refresh task and tear down one upstream session."""
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
        with contextlib.suppress(Exception):
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
    """Return the shared browser relay manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("whep_mirror_manager")
    if manager is None:
        manager = PetkitMirrorRelayManager(hass)
        domain_data["whep_mirror_manager"] = manager
    return manager


async def async_cleanup_whep_mirror_sessions(hass) -> None:
    """Close all active browser relay sessions."""
    manager = hass.data.get(DOMAIN, {}).pop("whep_mirror_manager", None)
    if manager is not None:
        await manager.close_all()
