"""WHEP endpoints for PetKit upstream ingest and shared go2rtc output."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from http import HTTPStatus
import secrets
from typing import TYPE_CHECKING

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from sdp_transform import parse as sdp_parse
from webrtc_models import RTCIceCandidateInit

from homeassistant.components.http import HomeAssistantView
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .agora_rtm import AgoraRTMSignaling
from .agora_websocket import AgoraWebSocketHandler
from .const import AGORA_APP_ID, DOMAIN, LOGGER
from .go2rtc_stream import get_go2rtc_stream_manager
from .webrtc_common import _get_live_feed_for_webrtc

if TYPE_CHECKING:
    from .camera import PetkitWebRTCCamera


TOKEN_REFRESH_INTERVAL_SECONDS = 20 * 60
_HA_MANAGED_URL = "http://127.0.0.1:11984/"
_GO2RTC_WHEP_PATH = "api/webrtc"
_REQUEST_TIMEOUT = ClientTimeout(total=15)


def _check_external_auth(request: web.Request) -> web.Response | None:
    """Allow authenticated HA users or explicit access tokens."""
    hass = request.app["hass"]
    if request.get("hass_user"):
        return None

    token = request.query.get("token")
    if token:
        if hass.auth.async_validate_access_token(token) is None:
            return web.Response(status=401, text="Invalid token")
        return None

    return web.Response(status=401, text="Authentication required")


@dataclass
class AgoraUpstreamSession:
    """One direct Agora session used by the shared go2rtc stream."""

    session_id: str
    camera: PetkitWebRTCCamera
    agora_handler: AgoraWebSocketHandler
    agora_rtm: AgoraRTMSignaling
    refresh_task: asyncio.Task[None] | None = None
    location_path: str = ""


@dataclass
class Go2RTCProxySession:
    """One public WHEP session proxied to the internal go2rtc stream."""

    session_id: str
    device_id: str
    upstream_location: str


class PetkitAgoraUpstreamManager:
    """Manage one direct Agora session per device for go2rtc ingest."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._sessions: dict[str, AgoraUpstreamSession] = {}

    async def create_session(
        self,
        camera: PetkitWebRTCCamera,
        offer_sdp: str,
    ) -> tuple[str, str]:
        """Create or replace one direct Agora session for the device."""
        device_id = str(camera.device.id)
        await self.close_session(device_id)

        live_feed = await camera._async_get_live_feed(refresh=True)
        if live_feed is None:
            raise RuntimeError("Live feed unavailable or missing RTM credentials")

        await camera._refresh_agora_context(live_feed)
        if camera._agora_response is None:
            raise RuntimeError("Failed to retrieve Agora edge servers")

        agora_rtm = AgoraRTMSignaling(AGORA_APP_ID)

        async def refresh_rtc_token() -> str | None:
            refreshed_live_feed = await camera._async_get_live_feed(refresh=True)
            if refreshed_live_feed is None or not refreshed_live_feed.rtc_token:
                return None
            await agora_rtm.update_tokens(refreshed_live_feed)
            return refreshed_live_feed.rtc_token

        agora_handler = AgoraWebSocketHandler(
            rtc_token_provider=refresh_rtc_token,
            prefer_instant_video=True,
            subscribe_retry_delay=1.0,
            subscribe_retry_attempts=3,
            declare_remote_video_ssrc=True,
            disable_audio_answer=True,
        )
        for line in offer_sdp.splitlines():
            stripped = line.strip()
            if stripped.startswith("a=candidate:"):
                agora_handler.add_ice_candidate(
                    RTCIceCandidateInit(candidate=stripped.removeprefix("a="))
                )

        agora_handler.candidates = camera.filter_agora_candidates(
            agora_handler.candidates,
            camera._agora_response,
        )

        rtm_started = await agora_rtm.start_live(live_feed)
        if not rtm_started:
            LOGGER.warning(
                "go2rtc upstream start_live/heartbeat not active for %s",
                device_id,
            )

        session_id = secrets.token_hex(16)
        try:
            answer_sdp = await agora_handler.connect_and_join(
                live_feed=live_feed,
                offer_sdp=offer_sdp,
                session_id=session_id,
                app_id=AGORA_APP_ID,
                agora_response=camera._agora_response,
            )
        except Exception:
            await asyncio.gather(
                agora_handler.disconnect(),
                agora_rtm.stop_live(send_stop=True),
                return_exceptions=True,
            )
            raise

        if not answer_sdp:
            await asyncio.gather(
                agora_handler.disconnect(),
                agora_rtm.stop_live(send_stop=True),
                return_exceptions=True,
            )
            raise RuntimeError("Agora upstream negotiation did not return an SDP answer")

        session = AgoraUpstreamSession(
            session_id=session_id,
            camera=camera,
            agora_handler=agora_handler,
            agora_rtm=agora_rtm,
            location_path=f"/api/petkit/whep_upstream/{device_id}/{session_id}",
        )
        session.refresh_task = self.hass.async_create_background_task(
            self._refresh_tokens(session),
            f"petkit go2rtc upstream refresh {device_id}",
        )

        async with self._lock:
            self._sessions[device_id] = session

        return session_id, answer_sdp

    async def close_session(self, device_id: str) -> bool:
        """Close one direct Agora session."""
        async with self._lock:
            session = self._sessions.pop(device_id, None)

        if session is None:
            return False

        if session.refresh_task is not None:
            session.refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.refresh_task

        await asyncio.gather(
            session.agora_handler.disconnect(),
            session.agora_rtm.stop_live(send_stop=True),
            return_exceptions=True,
        )
        return True

    async def add_session_candidates(
        self,
        device_id: str,
        session_id: str,
        sdp_fragment: str,
    ) -> bool:
        """Forward trickled ICE candidates for one active upstream session."""
        async with self._lock:
            session = self._sessions.get(device_id)
        if session is None or session.session_id != session_id:
            return False

        added = 0
        for candidate in _parse_trickle_candidates(sdp_fragment):
            session.agora_handler.add_ice_candidate(candidate)
            added += 1

        if added:
            LOGGER.debug(
                "Collected %d upstream PATCH candidates for %s",
                added,
                device_id,
            )
        return True

    async def close_all(self) -> None:
        """Close all active upstream sessions."""
        async with self._lock:
            device_ids = list(self._sessions)
        for device_id in device_ids:
            await self.close_session(device_id)

    async def _refresh_tokens(self, session: AgoraUpstreamSession) -> None:
        """Refresh RTM tokens while the upstream session is alive."""
        while True:
            await asyncio.sleep(TOKEN_REFRESH_INTERVAL_SECONDS)
            live_feed = await _get_live_feed_for_webrtc(session.camera)
            if live_feed is None:
                continue
            await session.agora_rtm.update_tokens(live_feed)


class PetkitGo2RTCProxyManager:
    """Proxy public WHEP sessions to the internal shared go2rtc stream."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._sessions: dict[tuple[str, str], Go2RTCProxySession] = {}

    @property
    def _session(self) -> ClientSession:
        return async_get_clientsession(self.hass)

    @property
    def _base_url(self) -> str:
        return _HA_MANAGED_URL

    async def create_session(
        self,
        device_id: str,
        offer_sdp: str,
        headers: web.BaseRequest.headers,
    ) -> tuple[str, str]:
        """Create one public WHEP session against the shared go2rtc stream."""
        stream_manager = get_go2rtc_stream_manager(self.hass)
        await stream_manager.async_ensure_stream(device_id, raise_on_failure=True)

        response = await self._request(
            "POST",
            self._stream_url(device_id),
            body=offer_sdp.encode(),
            headers=headers,
        )
        if response.status not in (HTTPStatus.OK, HTTPStatus.CREATED):
            raise RuntimeError(
                f"go2rtc WHEP setup failed with status {response.status}: {response.body_text}"
            )

        upstream_location = response.headers.get("Location")
        if not upstream_location:
            raise RuntimeError("go2rtc WHEP setup did not return a session location")

        proxy_session_id = secrets.token_hex(16)
        async with self._lock:
            self._sessions[(device_id, proxy_session_id)] = Go2RTCProxySession(
                session_id=proxy_session_id,
                device_id=device_id,
                upstream_location=self._normalize_location(upstream_location),
            )

        return proxy_session_id, response.body_text

    async def proxy_session_request(
        self,
        device_id: str,
        session_id: str,
        method: str,
        *,
        body: bytes = b"",
        headers: web.BaseRequest.headers,
        forget: bool = False,
    ) -> "_ProxyResponse | None":
        """Proxy one session-scoped request to go2rtc if the session exists."""
        async with self._lock:
            session = self._sessions.get((device_id, session_id))
        if session is None:
            return None

        response = await self._request(
            method,
            session.upstream_location,
            body=body,
            headers=headers,
        )

        if forget or method == "DELETE":
            async with self._lock:
                self._sessions.pop((device_id, session_id), None)

        return response

    async def close_all(self) -> None:
        """Close all active proxied go2rtc sessions."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            with contextlib.suppress(Exception):
                await self._request("DELETE", session.upstream_location, headers={})

    def _stream_url(self, device_id: str) -> str:
        """Return the internal go2rtc WHEP viewer URL for one device stream."""
        stream_name = get_go2rtc_stream_manager(self.hass).stream_name(device_id)
        return f"{self._base_url}{_GO2RTC_WHEP_PATH}?src={stream_name}"

    def _normalize_location(self, location: str) -> str:
        """Normalize a go2rtc Location header into an absolute internal URL."""
        if location.startswith(("http://", "https://")):
            return location
        if location.startswith("/"):
            return f"{self._base_url.rstrip('/')}{location}"
        return f"{self._base_url}{location}"

    async def _request(
        self,
        method: str,
        url: str,
        *,
        body: bytes = b"",
        headers,
    ) -> "_ProxyResponse":
        """Forward one HTTP request to go2rtc."""
        forward_headers = _filter_proxy_headers(headers)
        try:
            async with self._session.request(
                method,
                url,
                data=body if body else None,
                headers=forward_headers,
                timeout=_REQUEST_TIMEOUT,
            ) as response:
                raw_body = await response.read()
                response_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() in {"content-type", "etag", "location"}
                }
                return _ProxyResponse(
                    status=response.status,
                    body=raw_body,
                    body_text=raw_body.decode(errors="ignore"),
                    headers=response_headers,
                    content_type=response.content_type,
                )
        except (ClientError, TimeoutError) as err:
            raise RuntimeError(f"go2rtc proxy request failed: {err}") from err


@dataclass(frozen=True)
class _ProxyResponse:
    """Small transport object for proxied go2rtc responses."""

    status: int
    body: bytes
    body_text: str
    headers: dict[str, str]
    content_type: str | None


def _filter_proxy_headers(headers) -> dict[str, str]:
    """Forward only the headers relevant to WHEP semantics."""
    return {
        key: value
        for key, value in headers.items()
        if key.lower() in {"content-type", "accept", "if-match"}
    }


def _get_upstream_manager(hass) -> PetkitAgoraUpstreamManager:
    """Return the shared upstream manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("whep_upstream_manager")
    if manager is None:
        manager = PetkitAgoraUpstreamManager(hass)
        domain_data["whep_upstream_manager"] = manager
    return manager


def _get_proxy_manager(hass) -> PetkitGo2RTCProxyManager:
    """Return the shared public WHEP proxy manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("whep_proxy_manager")
    if manager is None:
        manager = PetkitGo2RTCProxyManager(hass)
        domain_data["whep_proxy_manager"] = manager
    return manager


def _parse_trickle_candidates(sdp_fragment: str) -> list[RTCIceCandidateInit]:
    """Extract trickled ICE candidates from a WHEP SDP fragment."""
    try:
        parsed = sdp_parse(sdp_fragment)
    except Exception:  # noqa: BLE001
        return []

    candidates: list[RTCIceCandidateInit] = []
    for media in parsed.get("media", []) or []:
        mid = media.get("mid")
        mline_index = media.get("mLineIndex")
        for candidate in media.get("candidates", []) or []:
            foundation = candidate.get("foundation", "0")
            component = candidate.get("component", 1)
            transport = candidate.get("transport", "udp")
            priority = candidate.get("priority", 0)
            ip = candidate.get("ip", "")
            port = candidate.get("port", 0)
            candidate_type = candidate.get("type", "host")
            candidate_line = (
                f"candidate:{foundation} {component} {transport} "
                f"{priority} {ip} {port} typ {candidate_type}"
            )
            candidates.append(
                RTCIceCandidateInit(
                    candidate=candidate_line,
                    sdp_mid=str(mid) if mid is not None else None,
                    sdp_m_line_index=int(mline_index)
                    if isinstance(mline_index, int)
                    else None,
                )
            )
    return candidates


async def async_cleanup_whep_proxy_sessions(hass) -> None:
    """Close all active WHEP proxy state."""
    proxy_manager = hass.data.get(DOMAIN, {}).pop("whep_proxy_manager", None)
    if proxy_manager is not None:
        await proxy_manager.close_all()

    upstream_manager = hass.data.get(DOMAIN, {}).pop("whep_upstream_manager", None)
    if upstream_manager is not None:
        await upstream_manager.close_all()


class PetkitUpstreamWhepView(HomeAssistantView):
    """Internal WHEP endpoint used by the shared go2rtc stream."""

    url = "/api/petkit/whep_upstream/{device_id}"
    name = "api:petkit:whep_upstream"
    requires_auth = False

    async def post(self, request: web.Request, device_id: str) -> web.Response:
        """Receive the internal go2rtc SDP offer and return the Agora SDP answer."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        hass = request.app["hass"]
        cameras = hass.data.get(DOMAIN, {}).get("cameras", {})
        camera = cameras.get(device_id)
        if camera is None:
            return web.Response(status=404, text="Camera not found")

        offer_sdp = await request.text()
        if not offer_sdp or not offer_sdp.strip():
            return web.Response(status=400, text="Empty SDP offer")

        try:
            session_id, answer_sdp = await _get_upstream_manager(hass).create_session(
                camera, offer_sdp
            )
        except (OSError, RuntimeError, ValueError) as err:
            LOGGER.error("go2rtc upstream WHEP failed for %s: %s", device_id, err)
            return web.Response(status=502, text=str(err))

        return web.Response(
            status=201,
            text=answer_sdp,
            content_type="application/sdp",
            headers={"Location": f"{request.path}/{session_id}"},
        )


class PetkitUpstreamWhepSessionView(HomeAssistantView):
    """Session-scoped internal upstream WHEP resource."""

    url = "/api/petkit/whep_upstream/{device_id}/{session_id}"
    name = "api:petkit:whep_upstream:session"
    requires_auth = False

    async def patch(
        self,
        request: web.Request,
        device_id: str,
        session_id: str,
    ) -> web.Response:
        """Accept trickled ICE candidates for one active upstream session."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        body = await request.text()
        if not await _get_upstream_manager(request.app["hass"]).add_session_candidates(
            device_id,
            session_id,
            body,
        ):
            return web.Response(status=404, text="No active upstream WHEP session")
        return web.Response(status=204)

    async def delete(
        self,
        request: web.Request,
        device_id: str,
        session_id: str,
    ) -> web.Response:
        """Tear down the active upstream WHEP session."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        if not await _get_upstream_manager(request.app["hass"]).close_session(device_id):
            return web.Response(status=404, text="No active upstream WHEP session")

        return web.Response(status=200, text="Session closed")


class PetkitDirectWhepProxyView(HomeAssistantView):
    """Public stable WHEP endpoint backed by the shared internal go2rtc stream."""

    url = "/api/petkit/whep_direct/{device_id}"
    name = "api:petkit:whep_direct"
    requires_auth = False

    async def post(self, request: web.Request, device_id: str) -> web.Response:
        """Receive SDP offer and return the shared go2rtc SDP answer."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        hass = request.app["hass"]
        cameras = hass.data.get(DOMAIN, {}).get("cameras", {})
        camera = cameras.get(device_id)
        if camera is None:
            return web.Response(status=404, text="Camera not found")

        offer_sdp = await request.text()
        if not offer_sdp or not offer_sdp.strip():
            return web.Response(status=400, text="Empty SDP offer")

        stream_manager = get_go2rtc_stream_manager(hass)
        if stream_manager.is_managed_available():
            try:
                session_id, answer_sdp = await _get_proxy_manager(hass).create_session(
                    device_id,
                    offer_sdp,
                    request.headers,
                )
            except RuntimeError as err:
                LOGGER.error("Direct WHEP proxy failed for %s: %s", device_id, err)
                return web.Response(status=502, text=str(err))

            return web.Response(
                status=201,
                text=answer_sdp,
                content_type="application/sdp",
                headers={"Location": f"{request.path}/{session_id}"},
            )

        try:
            session_id, answer_sdp = await _get_upstream_manager(hass).create_session(
                camera,
                offer_sdp,
            )
        except (OSError, RuntimeError, ValueError) as err:
            LOGGER.error("Direct WHEP fallback failed for %s: %s", device_id, err)
            return web.Response(status=502, text=str(err))

        return web.Response(
            status=201,
            text=answer_sdp,
            content_type="application/sdp",
            headers={"Location": f"{request.path}/{session_id}"},
        )


class PetkitDirectWhepProxySessionView(HomeAssistantView):
    """Session-scoped public WHEP resource for PATCH / DELETE."""

    url = "/api/petkit/whep_direct/{device_id}/{session_id}"
    name = "api:petkit:whep_direct:session"
    requires_auth = False

    async def patch(
        self,
        request: web.Request,
        device_id: str,
        session_id: str,
    ) -> web.Response:
        """Handle trickled ICE candidates for the public shared WHEP session."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        body = await request.read()
        proxied = await _get_proxy_manager(request.app["hass"]).proxy_session_request(
            device_id,
            session_id,
            "PATCH",
            body=body,
            headers=request.headers,
        )
        if proxied is not None:
            return web.Response(status=proxied.status, headers=proxied.headers)

        text_body = body.decode(errors="ignore")
        if not await _get_upstream_manager(request.app["hass"]).add_session_candidates(
            device_id,
            session_id,
            text_body,
        ):
            return web.Response(status=404, text="No active direct WHEP session")
        return web.Response(status=204)

    async def delete(
        self,
        request: web.Request,
        device_id: str,
        session_id: str,
    ) -> web.Response:
        """Handle session teardown for the public shared WHEP session."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        proxied = await _get_proxy_manager(request.app["hass"]).proxy_session_request(
            device_id,
            session_id,
            "DELETE",
            headers=request.headers,
            forget=True,
        )
        if proxied is not None:
            return web.Response(status=proxied.status, headers=proxied.headers)

        if not await _get_upstream_manager(request.app["hass"]).close_session(device_id):
            return web.Response(status=404, text="No active direct WHEP session")

        return web.Response(status=200, text="Session closed")
