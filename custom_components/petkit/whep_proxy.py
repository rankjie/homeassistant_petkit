"""Pure-signaling WHEP proxy for external receivers such as go2rtc."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import secrets
from typing import TYPE_CHECKING

from aiohttp import web
from webrtc_models import RTCIceCandidateInit

from homeassistant.components.http import HomeAssistantView

from .agora_rtm import AgoraRTMSignaling
from .agora_websocket import AgoraWebSocketHandler
from .const import AGORA_APP_ID, DOMAIN, LOGGER
from .whep_mirror import TOKEN_REFRESH_INTERVAL_SECONDS, _check_external_auth
from .webrtc_common import _get_live_feed_for_webrtc

if TYPE_CHECKING:
    from .camera import PetkitWebRTCCamera


@dataclass
class DirectWhepSession:
    """One direct Agora session proxied for an external WHEP client."""

    session_id: str
    camera: PetkitWebRTCCamera
    agora_handler: AgoraWebSocketHandler
    agora_rtm: AgoraRTMSignaling
    refresh_task: asyncio.Task[None] | None = None


class PetkitDirectWhepProxyManager:
    """Manage direct signaling-only WHEP sessions."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._sessions: dict[str, DirectWhepSession] = {}

    async def create_session(
        self,
        camera: PetkitWebRTCCamera,
        offer_sdp: str,
    ) -> tuple[str, str]:
        """Create or replace one direct external WHEP session."""
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

        agora_handler = AgoraWebSocketHandler(rtc_token_provider=refresh_rtc_token)
        for line in offer_sdp.splitlines():
            stripped = line.strip()
            if stripped.startswith("a=candidate:"):
                agora_handler.add_ice_candidate(
                    RTCIceCandidateInit(candidate=stripped.removeprefix("a="))
                )

        # Inline candidates are all we currently proxy to Agora. We accept PATCH
        # later for WHEP compatibility, but go2rtc should already include enough
        # ICE candidates in the initial offer for the first PoC.
        agora_handler.candidates = camera.filter_agora_candidates(
            agora_handler.candidates,
            camera._agora_response,
        )

        rtm_started = await agora_rtm.start_live(live_feed)
        if not rtm_started:
            LOGGER.warning(
                "Direct WHEP proxy start_live/heartbeat not active for %s",
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
            raise RuntimeError("Agora direct negotiation did not return an SDP answer")

        session = DirectWhepSession(
            session_id=session_id,
            camera=camera,
            agora_handler=agora_handler,
            agora_rtm=agora_rtm,
        )
        session.refresh_task = self.hass.async_create_background_task(
            self._refresh_tokens(session),
            f"petkit direct whep refresh {device_id}",
        )

        async with self._lock:
            self._sessions[device_id] = session

        return session_id, answer_sdp

    async def close_session(self, device_id: str) -> bool:
        """Close one direct external WHEP session."""
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

    async def close_all(self) -> None:
        """Close all active direct WHEP sessions."""
        async with self._lock:
            device_ids = list(self._sessions)
        for device_id in device_ids:
            await self.close_session(device_id)

    async def _refresh_tokens(self, session: DirectWhepSession) -> None:
        """Refresh RTM tokens while the direct session is alive."""
        while True:
            await asyncio.sleep(TOKEN_REFRESH_INTERVAL_SECONDS)
            live_feed = await _get_live_feed_for_webrtc(session.camera)
            if live_feed is None:
                continue
            await session.agora_rtm.update_tokens(live_feed)


def _get_manager(hass) -> PetkitDirectWhepProxyManager:
    """Return the shared direct WHEP proxy manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("whep_proxy_manager")
    if manager is None:
        manager = PetkitDirectWhepProxyManager(hass)
        domain_data["whep_proxy_manager"] = manager
    return manager


async def async_cleanup_whep_proxy_sessions(hass) -> None:
    """Close all active direct WHEP proxy sessions."""
    manager = hass.data.get(DOMAIN, {}).pop("whep_proxy_manager", None)
    if manager is not None:
        await manager.close_all()


class PetkitDirectWhepProxyView(HomeAssistantView):
    """Public WHEP endpoint that only proxies signaling to Agora."""

    url = "/api/petkit/whep_direct/{device_id}"
    name = "api:petkit:whep_direct"
    requires_auth = False

    async def post(self, request: web.Request, device_id: str) -> web.Response:
        """Receive SDP offer and return the direct Agora SDP answer."""
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
            _, answer_sdp = await _get_manager(hass).create_session(camera, offer_sdp)
        except (OSError, RuntimeError, ValueError) as err:
            LOGGER.error("Direct WHEP proxy failed for %s: %s", device_id, err)
            return web.Response(status=502, text=str(err))

        return web.Response(
            status=201,
            text=answer_sdp,
            content_type="application/sdp",
            headers={"Location": request.path},
        )

    async def patch(self, request: web.Request, device_id: str) -> web.Response:
        """Accept trickled ICE patches for WHEP compatibility."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        body = await request.text()
        if body.strip():
            LOGGER.debug(
                "Ignoring external WHEP PATCH candidates for %s in direct proxy PoC",
                device_id,
            )
        return web.Response(status=204)

    async def delete(self, request: web.Request, device_id: str) -> web.Response:
        """Tear down the active direct WHEP session."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        if not await _get_manager(request.app["hass"]).close_session(device_id):
            return web.Response(status=404, text="No active direct WHEP session")

        return web.Response(status=200, text="Session closed")
