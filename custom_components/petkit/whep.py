"""WHEP endpoint for PetKit camera streams.

Allows go2rtc and other standard WebRTC consumers to negotiate a media
session with PetKit cameras via HTTP-based SDP offer/answer exchange.
Each WHEP session creates its own independent Agora signaling instances.
"""

from __future__ import annotations

import asyncio
import secrets
from typing import TYPE_CHECKING

from aiohttp import web

from homeassistant.components.http import HomeAssistantView

from .agora_api import SERVICE_IDS, AgoraAPIClient, AgoraResponse
from .agora_rtm import AgoraRTMSignaling
from .agora_websocket import AgoraWebSocketHandler
from .const import AGORA_APP_ID, DOMAIN, LOGGER

if TYPE_CHECKING:
    from .camera import PetkitWebRTCCamera


class WhepSession:
    """Track one WHEP session's signaling resources."""

    def __init__(
        self,
        rtm: AgoraRTMSignaling,
        handler: AgoraWebSocketHandler,
        agora_response: AgoraResponse,
    ) -> None:
        self.rtm = rtm
        self.handler = handler
        self.agora_response = agora_response

    async def close(self) -> None:
        """Tear down RTM + WebSocket."""
        results = await asyncio.gather(
            self.rtm.stop_live(send_stop=True),
            self.handler.disconnect(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                LOGGER.debug("WHEP session cleanup error: %s", result)


class PetkitWhepView(HomeAssistantView):
    """WHEP signaling endpoint for go2rtc and external WebRTC consumers."""

    url = "/api/petkit/whep/{device_id}"
    name = "api:petkit:whep"
    requires_auth = False

    async def post(self, request: web.Request, device_id: str) -> web.Response:
        """Receive SDP offer, perform Agora signaling, return SDP answer."""
        hass = request.app["hass"]

        # Auth: accept standard Bearer header OR ?token= query param
        if not request.get("hass_user"):
            token = request.query.get("token")
            if token:
                refresh_token = hass.auth.async_validate_access_token(token)
                if refresh_token is None:
                    return web.Response(status=401, text="Invalid token")
            else:
                return web.Response(status=401, text="Authentication required")

        cameras: dict[str, PetkitWebRTCCamera] = hass.data.get(DOMAIN, {}).get(
            "cameras", {}
        )
        camera = cameras.get(device_id)
        if camera is None:
            return web.Response(status=404, text="Camera not found")

        offer_sdp = await request.text()
        if not offer_sdp or not offer_sdp.strip():
            return web.Response(status=400, text="Empty SDP offer")

        # Close any existing WHEP session for this device
        whep_sessions: dict[str, WhepSession] = hass.data.setdefault(
            DOMAIN, {}
        ).setdefault("whep_sessions", {})
        existing = whep_sessions.pop(device_id, None)
        if existing:
            await existing.close()

        try:
            live_feed = await camera._get_live_feed()
            if live_feed is None:
                return web.Response(status=503, text="Live feed unavailable")

            async with AgoraAPIClient() as agora_client:
                agora_response = await agora_client.choose_server(
                    app_id=AGORA_APP_ID,
                    token=live_feed.rtc_token,
                    channel_name=live_feed.channel_id,
                    user_id=live_feed.uid,
                    service_flags=[
                        SERVICE_IDS["CHOOSE_SERVER"],
                        SERVICE_IDS["CLOUD_PROXY_FALLBACK"],
                    ],
                )

            rtm = AgoraRTMSignaling(AGORA_APP_ID)
            handler = AgoraWebSocketHandler()

            rtm_started = await rtm.start_live(live_feed)
            if not rtm_started:
                LOGGER.warning(
                    "WHEP: RTM start_live failed for device %s", device_id
                )

            session_id = secrets.token_hex(16)
            answer_sdp = await handler.connect_and_join(
                live_feed=live_feed,
                offer_sdp=offer_sdp,
                session_id=session_id,
                app_id=AGORA_APP_ID,
                agora_response=agora_response,
            )

            if not answer_sdp:
                await asyncio.gather(
                    rtm.stop_live(send_stop=True),
                    handler.disconnect(),
                    return_exceptions=True,
                )
                return web.Response(status=502, text="Agora negotiation failed")

            whep_sessions[device_id] = WhepSession(rtm, handler, agora_response)

            return web.Response(
                status=201,
                body=answer_sdp,
                content_type="application/sdp",
                headers={"Location": f"/api/petkit/whep/{device_id}"},
            )

        except (OSError, ValueError, RuntimeError) as err:
            LOGGER.error("WHEP signaling failed for %s: %s", device_id, err)
            return web.Response(status=502, text=str(err))

    async def delete(self, request: web.Request, device_id: str) -> web.Response:
        """Tear down an active WHEP session."""
        hass = request.app["hass"]

        if not request.get("hass_user"):
            token = request.query.get("token")
            if token:
                refresh_token = hass.auth.async_validate_access_token(token)
                if refresh_token is None:
                    return web.Response(status=401, text="Invalid token")
            else:
                return web.Response(status=401, text="Authentication required")

        whep_sessions: dict[str, WhepSession] = hass.data.get(DOMAIN, {}).get(
            "whep_sessions", {}
        )
        session = whep_sessions.pop(device_id, None)
        if session is None:
            return web.Response(status=404, text="No active WHEP session")

        await session.close()
        return web.Response(status=200, text="Session closed")


async def async_cleanup_whep_sessions(hass) -> None:
    """Close all active WHEP sessions."""
    whep_sessions: dict[str, WhepSession] = hass.data.get(DOMAIN, {}).pop(
        "whep_sessions", {}
    )
    for device_id, session in whep_sessions.items():
        LOGGER.debug("Closing WHEP session for device %s", device_id)
        await session.close()
