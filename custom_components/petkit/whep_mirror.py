"""Mirror WHEP endpoint using the camera entity's browser WebRTC path."""

from __future__ import annotations

import secrets

from aiohttp import web

from homeassistant.components.camera import WebRTCAnswer, WebRTCError
from homeassistant.components.http import HomeAssistantView

from .const import DOMAIN, LOGGER


class PetkitWhepMirrorView(HomeAssistantView):
    """WHEP endpoint that delegates to camera entity's async_handle_async_webrtc_offer."""

    url = "/api/petkit/whep_mirror/{device_id}"
    name = "api:petkit:whep_mirror"
    requires_auth = False

    async def post(self, request: web.Request, device_id: str) -> web.Response:
        """Receive SDP offer, delegate to camera entity, return SDP answer."""
        hass = request.app["hass"]

        # Auth: Bearer header or ?token= query param
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

        session_id = secrets.token_hex(16)

        # Delegate to the camera entity's browser WebRTC handler, but defer the
        # media start until after the HTTP answer is written back to the client.
        result = {}

        def send_message(msg):
            if isinstance(msg, WebRTCAnswer):
                result["answer"] = msg.answer
            elif isinstance(msg, WebRTCError):
                result["error"] = msg.message

        await camera.async_handle_async_webrtc_offer(
            offer_sdp,
            session_id,
            send_message,
            defer_media_start=True,
        )

        # Track session for DELETE cleanup
        mirror_sessions = hass.data.setdefault(DOMAIN, {}).setdefault(
            "mirror_sessions", {}
        )
        mirror_sessions[device_id] = {"session_id": session_id, "camera": camera}

        if "answer" in result:
            response = web.StreamResponse(
                status=201,
                headers={
                    "Content-Type": "application/sdp",
                    "Location": f"/api/petkit/whep_mirror/{device_id}",
                },
            )
            await response.prepare(request)
            await response.write(result["answer"].encode())
            await response.write_eof()

            camera.schedule_deferred_media_start(delay=1.0)
            return response

        return web.Response(status=502, text=result.get("error", "Negotiation failed"))

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

        mirror_sessions = hass.data.get(DOMAIN, {}).get("mirror_sessions", {})
        session = mirror_sessions.pop(device_id, None)
        if session is None:
            return web.Response(status=404, text="No active mirror session")

        session["camera"].close_webrtc_session(session["session_id"])
        return web.Response(status=200, text="Session closed")
