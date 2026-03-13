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
from pypetkitapi import TEMP_CAMERA_TYPES
from webrtc_models import RTCIceCandidateInit

from homeassistant.components.http import HomeAssistantView

from .agora_api import SERVICE_IDS, AgoraAPIClient, AgoraResponse
from .agora_rtm import AgoraRTMSignaling
from .agora_websocket import AgoraWebSocketHandler
from .const import AGORA_APP_ID, DOMAIN, LOGGER

if TYPE_CHECKING:
    from .camera import PetkitWebRTCCamera


def _missing_live_feed_fields(live_feed) -> list[str]:
    """Return missing fields needed to start and sustain a WHEP stream."""
    required_fields = {
        "channel_id": getattr(live_feed, "channel_id", None),
        "rtc_token": getattr(live_feed, "rtc_token", None),
        "app_rtm_user_id": getattr(live_feed, "app_rtm_user_id", None),
        "dev_rtm_user_id": getattr(live_feed, "dev_rtm_user_id", None),
        "rtm_token": getattr(live_feed, "rtm_token", None),
    }
    return [field for field, value in required_fields.items() if not value]


def _live_feed_ready_for_whep(live_feed) -> bool:
    """Return whether a live feed has all fields required by WHEP."""
    return live_feed is not None and not _missing_live_feed_fields(live_feed)


def _resolve_agora_user_id(camera: PetkitWebRTCCamera, live_feed) -> int:
    """Pick the most reliable Agora uid available for choose_server."""
    if (live_feed_uid := getattr(live_feed, "uid", None)) not in (None, ""):
        try:
            return int(live_feed_uid)
        except (TypeError, ValueError):
            LOGGER.debug("WHEP: invalid live_feed uid=%s", live_feed_uid)

    client = camera.coordinator.config_entry.runtime_data.client
    session_user_id = getattr(getattr(client, "_session", None), "user_id", None)
    if session_user_id not in (None, ""):
        try:
            return int(str(session_user_id))
        except (TypeError, ValueError):
            LOGGER.debug("WHEP: invalid session user_id=%s", session_user_id)

    app_rtm_user_id = str(getattr(live_feed, "app_rtm_user_id", "") or "")
    digits = "".join(char for char in app_rtm_user_id if char.isdigit())
    if digits:
        return int(digits)

    return 0


def _add_offer_candidates(
    handler: AgoraWebSocketHandler,
    offer_sdp: str,
) -> int:
    """Extract inline ICE candidates from the SDP offer for WHEP clients."""
    seen_candidates = {
        candidate.candidate
        for candidate in handler.candidates
        if candidate.candidate
    }
    added = 0
    media_index = -1
    current_mid: str | None = None

    for raw_line in offer_sdp.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("m="):
            media_index += 1
            current_mid = None
            continue

        if line.startswith("a=mid:"):
            current_mid = line.removeprefix("a=mid:")
            continue

        if not line.startswith("a=candidate:"):
            continue

        candidate_line = line.removeprefix("a=")
        if candidate_line in seen_candidates:
            continue

        handler.add_ice_candidate(
            RTCIceCandidateInit(
                candidate=candidate_line,
                sdp_mid=current_mid,
                sdp_m_line_index=media_index if media_index >= 0 else None,
            )
        )
        seen_candidates.add(candidate_line)
        added += 1

    return added


async def _get_live_feed_for_whep(camera: PetkitWebRTCCamera):
    """Fetch live feed data and wake supported cameras if needed."""
    live_feed = await camera._get_live_feed()
    if _live_feed_ready_for_whep(live_feed):
        return live_feed

    device_id = camera.device.id
    missing_fields = _missing_live_feed_fields(live_feed)
    if missing_fields:
        LOGGER.debug(
            "WHEP: initial live feed for %s missing %s",
            device_id,
            ", ".join(missing_fields),
        )

    await camera.coordinator.async_request_refresh()
    live_feed = await camera._get_live_feed()
    if _live_feed_ready_for_whep(live_feed):
        return live_feed

    device_type = str(
        getattr(getattr(camera.device, "device_nfo", None), "device_type", "") or ""
    ).lower()
    if device_type not in TEMP_CAMERA_TYPES:
        LOGGER.debug(
            "WHEP: device %s (%s) does not support temporary_open_camera",
            device_id,
            device_type,
        )
        return None

    client = camera.coordinator.config_entry.runtime_data.client
    LOGGER.debug(
        "WHEP: requesting temporary_open_camera for %s (%s)",
        device_id,
        device_type,
    )
    try:
        await client.temporary_open_camera(device_type, device_id)
    except Exception as err:  # noqa: BLE001
        LOGGER.debug(
            "WHEP: temporary_open_camera failed for %s: %s",
            device_id,
            err,
        )
        return None

    await asyncio.sleep(3)
    await camera.coordinator.async_request_refresh()
    live_feed = await camera._get_live_feed()
    if _live_feed_ready_for_whep(live_feed):
        return live_feed

    LOGGER.debug(
        "WHEP: live feed for %s still missing %s after temporary_open_camera",
        device_id,
        ", ".join(_missing_live_feed_fields(live_feed)),
    )
    return None


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
            live_feed = await _get_live_feed_for_whep(camera)
            if live_feed is None:
                return web.Response(
                    status=503,
                    text="Live feed unavailable or missing RTM credentials",
                )

            agora_user_id = _resolve_agora_user_id(camera, live_feed)
            LOGGER.debug(
                "WHEP: starting device %s with Agora uid=%s",
                device_id,
                agora_user_id,
            )

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

            rtm = AgoraRTMSignaling(AGORA_APP_ID)
            handler = AgoraWebSocketHandler(
                rtc_token_provider=camera._refresh_rtc_token,
                prefer_instant_video=True,
                subscribe_retry_delay=1.0,
                subscribe_retry_attempts=3,
            )

            parsed_candidates = _add_offer_candidates(handler, offer_sdp)
            if parsed_candidates:
                LOGGER.debug(
                    "WHEP: extracted %d ICE candidates from offer for %s",
                    parsed_candidates,
                    device_id,
                )
            else:
                LOGGER.debug("WHEP: no inline ICE candidates found for %s", device_id)

            handler.candidates = camera._filter_candidates(
                handler.candidates,
                agora_response,
            )
            LOGGER.debug(
                "WHEP: using %d filtered ICE candidates for %s",
                len(handler.candidates),
                device_id,
            )

            rtm_started = await rtm.start_live(live_feed)
            if not rtm_started:
                LOGGER.warning("WHEP: RTM start_live failed for device %s", device_id)
                await asyncio.gather(
                    rtm.stop_live(send_stop=True),
                    handler.disconnect(),
                    return_exceptions=True,
                )
                return web.Response(
                    status=503,
                    text="PetKit RTM start_live failed",
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
            LOGGER.debug(
                "WHEP: negotiated device %s successfully (answer bytes=%d)",
                device_id,
                len(answer_sdp),
            )

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
