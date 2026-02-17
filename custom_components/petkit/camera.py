"""Camera platform for Petkit Smart Devices integration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from pypetkitapi import FEEDER_WITH_CAMERA, LITTER_WITH_CAMERA, Feeder, Litter, LiveFeed
from webrtc_models import RTCIceCandidateInit, RTCIceServer

from homeassistant.components.camera import (
    CameraEntityDescription,
    WebRTCAnswer,
    WebRTCError,
    WebRTCSendMessage,
)
from homeassistant.components.web_rtc import async_register_ice_servers
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .agora_api import SERVICE_IDS, AgoraAPIClient, AgoraResponse
from .agora_rtm import AgoraRTMSignaling
from .agora_websocket import AgoraWebSocketHandler
from .const import (
    CONF_STREAM_CONTROL_MODE,
    DEFAULT_STREAM_CONTROL_MODE,
    LOGGER,
    STREAM_CONTROL_EXCLUSIVE,
    STREAM_CONTROL_SHARED,
)
from .coordinator import PetkitDataUpdateCoordinator
from .entity import PetKitDescSensorBase, PetkitCameraBaseEntity

AGORA_APP_ID = "244c49951296440cbc1e3b937bf5e410"


@dataclass(frozen=True, kw_only=True)
class PetKitCameraDesc(PetKitDescSensorBase, CameraEntityDescription):
    """Description class for PetKit camera entities."""


CAMERA_MAPPING: dict[type[Feeder | Litter], list[PetKitCameraDesc]] = {
    Feeder: [
        PetKitCameraDesc(
            key="camera",
            translation_key="camera",
            only_for_types=FEEDER_WITH_CAMERA,
            value=lambda _device: True,
        )
    ],
    Litter: [
        PetKitCameraDesc(
            key="camera",
            translation_key="camera",
            only_for_types=LITTER_WITH_CAMERA,
            value=lambda _device: True,
        )
    ],
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up camera entities."""
    devices = entry.runtime_data.client.petkit_entities.values()

    entities: list[PetkitWebRTCCamera] = [
        PetkitWebRTCCamera(
            coordinator=entry.runtime_data.coordinator,
            device=device,
            entity_description=entity_description,
            hass=hass,
        )
        for device in devices
        for device_type, descriptions in CAMERA_MAPPING.items()
        if isinstance(device, device_type)
        for entity_description in descriptions
        if entity_description.is_supported(device)
    ]

    if entities:
        results = await asyncio.gather(
            *(entity.async_prepare_agora() for entity in entities),
            return_exceptions=True,
        )
        for entity, result in zip(entities, results, strict=False):
            if isinstance(result, Exception):
                LOGGER.debug(
                    "Failed to prefetch Agora context for %s: %s",
                    entity.entity_id,
                    result,
                )

    async_add_entities(entities)


class PetkitWebRTCCamera(PetkitCameraBaseEntity):
    """Native Home Assistant WebRTC camera backed by Agora signaling."""

    entity_description: PetKitCameraDesc

    def __init__(
        self,
        coordinator: PetkitDataUpdateCoordinator,
        device: Feeder | Litter,
        entity_description: PetKitCameraDesc,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the camera entity."""
        super().__init__(coordinator, device, entity_description.key)
        self.hass = hass
        self.coordinator = coordinator
        self.device = device
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.translation_key

        self._agora_rtm = AgoraRTMSignaling(AGORA_APP_ID)
        self._agora_handler = AgoraWebSocketHandler(
            rtc_token_provider=self._refresh_rtc_token
        )
        self._agora_response: AgoraResponse | None = None
        self._ice_servers: list[RTCIceServer] = []

        self._remove_ice_servers: Callable[[], None] | None = None

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return bool(super().available and self._get_live_feed() is not None)

    async def async_added_to_hass(self) -> None:
        """Register ICE callback when entity is added."""
        await super().async_added_to_hass()
        self._remove_ice_servers = async_register_ice_servers(
            self.hass,
            self.get_ice_servers,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup callbacks and websocket sessions."""
        if self._remove_ice_servers:
            self._remove_ice_servers()
            self._remove_ice_servers = None
        await self._async_close_stream()
        await super().async_will_remove_from_hass()

    async def async_prepare_agora(self) -> None:
        """Best-effort prefetch for ICE servers before first offer."""
        live_feed = self._get_live_feed()
        if live_feed is None:
            return
        await self._refresh_agora_context(live_feed)

    async def async_camera_image(
        self,
        width: int | None = None,
        height: int | None = None,
    ) -> bytes | None:
        """WebRTC cameras do not provide still snapshots directly."""
        return None

    async def async_handle_async_webrtc_offer(
        self,
        offer_sdp: str,
        session_id: str,
        send_message: WebRTCSendMessage,
    ) -> None:
        """Handle browser WebRTC offer and return SDP answer."""
        await self._agora_handler.disconnect()
        self._agora_handler.candidates = []

        try:
            live_feed = await self._async_get_live_feed(refresh=True)
            if live_feed is None:
                send_message(
                    WebRTCError(
                        code="live_feed_unavailable",
                        message="No PetKit live feed token available for this device",
                    )
                )
                return

            await self._refresh_agora_context(live_feed)
            if self._agora_response is None:
                send_message(
                    WebRTCError(
                        code="agora_context_failed",
                        message="Failed to retrieve Agora edge servers",
                    )
                )
                return

            self._agora_handler.candidates = self._filter_candidates(
                self._agora_handler.candidates,
                self._agora_response,
            )

            rtm_started = await self._agora_rtm.start_live(live_feed)
            if not rtm_started:
                LOGGER.warning(
                    "start_live/heartbeat not active for PetKit camera %s",
                    self.device.id,
                )

            answer_sdp = await self._agora_handler.connect_and_join(
                live_feed=live_feed,
                offer_sdp=offer_sdp,
                session_id=session_id,
                app_id=AGORA_APP_ID,
                agora_response=self._agora_response,
            )

            if answer_sdp:
                send_message(WebRTCAnswer(answer_sdp))
                return

            await self._async_close_stream()
            send_message(
                WebRTCError(
                    code="webrtc_negotiation_failed",
                    message="Agora negotiation did not return an SDP answer",
                )
            )
        except (OSError, ValueError, RuntimeError) as err:
            await self._async_close_stream()
            LOGGER.error("WebRTC offer handling failed: %s", err)
            send_message(
                WebRTCError(
                    code="webrtc_offer_error",
                    message=str(err),
                )
            )

    async def async_on_webrtc_candidate(
        self,
        session_id: str,
        candidate: RTCIceCandidateInit,
    ) -> None:
        """Collect browser ICE candidates for join_v3."""
        self._agora_handler.add_ice_candidate(candidate)

    @callback
    def close_webrtc_session(self, session_id: str) -> None:
        """Close and cleanup a WebRTC session."""
        self.hass.async_create_task(self._async_close_stream())

    def get_ice_servers(self) -> list[RTCIceServer]:
        """Return cached Agora ICE servers for Home Assistant frontend."""
        return self._ice_servers

    async def _async_close_stream(self) -> None:
        """Stop signaling control (mode-dependent) and close websocket session."""
        send_stop = self._stream_control_mode() == STREAM_CONTROL_EXCLUSIVE
        results = await asyncio.gather(
            self._agora_rtm.stop_live(send_stop=send_stop),
            self._agora_handler.disconnect(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                LOGGER.debug(
                    "Stream cleanup error for %s: %s",
                    self.device.id,
                    result,
                )

    def _stream_control_mode(self) -> str:
        """Return stream control mode from config entry options."""
        config_entry = self.coordinator.config_entry
        mode = config_entry.options.get(
            CONF_STREAM_CONTROL_MODE,
            DEFAULT_STREAM_CONTROL_MODE,
        )
        if mode not in (STREAM_CONTROL_SHARED, STREAM_CONTROL_EXCLUSIVE):
            return DEFAULT_STREAM_CONTROL_MODE
        return mode

    async def _refresh_rtc_token(self) -> str | None:
        """Fetch fresh live feed tokens and return the latest RTC token."""
        await self.coordinator.async_request_refresh()
        live_feed = self._get_live_feed()
        if live_feed is None or not live_feed.rtc_token:
            return None

        await self._agora_rtm.update_tokens(live_feed)
        return live_feed.rtc_token

    async def _async_get_live_feed(self, refresh: bool = False) -> LiveFeed | None:
        """Return current live feed token payload for this device."""
        live_feed = self._get_live_feed()
        if live_feed is not None:
            return live_feed

        if not refresh:
            return None

        await self.coordinator.async_request_refresh()
        return self._get_live_feed()

    def _get_live_feed(self) -> LiveFeed | None:
        """Read live feed from latest coordinator state."""
        if not self.coordinator.data:
            return None

        device_state = self.coordinator.data.get(self.device.id)
        if device_state is None:
            return None

        live_feed = getattr(device_state, "live_feed", None)
        if not isinstance(live_feed, LiveFeed):
            return None

        if not live_feed.channel_id or not live_feed.rtc_token:
            return None

        return live_feed

    def _resolve_rtc_uid(self, _live_feed: LiveFeed) -> int:
        """Resolve RTC uid matching PetKit behavior (server-assigned uid 0)."""
        return 0

    async def _refresh_agora_context(self, live_feed: LiveFeed) -> None:
        """Fetch Agora gateway + TURN endpoints and cache ICE servers."""
        rtc_uid = self._resolve_rtc_uid(live_feed)
        self._agora_response = None

        async with AgoraAPIClient() as agora_client:
            response = await agora_client.choose_server(
                app_id=AGORA_APP_ID,
                token=live_feed.rtc_token,
                channel_name=live_feed.channel_id,
                user_id=rtc_uid,
                service_flags=[
                    SERVICE_IDS["CHOOSE_SERVER"],
                    SERVICE_IDS["CLOUD_PROXY_FALLBACK"],
                ],
            )

        self._agora_response = response
        ice_servers = response.get_ice_servers(use_all_turn_servers=False)
        self._ice_servers = [
            RTCIceServer(
                urls=server.urls,
                username=server.username,
                credential=server.credential,
            )
            for server in ice_servers
        ]

        LOGGER.debug(
            "Cached %d ICE servers for PetKit camera %s",
            len(self._ice_servers),
            self.device.id,
        )

    @staticmethod
    def _filter_candidates(
        candidates: list[RTCIceCandidateInit],
        agora_response: AgoraResponse,
    ) -> list[RTCIceCandidateInit]:
        """Prefer relay/srflx candidates and drop host candidates."""
        valid_turn_ips = {
            address.ip for address in (agora_response.get_turn_addresses() or [])
        }

        filtered: list[RTCIceCandidateInit] = []
        for candidate in candidates:
            candidate_str = candidate.candidate or ""

            if "typ srflx" in candidate_str or "typ prflx" in candidate_str:
                filtered.append(candidate)
                continue

            if "typ relay" in candidate_str:
                if not valid_turn_ips or any(ip in candidate_str for ip in valid_turn_ips):
                    filtered.append(candidate)
                continue

        return filtered or candidates
