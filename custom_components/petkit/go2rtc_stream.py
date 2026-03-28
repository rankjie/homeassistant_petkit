"""Helpers for exposing PetKit streams through HA-managed go2rtc."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from http import HTTPStatus

from aiohttp import ClientError, ClientSession, ClientTimeout

from homeassistant.components.http.auth import async_sign_path
from homeassistant.components.go2rtc.const import HA_MANAGED_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN, LOGGER

_GO2RTC_DOMAIN = "go2rtc"
_HA_MANAGED_URL_ALIASES = {
    HA_MANAGED_URL,
    "http://127.0.0.1:11984/",
}
_SIGN_EXPIRATION = timedelta(days=365)
_GO2RTC_API_PATH = "api/streams"
_GO2RTC_RTSP_BASE = "rtsp://127.0.0.1:18554"
_REQUEST_TIMEOUT = ClientTimeout(total=10)


class PetkitGo2RTCStreamManager:
    """Manage fixed go2rtc streams per PetKit device."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def _session(self) -> ClientSession:
        """Return the aiohttp session for go2rtc API calls."""
        return async_get_clientsession(self.hass)

    @property
    def _base_url(self) -> str:
        """Return the go2rtc API base URL."""
        url = self._configured_url
        return HA_MANAGED_URL if url in _HA_MANAGED_URL_ALIASES or url is None else url

    def is_managed_available(self) -> bool:
        """Return whether HA-managed go2rtc is active."""
        return self._configured_url in _HA_MANAGED_URL_ALIASES

    @property
    def _configured_url(self) -> str | None:
        """Return the go2rtc URL stored by Home Assistant."""
        go2rtc_data = self.hass.data.get(_GO2RTC_DOMAIN)
        return getattr(go2rtc_data, "url", go2rtc_data)

    def stream_name(self, device_id: str) -> str:
        """Return the deterministic go2rtc stream name for one device."""
        return f"petkit_{device_id}"

    def rtsp_url(self, device_id: str) -> str:
        """Return the local RTSP URL exposed by go2rtc."""
        return f"{_GO2RTC_RTSP_BASE}/{self.stream_name(device_id)}"

    def internal_webrtc_source(self, device_id: str) -> str | None:
        """Return the signed internal upstream source URL for go2rtc."""
        http_server = getattr(self.hass, "http", None)
        if http_server is None or not getattr(http_server, "server_port", None):
            return None

        signed_path = async_sign_path(
            self.hass,
            f"/api/petkit/whep_upstream/{device_id}",
            _SIGN_EXPIRATION,
        )
        return f"webrtc:http://127.0.0.1:{http_server.server_port}{signed_path}"

    async def async_ensure_stream(
        self,
        device_id: str,
        *,
        raise_on_failure: bool = False,
    ) -> str | None:
        """Ensure the shared go2rtc stream exists and return its local RTSP URL."""
        if not self.is_managed_available():
            return None

        source = self.internal_webrtc_source(device_id)
        if source is None:
            message = f"PetKit go2rtc stream {device_id} unavailable: HA HTTP server not ready"
            LOGGER.debug(message)
            if raise_on_failure:
                raise RuntimeError(message)
            return None

        stream_name = self.stream_name(device_id)
        lock = self._locks.setdefault(stream_name, asyncio.Lock())
        async with lock:
            if await self._async_stream_matches(stream_name, source):
                return self.rtsp_url(device_id)

            methods: tuple[tuple[str, dict[str, str]], ...] = (
                ("post", {"dst": stream_name, "src": source}),
                ("put", {"name": stream_name, "src": source}),
                ("patch", {"name": stream_name, "src": source}),
                ("patch", {"dst": stream_name, "src": source}),
            )

            statuses: list[str] = []
            for method, params in methods:
                status, detail = await self._async_call_api(method, params)
                detail_suffix = f" ({detail})" if detail else ""
                statuses.append(f"{method.upper()}={status}{detail_suffix}")
                if status in (
                    HTTPStatus.OK,
                    HTTPStatus.CREATED,
                    HTTPStatus.NO_CONTENT,
                ):
                    return self.rtsp_url(device_id)
                if await self._async_stream_matches(stream_name, source):
                    return self.rtsp_url(device_id)

            LOGGER.warning(
                "Failed to register PetKit go2rtc stream %s (%s)",
                device_id,
                ", ".join(statuses),
            )
            if raise_on_failure:
                raise RuntimeError(
                    f"Failed to register PetKit go2rtc stream {device_id}"
                    f" ({', '.join(statuses)})"
                )
            return None

    async def async_remove_stream(self, device_id: str) -> bool:
        """Remove the shared go2rtc stream if it exists."""
        if not self.is_managed_available():
            return False

        stream_name = self.stream_name(device_id)
        lock = self._locks.setdefault(stream_name, asyncio.Lock())
        async with lock:
            for params in ({"dst": stream_name}, {"name": stream_name}):
                status, _ = await self._async_call_api("delete", params)
                if status in (HTTPStatus.OK, HTTPStatus.NO_CONTENT):
                    return True
                if status == HTTPStatus.NOT_FOUND:
                    continue
        return False

    async def _async_stream_matches(self, stream_name: str, source: str) -> bool:
        """Return whether go2rtc already has the expected producer source."""
        streams = await self._async_get_streams()
        if streams is None:
            return False

        stream = streams.get(stream_name)
        if not isinstance(stream, dict):
            return False

        producers = stream.get("producers") or []
        return any(
            isinstance(producer, dict) and producer.get("url") == source
            for producer in producers
        )

    async def _async_get_streams(self) -> dict[str, dict] | None:
        """Fetch the current go2rtc streams payload."""
        try:
            async with self._session.get(
                f"{self._base_url}{_GO2RTC_API_PATH}",
                timeout=_REQUEST_TIMEOUT,
            ) as response:
                if response.status != HTTPStatus.OK:
                    return None
                payload = await response.json()
        except (ClientError, TimeoutError, ValueError) as err:
            LOGGER.debug("Failed to query HA-managed go2rtc streams: %s", err)
            return None

        if not isinstance(payload, dict):
            return None
        return payload

    async def _async_call_api(
        self, method: str, params: dict[str, str]
    ) -> tuple[int, str | None]:
        """Call the go2rtc API and return the HTTP status code plus error detail."""
        request: Callable[..., object] = getattr(self._session, method)
        try:
            async with request(
                f"{self._base_url}{_GO2RTC_API_PATH}",
                params=params,
                timeout=_REQUEST_TIMEOUT,
            ) as response:
                await response.read()
                return response.status, None
        except (ClientError, TimeoutError) as err:
            LOGGER.debug("go2rtc %s failed for %s: %s", method.upper(), params, err)
            return 0, str(err)


def get_go2rtc_stream_manager(hass: HomeAssistant) -> PetkitGo2RTCStreamManager:
    """Return the shared HA-managed go2rtc helper."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("go2rtc_stream_manager")
    if manager is None:
        manager = PetkitGo2RTCStreamManager(hass)
        domain_data["go2rtc_stream_manager"] = manager
    return manager
