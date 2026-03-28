"""Lightweight RTSP/TCP proxy for exposing HA-managed go2rtc externally."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit
import zlib

from .const import DOMAIN, LOGGER
from .go2rtc_stream import get_go2rtc_stream_manager

if TYPE_CHECKING:
    from .camera import PetkitWebRTCCamera


_REQUEST_END = b"\r\n\r\n"
_LOCALHOST = "127.0.0.1"
_GO2RTC_RTSP_PORT = 18554
_LISTEN_HOST = "0.0.0.0"
_PORT_RANGE_START = 19000
_PORT_RANGE_SIZE = 20000


@dataclass
class RtspProxyClient:
    """One proxied RTSP client."""

    client_reader: asyncio.StreamReader
    client_writer: asyncio.StreamWriter
    upstream_reader: asyncio.StreamReader
    upstream_writer: asyncio.StreamWriter
    tasks: list[asyncio.Task[None]] = field(default_factory=list)


@dataclass
class RtspProxyServer:
    """One RTSP proxy listener bound to a camera."""

    camera: PetkitWebRTCCamera
    server: asyncio.AbstractServer
    port: int

    @property
    def device_id(self) -> str:
        return str(self.camera.device.id)


class PetkitRTSPProxyManager:
    """Expose one stable RTSP proxy port per PetKit device."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._servers: dict[str, RtspProxyServer] = {}
        self._last_errors: dict[str, str] = {}

    async def async_ensure_listener(self, camera: PetkitWebRTCCamera) -> str | None:
        """Ensure the RTSP proxy listener exists for one camera."""
        device_id = str(camera.device.id)
        async with self._lock:
            existing = self._servers.get(device_id)
            if existing is not None and existing.server.is_serving():
                return self.local_rtsp_url(device_id)

            server, port = await self._async_start_listener(camera)
            if server is None or port is None:
                return None

            self._servers[device_id] = RtspProxyServer(
                camera=camera,
                server=server,
                port=port,
            )
            return self.local_rtsp_url(device_id)

    async def async_close_listener(self, device_id: str) -> bool:
        """Close one RTSP proxy listener."""
        async with self._lock:
            proxy = self._servers.pop(device_id, None)

        if proxy is None:
            return False

        proxy.server.close()
        await proxy.server.wait_closed()
        return True

    async def async_close_all(self) -> None:
        """Close every RTSP proxy listener."""
        async with self._lock:
            device_ids = list(self._servers)
        for device_id in device_ids:
            await self.async_close_listener(device_id)

    def last_error(self, device_id: str) -> str | None:
        """Return the last RTSP proxy failure for one device."""
        return self._last_errors.get(device_id)

    def listener_port(self, device_id: str) -> int | None:
        """Return the active listener port for one device."""
        active = self._servers.get(device_id)
        return active.port if active is not None else None

    def local_rtsp_url(self, device_id: str) -> str:
        """Return the loopback RTSP proxy URL for one device."""
        return f"rtsp://{_LOCALHOST}:{self._port_for_device(device_id)}/{device_id}"

    def rtsp_url_for_host(self, device_id: str, host: str) -> str:
        """Return the externally reachable RTSP proxy URL for one device."""
        formatted_host = self._format_host(host)
        return f"rtsp://{formatted_host}:{self._port_for_device(device_id)}/{device_id}"

    async def _async_start_listener(
        self,
        camera: PetkitWebRTCCamera,
    ) -> tuple[asyncio.AbstractServer | None, int | None]:
        """Start the deterministic RTSP proxy listener for one device."""
        used_ports = {session.port for session in self._servers.values()}
        preferred_port = self._preferred_port(str(camera.device.id))
        last_error: OSError | None = None

        for offset in range(_PORT_RANGE_SIZE):
            port = _PORT_RANGE_START + (
                (preferred_port - _PORT_RANGE_START + offset) % _PORT_RANGE_SIZE
            )
            if port in used_ports:
                continue
            try:
                server = await asyncio.start_server(
                    lambda reader, writer: self._handle_client(camera, reader, writer),
                    host=_LISTEN_HOST,
                    port=port,
                )
            except OSError as err:
                last_error = err
                continue
            return server, port

        LOGGER.warning(
            "Failed to start RTSP proxy listener for %s: %s",
            camera.device.id,
            last_error,
        )
        return None, None

    async def _handle_client(
        self,
        camera: PetkitWebRTCCamera,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        """Proxy one RTSP/TCP client connection to HA-managed go2rtc."""
        device_id = str(camera.device.id)
        manager = get_go2rtc_stream_manager(self.hass)
        local_rtsp = await manager.async_ensure_stream(device_id, raise_on_failure=True)
        if local_rtsp is None:
            raise RuntimeError(
                f"HA-managed go2rtc stream unavailable for RTSP proxy {device_id}"
            )

        local_parts = urlsplit(local_rtsp)
        local_target = urlunsplit(
            ("rtsp", f"{_LOCALHOST}:{_GO2RTC_RTSP_PORT}", local_parts.path, "", "")
        )

        upstream_reader = upstream_writer = None
        proxy = None
        try:
            self._last_errors.pop(device_id, None)
            upstream_reader, upstream_writer = await asyncio.open_connection(
                _LOCALHOST,
                _GO2RTC_RTSP_PORT,
            )
            proxy = RtspProxyClient(
                client_reader=client_reader,
                client_writer=client_writer,
                upstream_reader=upstream_reader,
                upstream_writer=upstream_writer,
            )
            proxy.tasks = [
                self.hass.async_create_background_task(
                    self._client_to_upstream(proxy, local_target),
                    f"petkit rtsp upstream {device_id}",
                ),
                self.hass.async_create_background_task(
                    self._upstream_to_client(proxy),
                    f"petkit rtsp downstream {device_id}",
                ),
            ]
            await asyncio.wait(proxy.tasks, return_when=asyncio.FIRST_COMPLETED)
        except Exception as err:  # noqa: BLE001
            self._last_errors[device_id] = str(err)
            LOGGER.debug("RTSP proxy client failed for %s: %s", device_id, err)
        finally:
            if proxy is not None:
                for task in proxy.tasks:
                    task.cancel()
                for task in proxy.tasks:
                    try:
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                    except Exception as err:  # noqa: BLE001
                        self._last_errors.setdefault(device_id, str(err))
                        LOGGER.debug(
                            "RTSP proxy task cleanup failed for %s: %s",
                            device_id,
                            err,
                        )
            if upstream_writer is not None:
                upstream_writer.close()
                with contextlib.suppress(Exception):
                    await upstream_writer.wait_closed()
            client_writer.close()
            with contextlib.suppress(Exception):
                await client_writer.wait_closed()

    async def _client_to_upstream(
        self,
        proxy: RtspProxyClient,
        local_target: str,
    ) -> None:
        """Rewrite RTSP requests to the local go2rtc target and forward them."""
        while not proxy.client_reader.at_eof():
            chunk = await proxy.client_reader.readexactly(1)
            if chunk == b"$":
                header = await proxy.client_reader.readexactly(3)
                payload_size = int.from_bytes(header[1:], "big")
                payload = (
                    await proxy.client_reader.readexactly(payload_size)
                    if payload_size
                    else b""
                )
                proxy.upstream_writer.write(chunk + header + payload)
                await proxy.upstream_writer.drain()
                continue

            raw = chunk + await proxy.client_reader.readuntil(_REQUEST_END)
            header_blob, _, body = raw.partition(_REQUEST_END)
            header_lines = header_blob.decode("utf-8", errors="ignore").split("\r\n")
            request_line = header_lines[0]
            parts = request_line.split(" ")
            if len(parts) != 3:
                proxy.upstream_writer.write(raw)
                await proxy.upstream_writer.drain()
                continue

            headers: list[str] = []
            content_length = 0
            for line in header_lines[1:]:
                headers.append(line)
                if line.lower().startswith("content-length:"):
                    with contextlib.suppress(ValueError):
                        content_length = int(line.split(":", 1)[1].strip())

            if content_length > len(body):
                body += await proxy.client_reader.readexactly(content_length - len(body))

            method, _, version = parts
            rewritten = "\r\n".join([f"{method} {local_target} {version}", *headers])
            proxy.upstream_writer.write(rewritten.encode() + _REQUEST_END + body)
            await proxy.upstream_writer.drain()

    async def _upstream_to_client(self, proxy: RtspProxyClient) -> None:
        """Relay bytes from the local go2rtc RTSP server back to the client."""
        forwarded_any = False
        while not proxy.upstream_reader.at_eof():
            data = await proxy.upstream_reader.read(65536)
            if not data:
                break
            forwarded_any = True
            proxy.client_writer.write(data)
            await proxy.client_writer.drain()
        if not forwarded_any:
            raise RuntimeError("Local go2rtc RTSP server closed without a response")

    @staticmethod
    def _preferred_port(device_id: str) -> int:
        """Return the deterministic RTSP port for one device."""
        return _PORT_RANGE_START + (zlib.crc32(device_id.encode()) % _PORT_RANGE_SIZE)

    def _port_for_device(self, device_id: str) -> int:
        """Return the active proxy port for one device."""
        active = self._servers.get(device_id)
        if active is not None:
            return active.port
        return self._preferred_port(device_id)

    @staticmethod
    def _format_host(host: str) -> str:
        """Format IPv4/IPv6 hostnames for RTSP URLs."""
        if ":" in host and not host.startswith("["):
            return f"[{host}]"
        return host


def get_rtsp_proxy_manager(hass) -> PetkitRTSPProxyManager:
    """Return the shared RTSP proxy manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("rtsp_proxy_manager")
    if manager is None:
        manager = PetkitRTSPProxyManager(hass)
        domain_data["rtsp_proxy_manager"] = manager
    return manager
