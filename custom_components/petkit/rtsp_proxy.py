"""Local RTSP/TCP server for PetKit H.264 passthrough."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import random
import secrets
from typing import TYPE_CHECKING, Any
import zlib

from av import Packet
from aiortc.codecs.h264 import H264Encoder
from aiortc.rtp import HeaderExtensionsMap, RtpPacket

from .const import DOMAIN, LOGGER
from .whep_mirror import AIORTC_IMPORT_ERROR, _get_manager as _get_whep_manager

if TYPE_CHECKING:
    from .camera import PetkitWebRTCCamera


_REQUEST_END = b"\r\n\r\n"
_OPTIONS_PUBLIC = (
    "OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN, GET_PARAMETER, SET_PARAMETER"
)
_INTERLEAVED_DEFAULT = (0, 1)
_LISTEN_HOST = "0.0.0.0"
_LOCALHOST = "127.0.0.1"
_PORT_RANGE_START = 19000
_PORT_RANGE_SIZE = 20000


def _split_rtsp_request(data: bytes) -> tuple[str, dict[str, str], bytes]:
    header_blob, _, body = data.partition(_REQUEST_END)
    header_lines = header_blob.decode("utf-8", errors="ignore").split("\r\n")
    request_line = header_lines[0]
    headers: dict[str, str] = {}
    for line in header_lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return request_line, headers, body


def _build_rtsp_response(
    status_code: int,
    cseq: str,
    *,
    status_text: str = "OK",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> bytes:
    response_headers = {
        "CSeq": cseq,
        "Server": "petkit-ha-python",
        "Connection": "keep-alive",
    }
    if headers:
        response_headers.update(headers)
    if body is not None:
        response_headers["Content-Length"] = str(len(body))
    lines = [f"RTSP/1.0 {status_code} {status_text}"]
    lines.extend(f"{key}: {value}" for key, value in response_headers.items())
    payload = "\r\n".join(lines).encode() + b"\r\n\r\n"
    if body is not None:
        payload += body
    return payload


def _annexb_wrap(nalu: bytes) -> bytes:
    return b"\x00\x00\x00\x01" + nalu


def _iter_annexb_nalus(data: bytes) -> list[bytes]:
    nalus: list[bytes] = []
    i = 0
    while True:
        start = data.find(b"\x00\x00\x01", i)
        if start == -1:
            break
        start += 3
        if start < len(data) and data[start - 1] == 0:
            start -= 1
        end = data.find(b"\x00\x00\x01", start)
        if end == -1:
            nalus.append(data[start:])
            break
        if end > start and data[end - 1] == 0:
            nalus.append(data[start : end - 1])
        else:
            nalus.append(data[start:end])
        i = end
    return [nalu for nalu in nalus if nalu]


def _augment_h264_packet(
    packet: Packet,
    cached_sps: bytes | None,
    cached_pps: bytes | None,
) -> tuple[Packet, bytes | None, bytes | None]:
    data = bytes(packet)
    nalus = _iter_annexb_nalus(data)
    if not nalus:
        return packet, cached_sps, cached_pps

    has_idr = False
    has_sps = False
    has_pps = False
    for nalu in nalus:
        nalu_type = nalu[0] & 0x1F
        if nalu_type == 7:
            cached_sps = nalu
            has_sps = True
        elif nalu_type == 8:
            cached_pps = nalu
            has_pps = True
        elif nalu_type == 5:
            has_idr = True

    if has_idr and (cached_sps and cached_pps) and not (has_sps and has_pps):
        augmented = _annexb_wrap(cached_sps) + _annexb_wrap(cached_pps) + data
        new_packet = Packet(augmented)
        new_packet.pts = packet.pts
        new_packet.time_base = packet.time_base
        return new_packet, cached_sps, cached_pps

    return packet, cached_sps, cached_pps


@dataclass
class RtspClientSession:
    """One RTSP/TCP consumer session."""

    session_id: str
    writer: asyncio.StreamWriter
    writer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    interleaved_rtp_channel: int = _INTERLEAVED_DEFAULT[0]
    play_task: asyncio.Task[None] | None = None
    sequence_number: int = field(default_factory=lambda: random.randrange(0, 32768))
    ssrc: int = field(default_factory=lambda: random.randrange(1, 0xFFFFFFFF))
    last_timestamp: int = 0


@dataclass
class RtspServerSession:
    """One local RTSP server tied to a camera upstream."""

    camera: PetkitWebRTCCamera
    server: asyncio.AbstractServer
    port: int
    clients: dict[str, RtspClientSession] = field(default_factory=dict)

    @property
    def device_id(self) -> str:
        return str(self.camera.device.id)

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://{_LOCALHOST}:{self.port}/{self.device_id}"


class PetkitRTSPProxyManager:
    """Expose an active PetKit upstream as a stable RTSP/TCP stream."""

    def __init__(self, hass) -> None:
        self.hass = hass
        self._lock = asyncio.Lock()
        self._sessions: dict[str, RtspServerSession] = {}
        self._last_errors: dict[str, str] = {}

    async def async_ensure_listener(self, camera: PetkitWebRTCCamera) -> str | None:
        """Ensure a local RTSP listener exists for one camera."""
        if AIORTC_IMPORT_ERROR is not None:
            self._last_errors[str(camera.device.id)] = str(AIORTC_IMPORT_ERROR)
            camera.async_write_ha_state()
            return None

        device_id = str(camera.device.id)
        async with self._lock:
            existing = self._sessions.get(device_id)
            if existing is not None and existing.server.is_serving():
                return existing.rtsp_url

            server, port = await self._async_start_listener(camera)
            if server is None or port is None:
                return None

            session = RtspServerSession(camera=camera, server=server, port=port)
            self._sessions[device_id] = session

        if self._last_errors.pop(device_id, None) is not None:
            camera.async_write_ha_state()
        LOGGER.debug(
            "Started local RTSP passthrough server for %s at %s",
            device_id,
            session.rtsp_url,
        )
        return session.rtsp_url

    async def async_close_listener(self, device_id: str) -> bool:
        """Stop one local RTSP listener."""
        async with self._lock:
            session = self._sessions.pop(device_id, None)

        if session is None:
            return False

        session.server.close()
        await session.server.wait_closed()
        for client in list(session.clients.values()):
            if client.play_task is not None:
                client.play_task.cancel()
            client.writer.close()
            with contextlib.suppress(Exception):
                await client.writer.wait_closed()

        await _get_whep_manager(self.hass).maybe_close_idle_upstream(device_id)
        return True

    async def async_close_all(self) -> None:
        """Close all RTSP listeners."""
        async with self._lock:
            device_ids = list(self._sessions)
        for device_id in device_ids:
            await self.async_close_listener(device_id)

    def last_error(self, device_id: str) -> str | None:
        """Return the last RTSP failure for one device."""
        return self._last_errors.get(device_id)

    def listener_port(self, device_id: str) -> int | None:
        """Return the active listener port for one device."""
        session = self._sessions.get(device_id)
        return session.port if session is not None else None

    def local_rtsp_url(self, device_id: str) -> str:
        """Return the deterministic loopback RTSP URL for one device."""
        return f"rtsp://{_LOCALHOST}:{self._port_for_device(device_id)}/{device_id}"

    def rtsp_url_for_host(self, device_id: str, host: str) -> str:
        """Return the deterministic RTSP URL for one device and host."""
        formatted_host = self._format_host(host)
        return f"rtsp://{formatted_host}:{self._port_for_device(device_id)}/{device_id}"

    async def _async_start_listener(
        self,
        camera: PetkitWebRTCCamera,
    ) -> tuple[asyncio.AbstractServer | None, int | None]:
        """Start a deterministic RTSP listener for one device."""
        used_ports = {session.port for session in self._sessions.values()}
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

        message = (
            f"Failed to start local RTSP passthrough server for {camera.device.id}: "
            f"{last_error}"
        )
        self._last_errors[str(camera.device.id)] = message
        camera.async_write_ha_state()
        LOGGER.warning(message)
        return None, None

    async def _handle_client(
        self,
        camera: PetkitWebRTCCamera,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve one RTSP/TCP client."""
        device_id = str(camera.device.id)
        client = RtspClientSession(session_id=secrets.token_hex(8), writer=writer)
        session = self._sessions.get(device_id)
        if session is not None:
            session.clients[client.session_id] = client

        cached_sps: bytes | None = None
        cached_pps: bytes | None = None
        request_uri = "/"
        upstream = None
        encoder: H264Encoder | None = None

        def _set_cached(kind: str, value: bytes | None) -> None:
            nonlocal cached_sps, cached_pps
            if value is None:
                return
            if kind == "sps":
                cached_sps = value
            else:
                cached_pps = value

        try:
            while not reader.at_eof():
                request = await self._read_rtsp_request(reader)
                if request is None:
                    continue
                request_line, headers, _ = request
                parts = request_line.split(" ")
                if len(parts) < 2:
                    break
                method, request_uri = parts[0], parts[1]
                cseq = headers.get("cseq", "1")

                if method == "OPTIONS":
                    await self._write_response(
                        client,
                        _build_rtsp_response(
                            200,
                            cseq,
                            headers={"Public": _OPTIONS_PUBLIC},
                        ),
                    )
                elif method == "DESCRIBE":
                    body = self._build_sdp(request_uri)
                    await self._write_response(
                        client,
                        _build_rtsp_response(
                            200,
                            cseq,
                            headers={
                                "Content-Type": "application/sdp",
                                "Content-Base": request_uri,
                            },
                            body=body,
                        ),
                    )
                elif method == "SETUP":
                    transport = headers.get("transport", "")
                    client.interleaved_rtp_channel = self._parse_interleaved_channel(
                        transport
                    )
                    await self._write_response(
                        client,
                        _build_rtsp_response(
                            200,
                            cseq,
                            headers={
                                "Session": client.session_id,
                                "Transport": (
                                    "RTP/AVP/TCP;unicast;"
                                    f"interleaved={client.interleaved_rtp_channel}-"
                                    f"{client.interleaved_rtp_channel + 1};"
                                    f"ssrc={client.ssrc:08x};mode=play"
                                ),
                            },
                        ),
                    )
                elif method == "PLAY":
                    if client.play_task is None:
                        if upstream is None:
                            upstream = await _get_whep_manager(self.hass)._ensure_upstream(
                                camera
                            )
                        if encoder is None:
                            encoder = H264Encoder()
                        relay_track = upstream.relay.subscribe(
                            upstream.video_track,
                            buffered=False,
                        )
                        if self._last_errors.pop(device_id, None) is not None:
                            camera.async_write_ha_state()
                        client.play_task = self.hass.async_create_background_task(
                            self._stream_rtp(
                                track=relay_track,
                                client=client,
                                encoder=encoder,
                                get_cached_sps=lambda: cached_sps,
                                get_cached_pps=lambda: cached_pps,
                                set_cached_sps=lambda value: _set_cached("sps", value),
                                set_cached_pps=lambda value: _set_cached("pps", value),
                                device_id=device_id,
                                camera=camera,
                            ),
                            f"petkit rtsp play {camera.device.id}",
                        )
                    await self._write_response(
                        client,
                        _build_rtsp_response(
                            200,
                            cseq,
                            headers={
                                "Session": client.session_id,
                                "RTP-Info": (
                                    f"url={request_uri}/trackID=0;"
                                    f"seq={client.sequence_number};"
                                    f"rtptime={client.last_timestamp}"
                                ),
                            },
                        ),
                    )
                elif method in {"GET_PARAMETER", "SET_PARAMETER"}:
                    await self._write_response(
                        client,
                        _build_rtsp_response(
                            200,
                            cseq,
                            headers={"Session": client.session_id},
                        ),
                    )
                elif method == "TEARDOWN":
                    await self._write_response(
                        client,
                        _build_rtsp_response(
                            200,
                            cseq,
                            headers={"Session": client.session_id},
                        ),
                    )
                    break
                else:
                    await self._write_response(
                        client,
                        _build_rtsp_response(
                            405,
                            cseq,
                            status_text="Method Not Allowed",
                        ),
                    )
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as err:  # noqa: BLE001
            self._last_errors[device_id] = str(err)
            camera.async_write_ha_state()
            LOGGER.debug("Local RTSP session failed for %s: %s", device_id, err)
        finally:
            if session is not None:
                session.clients.pop(client.session_id, None)
            if client.play_task is not None:
                client.play_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await client.play_task
            if session is not None and not session.clients:
                await _get_whep_manager(self.hass).maybe_close_idle_upstream(device_id)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _stream_rtp(
        self,
        *,
        track: Any,
        client: RtspClientSession,
        encoder: H264Encoder,
        get_cached_sps,
        get_cached_pps,
        set_cached_sps,
        set_cached_pps,
        device_id: str,
        camera: PetkitWebRTCCamera,
    ) -> None:
        """Packetize one track into RTSP interleaved RTP."""
        try:
            while True:
                frame_or_packet = await track.recv()
                if isinstance(frame_or_packet, Packet):
                    packet, cached_sps, cached_pps = _augment_h264_packet(
                        frame_or_packet,
                        get_cached_sps(),
                        get_cached_pps(),
                    )
                    set_cached_sps(cached_sps)
                    set_cached_pps(cached_pps)
                    payloads, timestamp = encoder.pack(packet)
                else:
                    payloads, timestamp = encoder.encode(frame_or_packet)

                client.last_timestamp = timestamp
                for index, payload in enumerate(payloads):
                    rtp_packet = RtpPacket(
                        payload_type=96,
                        sequence_number=client.sequence_number,
                        timestamp=timestamp,
                    )
                    rtp_packet.ssrc = client.ssrc
                    rtp_packet.payload = payload
                    rtp_packet.marker = 1 if index == len(payloads) - 1 else 0
                    packet_bytes = rtp_packet.serialize(HeaderExtensionsMap())
                    interleaved = (
                        b"$"
                        + bytes([client.interleaved_rtp_channel])
                        + len(packet_bytes).to_bytes(2, "big")
                        + packet_bytes
                    )
                    await self._write_response(client, interleaved)
                    client.sequence_number = (client.sequence_number + 1) % 65536
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            self._last_errors[device_id] = str(err)
            camera.async_write_ha_state()
            LOGGER.debug("RTSP RTP streaming failed for %s: %s", device_id, err)
            raise

    async def _write_response(
        self,
        client: RtspClientSession,
        payload: bytes,
    ) -> None:
        """Serialize writes per client."""
        async with client.writer_lock:
            client.writer.write(payload)
            await client.writer.drain()

    @staticmethod
    async def _read_rtsp_request(
        reader: asyncio.StreamReader,
    ) -> tuple[str, dict[str, str], bytes] | None:
        """Read one RTSP request and skip interleaved RTCP frames from the client."""
        first_byte = await reader.readexactly(1)
        if first_byte == b"$":
            header = await reader.readexactly(3)
            frame_length = int.from_bytes(header[1:], "big")
            if frame_length:
                await reader.readexactly(frame_length)
            return None

        raw = first_byte + await reader.readuntil(_REQUEST_END)
        request_line, headers, body = _split_rtsp_request(raw)
        with contextlib.suppress(ValueError):
            content_length = int(headers.get("content-length", "0") or "0")
            if content_length > len(body):
                body += await reader.readexactly(content_length - len(body))
            return request_line, headers, body
        return request_line, headers, body

    @staticmethod
    def _parse_interleaved_channel(transport: str) -> int:
        """Return the RTP interleaved channel from a RTSP Transport header."""
        for section in transport.split(";"):
            if not section.lower().startswith("interleaved="):
                continue
            channels = section.partition("=")[2]
            first_channel = channels.partition("-")[0].strip()
            if first_channel.isdigit():
                return int(first_channel)
        return _INTERLEAVED_DEFAULT[0]

    @staticmethod
    def _build_sdp(request_uri: str) -> bytes:
        """Build a minimal SDP for one H.264 video track."""
        sdp = "\r\n".join(
            [
                "v=0",
                "o=- 0 0 IN IP4 127.0.0.1",
                "s=PetKit",
                "t=0 0",
                "a=control:*",
                "m=video 0 RTP/AVP 96",
                "c=IN IP4 0.0.0.0",
                "a=rtpmap:96 H264/90000",
                "a=fmtp:96 packetization-mode=1",
                "a=control:trackID=0",
                "",
                "",
            ]
        )
        return sdp.encode()

    @staticmethod
    def _preferred_port(device_id: str) -> int:
        """Return the deterministic RTSP port for one device."""
        return _PORT_RANGE_START + (zlib.crc32(device_id.encode()) % _PORT_RANGE_SIZE)

    def _port_for_device(self, device_id: str) -> int:
        """Return the active or deterministic RTSP port for one device."""
        active = self._sessions.get(device_id)
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
    """Return the shared RTSP passthrough manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get("rtsp_proxy_manager")
    if manager is None:
        manager = PetkitRTSPProxyManager(hass)
        domain_data["rtsp_proxy_manager"] = manager
    return manager
