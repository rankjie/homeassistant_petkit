"""Helpers for H.264 packet passthrough over aiortc."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import logging
from typing import Any

from av import Packet

from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from aiortc.mediastreams import VIDEO_TIME_BASE

LOGGER = logging.getLogger(__name__)

_PATCHED = False
_ORIGINAL_DECODER_WORKER = None


class EncodedPacketStreamTrack(MediaStreamTrack):
    """MediaStreamTrack that carries pre-encoded packets."""

    def __init__(
        self,
        kind: str,
        *,
        track_id: str | None = None,
        on_started: Callable[[], None] | None = None,
        maxsize: int = 1,
    ) -> None:
        """Initialize the packet-backed track."""
        super().__init__()
        self.kind = kind
        if track_id is not None:
            self._id = track_id
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._queue._petkit_encoded_passthrough = True  # type: ignore[attr-defined]
        self._queue._petkit_track = self  # type: ignore[attr-defined]
        self._on_started = on_started
        self._started = False

    async def recv(self) -> Packet:
        """Receive the next encoded packet."""
        if self.readyState != "live":
            raise MediaStreamError

        packet = await self._queue.get()
        if packet is None:
            self.stop()
            raise MediaStreamError
        return packet

    def enqueue_packet(self, packet: Packet) -> None:
        """Queue one packet, dropping stale backlog if needed."""
        if self.readyState != "live":
            return

        if not self._started:
            self._started = True
            if self._on_started is not None:
                self._on_started()

        while self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()

        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(packet)

    def enqueue_end_of_stream(self) -> None:
        """Signal track termination."""
        while self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)


def ensure_aiortc_decoder_passthrough_patch() -> None:
    """Patch aiortc's decoder worker to support H.264 packet passthrough."""
    global _PATCHED, _ORIGINAL_DECODER_WORKER
    if _PATCHED:
        return

    import aiortc.rtcrtpreceiver as rtcrtpreceiver

    _ORIGINAL_DECODER_WORKER = rtcrtpreceiver.decoder_worker

    def _patched_decoder_worker(
        loop: asyncio.AbstractEventLoop,
        input_q,
        output_q,
    ) -> None:
        passthrough_track = getattr(output_q, "_petkit_track", None)
        passthrough_enabled = bool(
            getattr(output_q, "_petkit_encoded_passthrough", False)
        )

        if not passthrough_enabled or passthrough_track is None:
            _ORIGINAL_DECODER_WORKER(loop, input_q, output_q)
            return

        codec_name = None
        decoder = None

        while True:
            task = input_q.get()
            if task is None:
                loop.call_soon_threadsafe(passthrough_track.enqueue_end_of_stream)
                break

            codec, encoded_frame = task

            if codec.name == "h264":
                packet = Packet(encoded_frame.data)
                packet.pts = encoded_frame.timestamp
                packet.time_base = VIDEO_TIME_BASE
                loop.call_soon_threadsafe(
                    passthrough_track.enqueue_packet, packet
                )
                continue

            if codec.name != codec_name:
                decoder = rtcrtpreceiver.get_decoder(codec)
                codec_name = codec.name

            for frame in decoder.decode(encoded_frame):
                loop.call_soon_threadsafe(
                    passthrough_track.enqueue_packet, frame
                )

        if decoder is not None:
            del decoder

    rtcrtpreceiver.decoder_worker = _patched_decoder_worker
    _PATCHED = True
    LOGGER.debug("Enabled aiortc H.264 packet passthrough patch")


def replace_track_with_encoded_passthrough(
    peer_connection: Any,
    source_track: Any,
    *,
    on_started: Callable[[], None] | None = None,
) -> EncodedPacketStreamTrack | None:
    """Replace the receiver track for one transceiver with a packet track."""
    for transceiver in peer_connection.getTransceivers():
        receiver = transceiver.receiver
        if receiver.track is not source_track:
            continue

        packet_track = EncodedPacketStreamTrack(
            source_track.kind,
            track_id=source_track.id,
            on_started=on_started,
        )
        receiver._track = packet_track
        return packet_track

    return None
