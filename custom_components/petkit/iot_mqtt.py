"""Petkit IoT MQTT listener (experimental).

The official Petkit mobile app connects to an MQTT broker (Aliyun-style topics) to
receive near real-time device events. We use those messages as a trigger to refresh
data from the REST API, providing faster state updates in Home Assistant without
having to reverse-engineer every message type.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass
import re

from pypetkitapi.client import PetKitClient

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import LOGGER
from .coordinator import PetkitDataUpdateCoordinator

try:  # pragma: no cover
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore[assignment]


_HOST_PORT_RE = re.compile(r"^(?P<host>.+?)(?::(?P<port>\d+))?$")


@dataclass(frozen=True)
class _MqttEndpoint:
    host: str
    port: int


def _parse_mqtt_host(raw: str, *, default_port: int = 1883) -> _MqttEndpoint:
    """Parse `host[:port]` as returned by Petkit's IoT endpoint."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty mqtt host")

    # Very small parser; Petkit appears to return host:port for IPv4/hostname.
    # If we ever encounter IPv6 literals, we can extend this to handle [::1]:1883.
    match = _HOST_PORT_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid mqtt host: {raw!r}")

    host = (match.group("host") or "").strip()
    port_raw = match.group("port")
    port = int(port_raw) if port_raw else default_port

    if not host:
        raise ValueError(f"Invalid mqtt host: {raw!r}")
    return _MqttEndpoint(host=host, port=port)


class PetkitIotMqttListener:
    """Connect to Petkit's MQTT broker and refresh HA data on events."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: PetKitClient,
        coordinator: PetkitDataUpdateCoordinator,
        *,
        refresh_debounce_s: float = 2.0,
    ) -> None:
        self.hass = hass
        self.client = client
        self.coordinator = coordinator
        self.refresh_debounce_s = refresh_debounce_s

        self._mqtt_client = None
        self._subscribe_topics: list[str] = []
        self._refresh_task: asyncio.Task | None = None
        self._started = False
        self._petkit_device_name: str | None = None
        self._petkit_product_key: str | None = None
        self._recent_messages: deque[dict] = deque(maxlen=200)

    async def async_start(self) -> None:
        """Start the MQTT connection in the background."""
        if self._started:
            return
        self._started = True

        if mqtt is None:  # pragma: no cover
            LOGGER.error("paho-mqtt not installed; Petkit MQTT listener cannot start")
            return

        try:
            iot = await self.client.get_iot_mqtt_config()
        except Exception as err:  # noqa: BLE001
            LOGGER.warning("Failed to fetch Petkit IoT MQTT config: %s", err)
            return

        if not (iot.mqtt_host and iot.device_name and iot.device_secret and iot.product_key):
            LOGGER.warning("Petkit IoT MQTT config is missing required fields; listener disabled")
            return

        try:
            endpoint = _parse_mqtt_host(iot.mqtt_host)
        except ValueError as err:
            LOGGER.warning("Invalid Petkit MQTT host %r: %s", iot.mqtt_host, err)
            return

        self._petkit_device_name = iot.device_name
        self._petkit_product_key = iot.product_key
        base = f"/{iot.product_key}/{iot.device_name}/user"
        # The official Android app subscribes to /user/get and uses /user/update as its will topic.
        # Subscribing to both gives us a better chance to observe relevant traffic if ACLs allow it.
        self._subscribe_topics = [f"{base}/get", f"{base}/update"]

        # MQTT 3.1.1 with persistent session (mirrors the Android app).
        client_id = iot.device_name
        paho_client = mqtt.Client(client_id=client_id, clean_session=False, protocol=mqtt.MQTTv311)
        paho_client.username_pw_set(iot.device_name, iot.device_secret)
        paho_client.reconnect_delay_set(min_delay=1, max_delay=30)

        paho_client.on_connect = self._on_connect
        paho_client.on_disconnect = self._on_disconnect
        paho_client.on_message = self._on_message

        self._mqtt_client = paho_client

        try:
            paho_client.connect_async(endpoint.host, endpoint.port, keepalive=60)
            paho_client.loop_start()
        except Exception as err:  # noqa: BLE001
            LOGGER.warning("Failed to start Petkit MQTT connection: %s", err)
            self._mqtt_client = None
            self._subscribe_topic = None
            return

        LOGGER.info(
            "Petkit MQTT listener started (broker=%s:%s, topics=%s)",
            endpoint.host,
            endpoint.port,
            self._subscribe_topics,
        )

    async def async_stop(self) -> None:
        """Stop the MQTT listener and cleanup resources."""
        self._started = False

        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None

        client = self._mqtt_client
        self._mqtt_client = None
        self._subscribe_topics = []
        self._petkit_device_name = None
        self._petkit_product_key = None

        if client is None:
            return

        try:
            client.disconnect()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Petkit MQTT disconnect raised", exc_info=True)

        try:
            client.loop_stop()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Petkit MQTT loop_stop raised", exc_info=True)

        LOGGER.info("Petkit MQTT listener stopped")

    def _on_connect(self, client, userdata, flags, rc, *args, **kwargs) -> None:  # noqa: ANN001
        topics = self._subscribe_topics
        if rc != 0:
            LOGGER.warning("Petkit MQTT connect failed (rc=%s)", rc)
            return
        if not topics:
            LOGGER.warning("Petkit MQTT connected but subscribe topics are missing")
            return
        try:
            for topic in topics:
                client.subscribe(topic, qos=0)
            LOGGER.debug("Petkit MQTT subscribed to %s", topics)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Petkit MQTT subscribe failed", exc_info=True)

    def _on_disconnect(self, client, userdata, rc, *args, **kwargs) -> None:  # noqa: ANN001
        # paho-mqtt will try to reconnect (reconnect_on_failure / reconnect_delay_set),
        # so we just log.
        LOGGER.debug("Petkit MQTT disconnected (rc=%s)", rc)

    def _on_message(self, client, userdata, msg) -> None:  # noqa: ANN001
        # Called from the paho-mqtt network thread. Always hop back onto HA's loop.
        topic = getattr(msg, "topic", None)
        payload = getattr(msg, "payload", b"")
        self.hass.loop.call_soon_threadsafe(self._handle_message, topic, payload)

    def _handle_message(self, topic: str | None, payload: bytes) -> None:
        """Handle an incoming MQTT message on the HA event loop thread."""
        payload_encoding = "utf-8"
        try:
            payload_text = payload.decode("utf-8")
        except UnicodeDecodeError:
            payload_text = base64.b64encode(payload).decode("ascii")
            payload_encoding = "base64"

        event_data = {
            "topic": topic or "",
            "payload": payload_text,
            "payload_encoding": payload_encoding,
            "received_at": dt_util.utcnow().isoformat(),
            "petkit_device_name": self._petkit_device_name or "",
            "petkit_product_key": self._petkit_product_key or "",
        }

        self._recent_messages.append(event_data)
        self.hass.bus.async_fire("petkit_mqtt_message", event_data)

        self._schedule_refresh()

    def get_recent_messages(
        self, *, limit: int = 1, topic_contains: str | None = None
    ) -> list[dict]:
        """Return up to `limit` most recent messages, optionally filtered by topic substring."""
        msgs = list(self._recent_messages)
        if topic_contains:
            msgs = [m for m in msgs if topic_contains in m.get("topic", "")]
        if limit <= 0:
            return []
        return msgs[-limit:]

    def _schedule_refresh(self) -> None:
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._refresh_task = self.hass.async_create_task(self._debounced_refresh())

    async def _debounced_refresh(self) -> None:
        # Small debounce so bursts of MQTT events don't spam the REST API.
        await asyncio.sleep(self.refresh_debounce_s)
        try:
            # Also temporarily increase polling rate after events for better UX.
            self.coordinator.enable_smart_polling(12)
            await self.coordinator.async_request_refresh()
        except Exception:  # noqa: BLE001
            LOGGER.debug("MQTT-triggered refresh failed", exc_info=True)
