"""Custom integration to integrate Petkit Smart Devices with Home Assistant."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from aiohttp import web
from pypetkitapi import PetKitClient

import voluptuous as vol

from homeassistant.components.http import HomeAssistantView
from homeassistant.const import (
    CONF_PASSWORD,
    CONF_REGION,
    CONF_SCAN_INTERVAL,
    CONF_TIME_ZONE,
    CONF_USERNAME,
    Platform,
)
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.loader import async_get_loaded_integration

from .const import (
    BT_SECTION,
    CONF_SCAN_INTERVAL_BLUETOOTH,
    CONF_SCAN_INTERVAL_MEDIA,
    CONF_REALTIME_MQTT,
    COORDINATOR,
    COORDINATOR_BLUETOOTH,
    COORDINATOR_MEDIA,
    DEFAULT_REALTIME_MQTT,
    DOMAIN,
    EVENT_MQTT_DUMP,
    LOGGER,
    MEDIA_SECTION,
    SERVICE_MQTT_DUMP,
)
from .coordinator import (
    PetkitBluetoothUpdateCoordinator,
    PetkitDataUpdateCoordinator,
    PetkitMediaUpdateCoordinator,
)
from .data import PetkitData

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .data import PetkitConfigEntry

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.SWITCH,
    Platform.LIGHT,
    Platform.TEXT,
    Platform.BUTTON,
    Platform.NUMBER,
    Platform.SELECT,
    Platform.IMAGE,
    Platform.FAN,
]


class PetkitSessionView(HomeAssistantView):
    """Expose PetKit session tokens for external consumers (e.g. Scrypted)."""

    url = "/api/petkit/session"
    name = "api:petkit:session"
    requires_auth = True

    async def get(self, request: web.Request) -> web.Response:
        """Return session info for one or all PetKit accounts."""
        hass = request.app["hass"]
        username_filter = request.query.get("username")

        sessions = []
        for entry in hass.config_entries.async_entries(DOMAIN):
            runtime = getattr(entry, "runtime_data", None)
            if not runtime or not runtime.client:
                continue
            client: PetKitClient = runtime.client
            account_username = entry.data.get(CONF_USERNAME, "")

            if username_filter and account_username != username_filter:
                continue

            await client.validate_session()
            if client._session:
                sessions.append({
                    "username": account_username,
                    "session_id": client._session.id,
                    "base_url": client.req.base_url,
                })

        if username_filter:
            if not sessions:
                return self.json_message(
                    f"No active session for {username_filter}", 404
                )
            return self.json(sessions[0])

        if not sessions:
            return self.json_message("No active PetKit sessions", 404)

        return self.json(sessions)


class PetkitIotView(HomeAssistantView):
    """Expose PetKit IoT/MQTT credentials for external consumers."""

    url = "/api/petkit/iot"
    name = "api:petkit:iot"
    requires_auth = True

    async def get(self, request: web.Request) -> web.Response:
        """Return IoT MQTT config for one or all PetKit accounts."""
        hass = request.app["hass"]
        username_filter = request.query.get("username")

        results = []
        for entry in hass.config_entries.async_entries(DOMAIN):
            runtime = getattr(entry, "runtime_data", None)
            if not runtime or not runtime.client:
                continue
            client: PetKitClient = runtime.client
            account_username = entry.data.get(CONF_USERNAME, "")

            if username_filter and account_username != username_filter:
                continue

            try:
                iot = await client.get_iot_mqtt_config()
                results.append({
                    "username": account_username,
                    "deviceName": iot.device_name,
                    "deviceSecret": iot.device_secret,
                    "productKey": iot.product_key,
                    "mqttHost": iot.mqtt_host,
                    "iotInstanceId": iot.iot_instance_id,
                })
            except Exception as err:  # noqa: BLE001
                results.append({
                    "username": account_username,
                    "error": str(err),
                })

        if username_filter:
            if not results:
                return self.json_message(
                    f"No IoT config for {username_filter}", 404
                )
            return self.json(results[0])

        if not results:
            return self.json_message("No PetKit accounts found", 404)

        return self.json(results)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: PetkitConfigEntry,
) -> bool:
    """Set up this integration using UI."""

    # Register API views once (idempotent — HA deduplicates by name)
    hass.http.register_view(PetkitSessionView())
    hass.http.register_view(PetkitIotView())

    country_from_ha = hass.config.country
    tz_from_ha = hass.config.time_zone

    coordinator = PetkitDataUpdateCoordinator(
        hass=hass,
        logger=LOGGER,
        name=f"{DOMAIN}.devices",
        update_interval=timedelta(seconds=entry.options[CONF_SCAN_INTERVAL]),
        config_entry=entry,
    )
    coordinator_media = PetkitMediaUpdateCoordinator(
        hass=hass,
        logger=LOGGER,
        name=f"{DOMAIN}.medias",
        update_interval=timedelta(
            minutes=entry.options[MEDIA_SECTION][CONF_SCAN_INTERVAL_MEDIA]
        ),
        config_entry=entry,
        data_coordinator=coordinator,
    )
    coordinator_bluetooth = PetkitBluetoothUpdateCoordinator(
        hass=hass,
        logger=LOGGER,
        name=f"{DOMAIN}.bluetooth",
        update_interval=timedelta(
            minutes=entry.options[BT_SECTION][CONF_SCAN_INTERVAL_BLUETOOTH]
        ),
        config_entry=entry,
        data_coordinator=coordinator,
    )
    entry.runtime_data = PetkitData(
        client=PetKitClient(
            username=entry.data[CONF_USERNAME],
            password=entry.data[CONF_PASSWORD],
            region=entry.data.get(CONF_REGION, country_from_ha),
            timezone=entry.data.get(CONF_TIME_ZONE, tz_from_ha),
            session=async_get_clientsession(hass),
        ),
        integration=async_get_loaded_integration(hass, entry.domain),
        coordinator=coordinator,
        coordinator_media=coordinator_media,
        coordinator_bluetooth=coordinator_bluetooth,
    )

    await coordinator.async_config_entry_first_refresh()
    await coordinator_media.async_config_entry_first_refresh()
    await coordinator_bluetooth.async_config_entry_first_refresh()

    if entry.options.get(CONF_REALTIME_MQTT, DEFAULT_REALTIME_MQTT):
        from .iot_mqtt import PetkitIotMqttListener

        mqtt_listener = PetkitIotMqttListener(
            hass=hass,
            client=entry.runtime_data.client,
            coordinator=coordinator,
        )
        entry.runtime_data.mqtt_listener = mqtt_listener
        await mqtt_listener.async_start()

    if not hass.services.has_service(DOMAIN, SERVICE_MQTT_DUMP):
        schema = vol.Schema(
            {
                vol.Optional("entry_id"): str,
                vol.Optional("limit", default=1): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=200)
                ),
                vol.Optional("topic_contains"): str,
            }
        )

        async def _async_mqtt_dump(call) -> None:  # noqa: ANN001
            target_entry_id = call.data.get("entry_id")
            limit = call.data["limit"]
            topic_contains = call.data.get("topic_contains")

            messages: list[dict] = []
            found_listener = False
            for ent in hass.config_entries.async_entries(DOMAIN):
                if target_entry_id and ent.entry_id != target_entry_id:
                    continue
                runtime = getattr(ent, "runtime_data", None)
                mqtt_enabled = ent.options.get(CONF_REALTIME_MQTT, DEFAULT_REALTIME_MQTT)
                listener = getattr(runtime, "mqtt_listener", None)

                if mqtt_enabled and listener is None:
                    LOGGER.warning(
                        "Petkit MQTT dump: MQTT is enabled for %s but no listener is running",
                        ent.title,
                    )
                    continue

                if listener is None:
                    continue

                found_listener = True
                diag = listener.diagnostics
                LOGGER.info(
                    "Petkit MQTT dump: account=%s status=%s messages_received=%s buffer_size=%s",
                    ent.title,
                    diag["status"],
                    diag["messages_received"],
                    diag["buffer_size"],
                )

                for msg in listener.get_recent_messages(
                    limit=limit, topic_contains=topic_contains
                ):
                    enriched = dict(msg)
                    enriched["entry_id"] = ent.entry_id
                    enriched["account"] = ent.title
                    messages.append(enriched)

            if not found_listener:
                LOGGER.info("Petkit MQTT dump: no MQTT listeners active")
            elif not messages:
                LOGGER.info("Petkit MQTT dump: no messages available")
            else:
                LOGGER.info("Petkit MQTT dump: dumping %s message(s)", len(messages))
                for msg in messages:
                    LOGGER.info(
                        "Petkit MQTT dump entry=%s topic=%s payload_encoding=%s received_at=%s payload=%s",
                        msg.get("entry_id"),
                        msg.get("topic"),
                        msg.get("payload_encoding"),
                        msg.get("received_at"),
                        msg.get("payload"),
                    )

            hass.bus.async_fire(EVENT_MQTT_DUMP, {"messages": messages})

        hass.services.async_register(
            DOMAIN,
            SERVICE_MQTT_DUMP,
            _async_mqtt_dump,
            schema=schema,
        )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    hass.data[DOMAIN][COORDINATOR] = coordinator
    hass.data[DOMAIN][COORDINATOR_MEDIA] = coordinator
    hass.data[DOMAIN][COORDINATOR_BLUETOOTH] = coordinator

    return True


async def async_unload_entry(
    hass: HomeAssistant,
    entry: PetkitConfigEntry,
) -> bool:
    """Handle removal of an entry."""
    mqtt_listener = getattr(entry.runtime_data, "mqtt_listener", None)
    if mqtt_listener is not None:
        await mqtt_listener.async_stop()

    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant,
    entry: PetkitConfigEntry,
) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)


async def async_update_options(hass: HomeAssistant, entry: PetkitConfigEntry) -> None:
    """Update options."""

    await hass.config_entries.async_reload(entry.entry_id)


async def async_remove_config_entry_device(
    hass: HomeAssistant, config_entry: PetkitConfigEntry, device_entry: dr.DeviceEntry
) -> bool:
    """Remove a config entry from a device."""
    return True
