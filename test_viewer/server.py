"""Local test server for PetKit camera WebRTC streaming.

Usage:
    # Direct login:
    python server.py --username USER --password PASS [--region cn] [--timezone Asia/Shanghai]

    # Reuse session from running HA instance:
    python server.py --ha-url http://192.168.1.28:8123 --ha-token YOUR_LONG_LIVED_TOKEN

Dependencies:
    pip install aiohttp rankjie-pypetkitapi websockets sdp-transform webrtc-models

The server imports agora modules from ../custom_components/petkit via sys.path shimming.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Path shimming: allow importing agora_* modules from the HA integration
# without pulling in homeassistant itself.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PETKIT_PKG = _REPO_ROOT / "custom_components" / "petkit"

# Create a minimal 'petkit' package so relative imports inside agora_* work.
_petkit_mod = types.ModuleType("petkit")
_petkit_mod.__path__ = [str(_PETKIT_PKG)]
_petkit_mod.__package__ = "petkit"
sys.modules["petkit"] = _petkit_mod

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402
from pypetkitapi import (  # noqa: E402
    FEEDER_WITH_CAMERA,
    LITTER_WITH_CAMERA,
    Feeder,
    Litter,
    LiveFeed,
    PetKitClient,
)

from petkit.agora_api import SERVICE_IDS, AgoraAPIClient, AgoraResponse  # noqa: E402
from petkit.agora_rtm import AgoraRTMSignaling  # noqa: E402
from petkit.agora_websocket import AgoraWebSocketHandler  # noqa: E402

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
LOG = logging.getLogger("test_viewer")

AGORA_APP_ID = "244c49951296440cbc1e3b937bf5e410"

# ---------------------------------------------------------------------------
# Global state (single-user test server)
# ---------------------------------------------------------------------------
_clients: dict[str, PetKitClient] = {}  # keyed by account username/label
_device_to_client: dict[int, PetKitClient] = {}  # device_id -> client
_http_session: aiohttp.ClientSession | None = None
_rtm: AgoraRTMSignaling | None = None
_ws_handler: AgoraWebSocketHandler | None = None
_agora_response: AgoraResponse | None = None
_live_feed: LiveFeed | None = None
_current_device_id: int | None = None


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------
async def handle_index(request: web.Request) -> web.Response:
    html = (Path(__file__).parent / "index.html").read_text()
    return web.Response(text=html, content_type="text/html")


async def _make_ha_client(
    session_id: str, base_url: str, http_session: aiohttp.ClientSession,
    timezone: str = "Asia/Shanghai",
) -> PetKitClient:
    """Create a PetKitClient from an HA session."""
    from datetime import datetime, timezone as tz
    from pypetkitapi.containers import SessionInfo

    client = PetKitClient(
        username="ha-session",
        password="unused",
        region="cn",
        timezone=timezone,
        session=http_session,
    )
    client._session = SessionInfo(
        id=session_id,
        userId="0",
        expiresIn=86400 * 365,
        createdAt=datetime.now(tz=tz.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
    )
    client.req.base_url = base_url
    return client


async def handle_login(request: web.Request) -> web.Response:
    """Login and return camera-capable devices."""
    global _http_session, _clients, _device_to_client

    body = await request.json()
    args = request.app["args"]

    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()

    _clients.clear()
    _device_to_client.clear()

    ha_sessions = getattr(args, "ha_sessions", None)

    if ha_sessions:
        # Mode 1: multiple HA sessions
        for ha in ha_sessions:
            label = ha.get("username", ha["session_id"][:12])
            client = await _make_ha_client(
                ha["session_id"], ha["base_url"], _http_session,
                timezone=getattr(args, "timezone", "Asia/Shanghai"),
            )
            LOG.info("Using HA session: %s -> %s", label, ha["base_url"])
            try:
                await client.get_devices_data()
                _clients[label] = client
            except Exception as e:
                LOG.error("Failed to load devices for %s: %s", label, e)
    else:
        # Mode 2: normal username/password login
        username = body.get("username") or args.username
        password = body.get("password") or args.password
        region = body.get("region") or args.region
        timezone = body.get("timezone") or args.timezone

        client = PetKitClient(
            username=username,
            password=password,
            region=region,
            timezone=timezone,
            session=_http_session,
        )
        await client.login()
        await client.get_devices_data()
        _clients[username] = client

    devices = []
    for label, client in _clients.items():
        for device_id, entity in client.petkit_entities.items():
            if not isinstance(entity, (Feeder, Litter)):
                continue
            dt = getattr(entity.device_nfo, "device_type", "") if entity.device_nfo else ""
            if dt not in (*FEEDER_WITH_CAMERA, *LITTER_WITH_CAMERA):
                continue
            name = getattr(entity.device_nfo, "device_name", "") if entity.device_nfo else ""
            has_live = entity.live_feed is not None
            _device_to_client[device_id] = client
            devices.append({
                "id": device_id,
                "name": name or f"Device {device_id}",
                "type": dt,
                "hasLiveFeed": has_live,
                "account": label,
            })

    LOG.info("Found %d camera-capable devices across %d accounts", len(devices), len(_clients))
    return web.json_response({"devices": devices})


async def handle_start(request: web.Request) -> web.Response:
    """Start live: fetch tokens, Agora choose_server, RTM start_live."""
    global _rtm, _ws_handler, _agora_response, _live_feed, _current_device_id

    body = await request.json()
    device_id = int(body["deviceId"])
    _current_device_id = device_id

    client = _device_to_client.get(device_id)
    if client is None:
        return web.json_response({"error": "Device not found or not logged in"}, status=400)

    # Re-fetch device data to get fresh live feed tokens (uses GET now)
    await client.get_devices_data()

    entity = client.petkit_entities.get(device_id)
    if not entity or not isinstance(entity, (Feeder, Litter)):
        return web.json_response({"error": "Device not found"}, status=404)

    live_feed = entity.live_feed
    if not live_feed or not live_feed.channel_id or not live_feed.rtc_token:
        # Try temporaryOpenCamera
        dt = entity.device_nfo.device_type if entity.device_nfo else ""
        LOG.info("No live feed, trying temporaryOpenCamera for %s", dt)
        try:
            await client.temporary_open_camera(dt, device_id)
            await asyncio.sleep(2)
            await client.get_devices_data()
            entity = client.petkit_entities.get(device_id)
            live_feed = entity.live_feed if entity else None
        except Exception as e:
            LOG.warning("temporaryOpenCamera failed: %s", e)

        if not live_feed or not live_feed.channel_id or not live_feed.rtc_token:
            return web.json_response({"error": "No live feed tokens available"}, status=503)

    _live_feed = live_feed

    # Agora choose_server
    async with AgoraAPIClient() as agora_client:
        _agora_response = await agora_client.choose_server(
            app_id=AGORA_APP_ID,
            token=live_feed.rtc_token,
            channel_name=live_feed.channel_id,
            user_id=0,
            service_flags=[
                SERVICE_IDS["CHOOSE_SERVER"],
                SERVICE_IDS["CLOUD_PROXY_FALLBACK"],
            ],
        )

    # ICE servers for the browser
    ice_servers_raw = _agora_response.get_ice_servers(use_all_turn_servers=False)
    ice_servers = [
        {"urls": s.urls, "username": s.username, "credential": s.credential}
        for s in ice_servers_raw
    ]

    # RTM start_live
    _rtm = AgoraRTMSignaling(AGORA_APP_ID)
    rtm_ok = await _rtm.start_live(live_feed)

    # Prepare WebSocket handler
    _ws_handler = AgoraWebSocketHandler()

    LOG.info(
        "Live started: channel=%s rtm=%s ice_servers=%d",
        live_feed.channel_id,
        rtm_ok,
        len(ice_servers),
    )

    return web.json_response({
        "channel": live_feed.channel_id,
        "rtmOk": rtm_ok,
        "iceServers": ice_servers,
    })


async def handle_offer(request: web.Request) -> web.Response:
    """Accept browser SDP offer, perform Agora join_v3, return SDP answer."""
    if _ws_handler is None or _agora_response is None or _live_feed is None:
        return web.json_response({"error": "Stream not started"}, status=400)

    body = await request.json()
    offer_sdp = body["offer"]
    session_id = body.get("sessionId", "test-session-1")

    # Add any trickle candidates collected before offer
    candidates = body.get("candidates", [])
    from webrtc_models import RTCIceCandidateInit
    for c in candidates:
        _ws_handler.add_ice_candidate(RTCIceCandidateInit(
            candidate=c.get("candidate", ""),
            sdp_mid=c.get("sdpMid"),
            sdp_m_line_index=c.get("sdpMLineIndex"),
        ))

    answer_sdp = await _ws_handler.connect_and_join(
        live_feed=_live_feed,
        offer_sdp=offer_sdp,
        session_id=session_id,
        app_id=AGORA_APP_ID,
        agora_response=_agora_response,
    )

    if answer_sdp:
        LOG.info("WebRTC answer generated (%d bytes)", len(answer_sdp))
        return web.json_response({"answer": answer_sdp})

    return web.json_response(
        {"error": "Agora negotiation failed - no SDP answer"},
        status=502,
    )


async def handle_candidate(request: web.Request) -> web.Response:
    """Accept trickle ICE candidate from browser."""
    if _ws_handler is None:
        return web.json_response({"error": "No active session"}, status=400)

    body = await request.json()
    from webrtc_models import RTCIceCandidateInit
    _ws_handler.add_ice_candidate(RTCIceCandidateInit(
        candidate=body.get("candidate", ""),
        sdp_mid=body.get("sdpMid"),
        sdp_m_line_index=body.get("sdpMLineIndex"),
    ))
    return web.json_response({"ok": True})


async def handle_stop(request: web.Request) -> web.Response:
    """Stop live stream and clean up."""
    global _rtm, _ws_handler, _agora_response, _live_feed

    tasks = []
    if _ws_handler:
        tasks.append(_ws_handler.disconnect())
    if _rtm:
        tasks.append(_rtm.stop_live(send_stop=True))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    _ws_handler = None
    _rtm = None
    _agora_response = None
    _live_feed = None

    LOG.info("Stream stopped")
    return web.json_response({"ok": True})


async def on_shutdown(app: web.Application) -> None:
    global _http_session
    tasks = []
    if _ws_handler:
        tasks.append(_ws_handler.disconnect())
    if _rtm:
        tasks.append(_rtm.stop_live(send_stop=True))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None


async def _fetch_ha_sessions(ha_url: str, ha_token: str, account: str | None) -> list[dict]:
    """Fetch PetKit sessions from running HA instance."""
    url = f"{ha_url.rstrip('/')}/api/petkit/session"
    if account:
        url += f"?username={account}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"Authorization": f"Bearer {ha_token}"}) as resp:
            data = await resp.json()
            if isinstance(data, dict):
                return [data]
            if isinstance(data, list):
                if not data:
                    raise ValueError("No active PetKit sessions in HA")
                return data
            raise ValueError(f"Unexpected response: {data}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PetKit Camera Test Viewer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--username", "-u", help="PetKit username (email/phone)")
    group.add_argument("--ha-url", help="HA base URL (e.g. http://192.168.1.28:8123)")
    parser.add_argument("--password", "-p", default="")
    parser.add_argument("--ha-token", help="HA long-lived access token")
    parser.add_argument("--ha-account", help="Filter HA session by PetKit username")
    parser.add_argument("--region", "-r", default="cn")
    parser.add_argument("--timezone", "-t", default="Asia/Shanghai")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # If using HA session, fetch all sessions
    args.ha_sessions = None
    if args.ha_url:
        if not args.ha_token:
            parser.error("--ha-token is required when using --ha-url")
        args.ha_sessions = asyncio.run(_fetch_ha_sessions(args.ha_url, args.ha_token, args.ha_account))
        LOG.info("Fetched %d HA session(s)", len(args.ha_sessions))
        for s in args.ha_sessions:
            LOG.info("  %s -> %s", s.get("username", "?"), s.get("base_url", "?"))

    app = web.Application()
    app["args"] = args
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", handle_index)
    app.router.add_post("/api/login", handle_login)
    app.router.add_post("/api/start", handle_start)
    app.router.add_post("/api/offer", handle_offer)
    app.router.add_post("/api/candidate", handle_candidate)
    app.router.add_post("/api/stop", handle_stop)

    print(f"\n  Open http://127.0.0.1:{args.port} in your browser\n")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
