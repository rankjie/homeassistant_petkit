"""Authenticated HTTP proxy for HLS playlists served by HA-managed go2rtc."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from aiohttp import web

from homeassistant.components.http import HomeAssistantView
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN
from .go2rtc_stream import get_go2rtc_stream_manager
from .whep_proxy import _check_external_auth

_AUTH_QUERY_KEYS = frozenset({"token", "authSig"})
_FORWARDED_RESPONSE_HEADERS = frozenset(
    {
        "cache-control",
        "content-length",
        "content-range",
        "content-type",
        "etag",
        "last-modified",
    }
)


class PetkitHlsProxyView(HomeAssistantView):
    """Expose fixed authenticated HLS URLs for PetKit streams."""

    url = "/api/petkit/hls/{device_id}"
    extra_urls = [
        "/api/petkit/hls/{device_id}/",
        "/api/petkit/hls/{device_id}/index.m3u8",
        "/api/petkit/hls/{device_id}/{tail:.*}",
    ]
    name = "api:petkit:hls"
    requires_auth = False

    async def get(
        self,
        request: web.Request,
        device_id: str,
        tail: str | None = None,
    ) -> web.StreamResponse:
        """Proxy HLS playlists and segments from HA-managed go2rtc."""
        auth_error = _check_external_auth(request)
        if auth_error is not None:
            return auth_error

        stream_manager = get_go2rtc_stream_manager(request.app["hass"])
        if not stream_manager.is_available(device_id):
            return web.Response(status=404, text="Shared PetKit stream unavailable")

        try:
            await stream_manager.async_ensure_stream(device_id, raise_on_failure=True)
        except RuntimeError as err:
            return web.Response(status=502, text=str(err))

        if tail in (None, "") and not request.path.endswith("/index.m3u8"):
            redirect_target = f"/api/petkit/hls/{device_id}/index.m3u8"
            if request.query_string:
                redirect_target = f"{redirect_target}?{request.query_string}"
            raise web.HTTPTemporaryRedirect(redirect_target)

        upstream_url = await self._upstream_url(stream_manager, device_id, tail or "")
        if upstream_url is None:
            return web.Response(status=404, text="Shared PetKit stream unavailable")

        auth_query = {
            key: value
            for key, value in request.query.items()
            if key in _AUTH_QUERY_KEYS
        }
        upstream_query = {
            key: value
            for key, value in request.query.items()
            if key not in _AUTH_QUERY_KEYS
        }

        session = async_get_clientsession(request.app["hass"])
        async with session.get(upstream_url, params=upstream_query or None) as response:
            body = await response.read()
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() in _FORWARDED_RESPONSE_HEADERS
            }

            if response.status != HTTPStatus.OK:
                return web.Response(
                    status=response.status,
                    body=body,
                    headers=headers,
                    content_type=response.content_type,
                )

            if self._is_playlist_path(tail):
                body = self._rewrite_playlist(body, auth_query)

            return web.Response(
                status=response.status,
                body=body,
                headers=headers,
                content_type=response.content_type,
            )

    async def _upstream_url(
        self,
        stream_manager,
        device_id: str,
        tail: str,
    ) -> str | None:
        """Build the upstream go2rtc URL for one proxied HLS request."""
        if tail in ("", "index.m3u8"):
            return await stream_manager.hls_master_url(device_id)

        base_url = stream_manager.api_base_url(device_id)
        if base_url is None:
            return None

        return urljoin(base_url, f"api/{tail.lstrip('/')}")

    @staticmethod
    def _is_playlist_path(tail: str | None) -> bool:
        """Return whether one request targets an HLS playlist."""
        return tail in (None, "", "index.m3u8") or (tail or "").endswith(".m3u8")

    @staticmethod
    def _rewrite_playlist(body: bytes, auth_query: Mapping[str, str]) -> bytes:
        """Append auth query parameters to relative HLS playlist entries."""
        if not auth_query:
            return body

        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return body

        rewritten: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                rewritten.append(line)
                continue

            rewritten.append(PetkitHlsProxyView._append_query(line, auth_query))

        suffix = "\n" if text.endswith("\n") else ""
        return ("\n".join(rewritten) + suffix).encode("utf-8")

    @staticmethod
    def _append_query(url: str, auth_query: Mapping[str, str]) -> str:
        """Append auth query parameters to one playlist URL."""
        split = urlsplit(url)
        query = dict(parse_qsl(split.query, keep_blank_values=True))
        query.update(auth_query)
        return urlunsplit(
            (
                split.scheme,
                split.netloc,
                split.path,
                urlencode(query),
                split.fragment,
            )
        )


def get_hls_proxy_url(device_id: str) -> str:
    """Return the fixed HLS proxy path for one device."""
    return f"/api/petkit/hls/{device_id}/index.m3u8"
