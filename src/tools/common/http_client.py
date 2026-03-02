from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from src.tools.base import ToolError

DEFAULT_TIMEOUT = float(os.getenv("TOOLS_HTTP_TIMEOUT", "15"))
MAX_BYTES = int(os.getenv("TOOLS_HTTP_MAX_BYTES", str(2 * 1024 * 1024)))

_ALLOWLIST = {
    host.strip()
    for host in os.getenv("TOOLS_HTTP_ALLOWLIST", "localhost,127.0.0.1,example.com").split(",")
    if host.strip()
}
_DENYLIST = {
    host.strip()
    for host in os.getenv(
        "TOOLS_HTTP_DENYLIST",
        "169.254.169.254,metadata.google.internal,metadata.azure.internal"
    ).split(",")
    if host.strip()
}


def _validate_host(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        raise ToolError("URL host is required", code="invalid_url")
    lower_host = host.lower()
    if lower_host in _DENYLIST:
        raise ToolError(f"Requests to host '{host}' are blocked", code="host_blocked", status_code=403)
    if _ALLOWLIST and lower_host not in _ALLOWLIST:
        raise ToolError(f"Host '{host}' is not allowlisted", code="host_not_allowed", status_code=403)
    return lower_host


async def fetch_bytes(
    url: str,
    *,
    timeout: Optional[float] = None,
    max_bytes: int = MAX_BYTES,
    headers: Optional[Dict[str, str]] = None,
) -> bytes:
    _validate_host(url)
    client_timeout = timeout or DEFAULT_TIMEOUT
    async with httpx.AsyncClient(follow_redirects=True, timeout=client_timeout, headers=headers) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ToolError(f"Request to {url} timed out", code="timeout", status_code=504) from exc
        except httpx.HTTPError as exc:  # noqa: BLE001
            raise ToolError(f"Failed to fetch URL {url}: {exc}", code="http_error", status_code=502) from exc

    content = response.content or b""
    if len(content) > max_bytes:
        raise ToolError("Fetched payload exceeds configured limit", code="payload_too_large")
    return content


async def fetch_text(
    url: str,
    *,
    timeout: Optional[float] = None,
    max_bytes: int = MAX_BYTES,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    raw = await fetch_bytes(url, timeout=timeout, max_bytes=max_bytes, headers=headers)
    try:
        decoded = raw.decode("utf-8")
    except Exception:
        decoded = raw.decode("latin-1", errors="ignore")
    return {"url": url, "content": decoded, "bytes": len(raw)}
