from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaUnavailable(OllamaError):
    pass


@dataclass
class OllamaResponse:
    data: Dict[str, Any]
    latency_ms: int


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds)
        self.retries = retries
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> OllamaResponse:
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                start = time.perf_counter()
                response = await self._client.request(method, path, json=json_body)
                latency_ms = int((time.perf_counter() - start) * 1000)
                if response.status_code >= 500:
                    raise OllamaUnavailable(f"Ollama server error {response.status_code}")
                response.raise_for_status()
                return OllamaResponse(data=response.json(), latency_ms=latency_ms)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                await asyncio.sleep(0.4 * (attempt + 1))
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code >= 500:
                    await asyncio.sleep(0.4 * (attempt + 1))
                else:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await asyncio.sleep(0.2 * (attempt + 1))
        raise OllamaUnavailable(f"Ollama request failed: {last_exc}")

    async def list_models(self) -> List[Dict[str, Any]]:
        response = await self._request("GET", "/api/tags")
        models = response.data.get("models")
        if not isinstance(models, list):
            raise OllamaError("Malformed response from /api/tags")
        return models

    async def list_running(self) -> List[Dict[str, Any]]:
        try:
            response = await self._request("GET", "/api/ps")
        except OllamaUnavailable:
            return []
        models = response.data.get("models")
        if not isinstance(models, list):
            return []
        return models

    async def warmup_chat(self, model: str, keep_alive: str, prompt: str = "warmup") -> OllamaResponse:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": keep_alive,
        }
        return await self._request("POST", "/api/chat", json_body=payload)
