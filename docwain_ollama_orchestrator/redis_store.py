from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import redis

logger = logging.getLogger(__name__)

REGISTRY_KEY = "ollama:session:registry"
META_KEY = "ollama:session:meta"
MODEL_KEY_PREFIX = "ollama:model:"


@dataclass
class RedisStore:
    redis_url: str

    def _client(self) -> redis.Redis:
        return redis.from_url(self.redis_url, decode_responses=True)

    def save_registry(self, registry_json: str, meta_json: str, ttl_seconds: int, models: Dict[str, Dict[str, Any]]) -> None:
        try:
            client = self._client()
            pipe = client.pipeline(transaction=True)
            pipe.set(REGISTRY_KEY, registry_json)
            pipe.set(META_KEY, meta_json)
            if ttl_seconds:
                pipe.expire(REGISTRY_KEY, ttl_seconds)
                pipe.expire(META_KEY, ttl_seconds)
            for name, model in models.items():
                key = f"{MODEL_KEY_PREFIX}{name}"
                payload = json.dumps(model, ensure_ascii=True)
                pipe.set(key, payload)
                if ttl_seconds:
                    pipe.expire(key, ttl_seconds)
            pipe.execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis save failed: %s", exc)

    def load_registry(self) -> Optional[Tuple[str, str]]:
        try:
            client = self._client()
            registry_json = client.get(REGISTRY_KEY)
            meta_json = client.get(META_KEY)
            if registry_json and meta_json:
                return registry_json, meta_json
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis load failed: %s", exc)
        return None

    def load_model(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            client = self._client()
            data = client.get(f"{MODEL_KEY_PREFIX}{name}")
            if data:
                return json.loads(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis model load failed: %s", exc)
        return None
