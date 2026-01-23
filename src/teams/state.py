import json
import logging
import time
from typing import Any, Dict, List, Optional

import redis

from src.api.config import Config

logger = logging.getLogger(__name__)


def _build_redis_client() -> Optional[redis.Redis]:
    """Create a Redis client from Config.Redis settings."""
    try:
        return redis.Redis(
            host=getattr(Config.Redis, "HOST", "localhost"),
            port=int(getattr(Config.Redis, "PORT", 6379)),
            password=getattr(Config.Redis, "PASSWORD", None),
            ssl=bool(getattr(Config.Redis, "SSL", False)),
            db=int(getattr(Config.Redis, "DB", 0)),
            decode_responses=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis client unavailable; falling back to in-memory store: %s", exc)
        return None


class TeamsStateStore:
    """Tracks Teams uploads and per-session preferences."""

    def __init__(self, max_uploads: int = 50, ttl_seconds: int = 7 * 24 * 3600):
        self.client = _build_redis_client()
        self.max_uploads = max_uploads
        self.ttl_seconds = ttl_seconds
        self._memory_uploads: Dict[str, List[Dict[str, Any]]] = {}
        self._memory_prefs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _uploads_key(subscription_id: str, profile_id: str) -> str:
        return f"teams:{subscription_id}:{profile_id}:uploads"

    @staticmethod
    def _prefs_key(subscription_id: str, profile_id: str) -> str:
        return f"teams:{subscription_id}:{profile_id}:prefs"

    def record_upload(
        self,
        subscription_id: str,
        profile_id: str,
        filename: str,
        doc_tag: str,
        documents_created: int,
    ) -> None:
        entry = {
            "filename": filename,
            "doc_tag": doc_tag,
            "documents_created": documents_created,
            "timestamp": time.time(),
        }
        key = self._uploads_key(subscription_id, profile_id)
        if self.client:
            try:
                pipe = self.client.pipeline()
                pipe.lpush(key, json.dumps(entry))
                pipe.ltrim(key, 0, self.max_uploads - 1)
                pipe.expire(key, self.ttl_seconds)
                pipe.execute()
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable while recording upload; using memory: %s", exc)
        uploads = self._memory_uploads.setdefault(key, [])
        uploads.insert(0, entry)
        self._memory_uploads[key] = uploads[: self.max_uploads]

    def list_uploads(self, subscription_id: str, profile_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        key = self._uploads_key(subscription_id, profile_id)
        records: List[Dict[str, Any]] = []
        if self.client:
            try:
                raw = self.client.lrange(key, 0, max(limit - 1, 0))
                for item in raw:
                    try:
                        records.append(json.loads(item))
                    except Exception:
                        continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable while listing uploads; using memory: %s", exc)
        if not records and key in self._memory_uploads:
            records = self._memory_uploads.get(key, [])[:limit]
        return records

    def get_preferences(self, subscription_id: str, profile_id: str) -> Dict[str, Any]:
        key = self._prefs_key(subscription_id, profile_id)
        if self.client:
            try:
                raw = self.client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable while reading prefs; using memory: %s", exc)
        return self._memory_prefs.get(key, {})

    def set_preferences(
        self,
        subscription_id: str,
        profile_id: str,
        model_name: Optional[str] = None,
        persona: Optional[str] = None,
    ) -> Dict[str, Any]:
        prefs = self.get_preferences(subscription_id, profile_id).copy()
        if model_name:
            prefs["model_name"] = model_name
        if persona:
            prefs["persona"] = persona

        key = self._prefs_key(subscription_id, profile_id)
        if self.client:
            try:
                self.client.setex(key, self.ttl_seconds, json.dumps(prefs))
                return prefs
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable while setting prefs; using memory: %s", exc)
        self._memory_prefs[key] = prefs
        return prefs
