import json
from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, List, Optional

import redis

from src.api.config import Config

logger = get_logger(__name__)

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
        logger.debug("Redis client unavailable; falling back to in-memory store: %s", exc)
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

    def _cleanup_expired_memory(self) -> None:
        """Remove expired entries from in-memory fallback stores."""
        now = time.time()
        expired_upload_keys = []
        for key, uploads in self._memory_uploads.items():
            self._memory_uploads[key] = [
                u for u in uploads if now - u.get("timestamp", 0) < self.ttl_seconds
            ]
            if not self._memory_uploads[key]:
                expired_upload_keys.append(key)
        for key in expired_upload_keys:
            del self._memory_uploads[key]

        # Preferences don't have timestamps, so track insertion time
        expired_pref_keys = [
            key for key, val in self._memory_prefs.items()
            if now - val.get("_ts", 0) > self.ttl_seconds
        ]
        for key in expired_pref_keys:
            del self._memory_prefs[key]

    def record_upload(
        self,
        subscription_id: str,
        profile_id: str,
        filename: str,
        doc_tag: str,
        documents_created: int,
        document_type: Optional[str] = None,
    ) -> None:
        entry = {
            "filename": filename,
            "doc_tag": doc_tag,
            "documents_created": documents_created,
            "document_type": document_type or "general",
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
        if len(self._memory_uploads) > 100:
            self._cleanup_expired_memory()

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

    def clear_uploads(self, subscription_id: str, profile_id: str) -> int:
        """Remove all upload records for a Teams user. Returns count removed."""
        key = self._uploads_key(subscription_id, profile_id)
        removed = 0
        if self.client:
            try:
                removed = self.client.llen(key) or 0
                self.client.delete(key)
                return removed
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable while clearing uploads; using memory: %s", exc)
        entries = self._memory_uploads.pop(key, [])
        return len(entries) or removed

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
        prefs["_ts"] = time.time()
        self._memory_prefs[key] = prefs
        if len(self._memory_prefs) > 100:
            self._cleanup_expired_memory()
        return prefs
