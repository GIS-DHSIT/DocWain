from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.api.config import Config
from src.intelligence.redis_schema import RedisSchema

logger = get_logger(__name__)

def _now_ts() -> float:
    return float(time.time())

def _safe_json_loads(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis JSON decode failed: %s", exc)
        return None

def _safe_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

@dataclass
class SessionState:
    active_profile_id: Optional[str] = None
    active_document_id: Optional[str] = None
    active_domain: Optional[str] = None
    active_docset: List[str] = field(default_factory=list)
    recent_routes: List[Dict[str, Any]] = field(default_factory=list)
    recent_entities: List[Dict[str, Any]] = field(default_factory=list)
    style_prefs: Dict[str, Any] = field(
        default_factory=lambda: {"verbosity": "medium", "format_bias": "bullets"}
    )
    last_seen_ts: float = field(default_factory=_now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_profile_id": self.active_profile_id,
            "active_document_id": self.active_document_id,
            "active_domain": self.active_domain,
            "active_docset": self.active_docset,
            "recent_routes": self.recent_routes,
            "recent_entities": self.recent_entities,
            "style_prefs": self.style_prefs,
            "last_seen_ts": self.last_seen_ts,
        }

    @staticmethod
    def from_dict(payload: Optional[Dict[str, Any]]) -> "SessionState":
        payload = payload or {}
        return SessionState(
            active_profile_id=payload.get("active_profile_id"),
            active_document_id=payload.get("active_document_id"),
            active_domain=payload.get("active_domain"),
            active_docset=list(payload.get("active_docset") or []),
            recent_routes=list(payload.get("recent_routes") or []),
            recent_entities=list(payload.get("recent_entities") or []),
            style_prefs=payload.get("style_prefs") or {"verbosity": "medium", "format_bias": "bullets"},
            last_seen_ts=float(payload.get("last_seen_ts") or _now_ts()),
        )

class RedisIntelCache:
    """Redis helper for DocWain intelligent-by-default cache (dwx:*)."""

    def __init__(
        self,
        redis_client: Optional[Any],
        *,
        prefix: str = "dwx",
        session_ttl: Optional[int] = None,
        catalog_ttl: Optional[int] = None,
        summary_ttl: Optional[int] = None,
        entities_ttl: Optional[int] = None,
    ) -> None:
        self.redis = redis_client
        self.prefix = prefix
        self.schema = RedisSchema(prefix=prefix)
        self.session_ttl = int(session_ttl or getattr(Config.Intelligence, "SESSION_TTL_SECONDS", 604800))
        self.catalog_ttl = int(catalog_ttl or getattr(Config.Intelligence, "CATALOG_TTL_SECONDS", 2592000))
        self.summary_ttl = int(summary_ttl or getattr(Config.Intelligence, "SUMMARY_TTL_SECONDS", 2592000))
        self.entities_ttl = int(entities_ttl or getattr(Config.Intelligence, "ENTITIES_TTL_SECONDS", 2592000))
        self.route_history_max = int(getattr(Config.Intelligence, "ROUTE_HISTORY_MAX", 20))
        self.entity_history_max = int(getattr(Config.Intelligence, "ENTITY_HISTORY_MAX", 50))

    def _key(self, *parts: str) -> str:
        return self.schema.key(*parts)

    def session_key(self, subscription_id: str, session_id: str) -> str:
        return self.schema.session_key(subscription_id, session_id)

    def catalog_key(self, subscription_id: str, profile_id: str) -> str:
        return self.schema.catalog_key(subscription_id, profile_id)

    def sections_key(self, subscription_id: str, profile_id: str, document_id: str) -> str:
        return self.schema.sections_key(subscription_id, profile_id, document_id)

    def layout_key(self, subscription_id: str, profile_id: str, document_id: str) -> str:
        return self.schema.layout_key(subscription_id, profile_id, document_id)

    def docsum_key(self, document_id: str) -> str:
        return self.schema.docsum_key(document_id)

    def profilesum_key(self, subscription_id: str, profile_id: str) -> str:
        return self.schema.profilesum_key(subscription_id, profile_id)

    def entities_key(self, subscription_id: str, profile_id: str) -> str:
        return self.schema.entities_key(subscription_id, profile_id)

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.redis:
            return None
        try:
            payload = self.redis.get(key)
            return _safe_json_loads(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis get failed for %s: %s", key, exc)
            return None

    def set_json(self, key: str, payload: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        if not self.redis:
            return
        ttl_val = int(ttl_seconds or self.catalog_ttl)
        try:
            self.redis.setex(key, max(1, ttl_val), _safe_json_dumps(payload))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis set failed for %s: %s", key, exc)

    def scan_prefix(self, prefix: str, count: int = 200) -> List[str]:
        if not self.redis:
            return []
        keys: List[str] = []
        try:
            cursor = 0
            pattern = f"{prefix}*"
            while True:
                cursor, batch = self.redis.scan(cursor=cursor, match=pattern, count=count)
                if batch:
                    keys.extend([k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in batch])
                if cursor == 0:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis scan failed for %s: %s", prefix, exc)
        return keys

    def get_session_state(self, subscription_id: str, session_id: str) -> SessionState:
        key = self.session_key(subscription_id, session_id)
        payload = self.get_json(key)
        return SessionState.from_dict(payload)

    def set_session_state(self, subscription_id: str, session_id: str, state: SessionState) -> None:
        key = self.session_key(subscription_id, session_id)
        self.set_json(key, state.to_dict(), ttl_seconds=self.session_ttl)

    def touch_session_state(
        self,
        subscription_id: str,
        session_id: str,
        *,
        active_profile_id: Optional[str] = None,
        active_document_id: Optional[str] = None,
        active_domain: Optional[str] = None,
        active_docset: Optional[Iterable[str]] = None,
        recent_route: Optional[Dict[str, Any]] = None,
        recent_entities: Optional[List[Dict[str, Any]]] = None,
        style_prefs: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        state = self.get_session_state(subscription_id, session_id)
        if active_profile_id is not None:
            state.active_profile_id = active_profile_id
        if active_document_id is not None:
            state.active_document_id = active_document_id
        if active_domain is not None:
            state.active_domain = active_domain
        if active_docset is not None:
            state.active_docset = list(active_docset)
        if recent_route:
            state.recent_routes.append(recent_route)
            state.recent_routes = state.recent_routes[-self.route_history_max :]
        if recent_entities:
            state.recent_entities.extend(recent_entities)
            state.recent_entities = state.recent_entities[-self.entity_history_max :]
        if style_prefs:
            state.style_prefs.update(style_prefs)
        state.last_seen_ts = _now_ts()
        self.set_session_state(subscription_id, session_id, state)
        return state

__all__ = ["RedisIntelCache", "SessionState"]
