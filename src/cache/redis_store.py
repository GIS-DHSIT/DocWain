from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .redis_keys import RedisKeys

CATALOG_TTL_SEC = 7 * 24 * 60 * 60
SECTION_INDEX_TTL_SEC = 7 * 24 * 60 * 60
ENTITY_INDEX_TTL_SEC = 7 * 24 * 60 * 60
LAYOUT_REF_TTL_SEC = 30 * 24 * 60 * 60
PLAN_CACHE_TTL_SEC = 10 * 60
KG_ANCHOR_STATS_TTL_SEC = 7 * 24 * 60 * 60

SCHEMA_VERSION = "dwx-redis-2026-02-06"


@dataclass
class RedisStore:
    redis_client: Any

    def _ensure_client(self) -> None:
        if self.redis_client is None:
            raise RuntimeError("Redis client is required for RedisStore")

    def safe_json_set(self, key: str, value: Any, *, ttl: Optional[int] = None, schema_version: str = SCHEMA_VERSION) -> None:
        self._ensure_client()
        payload = {"schema_version": schema_version, "payload": value, "updated_at": int(time.time())}
        raw = json.dumps(payload)
        if ttl:
            self.redis_client.setex(key, int(ttl), raw)
        else:
            self.redis_client.set(key, raw)

    def safe_json_get(self, key: str, *, schema_version: str = SCHEMA_VERSION) -> Optional[Any]:
        self._ensure_client()
        raw = self.redis_client.get(key)
        if not raw:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if payload.get("schema_version") != schema_version:
            return None
        return payload.get("payload")

    def get_catalog(self, keys: RedisKeys) -> Optional[Dict[str, Any]]:
        return self.safe_json_get(keys.catalog())

    def set_catalog(self, keys: RedisKeys, catalog: Dict[str, Any]) -> None:
        self.safe_json_set(keys.catalog(), catalog, ttl=CATALOG_TTL_SEC)

    def get_layout_ref(self, keys: RedisKeys, document_id: str) -> Optional[Dict[str, Any]]:
        return self.safe_json_get(keys.layout_ref(document_id))

    def set_layout_ref(self, keys: RedisKeys, document_id: str, layout_ref: Dict[str, Any]) -> None:
        self.safe_json_set(keys.layout_ref(document_id), layout_ref, ttl=LAYOUT_REF_TTL_SEC)

    def get_section_index(self, keys: RedisKeys, document_id: str) -> Optional[Dict[str, Any]]:
        return self.safe_json_get(keys.section_index(document_id))

    def set_section_index(self, keys: RedisKeys, document_id: str, section_index: Dict[str, Any]) -> None:
        self.safe_json_set(keys.section_index(document_id), section_index, ttl=SECTION_INDEX_TTL_SEC)

    def get_entity_index(self, keys: RedisKeys) -> Dict[str, Any]:
        return self.safe_json_get(keys.entity_index()) or {}

    def set_entity_index(self, keys: RedisKeys, entity_index: Dict[str, Any]) -> None:
        self.safe_json_set(keys.entity_index(), entity_index, ttl=ENTITY_INDEX_TTL_SEC)

    def update_entity_index(self, keys: RedisKeys, new_entities: Dict[str, list[Dict[str, Any]]]) -> Dict[str, Any]:
        current = self.get_entity_index(keys)
        entities = dict(current.get("entities") or {})
        for entity_norm, entries in new_entities.items():
            merged = list(entities.get(entity_norm) or []) + list(entries or [])
            entities[entity_norm] = _dedupe_entity_entries(merged)
        updated = {
            "entities": entities,
            "updated_at": int(time.time()),
        }
        self.set_entity_index(keys, updated)
        return updated

    def get_kg_anchor_stats(self, keys: RedisKeys) -> Optional[Dict[str, Any]]:
        return self.safe_json_get(keys.kg_anchor_stats())

    def set_kg_anchor_stats(self, keys: RedisKeys, stats: Dict[str, Any]) -> None:
        self.safe_json_set(keys.kg_anchor_stats(), stats, ttl=KG_ANCHOR_STATS_TTL_SEC)

    def get_plan_cache(self, keys: RedisKeys, session_id: str, query: str) -> Optional[Dict[str, Any]]:
        return self.safe_json_get(keys.plan_cache(session_id, query))

    def set_plan_cache(self, keys: RedisKeys, session_id: str, query: str, plan: Dict[str, Any]) -> None:
        self.safe_json_set(keys.plan_cache(session_id, query), plan, ttl=PLAN_CACHE_TTL_SEC)


def _dedupe_entity_entries(entries: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen = set()
    deduped: list[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = (
            str(entry.get("file_name") or ""),
            str(entry.get("document_id") or ""),
            str(entry.get("section_id") or ""),
            str(entry.get("page") or ""),
            str(entry.get("snippet_sha") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


__all__ = ["RedisStore", "RedisKeys", "SCHEMA_VERSION"]
