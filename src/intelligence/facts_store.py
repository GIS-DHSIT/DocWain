from __future__ import annotations

import hashlib
import json
from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, Iterable, List, Optional

from src.utils.payload_utils import token_count
from src.intelligence.redis_schema import RedisSchema

logger = get_logger(__name__)

def _now_ts() -> float:
    return float(time.time())

def _safe_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)

class FactsStore:
    def __init__(
        self,
        *,
        redis_client: Optional[Any] = None,
        db: Optional[Any] = None,
        ttl_days: int = 30,
    ) -> None:
        self.redis = redis_client
        self.db = db
        self.ttl_seconds = max(1, int(ttl_days) * 24 * 3600)
        self.prefix = "dwx"
        self.schema = RedisSchema(prefix=self.prefix)

    def _key(self, *parts: str) -> str:
        return self.schema.key(*parts)

    def _get_db(self) -> Optional[Any]:
        if self.db is not None:
            return self.db
        try:
            import sys

            data_handler = sys.modules.get("src.api.dataHandler")
            if data_handler is not None:
                return getattr(data_handler, "db", None)
        except Exception:
            return None
        return None

    def _set_json(self, key: str, payload: Dict[str, Any]) -> None:
        if not self.redis:
            return
        try:
            self.redis.setex(key, self.ttl_seconds, _safe_json(payload))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis set failed for %s: %s", key, exc)

    def _get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.redis:
            return None
        try:
            raw = self.redis.get(key)
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis get failed for %s: %s", key, exc)
            return None

    def persist_document_sections(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        source_name: str,
        doc_domain: str,
        doc_version_hash: Optional[str] = None,
        sections: List[Dict[str, Any]],
        section_facts: List[Dict[str, Any]],
        section_summaries: Optional[Dict[str, str]] = None,
    ) -> None:
        section_summaries = section_summaries or {}
        updated_at = _now_ts()

        # Version hash for cache invalidation
        version_seed = "|".join(
            [f"{sec.get('section_id')}:{hashlib.sha1((sec.get('raw_text') or '').encode('utf-8')).hexdigest()}" for sec in sections]
        )
        version_hash = hashlib.sha1(version_seed.encode("utf-8")).hexdigest()[:12]

        # DB persistence
        db = self._get_db()
        if db is not None:
            try:
                doc_sections = db["document_sections"]
                for sec in sections:
                    digest = hashlib.sha1((sec.get("raw_text") or "").encode("utf-8")).hexdigest()
                    record = {
                        "document_id": document_id,
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                        "doc_domain": doc_domain,
                        "section_id": sec.get("section_id"),
                        "section_kind": sec.get("section_kind"),
                        "section_title": sec.get("section_title"),
                        "page_range": sec.get("page_range"),
                        "section_text_digest": digest,
                        "section_facts_json": next(
                            (fact for fact in section_facts if fact.get("provenance", {}).get("section_id") == sec.get("section_id")),
                            {},
                        ),
                        "doc_version_hash": doc_version_hash,
                        "updated_at": updated_at,
                    }
                    doc_sections.update_one(
                        {
                            "document_id": document_id,
                            "section_id": sec.get("section_id"),
                            "subscription_id": subscription_id,
                            "profile_id": profile_id,
                        },
                        {"$set": record},
                        upsert=True,
                    )

                doc_facts = db["doc_facts"]
                doc_facts.update_one(
                    {
                        "document_id": document_id,
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                    },
                    {
                        "$set": {
                            "document_id": document_id,
                            "subscription_id": subscription_id,
                            "profile_id": profile_id,
                            "doc_domain": doc_domain,
                            "facts_json": section_facts,
                            "doc_version_hash": doc_version_hash,
                            "updated_at": updated_at,
                        }
                    },
                    upsert=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to persist document sections for %s: %s", document_id, exc)

        section_inventory = [
            {
                "section_id": sec.get("section_id"),
                "section_kind": sec.get("section_kind"),
                "section_title": sec.get("section_title"),
                "page_range": sec.get("page_range"),
            }
            for sec in sections
        ]
        total_chars = sum(len(sec.get("raw_text") or "") for sec in sections)
        total_tokens = sum(token_count(sec.get("raw_text") or "") for sec in sections)
        quality = {
            "extracted_chars": total_chars,
            "token_count": total_tokens,
            "coverage": len(sections),
        }

        # Redis caches
        if self.redis:
            self._set_json(
                self.schema.sections_key(subscription_id, profile_id, document_id),
                {
                    "document_id": document_id,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "sections": section_inventory,
                    "doc_version_hash": doc_version_hash,
                    "version_hash": version_hash,
                    "updated_at": updated_at,
                },
            )
            self._set_json(
                self.schema.facts_doc_key(subscription_id, profile_id, document_id),
                {
                    "document_id": document_id,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "section_facts": section_facts,
                    "doc_version_hash": doc_version_hash,
                    "version_hash": version_hash,
                    "updated_at": updated_at,
                },
            )

            # Update catalog
            catalog_key = self._key("catalog", subscription_id, profile_id)
            catalog = self._get_json(catalog_key) or {
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "documents": [],
                "dominant_domains": {},
            }
            docs = catalog.get("documents") or []
            doc_entry = {
                "document_id": document_id,
                "source_name": source_name,
                "doc_domain": doc_domain,
                "sections": section_inventory,
                "summary_short": section_summaries.get(sections[0].get("section_id")) if sections else "",
                "quality": quality,
                "doc_version_hash": doc_version_hash,
                "updated_at": updated_at,
            }
            updated_docs = [d for d in docs if str(d.get("document_id")) != str(document_id)]
            updated_docs.append(doc_entry)
            catalog["documents"] = updated_docs
            domain_counts = catalog.get("dominant_domains") or {}
            domain_counts[doc_domain] = domain_counts.get(doc_domain, 0) + 1
            catalog["dominant_domains"] = domain_counts
            catalog["version_hash"] = version_hash
            self._set_json(catalog_key, catalog)

        # Profile aggregates (DB + Redis)
        existing_profile = self.get_profile_facts(subscription_id, profile_id) or {}
        existing_agg = existing_profile.get("aggregated_facts") or {}
        aggregated = self.merge_profile_facts(existing_agg, section_facts)
        self.persist_profile_sections(
            subscription_id=subscription_id,
            profile_id=profile_id,
            aggregated_facts=aggregated,
        )

    def persist_profile_sections(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        aggregated_facts: Dict[str, Any],
    ) -> None:
        updated_at = _now_ts()
        version_hash = hashlib.sha1(_safe_json(aggregated_facts).encode("utf-8")).hexdigest()[:12]
        db = self._get_db()
        if db is not None:
            try:
                profile_facts = db["profile_facts"]
                profile_facts.update_one(
                    {"subscription_id": subscription_id, "profile_id": profile_id},
                    {
                        "$set": {
                            "profile_id": profile_id,
                            "subscription_id": subscription_id,
                            "merged_facts_json": aggregated_facts,
                            "version_hash": version_hash,
                            "updated_at": updated_at,
                        }
                    },
                    upsert=True,
                )
                profile_sections = db["profile_sections"]
                profile_sections.update_one(
                    {"subscription_id": subscription_id, "profile_id": profile_id},
                    {
                        "$set": {
                            "profile_id": profile_id,
                            "subscription_id": subscription_id,
                            "aggregated_facts_json": aggregated_facts,
                            "version_hash": version_hash,
                            "updated_at": updated_at,
                        }
                    },
                    upsert=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to persist profile facts for %s: %s", profile_id, exc)

        if self.redis:
            self._set_json(
                self.schema.facts_profile_key(subscription_id, profile_id),
                {
                    "profile_id": profile_id,
                    "aggregated_facts": aggregated_facts,
                    "version_hash": version_hash,
                    "updated_at": updated_at,
                },
            )

    def get_document_facts(
        self,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        *,
        doc_version_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        key = self.schema.facts_doc_key(subscription_id, profile_id, document_id)
        cached = self._get_json(key)
        if cached:
            if doc_version_hash and cached.get("doc_version_hash") and cached.get("doc_version_hash") != doc_version_hash:
                return None
            return cached
        db = self._get_db()
        if db is None:
            return None
        try:
            doc_facts = db["doc_facts"]
            row = doc_facts.find_one(
                {
                    "document_id": document_id,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                }
            )
            if row and row.get("facts_json"):
                return {
                    "document_id": document_id,
                    "section_facts": row.get("facts_json"),
                    "doc_version_hash": row.get("doc_version_hash"),
                    "updated_at": _now_ts(),
                }
            doc_sections = db["document_sections"]
            rows = list(
                doc_sections.find(
                    {
                        "document_id": document_id,
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                    }
                )
            )
            section_facts = [row.get("section_facts_json") for row in rows if row.get("section_facts_json")]
            if not section_facts:
                return None
            return {
                "document_id": document_id,
                "section_facts": section_facts,
                "doc_version_hash": doc_version_hash,
                "updated_at": _now_ts(),
            }
        except Exception:
            return None

    def get_profile_facts(self, subscription_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
        key = self.schema.facts_profile_key(subscription_id, profile_id)
        cached = self._get_json(key)
        if cached:
            return cached
        db = self._get_db()
        if db is None:
            return None
        try:
            profile_facts = db["profile_facts"]
            row = profile_facts.find_one({"subscription_id": subscription_id, "profile_id": profile_id})
            if row:
                return {"profile_id": profile_id, "aggregated_facts": row.get("merged_facts_json") or {}, "updated_at": _now_ts()}
            profile_sections = db["profile_sections"]
            row = profile_sections.find_one({"subscription_id": subscription_id, "profile_id": profile_id})
            if not row:
                return None
            return {"profile_id": profile_id, "aggregated_facts": row.get("aggregated_facts_json") or {}, "updated_at": _now_ts()}
        except Exception:
            return None

    @staticmethod
    def merge_profile_facts(existing: Dict[str, Any], new_facts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(existing or {})
        by_kind = merged.get("by_section_kind") or {}
        for fact in new_facts:
            kind = fact.get("section_kind") or "misc"
            entry = by_kind.get(kind) or {"entities": [], "attributes": {}}
            entities = entry.get("entities") or []
            seen = {f"{e.get('type')}::{e.get('normalized')}" for e in entities}
            for ent in fact.get("entities") or []:
                key = f"{ent.get('type')}::{ent.get('normalized')}"
                if key in seen:
                    continue
                seen.add(key)
                entities.append(ent)
            attributes = entry.get("attributes") or {}
            for key, value in (fact.get("attributes") or {}).items():
                if key not in attributes:
                    attributes[key] = value
            entry["entities"] = entities[:50]
            entry["attributes"] = attributes
            by_kind[kind] = entry
        merged["by_section_kind"] = by_kind
        merged["updated_at"] = _now_ts()
        return merged

__all__ = ["FactsStore"]
