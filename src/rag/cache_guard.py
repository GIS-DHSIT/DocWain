from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.rag.entity_detector import EntityDetectionResult
from src.rag.query_cache import normalize_query


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_list(values: Iterable[str]) -> List[str]:
    output = [str(v) for v in values if v]
    return sorted(set(output))


def _hash_list(values: Iterable[str]) -> str:
    return _hash("|".join(_stable_list(values)))


def _entities_hash(entities: EntityDetectionResult) -> str:
    parts = []
    parts.extend(_stable_list(entities.people))
    parts.extend(_stable_list(entities.products))
    parts.extend(_stable_list(entities.documents))
    return _hash("|".join(parts)) if parts else _hash("")


def _target_docs_list(target_docs: Sequence[Any]) -> List[str]:
    values: List[str] = []
    for doc in target_docs or []:
        if isinstance(doc, str):
            values.append(doc)
            continue
        doc_id = getattr(doc, "doc_id", None) or ""
        source_file = getattr(doc, "source_file", None) or ""
        document_name = getattr(doc, "document_name", None) or ""
        values.append(doc_id or source_file or document_name)
    return _stable_list(values)


def _doc_set_list(doc_set: Sequence[str]) -> List[str]:
    return _stable_list(doc_set)


@dataclass(frozen=True)
class CacheContext:
    subscription_id: str
    profile_id: str
    session_id: str
    query_hash: str
    normalized_query: str
    intent_type: str
    scope: str
    target_docs: List[str]
    entities_hash: str
    corpus_fingerprint: str
    doc_set: List[str]
    doc_set_hash: str
    model_id: str
    wants_all_docs: bool
    wants_rank_top_n: Optional[int]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "query_hash": self.query_hash,
            "intent_type": self.intent_type,
            "scope": self.scope,
            "target_docs": self.target_docs,
            "entities_hash": self.entities_hash,
            "corpus_fingerprint": self.corpus_fingerprint,
            "doc_set_hash": self.doc_set_hash,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class CacheDecision:
    hit: bool
    reason: str


class CacheGuard:
    def __init__(self, ttl_seconds: int = 600):
        self.ttl_seconds = max(1, int(ttl_seconds))

    def build_context(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        session_id: Optional[str],
        query_text: str,
        intent_type: str,
        scope: str,
        target_docs: Sequence[Any],
        entities: EntityDetectionResult,
        corpus_fingerprint: str,
        model_id: str,
        doc_set: Sequence[str],
        wants_all_docs: bool = False,
        wants_rank_top_n: Optional[int] = None,
    ) -> CacheContext:
        normalized = normalize_query(query_text or "")
        query_hash = _hash(normalized)
        target_docs_list = _target_docs_list(target_docs)
        doc_set_list = _doc_set_list(doc_set)
        doc_set_hash = _hash_list(doc_set_list) if doc_set_list else ""
        return CacheContext(
            subscription_id=subscription_id or "",
            profile_id=profile_id or "",
            session_id=session_id or "default",
            query_hash=query_hash,
            normalized_query=normalized,
            intent_type=intent_type or "",
            scope=scope or "",
            target_docs=target_docs_list,
            entities_hash=_entities_hash(entities),
            corpus_fingerprint=corpus_fingerprint or "",
            doc_set=doc_set_list,
            doc_set_hash=doc_set_hash,
            model_id=model_id or "",
            wants_all_docs=bool(wants_all_docs),
            wants_rank_top_n=wants_rank_top_n,
        )

    def build_cache_key(self, context: CacheContext) -> str:
        payload = "|".join(
            [
                context.subscription_id,
                context.profile_id,
                context.session_id,
                context.query_hash,
                context.intent_type,
                context.scope,
                _hash_list(context.target_docs),
                context.entities_hash,
                context.corpus_fingerprint,
                context.doc_set_hash,
                context.model_id,
            ]
        )
        return _hash(payload)

    def _cache_allowed(self, context: CacheContext) -> bool:
        if context.intent_type in {"compare", "rank"}:
            return bool(context.doc_set_hash)
        if context.wants_all_docs or context.scope == "multi_doc":
            return bool(context.doc_set_hash)
        return True

    def evaluate_cached_payload(
        self,
        *,
        context: CacheContext,
        cache_key: str,
        cached_payload: Optional[Dict[str, Any]],
    ) -> CacheDecision:
        if not cached_payload:
            return CacheDecision(hit=False, reason="miss")
        if cached_payload.get("cache_key") != cache_key:
            return CacheDecision(hit=False, reason="key_mismatch")
        if not self._cache_allowed(context):
            return CacheDecision(hit=False, reason="cache_disallowed")

        created_at = cached_payload.get("created_at")
        if created_at is not None:
            age = time.time() - float(created_at)
            if age > self.ttl_seconds:
                return CacheDecision(hit=False, reason="stale_ttl")

        metadata = cached_payload.get("metadata") or {}
        required = context.to_metadata()
        for key, value in required.items():
            if metadata.get(key) != value:
                return CacheDecision(hit=False, reason=f"metadata_mismatch:{key}")

        return CacheDecision(hit=True, reason="cache_hit")


__all__ = ["CacheContext", "CacheDecision", "CacheGuard"]
