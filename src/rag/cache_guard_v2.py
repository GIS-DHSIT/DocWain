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
    parts: List[str] = []
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


def _meta_from_chunk(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk.get("metadata") or chunk
    return getattr(chunk, "metadata", {}) or {}


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return str(getattr(chunk, "text", "") or "")


def compute_retrieval_signature(chunks: Sequence[Any], top_k: int = 8) -> str:
    if not chunks:
        return ""

    doc_ids: List[str] = []
    for chunk in chunks:
        meta = _meta_from_chunk(chunk)
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or "")
        source_file = str(meta.get("source_file") or meta.get("source") or meta.get("file_name") or "")
        doc_ids.append(doc_id or source_file)
    doc_ids = _stable_list(doc_ids)

    scored = []
    for chunk in chunks:
        score = 0.0
        if isinstance(chunk, dict):
            score = float(chunk.get("score") or 0.0)
        else:
            score = float(getattr(chunk, "score", 0.0) or 0.0)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[: max(1, int(top_k))]]

    top_hashes: List[str] = []
    for chunk in top_chunks:
        meta = _meta_from_chunk(chunk)
        chunk_id = meta.get("chunk_id") or meta.get("chunkId") or meta.get("id")
        text_hash = meta.get("text_hash")
        if not text_hash:
            text = " ".join(_chunk_text(chunk).split())
            if text:
                text_hash = _hash(text)
        if not chunk_id and not text_hash:
            page = meta.get("page") or meta.get("page_start") or meta.get("page_end") or ""
            chunk_index = meta.get("chunk_index") or ""
            part = f"{page}:{chunk_index}:{len(_chunk_text(chunk))}"
        else:
            part = f"{chunk_id or ''}:{text_hash or ''}"
        top_hashes.append(part)

    signature = "|".join(doc_ids) + "||" + "|".join(top_hashes)
    return _hash(signature)


@dataclass(frozen=True)
class CacheContextV2:
    subscription_id: str
    profile_id: str
    normalized_query_hash: str
    intent_type: str
    scope_type: str
    target_doc_ids_hash: str
    entities_hash: str
    corpus_fingerprint: str
    model_id: str
    retrieval_signature: str
    target_doc_ids: List[str]
    is_vague: bool

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "profile_id": self.profile_id,
            "normalized_query_hash": self.normalized_query_hash,
            "intent_type": self.intent_type,
            "scope_type": self.scope_type,
            "target_doc_ids_hash": self.target_doc_ids_hash,
            "entities_hash": self.entities_hash,
            "corpus_fingerprint": self.corpus_fingerprint,
            "model_id": self.model_id,
            "retrieval_signature": self.retrieval_signature,
        }


@dataclass(frozen=True)
class CacheDecision:
    hit: bool
    reason: str


class CacheGuardV2:
    def __init__(self, ttl_seconds: int = 600):
        ttl = int(ttl_seconds)
        ttl = max(1, ttl)
        self.ttl_seconds = min(600, ttl)

    def build_context(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        query_text: str,
        intent_type: str,
        scope_type: str,
        target_docs: Sequence[Any],
        entities: EntityDetectionResult,
        corpus_fingerprint: str,
        model_id: str,
        retrieval_signature: str,
        is_vague: bool,
    ) -> CacheContextV2:
        normalized = normalize_query(query_text or "")
        normalized_hash = _hash(normalized)
        target_doc_ids = _target_docs_list(target_docs)
        return CacheContextV2(
            subscription_id=subscription_id or "",
            profile_id=profile_id or "",
            normalized_query_hash=normalized_hash,
            intent_type=intent_type or "",
            scope_type=scope_type or "",
            target_doc_ids_hash=_hash_list(target_doc_ids) if target_doc_ids else "",
            entities_hash=_entities_hash(entities),
            corpus_fingerprint=corpus_fingerprint or "",
            model_id=model_id or "",
            retrieval_signature=retrieval_signature or "",
            target_doc_ids=target_doc_ids,
            is_vague=bool(is_vague),
        )

    def build_cache_key(self, context: CacheContextV2) -> str:
        payload = "|".join(
            [
                context.subscription_id,
                context.profile_id,
                context.normalized_query_hash,
                context.intent_type,
                context.scope_type,
                context.target_doc_ids_hash,
                context.entities_hash,
                context.corpus_fingerprint,
                context.model_id,
                context.retrieval_signature,
            ]
        )
        return _hash(payload)

    def is_cacheable(self, context: CacheContextV2) -> bool:
        if context.is_vague:
            return False
        if not context.retrieval_signature:
            return False
        return True

    def evaluate_cached_payload(
        self,
        *,
        context: CacheContextV2,
        cache_key: str,
        cached_payload: Optional[Dict[str, Any]],
    ) -> CacheDecision:
        if context.is_vague:
            return CacheDecision(hit=False, reason="vague_query")
        if not cached_payload:
            return CacheDecision(hit=False, reason="miss")
        if cached_payload.get("cache_key") != cache_key:
            return CacheDecision(hit=False, reason="key_mismatch")
        if not self.is_cacheable(context):
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


__all__ = [
    "CacheContextV2",
    "CacheDecision",
    "CacheGuardV2",
    "compute_retrieval_signature",
]
