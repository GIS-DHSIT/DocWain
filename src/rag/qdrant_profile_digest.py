from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.rag.doc_inventory import DocInventoryItem, fetch_doc_inventory


_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "these", "those",
    "section", "summary", "report", "document", "page", "pages",
}
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_ORG_RE = re.compile(r"\b([A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+){0,3})\b")


@dataclass(frozen=True)
class ProfileDigest:
    corpus_fingerprint: str
    doc_count: int
    documents: List[Dict[str, Any]]
    dominant_topics: List[str]
    last_updated: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "corpus_fingerprint": self.corpus_fingerprint,
            "doc_count": self.doc_count,
            "documents": self.documents,
            "dominant_topics": self.dominant_topics,
            "last_updated": self.last_updated,
        }


def _hash(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _doc_key(doc: DocInventoryItem) -> str:
    return doc.doc_id or doc.source_file or doc.document_name or ""


def compute_corpus_fingerprint(doc_inventory: Sequence[DocInventoryItem]) -> str:
    parts: List[str] = []
    for doc in sorted(doc_inventory or [], key=_doc_key):
        parts.append("|".join([doc.doc_id or "", doc.source_file or "", doc.document_name or "", doc.doc_type or ""]))
    parts.append(f"count={len(doc_inventory or [])}")
    return _hash("|".join(parts))


def _page_range(pages: Iterable[int]) -> Optional[tuple[int, int]]:
    values = [p for p in pages if isinstance(p, int)]
    if not values:
        return None
    return (min(values), max(values))


def _extract_keywords(text: str, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z]{3,}", text or "")
    filtered = [t.lower() for t in tokens if t.lower() not in _STOPWORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(limit)]


def _extract_entities(text: str, limit: int = 6) -> List[str]:
    candidates = _ENTITY_RE.findall(text or "") + _ORG_RE.findall(text or "")
    cleaned: List[str] = []
    for cand in candidates:
        value = cand.strip()
        if not value or value.lower() in _STOPWORDS:
            continue
        cleaned.append(value)
    unique: List[str] = []
    seen = set()
    for item in cleaned:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:limit]


def _detect_doc_signal(doc: DocInventoryItem, snippets: List[str]) -> tuple[str, float]:
    haystack = " ".join([doc.source_file or "", doc.document_name or "", doc.doc_type or ""] + snippets).lower()
    if any(tok in haystack for tok in ["resume", "cv", "curriculum"]):
        return "resume", 0.8
    if any(tok in haystack for tok in ["invoice", "bill", "statement", "receipt"]):
        return "invoice", 0.8
    if any(tok in haystack for tok in ["report", "analysis", "summary"]):
        return "report", 0.6
    return "unknown", 0.3


def _point_payload(point: Any) -> Dict[str, Any]:
    if hasattr(point, "payload"):
        return point.payload or {}
    if isinstance(point, dict):
        return point.get("payload") or point
    return {}


def _scroll_points(
    qdrant_client: Any,
    *,
    collection_name: str,
    query_filter: Dict[str, Any],
    limit: int,
) -> List[Any]:
    if not qdrant_client:
        return []
    try:
        scroll = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return []
    if hasattr(scroll, "points"):
        return scroll.points or []
    if isinstance(scroll, tuple):
        return scroll[0] if len(scroll) > 0 else []
    return []


def _build_doc_entry(doc: DocInventoryItem, points: Sequence[Any]) -> Dict[str, Any]:
    snippets: List[str] = []
    pages: List[int] = []
    has_tables = False
    entities: List[str] = []
    for point in points:
        payload = _point_payload(point)
        text = str(payload.get("text") or payload.get("content") or "")
        if text:
            snippets.append(text[:200])
            entities.extend(_extract_entities(text, limit=6))
        page = payload.get("page") or payload.get("page_start") or payload.get("page_end")
        if isinstance(page, int):
            pages.append(page)
        chunk_type = str(payload.get("chunk_type") or payload.get("chunk_kind") or "")
        if chunk_type in {"table", "table_row", "table_header", "table_text"}:
            has_tables = True

    entity_list = list(dict.fromkeys(entities))[:6]
    page_range = _page_range(pages)
    doc_signal, confidence = _detect_doc_signal(doc, snippets)
    keywords = _extract_keywords(" ".join(snippets))

    return {
        "document_id": doc.doc_id,
        "source_file": doc.source_file or doc.document_name,
        "top_entities": entity_list,
        "doc_signal": doc_signal,
        "doc_signal_confidence": confidence,
        "has_tables": has_tables,
        "page_range": page_range,
        "section_titles": keywords,
    }


def build_profile_digest(
    *,
    qdrant_client: Any,
    collection_name: str,
    profile_id: str,
    subscription_id: Optional[str],
    redis_client: Optional[Any] = None,
    doc_inventory: Optional[Sequence[DocInventoryItem]] = None,
    cache_ttl_seconds: int = 1800,
    per_doc_limit: int = 3,
) -> Dict[str, Any]:
    doc_inventory = list(doc_inventory or fetch_doc_inventory(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        profile_id=profile_id,
        subscription_id=subscription_id,
        redis_client=redis_client,
        cache_ttl_seconds=900,
    ))

    corpus_fingerprint = compute_corpus_fingerprint(doc_inventory)
    cache_key = f"profile_digest:{collection_name}:{profile_id}:{subscription_id or 'default'}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                payload = json.loads(cached)
                if payload.get("corpus_fingerprint") == corpus_fingerprint:
                    return payload
        except Exception:
            pass

    documents: List[Dict[str, Any]] = []
    for doc in doc_inventory:
        must_conditions = [{"key": "profile_id", "match": {"value": str(profile_id)}}]
        if subscription_id:
            must_conditions.append({"key": "subscription_id", "match": {"value": str(subscription_id)}})
        if doc.doc_id:
            must_conditions.append({"key": "document_id", "match": {"value": str(doc.doc_id)}})
        elif doc.source_file:
            must_conditions.append({"key": "source_file", "match": {"value": str(doc.source_file)}})
        query_filter = {"must": must_conditions}
        points = _scroll_points(
            qdrant_client,
            collection_name=collection_name,
            query_filter=query_filter,
            limit=max(1, int(per_doc_limit)),
        )
        documents.append(_build_doc_entry(doc, points))

    topic_counter = Counter()
    for doc in documents:
        topic_counter.update(doc.get("section_titles") or [])
        topic_counter.update([e.lower() for e in (doc.get("top_entities") or [])])
    dominant_topics = [topic for topic, _ in topic_counter.most_common(6)]

    digest = ProfileDigest(
        corpus_fingerprint=corpus_fingerprint,
        doc_count=len(documents),
        documents=documents,
        dominant_topics=dominant_topics,
        last_updated=time.time(),
    ).as_dict()

    if redis_client:
        try:
            redis_client.setex(cache_key, max(60, int(cache_ttl_seconds)), json.dumps(digest))
        except Exception:
            pass

    return digest


__all__ = ["ProfileDigest", "build_profile_digest", "compute_corpus_fingerprint"]
