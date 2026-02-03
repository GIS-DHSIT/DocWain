from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DocInventoryItem:
    doc_id: str
    source_file: str
    document_name: str
    doc_type: str


def _payload_value(payload: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def fetch_doc_inventory(
    *,
    qdrant_client: Any,
    collection_name: str,
    profile_id: str,
    subscription_id: Optional[str] = None,
    redis_client: Optional[Any] = None,
    cache_ttl_seconds: int = 900,
    max_docs: int = 500,
    max_points: int = 1600,
) -> List[DocInventoryItem]:
    """
    Return distinct documents for a subscription/profile with lightweight caching.
    """
    if not qdrant_client or not collection_name or not profile_id:
        return []

    cache_key = f"doc_inventory:{collection_name}:{profile_id}:{subscription_id or 'default'}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                payload = json.loads(cached)
                return [
                    DocInventoryItem(
                        doc_id=str(item.get("doc_id") or ""),
                        source_file=str(item.get("source_file") or ""),
                        document_name=str(item.get("document_name") or ""),
                        doc_type=str(item.get("doc_type") or ""),
                    )
                    for item in payload
                ]
        except Exception:
            pass

    must_conditions: List[Dict[str, Any]] = [{"key": "profile_id", "match": {"value": str(profile_id)}}]
    if subscription_id:
        must_conditions.append({"key": "subscription_id", "match": {"value": str(subscription_id)}})
    query_filter = {"must": must_conditions}

    unique: Dict[str, DocInventoryItem] = {}
    next_offset = None
    remaining = int(max_points)

    while remaining > 0 and len(unique) < max_docs:
        limit = min(200, remaining)
        try:
            scroll = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                offset=next_offset,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            break

        points = []
        if hasattr(scroll, "points"):
            points = scroll.points or []
            next_offset = getattr(scroll, "next_page_offset", None)
        elif isinstance(scroll, tuple):
            points = scroll[0] if len(scroll) > 0 else []
            next_offset = scroll[1] if len(scroll) > 1 else None
        else:
            points = []
            next_offset = None

        if not points:
            break

        for pt in points:
            payload = pt.payload or {}
            doc_id = _payload_value(payload, ("document_id", "doc_id", "docId", "documentId"))
            source_file = _payload_value(payload, ("source_file", "source", "file_name", "filename"))
            if not doc_id and not source_file:
                continue
            document_name = _payload_value(payload, ("document_name", "doc_name", "title")) or source_file
            doc_type = _payload_value(payload, ("doc_type", "document_type", "source_type"))
            key = doc_id or source_file
            if key in unique:
                continue
            unique[key] = DocInventoryItem(
                doc_id=doc_id,
                source_file=source_file,
                document_name=document_name,
                doc_type=doc_type,
            )
            if len(unique) >= max_docs:
                break

        remaining -= limit
        if not next_offset:
            break

    items = sorted(
        unique.values(),
        key=lambda item: (item.source_file or item.document_name or item.doc_id),
    )

    if redis_client:
        try:
            payload = [
                {
                    "doc_id": item.doc_id,
                    "source_file": item.source_file,
                    "document_name": item.document_name,
                    "doc_type": item.doc_type,
                }
                for item in items
            ]
            redis_client.setex(cache_key, max(60, int(cache_ttl_seconds)), json.dumps(payload))
        except Exception:
            pass

    return items


__all__ = ["DocInventoryItem", "fetch_doc_inventory"]
