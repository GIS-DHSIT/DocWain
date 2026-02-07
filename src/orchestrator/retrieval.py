from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.api.config import Config
from src.api.dataHandler import encode_with_fallback
from src.api.vector_store import build_collection_name, build_qdrant_filter
from src.router.schema import DocumentFilters
from src.utils.payload_utils import get_canonical_text, get_source_name


def _build_filter(
    *,
    subscription_id: str,
    profile_id: str,
    filters: DocumentFilters,
    chunk_kind: Optional[str],
) -> Filter:
    base = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id))
    must = list(getattr(base, "must", []) or [])
    should = list(getattr(base, "should", []) or [])
    if filters.document_type_hints:
        values = filters.document_type_hints
        should.append(FieldCondition(key="document.type", match=MatchAny(any=values)))
        should.append(FieldCondition(key="document_type", match=MatchAny(any=values)))
        should.append(FieldCondition(key="doc_type", match=MatchAny(any=values)))
    if filters.file_name_hints:
        values = filters.file_name_hints
        should.append(FieldCondition(key="source.name", match=MatchAny(any=values)))
        should.append(FieldCondition(key="file_name", match=MatchAny(any=values)))
        should.append(FieldCondition(key="source_file", match=MatchAny(any=values)))
    if chunk_kind:
        must.append(FieldCondition(key="chunk_kind", match=MatchValue(value=chunk_kind)))
    return Filter(must=must, should=should or None)


def _map_hit(hit: Any) -> Dict[str, Any]:
    payload = hit.payload or {}
    return {
        "text": get_canonical_text(payload),
        "file_name": get_source_name(payload) or payload.get("file_name"),
        "section_title": payload.get("section_title") or "",
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "chunk_kind": payload.get("chunk_kind") or payload.get("chunk_type"),
    }


def retrieve_chunks(
    *,
    subscription_id: str,
    profile_id: str,
    queries: Iterable[str],
    document_filters: DocumentFilters,
    chunk_kinds: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
    collection = build_collection_name(subscription_id)

    hits: List[Dict[str, Any]] = []
    query_list = [q for q in queries if q]
    if not query_list:
        return hits

    if not chunk_kinds:
        chunk_kinds = [""]

    for query in query_list:
        query_vec = encode_with_fallback([query], normalize_embeddings=True, convert_to_numpy=False)[0]
        for kind in chunk_kinds:
            q_filter = _build_filter(
                subscription_id=subscription_id,
                profile_id=profile_id,
                filters=document_filters,
                chunk_kind=kind or None,
            )
            results = client.search(
                collection_name=collection,
                query_vector=("content_vector", list(query_vec)),
                query_filter=q_filter,
                limit=top_k,
            )
            hits.extend(_map_hit(hit) for hit in results)
    return hits


__all__ = ["retrieve_chunks"]
