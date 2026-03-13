from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from src.api.config import Config
from src.api.vector_store import build_collection_name, build_qdrant_filter
from src.utils.logging_utils import get_logger
from src.utils.payload_utils import get_document_type, get_source_name
from src.utils.redis_cache import RedisJsonCache, stamp_cache_payload

logger = get_logger(__name__)


@dataclass
class DocumentIndexEntry:
    document_id: str
    source_name: Optional[str] = None
    document_type: Optional[str] = None
    ingestion_source: Optional[str] = None
    total_points: int = 0
    chunk_count_max: Optional[int] = None
    chunk_kind_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProfileDocumentIndex:
    subscription_id: str
    profile_id: str
    document_ids: List[str]
    documents: Dict[str, DocumentIndexEntry]
    total_points: int
    cached_at: float

    def to_payload(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "profile_id": self.profile_id,
            "document_ids": self.document_ids,
            "documents": {
                doc_id: {
                    "document_id": entry.document_id,
                    "source_name": entry.source_name,
                    "document_type": entry.document_type,
                    "ingestion_source": entry.ingestion_source,
                    "total_points": entry.total_points,
                    "chunk_count_max": entry.chunk_count_max,
                    "chunk_kind_counts": entry.chunk_kind_counts,
                }
                for doc_id, entry in self.documents.items()
            },
            "total_points": self.total_points,
            "cached_at": self.cached_at,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ProfileDocumentIndex":
        documents_payload = payload.get("documents") or {}
        documents = {
            doc_id: DocumentIndexEntry(
                document_id=doc_id,
                source_name=entry.get("source_name"),
                document_type=entry.get("document_type"),
                ingestion_source=entry.get("ingestion_source"),
                total_points=int(entry.get("total_points") or 0),
                chunk_count_max=entry.get("chunk_count_max"),
                chunk_kind_counts=entry.get("chunk_kind_counts") or {},
            )
            for doc_id, entry in documents_payload.items()
        }
        return cls(
            subscription_id=payload.get("subscription_id") or "",
            profile_id=payload.get("profile_id") or "",
            document_ids=list(payload.get("document_ids") or []),
            documents=documents,
            total_points=int(payload.get("total_points") or 0),
            cached_at=float(payload.get("cached_at") or time.time()),
        )


def _build_profile_filter(subscription_id: str, profile_id: str) -> Filter:
    return build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id))


def _summarize_payload(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[int]]:
    source_name = get_source_name(payload)
    document_type = get_document_type(payload)
    ingestion_source = payload.get("ingestion_source") or (payload.get("document") or {}).get("ingestion_source")
    chunk_kind = payload.get("chunk_kind") or (payload.get("chunk") or {}).get("role")
    chunk_count = payload.get("chunk_count") or (payload.get("chunk") or {}).get("count")
    return source_name, document_type, ingestion_source, chunk_kind, payload.get("document_id"), chunk_count


def get_redis_client():
    try:
        from src.api.dw_newron import get_redis_client as _get_redis_client

        return _get_redis_client()
    except Exception:  # noqa: BLE001
        logger.debug("get_redis_client: failed to get redis client", exc_info=True)
        return None


def build_profile_document_index(subscription_id: str, profile_id: str) -> ProfileDocumentIndex:
    logger.debug("build_profile_document_index: subscription_id=%s, profile_id=%s", subscription_id, profile_id)
    cache_key = f"docwain:pdi:v1:{subscription_id}:{profile_id}"
    redis_client = None
    redis_client = get_redis_client()
    cache = RedisJsonCache(redis_client, default_ttl=300)
    cached = cache.get_json(cache_key, feature="profile_document_index")
    if cached:
        logger.debug("build_profile_document_index: cache hit for %s", cache_key)
        payload = cached.get("payload") or cached
        return ProfileDocumentIndex.from_payload(payload)

    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
    collection = build_collection_name(subscription_id)
    scroll_filter = _build_profile_filter(subscription_id, profile_id)
    offset: Optional[Any] = None

    documents: Dict[str, DocumentIndexEntry] = {}
    total_points = 0
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("document_id")
            if not doc_id:
                continue
            source_name, document_type, ingestion_source, chunk_kind, _doc_id, chunk_count = _summarize_payload(payload)
            entry = documents.get(doc_id)
            if not entry:
                entry = DocumentIndexEntry(document_id=str(doc_id))
                documents[doc_id] = entry
            if source_name and not entry.source_name:
                entry.source_name = source_name
            if document_type and not entry.document_type:
                entry.document_type = document_type
            if ingestion_source and not entry.ingestion_source:
                entry.ingestion_source = ingestion_source
            if chunk_count is not None:
                chunk_count_val = int(chunk_count)
                entry.chunk_count_max = max(entry.chunk_count_max or 0, chunk_count_val)
            if chunk_kind:
                entry.chunk_kind_counts[chunk_kind] = entry.chunk_kind_counts.get(chunk_kind, 0) + 1
            entry.total_points += 1
            total_points += 1
        if next_offset is None:
            break
        offset = next_offset

    document_ids = sorted(documents.keys())
    index = ProfileDocumentIndex(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        document_ids=document_ids,
        documents=documents,
        total_points=total_points,
        cached_at=time.time(),
    )
    cache.set_json(cache_key, stamp_cache_payload({"payload": index.to_payload()}), feature="profile_document_index")
    logger.debug("build_profile_document_index: built index with %d documents, %d total_points", len(document_ids), total_points)
    return index


__all__ = ["ProfileDocumentIndex", "DocumentIndexEntry", "build_profile_document_index", "get_redis_client"]
