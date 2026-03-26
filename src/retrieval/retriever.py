"""Unified retriever with hybrid dense + keyword fallback search."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.api.vector_store import build_collection_name

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """A single piece of retrieved evidence."""

    text: str
    source_name: str
    document_id: str
    profile_id: str
    section: str
    page_start: int
    page_end: int
    score: float
    chunk_id: str
    chunk_type: str = "text"
    profile_name: str = ""


@dataclass
class RetrievalResult:
    """Aggregated retrieval output."""

    chunks: List[EvidenceChunk]
    profiles_searched: List[str]
    total_found: int


class UnifiedRetriever:
    """Single retriever combining dense vector search with keyword fallback."""

    # Minimum number of high-quality results before triggering keyword fallback
    _DENSE_MIN = 3
    # Score threshold for "high quality" dense result
    _HIGH_QUALITY_THRESHOLD = 0.5

    def __init__(self, qdrant_client, embedder):
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self._collection_exists_cache: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        subscription_id: str,
        profile_ids: List[str],
        *,
        document_ids: Optional[List[str]] = None,
        top_k: int = 50,
        correlation_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Search across one or more profiles and return merged results."""
        if not subscription_id or not str(subscription_id).strip():
            raise ValueError("subscription_id is required for retrieval")

        collection_name = build_collection_name(subscription_id)

        # Guard: verify collection exists before querying Qdrant.
        # Cache the result to avoid repeated round-trips.
        if collection_name not in self._collection_exists_cache:
            try:
                self._collection_exists_cache[collection_name] = (
                    self.qdrant_client.collection_exists(collection_name)
                )
            except Exception:
                logger.warning(
                    "Could not verify collection existence: %s", collection_name,
                )
                self._collection_exists_cache[collection_name] = False

        if not self._collection_exists_cache.get(collection_name):
            logger.warning(
                "Collection %s does not exist — returning empty results for subscription=%s",
                collection_name, subscription_id,
            )
            return RetrievalResult(
                chunks=[],
                profiles_searched=list(profile_ids),
                total_found=0,
            )

        query_vector = self.embedder.encode([query])[0]

        all_chunks: List[EvidenceChunk] = []
        per_profile = max(1, top_k // max(len(profile_ids), 1))

        for pid in profile_ids:
            chunks = self._search_profile(
                collection_name, query, query_vector, subscription_id, pid,
                document_ids=document_ids,
                top_k=per_profile,
                correlation_id=correlation_id,
            )
            all_chunks.extend(chunks)

        # Sort globally by score descending and trim to top_k
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        all_chunks = all_chunks[:top_k]

        return RetrievalResult(
            chunks=all_chunks,
            profiles_searched=list(profile_ids),
            total_found=len(all_chunks),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        subscription_id: str,
        profile_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> Filter:
        """Build Qdrant filter scoped to subscription + single profile."""
        must = [
            FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        ]
        if document_ids:
            if len(document_ids) == 1:
                must.append(FieldCondition(key="document_id", match=MatchValue(value=document_ids[0])))
            else:
                must.append(FieldCondition(key="document_id", match=MatchAny(any=document_ids)))
        return Filter(must=must)

    def _search_profile(
        self,
        collection_name: str,
        query: str,
        query_vector: list,
        subscription_id: str,
        profile_id: str,
        *,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
        correlation_id: Optional[str] = None,
    ) -> List[EvidenceChunk]:
        """Dense search for a single profile, with keyword fallback."""
        qfilter = self._build_filter(subscription_id, profile_id, document_ids)

        try:
            result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="content_vector",
                query_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            points = result.points if hasattr(result, "points") else []
        except Exception:
            logger.exception(
                "Dense search failed collection=%s profile=%s cid=%s",
                collection_name, profile_id, correlation_id,
            )
            points = []

        # Filter out doc_index/doc_intelligence points — those are fetched separately
        points = [
            pt for pt in points
            if (pt.payload or {}).get("resolution", "chunk") not in ("doc_index", "doc_intelligence")
        ]
        chunks = [self._point_to_chunk(pt, profile_id) for pt in points]

        # Keyword fallback when dense returns too few high-quality hits
        high_quality = [c for c in chunks if c.score >= self._HIGH_QUALITY_THRESHOLD]
        if len(high_quality) < self._DENSE_MIN:
            fallback = self._keyword_fallback(
                collection_name, query, qfilter, top_k, existing_ids={c.chunk_id for c in chunks},
            )
            chunks.extend(fallback)

        return chunks

    def _keyword_fallback(
        self,
        collection_name: str,
        query: str,
        qfilter: Filter,
        top_k: int,
        existing_ids: set,
    ) -> List[EvidenceChunk]:
        """Scroll-based keyword fallback for low-confidence dense results."""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            records = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
        except Exception:
            logger.exception("Keyword fallback scroll failed collection=%s", collection_name)
            return []

        query_tokens = set(query.lower().split())
        fallback_chunks: List[EvidenceChunk] = []

        for record in records:
            payload = record.payload or {}
            text = payload.get("canonical_text") or payload.get("embedding_text") or ""
            chunk_id = (payload.get("chunk") or {}).get("id", "")
            if chunk_id in existing_ids:
                continue

            # Simple keyword overlap score
            text_tokens = set(text.lower().split())
            overlap = query_tokens & text_tokens
            if not overlap:
                continue

            precision = len(overlap) / len(query_tokens) if query_tokens else 0
            recall = len(overlap) / len(text_tokens) if text_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            chunk = self._point_to_chunk(record, payload.get("profile_id", ""))
            chunk.score = f1 * 0.5  # Scale keyword scores below dense scores
            fallback_chunks.append(chunk)

        fallback_chunks.sort(key=lambda c: c.score, reverse=True)
        return fallback_chunks[:top_k]

    @staticmethod
    def _point_to_chunk(point, profile_id: str) -> EvidenceChunk:
        """Convert a Qdrant point/record to an EvidenceChunk."""
        payload = point.payload or {}
        chunk_meta = payload.get("chunk") or {}
        section_meta = payload.get("section") or {}
        provenance = payload.get("provenance") or {}

        text = payload.get("canonical_text") or payload.get("embedding_text") or ""
        source_name = (
            payload.get("source_name")
            or provenance.get("source_file")
            or payload.get("source_file")
            or ""
        )

        return EvidenceChunk(
            text=text,
            source_name=source_name,
            document_id=payload.get("document_id", ""),
            profile_id=payload.get("profile_id", profile_id),
            section=section_meta.get("title", ""),
            page_start=provenance.get("page_start", 0),
            page_end=provenance.get("page_end", 0),
            score=getattr(point, "score", 0.0) or 0.0,
            chunk_id=chunk_meta.get("id", ""),
            chunk_type=chunk_meta.get("type", "text"),
        )
