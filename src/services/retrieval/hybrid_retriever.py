from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range

try:
    from rank_bm25 import BM25Okapi
except Exception:  # noqa: BLE001
    BM25Okapi = None

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrieverConfig:
    topk_dense: int = 50
    hybrid_alpha: float = 0.75
    lexical_weight: float = 0.25
    score_threshold: Optional[float] = None
    metadata_boost: float = 0.12
    strict_filter_boost: float = 0.2


@dataclass
class RetrievalCandidate:
    id: str
    text: str
    score: float
    vector_score: float
    lexical_score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None
    method: str = "dense"
    boosts: Dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    """Dense-first retrieval with lexical and metadata-aware re-scoring."""

    def __init__(self, client: Any, embedder: Any, config: Optional[HybridRetrieverConfig] = None):
        self.client = client
        self.embedder = embedder
        self.config = config or HybridRetrieverConfig()

    @staticmethod
    def _normalize_scores(values: List[float]) -> List[float]:
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if math.isclose(min_val, max_val):
            return [1.0 for _ in values]
        span = max_val - min_val
        return [(v - min_val) / span for v in values]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]{2,}", (text or "").lower())

    def _lexical_scores(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        tokens = [self._tokenize(text) for text in texts]
        if BM25Okapi:
            bm25 = BM25Okapi(tokens)
            scores = bm25.get_scores(self._tokenize(query)).tolist()
            return [float(s) for s in scores]
        # Fallback: simple token overlap ratio
        query_tokens = set(self._tokenize(query))
        scores = []
        for token_list in tokens:
            if not token_list or not query_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens.intersection(token_list))
            scores.append(overlap / max(len(query_tokens), 1))
        return scores

    def _build_filter(
        self,
        profile_id: str,
        document_ids: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
        section_titles: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        min_confidence: Optional[float] = None,
    ) -> Filter:
        if not profile_id:
            raise ValueError("profile_id is required for retrieval")
        conditions = [FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id)))]
        if document_ids:
            conditions.append(FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids])))
        if source_files:
            conditions.append(FieldCondition(key="source_file", match=MatchAny(any=[str(s) for s in source_files])))
        if doc_types:
            conditions.append(FieldCondition(key="doc_type", match=MatchAny(any=[str(d) for d in doc_types])))
        if section_titles:
            conditions.append(FieldCondition(key="section_title", match=MatchAny(any=[str(s) for s in section_titles])))
        if page_numbers:
            values = []
            for page in page_numbers:
                try:
                    values.append(int(page))
                except Exception:
                    continue
                values.append(str(page))
            if values:
                conditions.append(FieldCondition(key="page", match=MatchAny(any=values)))
        if min_confidence is not None:
            try:
                min_conf = float(min_confidence)
                conditions.append(FieldCondition(key="ocr_confidence", range=Range(gte=min_conf)))
            except Exception:
                pass
        return Filter(must=conditions)

    def _metadata_boost(self, query: str, payload: Dict[str, Any], hints: Dict[str, Any]) -> Dict[str, float]:
        boosts: Dict[str, float] = {}
        lowered = (query or "").lower()

        for field in ("doc_type", "document_type"):
            value = payload.get(field)
            if value and str(value).lower() in lowered:
                boosts["doc_type"] = self.config.metadata_boost

        for field in ("section_title", "section", "heading"):
            value = payload.get(field)
            if value and str(value).lower() in lowered:
                boosts["section"] = self.config.metadata_boost

        for field in ("title", "document_title"):
            value = payload.get(field)
            if value and str(value).lower() in lowered:
                boosts["title"] = self.config.metadata_boost * 0.8

        for field in ("product_name", "product"):
            value = payload.get(field)
            if value and str(value).lower() in lowered:
                boosts["product"] = self.config.metadata_boost * 0.9

        # Soft boosts for hinted docs/sections when no explicit filter applied
        for hint in hints.get("document_hints", []) or []:
            if hint and hint.lower() in (payload.get("source_file") or "").lower():
                boosts["document_hint"] = self.config.metadata_boost * 0.6
        return boosts

    def retrieve(
        self,
        *,
        collection_name: str,
        query: str,
        profile_id: str,
        query_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        explicit_hints: Optional[Dict[str, List[str]]] = None,
    ) -> List[RetrievalCandidate]:
        start = time.time()
        filters = filters or {}
        explicit_hints = explicit_hints or {}
        top_k = int(top_k or self.config.topk_dense)
        try:
            query_vector = self.embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            if hasattr(query_vector, "astype"):
                query_vector = query_vector.astype(np.float32).tolist()
            elif isinstance(query_vector, (list, tuple)):
                query_vector = [float(v) for v in query_vector]
            else:
                query_vector = [float(v) for v in list(query_vector)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hybrid retrieval skipped (embedder unavailable): %s", exc)
            return []

        strict_filters = {
            key: filters.get(key)
            for key in ("document_ids", "source_files", "doc_types", "section_titles", "page_numbers", "min_confidence")
            if filters.get(key)
        }
        use_strict = bool(explicit_hints) or bool(strict_filters)

        query_filter = self._build_filter(
            profile_id,
            document_ids=strict_filters.get("document_ids") if use_strict else None,
            source_files=strict_filters.get("source_files") if use_strict else None,
            doc_types=strict_filters.get("doc_types") if use_strict else None,
            section_titles=strict_filters.get("section_titles") if use_strict else None,
            page_numbers=strict_filters.get("page_numbers") if use_strict else None,
            min_confidence=strict_filters.get("min_confidence") if use_strict else None,
        )

        kwargs = dict(
            collection_name=collection_name,
            query=query_vector,
            using="content_vector",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        if self.config.score_threshold is not None:
            kwargs["score_threshold"] = self.config.score_threshold
        results = self.client.query_points(**kwargs)

        points = []
        if hasattr(results, "points"):
            points = results.points or []
        elif isinstance(results, tuple) and results:
            points = results[0] or []

        texts = [getattr(pt, "payload", {}) or {} for pt in points]
        text_values = [payload.get("text", "") for payload in texts]
        vector_scores = [float(getattr(pt, "score", 0.0)) for pt in points]
        lexical_scores = self._lexical_scores(query, text_values)

        vector_norm = self._normalize_scores(vector_scores)
        lexical_norm = self._normalize_scores(lexical_scores)
        candidates: List[RetrievalCandidate] = []

        for idx, pt in enumerate(points):
            payload = getattr(pt, "payload", {}) or {}
            text = payload.get("text", "") or ""
            if not text.strip():
                continue
            vector_score = vector_norm[idx] if idx < len(vector_norm) else 0.0
            lexical_score = lexical_norm[idx] if idx < len(lexical_norm) else 0.0
            hybrid = (self.config.hybrid_alpha * vector_score) + ((1 - self.config.hybrid_alpha) * lexical_score)

            boosts = self._metadata_boost(query, payload, filters)
            boost_value = sum(boosts.values())
            if use_strict and boost_value > 0:
                boost_value += self.config.strict_filter_boost

            score = hybrid + boost_value
            candidates.append(
                RetrievalCandidate(
                    id=str(getattr(pt, "id", "")),
                    text=text,
                    score=float(score),
                    vector_score=float(vector_score),
                    lexical_score=float(lexical_score),
                    metadata=payload,
                    source=payload.get("source_file") or payload.get("source"),
                    method="dense+lexical",
                    boosts=boosts,
                )
            )

        candidates.sort(key=lambda c: float(c.score), reverse=True)

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Hybrid retrieval completed",
            extra={
                "query_id": query_id,
                "candidates": len(candidates),
                "top_k": top_k,
                "use_strict_filters": use_strict,
                "latency_ms": round(elapsed_ms, 2),
            },
        )
        return candidates
