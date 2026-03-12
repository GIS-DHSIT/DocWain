from __future__ import annotations

from src.utils.logging_utils import get_logger
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range
from src.api.qdrant_indexes import autoheal_missing_index
from src.retrieval.filter_builder import build_qdrant_filter
from src.retrieval.errors import RetrievalFilterError
from src.utils.payload_utils import get_content_text, get_document_type, get_embedding_text, get_source_name

try:
    from rank_bm25 import BM25Okapi
except Exception:  # noqa: BLE001
    BM25Okapi = None

logger = get_logger(__name__)

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
        self._profile_key_cache: Dict[tuple[str, str], Optional[str]] = {}
        self._profile_identity_cache: Dict[tuple[str, str], Optional[Tuple[str, List[Any]]]] = {}
        self._inferred_profile_cache: Dict[str, Optional[Tuple[str, str]]] = {}

    @staticmethod
    def _profile_value_variants(profile_id: str) -> List[Any]:
        variants: List[Any] = [str(profile_id)]
        try:
            if isinstance(profile_id, str) and profile_id.isdigit():
                variants.append(int(profile_id))
            elif isinstance(profile_id, (int, float)):
                variants.append(str(profile_id))
        except Exception:
            pass
        return variants

    def _resolve_profile_key(self, collection_name: Optional[str], profile_id: str) -> Optional[str]:
        _ = collection_name, profile_id
        logger.debug("Profile key inference disabled; enforcing strict profile filters.")
        return None

    def _infer_single_profile(self, collection_name: str, max_points: int = 200) -> Optional[Tuple[str, str]]:
        _ = collection_name, max_points
        logger.debug("Profile inference disabled; enforcing strict profile filters.")
        return None

    def _resolve_profile_identity(
        self, collection_name: Optional[str], profile_id: str
    ) -> Optional[Tuple[str, List[Any]]]:
        _ = collection_name, profile_id
        logger.debug("Profile identity inference disabled; enforcing strict profile filters.")
        return None

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
        if not any(tokens):
            return [0.0 for _ in texts]
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
        collection_name: Optional[str] = None,
        subscription_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
        doc_domains: Optional[List[str]] = None,
        section_ids: Optional[List[str]] = None,
        section_titles: Optional[List[str]] = None,
        section_kinds: Optional[List[str]] = None,
        chunk_kinds: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        min_confidence: Optional[float] = None,
    ) -> Filter:
        if not profile_id:
            raise ValueError("profile_id is required for retrieval")
        if not subscription_id:
            raise ValueError("subscription_id is required for retrieval")
        profile_value = str(profile_id)
        base = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=profile_value)
        conditions: List[Any] = list(getattr(base, "must", []) or [])
        should: List[FieldCondition] = list(getattr(base, "should", []) or [])
        if document_ids:
            conditions.append(FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids])))
        if source_files:
            values = [str(s) for s in source_files]
            should.append(FieldCondition(key="source.name", match=MatchAny(any=values)))
            should.append(FieldCondition(key="source_file", match=MatchAny(any=values)))
        if doc_types:
            values = [str(d) for d in doc_types]
            should.append(FieldCondition(key="document.type", match=MatchAny(any=values)))
            should.append(FieldCondition(key="doc_type", match=MatchAny(any=values)))
        if doc_domains:
            values = [str(d) for d in doc_domains]
            conditions.append(FieldCondition(key="doc_domain", match=MatchAny(any=values)))
        if section_ids:
            values = [str(s) for s in section_ids]
            conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="section_id", match=MatchAny(any=values)),
                        FieldCondition(key="section.id", match=MatchAny(any=values)),
                    ],
                )
            )
        if section_titles:
            conditions.append(FieldCondition(key="section_title", match=MatchAny(any=[str(s) for s in section_titles])))
        if section_kinds:
            values = [str(k) for k in section_kinds]
            conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="section_kind", match=MatchAny(any=values)),
                        FieldCondition(key="section.kind", match=MatchAny(any=values)),
                    ],
                )
            )
        if chunk_kinds:
            values = [str(k) for k in chunk_kinds]
            conditions.append(FieldCondition(key="chunk_kind", match=MatchAny(any=values)))
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
        return Filter(must=conditions, should=should or None)

    def _metadata_boost(self, query: str, payload: Dict[str, Any], hints: Dict[str, Any]) -> Dict[str, float]:
        boosts: Dict[str, float] = {}
        lowered = (query or "").lower()

        value = get_document_type(payload)
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
            source_name = get_source_name(payload) or ""
            if hint and hint.lower() in source_name.lower():
                boosts["document_hint"] = self.config.metadata_boost * 0.6
        for hint in hints.get("document_ids", []) or []:
            doc_id = payload.get("document_id")
            if doc_id and str(doc_id) == str(hint):
                boosts["document_id"] = self.config.metadata_boost * 0.8
        for hint in hints.get("chunk_ids", []) or []:
            chunk_id = payload.get("chunk_id")
            if chunk_id and str(chunk_id) == str(hint):
                boosts["chunk_id"] = self.config.metadata_boost
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
        subscription_id: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        start = time.time()
        filters = filters or {}
        explicit_hints = explicit_hints or {}
        top_k = int(top_k or self.config.topk_dense)

        strict_filters = {
            key: filters.get(key)
            for key in (
                "document_ids",
                "source_files",
                "doc_types",
                "doc_domains",
                "section_ids",
                "section_titles",
                "section_kinds",
                "chunk_kinds",
                "page_numbers",
                "min_confidence",
            )
            if filters.get(key)
        }
        use_strict = bool(strict_filters)

        effective_subscription = subscription_id or filters.get("subscription_id") or collection_name
        query_filter = self._build_filter(
            profile_id,
            collection_name=collection_name,
            subscription_id=effective_subscription,
            document_ids=strict_filters.get("document_ids") if use_strict else None,
            source_files=strict_filters.get("source_files") if use_strict else None,
            doc_types=strict_filters.get("doc_types") if use_strict else None,
            doc_domains=strict_filters.get("doc_domains") if use_strict else None,
            section_ids=strict_filters.get("section_ids") if use_strict else None,
            section_titles=strict_filters.get("section_titles") if use_strict else None,
            section_kinds=strict_filters.get("section_kinds") if use_strict else None,
            chunk_kinds=strict_filters.get("chunk_kinds") if use_strict else None,
            page_numbers=strict_filters.get("page_numbers") if use_strict else None,
            min_confidence=strict_filters.get("min_confidence") if use_strict else None,
        )

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
            return self._lexical_fallback(
                collection_name=collection_name,
                query=query,
                query_filter=query_filter,
                filters=filters,
                top_k=top_k,
                use_strict=use_strict,
                query_id=query_id,
                start=start,
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
        try:
            results = self.client.query_points(**kwargs)
        except UnexpectedResponse as exc:
            detail = getattr(exc, "content", None) or str(exc)
            missing_field = autoheal_missing_index(self.client, collection_name, detail)
            if missing_field:
                logger.warning("Retrying hybrid retrieval after auto-heal for %s", missing_field)
                try:
                    results = self.client.query_points(**kwargs)
                except UnexpectedResponse as retry_exc:
                    retry_detail = getattr(retry_exc, "content", None) or str(retry_exc)
                    raise RetrievalFilterError(
                        "Index required but not found; retrieval blocked.",
                        code="RETRIEVAL_INDEX_MISSING",
                        details=f"missing_index={missing_field}; {retry_detail}",
                    ) from retry_exc
            else:
                raise RetrievalFilterError(
                    "Profile isolation enforced; cannot search outside profile.",
                    code="RETRIEVAL_FILTER_FAILED",
                    details=str(detail),
                ) from exc

        points = []
        if hasattr(results, "points"):
            points = results.points or []
        elif isinstance(results, tuple) and results:
            points = results[0] or []

        texts = [getattr(pt, "payload", {}) or {} for pt in points]
        text_values = [get_embedding_text(payload) or get_content_text(payload) for payload in texts]
        vector_scores = [float(getattr(pt, "score", 0.0)) for pt in points]
        lexical_scores = self._lexical_scores(query, text_values)

        vector_norm = self._normalize_scores(vector_scores)
        lexical_norm = self._normalize_scores(lexical_scores)
        candidates: List[RetrievalCandidate] = []

        for idx, pt in enumerate(points):
            payload = getattr(pt, "payload", {}) or {}
            text = get_embedding_text(payload) or get_content_text(payload) or payload.get("text") or payload.get("content") or ""
            if not str(text).strip():
                continue
            vector_score = vector_norm[idx] if idx < len(vector_norm) else 0.0
            lexical_score = lexical_norm[idx] if idx < len(lexical_norm) else 0.0
            hybrid = (self.config.hybrid_alpha * vector_score) + ((1 - self.config.hybrid_alpha) * lexical_score)

            boost_hints = {**filters, **explicit_hints}
            boosts = self._metadata_boost(query, payload, boost_hints)
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
                    source=get_source_name(payload) or payload.get("source"),
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

    def _lexical_fallback(
        self,
        *,
        collection_name: str,
        query: str,
        query_filter: Filter,
        filters: Dict[str, Any],
        top_k: int,
        use_strict: bool,
        query_id: Optional[str],
        start: float,
    ) -> List[RetrievalCandidate]:
        try:
            scroll = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=max(50, int(top_k * 5)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hybrid lexical fallback failed: %s", exc)
            return []

        points = []
        if hasattr(scroll, "points"):
            points = scroll.points or []
        elif isinstance(scroll, tuple) and scroll:
            points = scroll[0] or []

        if not points:
            return []

        payloads = [getattr(pt, "payload", {}) or {} for pt in points]
        texts = [get_embedding_text(payload) for payload in payloads]
        lexical_scores = self._lexical_scores(query, texts)
        if any(score > 0 for score in lexical_scores):
            lexical_norm = self._normalize_scores(lexical_scores)
        else:
            lexical_norm = [0.0 for _ in lexical_scores]

        candidates: List[RetrievalCandidate] = []
        for idx, pt in enumerate(points):
            payload = getattr(pt, "payload", {}) or {}
            text = get_embedding_text(payload)
            if not text.strip():
                continue
            lexical_score = lexical_norm[idx] if idx < len(lexical_norm) else 0.0
            boosts = self._metadata_boost(query, payload, filters)
            boost_value = sum(boosts.values())
            if use_strict and boost_value > 0:
                boost_value += self.config.strict_filter_boost
            score = lexical_score + boost_value
            candidates.append(
                RetrievalCandidate(
                    id=str(getattr(pt, "id", "")),
                    text=text,
                    score=float(score),
                    vector_score=0.0,
                    lexical_score=float(lexical_score),
                    metadata=payload,
                    source=get_source_name(payload) or payload.get("source"),
                    method="lexical_fallback",
                    boosts=boosts,
                )
            )

        candidates.sort(key=lambda c: float(c.score), reverse=True)

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Hybrid lexical fallback completed",
            extra={
                "query_id": query_id,
                "candidates": len(candidates),
                "top_k": top_k,
                "use_strict_filters": use_strict,
                "latency_ms": round(elapsed_ms, 2),
            },
        )
        return candidates[:top_k]
