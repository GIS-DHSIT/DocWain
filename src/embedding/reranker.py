"""
Advanced reranking system with dual-model ensemble and cross-encoder scoring.

Implements hybrid dense-sparse scoring, cross-encoder re-ranking, and
semantic similarity boosting for improved retrieval accuracy.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config

logger = get_logger(__name__)

@dataclass(frozen=True)
class RerankingConfig:
    """Configuration for reranking strategy."""

    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_threshold: float = 0.3
    alpha_dense: float = 0.6  # Weight for dense vector similarity
    alpha_sparse: float = 0.2  # Weight for sparse (keyword) similarity
    alpha_semantic: float = 0.2  # Weight for semantic coherence
    top_k_rerank: int = 50  # Number of results to rerank
    min_rank_score: float = 0.1
    enable_diversity_boost: bool = True
    diversity_penalty: float = 0.05

@dataclass(frozen=True)
class RerankResult:
    """Result of reranking operation."""

    chunk_id: str
    original_score: float
    rerank_score: float
    cross_encoder_score: Optional[float] = None
    final_score: float = 0.0
    boost_applied: str = ""

class RetrievalReranker:
    """Reranks retrieved chunks using ensemble scoring and cross-encoder models."""

    def __init__(
        self,
        config: Optional[RerankingConfig] = None,
        cross_encoder_model: Optional[Any] = None,
    ):
        self.config = config or RerankingConfig()
        self.cross_encoder = cross_encoder_model
        self._cross_encoder_cache: Dict[str, float] = {}

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Any],
        *,
        profile_context: Optional[Dict[str, Any]] = None,
        query_intent: Optional[str] = None,
    ) -> List[Any]:
        """Rerank chunks using ensemble scoring and optional cross-encoder."""
        if not chunks:
            return []

        # Limit reranking to top-k for efficiency
        chunks_to_rerank = chunks[: self.config.top_k_rerank]

        # Compute base dense scores
        dense_scores = {
            str(c.chunk_id): float(c.score or 0.0)
            for c in chunks_to_rerank
            if hasattr(c, "chunk_id") and hasattr(c, "score")
        }

        # Compute sparse keyword overlap scores
        sparse_scores = self._compute_sparse_scores(query, chunks_to_rerank)

        # Compute semantic coherence scores
        semantic_scores = self._compute_semantic_scores(
            query,
            chunks_to_rerank,
            query_intent=query_intent,
        )

        # Apply cross-encoder if available
        cross_encoder_scores = {}
        if self.config.use_cross_encoder and self.cross_encoder:
            cross_encoder_scores = self._compute_cross_encoder_scores(
                query,
                chunks_to_rerank,
            )

        # Normalize and combine scores
        reranked = self._ensemble_score(
            query,
            chunks_to_rerank,
            dense_scores,
            sparse_scores,
            semantic_scores,
            cross_encoder_scores,
            profile_context=profile_context,
        )

        # Apply diversity boost if enabled
        if self.config.enable_diversity_boost:
            reranked = self._apply_diversity_boost(reranked, chunks_to_rerank)

        # Sort by final score and reattach remaining chunks
        reranked = sorted(reranked, key=lambda x: x.final_score, reverse=True)
        reranked = [r for r in reranked if r.final_score >= self.config.min_rank_score]

        # Return reranked chunks plus any remainder
        result = []
        reranked_ids = {r.chunk_id for r in reranked}
        for i, c in enumerate(reranked):
            # Update chunk score
            if hasattr(c, "score"):
                c.score = c.final_score
            result.append(c)

        # Add remaining chunks beyond rerank window
        for c in chunks[self.config.top_k_rerank :]:
            if hasattr(c, "chunk_id") and str(c.chunk_id) not in reranked_ids:
                result.append(c)

        return result

    def _compute_sparse_scores(
        self,
        query: str,
        chunks: List[Any],
    ) -> Dict[str, float]:
        """Compute BM25-like keyword overlap scores."""
        q_tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        if not q_tokens:
            return {str(c.chunk_id): 0.0 for c in chunks}

        scores = {}
        for chunk in chunks:
            chunk_id = str(chunk.chunk_id)
            text = getattr(chunk, "text", "") or ""
            chunk_tokens = set(re.findall(r"[a-z0-9]{3,}", text.lower()))

            if not chunk_tokens:
                scores[chunk_id] = 0.0
                continue

            overlap = len(q_tokens & chunk_tokens)
            precision = overlap / max(len(q_tokens), 1)
            recall = overlap / max(len(chunk_tokens), 1)
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            scores[chunk_id] = f1

        return scores

    def _compute_semantic_scores(
        self,
        query: str,
        chunks: List[Any],
        query_intent: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute semantic coherence scores based on query intent and chunk kind."""
        scores = {}
        intent_weights = self._get_intent_weights(query_intent)

        for chunk in chunks:
            chunk_id = str(chunk.chunk_id)
            chunk_kind = (chunk.meta or {}).get("chunk_kind", "section_text")

            # Boost highly relevant chunk kinds for specific intents
            kind_boost = intent_weights.get(chunk_kind, 1.0)

            # Check for entity matching
            entity_boost = self._compute_entity_boost(query, chunk)

            # Combine semantic signals
            semantic_score = 0.7 * kind_boost + 0.3 * entity_boost
            scores[chunk_id] = min(1.0, semantic_score)

        return scores

    def _get_intent_weights(self, query_intent: Optional[str]) -> Dict[str, float]:
        """Get chunk kind weights based on query intent."""
        weights = {
            "skills": 1.0,
            "experience": 1.0,
            "certification": 0.9,
            "education": 0.85,
            "project": 0.8,
            "kv": 0.7,
            "list_item": 0.8,
            "table_row": 0.75,
            "section_text": 0.6,
            "section_summary": 0.7,
        }

        if query_intent == "skills":
            weights.update({"skills": 1.0, "certification": 0.95, "experience": 0.85})
        elif query_intent == "experience":
            weights.update({"experience": 1.0, "project": 0.9, "skills": 0.8})
        elif query_intent == "education":
            weights.update({"education": 1.0, "certification": 0.9})

        return weights

    def _compute_entity_boost(self, query: str, chunk: Any) -> float:
        """Compute entity matching boost."""
        text = getattr(chunk, "text", "") or ""
        if not text or not query:
            return 0.5

        # Simple entity matching: check for named entities or technical terms
        query_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        chunk_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

        if not query_entities:
            return 0.5

        matches = sum(1 for qe in query_entities if any(qe in ce for ce in chunk_entities))
        entity_boost = matches / len(query_entities) if query_entities else 0.5
        return min(1.0, 0.5 + entity_boost * 0.5)

    def _compute_cross_encoder_scores(
        self,
        query: str,
        chunks: List[Any],
    ) -> Dict[str, float]:
        """Compute cross-encoder relevance scores."""
        scores = {}

        if not self.cross_encoder or not chunks:
            return scores

        try:
            texts = []
            chunk_ids = []
            for chunk in chunks:
                text = getattr(chunk, "text", "") or ""
                if text:
                    texts.append(text)
                    chunk_ids.append(str(chunk.chunk_id))

            if not texts:
                return scores

            # Check cache first
            ce_scores = []
            uncached_indices = []
            for i, text in enumerate(texts):
                cache_key = f"{query}|{text[:50]}"
                if cache_key in self._cross_encoder_cache:
                    ce_scores.append(self._cross_encoder_cache[cache_key])
                else:
                    ce_scores.append(None)
                    uncached_indices.append(i)

            # Batch compute uncached scores
            if uncached_indices:
                uncached_texts = [texts[i] for i in uncached_indices]
                pairs = [[query, text] for text in uncached_texts]

                # Assume cross_encoder has a predict method
                ce_predictions = self.cross_encoder.predict(pairs)

                for idx, ce_idx in enumerate(uncached_indices):
                    ce_scores[ce_idx] = float(ce_predictions[idx])
                    cache_key = f"{query}|{texts[ce_idx][:50]}"
                    self._cross_encoder_cache[cache_key] = ce_scores[ce_idx]

            # Normalize cross-encoder scores to [0, 1]
            ce_min = min(ce_scores) if ce_scores else 0.0
            ce_max = max(ce_scores) if ce_scores else 1.0
            ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

            for chunk_id, ce_score in zip(chunk_ids, ce_scores):
                normalized = (ce_score - ce_min) / ce_range if ce_range > 0 else 0.5
                scores[chunk_id] = float(normalized)

        except Exception as exc:
            logger.warning("Cross-encoder scoring failed: %s", exc)

        return scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores

        min_score = min(scores.values())
        max_score = max(scores.values())
        score_range = max_score - min_score if max_score > min_score else 1.0

        return {
            k: (v - min_score) / score_range if score_range > 0 else 0.5
            for k, v in scores.items()
        }

    def _ensemble_score(
        self,
        query: str,
        chunks: List[Any],
        dense_scores: Dict[str, float],
        sparse_scores: Dict[str, float],
        semantic_scores: Dict[str, float],
        cross_encoder_scores: Dict[str, float],
        profile_context: Optional[Dict[str, Any]] = None,
    ) -> List[RerankResult]:
        """Combine multiple scoring signals into final relevance score."""
        # Normalize all scores
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)
        semantic_norm = self._normalize_scores(semantic_scores)
        cross_encoder_norm = (
            self._normalize_scores(cross_encoder_scores)
            if cross_encoder_scores
            else {}
        )

        results = []
        for chunk in chunks:
            chunk_id = str(chunk.chunk_id)

            # Get component scores
            d_score = dense_norm.get(chunk_id, 0.0)
            s_score = sparse_norm.get(chunk_id, 0.0)
            sem_score = semantic_norm.get(chunk_id, 0.5)
            ce_score = cross_encoder_norm.get(chunk_id) if cross_encoder_scores else None

            # Compute ensemble score
            base_score = (
                self.config.alpha_dense * d_score
                + self.config.alpha_sparse * s_score
                + self.config.alpha_semantic * sem_score
            )

            final_score = base_score
            boost_applied = ""

            # Apply cross-encoder if available
            if ce_score is not None and ce_score >= self.config.cross_encoder_threshold:
                final_score = 0.7 * base_score + 0.3 * ce_score
                boost_applied = "cross_encoder"

            # Apply profile context boost if provided
            if profile_context:
                context_boost = self._compute_context_boost(chunk, profile_context)
                if context_boost > 1.0:
                    final_score = min(1.0, final_score * context_boost)
                    boost_applied += "_context" if boost_applied else "context"

            results.append(
                RerankResult(
                    chunk_id=chunk_id,
                    original_score=float(chunk.score or 0.0),
                    rerank_score=float(base_score),
                    cross_encoder_score=ce_score,
                    final_score=float(final_score),
                    boost_applied=boost_applied,
                )
            )

        return results

    def _compute_context_boost(
        self,
        chunk: Any,
        profile_context: Dict[str, Any],
    ) -> float:
        """Compute boost factor based on profile context."""
        boost = 1.0

        # Boost if chunk is from recently accessed documents
        recent_docs = profile_context.get("recent_documents", set())
        if hasattr(chunk, "meta") and (chunk.meta or {}).get("document_id") in recent_docs:
            boost *= 1.1

        # Boost if chunk matches profile's document domain
        doc_domain = profile_context.get("inferred_domain", "generic")
        chunk_domain = (chunk.meta or {}).get("doc_domain", "generic")
        if chunk_domain == doc_domain and doc_domain != "generic":
            boost *= 1.05

        return boost

    def _apply_diversity_boost(
        self,
        reranked: List[RerankResult],
        chunks: List[Any],
    ) -> List[RerankResult]:
        """Apply diversity penalty to reduce redundant sections."""
        chunk_map = {str(c.chunk_id): c for c in chunks}
        seen_sections = set()

        for result in reranked:
            chunk = chunk_map.get(result.chunk_id)
            if not chunk or not hasattr(chunk, "meta"):
                continue

            section_id = (chunk.meta or {}).get("section_id")
            if section_id and section_id in seen_sections:
                # Apply penalty for already-seen sections
                result.final_score = (
                    result.final_score
                    * (1.0 - self.config.diversity_penalty)
                )
            elif section_id:
                seen_sections.add(section_id)

        return reranked

__all__ = ["RetrievalReranker", "RerankingConfig", "RerankResult"]

