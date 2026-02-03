from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .evidence_constraints import EvidenceConstraints, EvidenceRequirements
from src.utils.payload_utils import get_source_name


@dataclass
class HybridRankerConfig:
    lexical_weight: float = 0.25
    metadata_weight: float = 0.15
    evidence_weight: float = 0.25
    section_boost: float = 0.12
    keyword_penalty: float = 0.15


class HybridRanker:
    """Lightweight hybrid re-ranker using lexical overlap and evidence gating."""

    def __init__(self, config: Optional[HybridRankerConfig] = None):
        self.config = config or HybridRankerConfig()
        self.constraints = EvidenceConstraints()
        self.section_terms = {
            "summary",
            "overview",
            "abstract",
            "details",
            "analysis",
            "results",
            "conclusion",
            "requirements",
        }

    def rank(
        self,
        query: str,
        chunks: List[Any],
        requirements: EvidenceRequirements,
        intent_type: str = "factual",
        *,
        relax_evidence: bool = False,
    ) -> List[Any]:
        if not chunks:
            return []
        query_tokens = self._tokenize(query)
        ranked: List[tuple[float, Any]] = []

        for chunk in chunks:
            text = getattr(chunk, "text", None) or ""
            meta = getattr(chunk, "metadata", {}) or {}
            base_score = float(getattr(chunk, "score", 0.0))
            lex_score = self._lexical_overlap(query_tokens, text)
            meta_score = self._metadata_boost(meta, query, intent_type)
            evidence_score = self.constraints.score_chunk(text, requirements)

            combined = base_score
            combined += self.config.lexical_weight * lex_score
            combined += self.config.metadata_weight * meta_score
            if not relax_evidence:
                combined += self.config.evidence_weight * evidence_score

            if requirements.required_keywords and lex_score < 0.05:
                combined -= self.config.keyword_penalty

            meta_features = meta.get("rank_features") or {}
            meta_features.update(
                {
                    "lexical_overlap": round(lex_score, 4),
                    "metadata_boost": round(meta_score, 4),
                    "evidence_score": round(evidence_score, 4),
                }
            )
            meta["rank_features"] = meta_features
            chunk.metadata = meta
            chunk.score = float(combined)
            ranked.append((combined, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked]

    def _metadata_boost(self, meta: Dict[str, Any], query: str, intent_type: str) -> float:
        score = 0.0
        section = str(meta.get("section_title") or meta.get("section_path") or meta.get("section") or "").lower()
        source = str(get_source_name(meta) or "").lower()
        query_lower = (query or "").lower()
        chunk_kind = str(meta.get("chunk_kind") or "").lower()
        importance = meta.get("section_importance_score") or meta.get("element_importance_score")
        try:
            importance_score = float(importance) if importance is not None else 0.0
        except (TypeError, ValueError):
            importance_score = 0.0

        score += min(0.35, max(0.0, importance_score)) * 0.6

        for term in self.section_terms:
            if term in section:
                score += self.config.section_boost
                break

        if intent_type in {"summarization", "deep_analysis"}:
            if chunk_kind in {"doc_summary", "section_summary"}:
                score += self.config.section_boost
        if intent_type in {"numeric_lookup", "field_extraction"}:
            if chunk_kind in {"table_text", "structured_field"}:
                score += self.config.section_boost

        for field in ("title", "document_title", "heading"):
            value = meta.get(field)
            if value and str(value).lower() in query_lower:
                score += self.config.section_boost * 0.8

        if source and source in query_lower:
            score += self.config.section_boost * 0.7

        return score

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]{3,}", (text or "").lower())

    @staticmethod
    def _lexical_overlap(query_tokens: List[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))
        if not text_tokens:
            return 0.0
        overlap = len(set(query_tokens) & text_tokens)
        return overlap / max(len(set(query_tokens)), 1)
