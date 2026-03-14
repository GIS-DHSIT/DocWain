from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalVerification:
    alignment_score: float
    issues: List[str] = field(default_factory=list)
    should_refine: bool = False
    refined_filters: Dict[str, Any] = field(default_factory=dict)


class VerificationAgent:
    """Checks whether retrieved context aligns with query intent and content."""

    def __init__(self, min_alignment: float = 0.18):
        self.min_alignment = min_alignment

    def verify(
        self,
        query: str,
        intent: str,
        chunks: List[Any],
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalVerification:
        logger.debug("verify called with intent=%s, chunks=%s", intent, len(chunks) if chunks else 0)
        if not chunks:
            return RetrievalVerification(0.0, ["no_chunks"], True, metadata_filters or {})

        query_tokens = self._tokenize(query)
        top_chunks = chunks[:5]
        overlap_scores: List[float] = []

        for chunk in top_chunks:
            text = getattr(chunk, "text", "") or ""
            overlap_scores.append(self._overlap(query_tokens, self._tokenize(text)))

        alignment = sum(overlap_scores) / max(len(overlap_scores), 1)
        issues: List[str] = []
        should_refine = alignment < self.min_alignment

        if intent == "comparison":
            doc_ids = {
                str((getattr(chunk, "metadata", {}) or {}).get("document_id"))
                for chunk in top_chunks
                if (getattr(chunk, "metadata", {}) or {}).get("document_id")
            }
            if len(doc_ids) < 2:
                issues.append("comparison_single_doc")
                should_refine = True

        if alignment < self.min_alignment:
            issues.append("low_alignment")

        refined = metadata_filters or {}
        logger.debug("verify returning alignment=%.4f, should_refine=%s, issues=%s",
                     alignment, should_refine, issues)
        return RetrievalVerification(
            alignment_score=round(alignment, 4),
            issues=issues,
            should_refine=should_refine,
            refined_filters=refined,
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]{3,}", (text or "").lower())

    @staticmethod
    def _overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        qset = set(query_tokens)
        dset = set(doc_tokens)
        return len(qset & dset) / max(len(qset), 1)
