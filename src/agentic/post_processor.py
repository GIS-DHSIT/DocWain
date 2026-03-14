from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.api.reasoning_layer import AnswerVerifier, ConfidenceScorer, VerificationReport
from src.chat.opener_generator import contains_banned_opener
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateScore:
    model: str
    confidence: float
    grounded: bool
    verification: VerificationReport


@dataclass
class PostProcessResult:
    answer: str
    confidence: float
    verification: VerificationReport
    candidate_scores: List[CandidateScore] = field(default_factory=list)


class PostProcessor:
    """Verify, rerank, and filter candidate answers."""

    def __init__(self, verifier: AnswerVerifier, scorer: ConfidenceScorer):
        self.verifier = verifier
        self.scorer = scorer

    def select_best(
        self,
        candidates: List[Tuple[str, str]],
        *,
        sources: List[Dict[str, Any]],
        retrieved_chunks: List[Any],
        confidence_threshold: float,
        opener: Optional[str] = None,
    ) -> PostProcessResult:
        logger.debug("select_best called with candidates=%s, sources=%s, threshold=%s",
                     len(candidates), len(sources), confidence_threshold)
        scored: List[CandidateScore] = []
        best_answer = ""
        best_confidence = 0.0
        best_verification: Optional[VerificationReport] = None

        for model_name, text in candidates:
            if not text or not text.strip():
                continue
            verification = self.verifier.verify(text, sources)
            confidence, _breakdown = self.scorer.score(retrieved_chunks, sources, verification)
            scored.append(
                CandidateScore(
                    model=model_name,
                    confidence=confidence,
                    grounded=verification.overall_grounded,
                    verification=verification,
                )
            )

            if verification.overall_grounded and confidence >= best_confidence:
                best_confidence = confidence
                best_answer = text
                best_verification = verification

        if not best_answer and scored:
            # Try to salvage the strongest candidate by stripping unsupported sentences.
            strongest = max(scored, key=lambda s: s.confidence)
            candidate_text = next((t for m, t in candidates if m == strongest.model), "")
            cleaned = self._strip_unsupported(candidate_text, strongest.verification)
            if cleaned:
                verification = self.verifier.verify(cleaned, sources)
                confidence, _breakdown = self.scorer.score(retrieved_chunks, sources, verification)
                if verification.overall_grounded:
                    best_answer = cleaned
                    best_confidence = confidence
                    best_verification = verification

        if not best_answer and candidates and sources:
            # Attempt auto-citation for the strongest candidate when citations are missing.
            best_candidate = max(scored, key=lambda s: s.confidence) if scored else None
            candidate_text = ""
            if best_candidate:
                candidate_text = next((t for m, t in candidates if m == best_candidate.model), "")
            if candidate_text:
                auto_cited = self._auto_cite(candidate_text, sources)
                verification = self.verifier.verify(auto_cited, sources)
                confidence, _breakdown = self.scorer.score(retrieved_chunks, sources, verification)
                if verification.overall_grounded:
                    best_answer = auto_cited
                    best_confidence = confidence
                    best_verification = verification

        if not best_answer and scored:
            strongest = max(scored, key=lambda s: s.confidence)
            candidate_text = next((t for m, t in candidates if m == strongest.model), "")
            if candidate_text:
                cleaned = self._strip_unsupported(candidate_text, strongest.verification)
                candidate_text = cleaned or candidate_text
                if sources:
                    candidate_text = self._auto_cite(candidate_text, sources)
                verification = self.verifier.verify(candidate_text, sources) if sources else strongest.verification
                confidence, _breakdown = (
                    self.scorer.score(retrieved_chunks, sources, verification)
                    if sources
                    else (strongest.confidence, {})
                )
                best_answer = candidate_text
                best_confidence = confidence
                best_verification = verification

        if not best_answer:
            fallback_verification = VerificationReport(
                citations_valid=True,
                invalid_citations=[],
                missing_citations=0,
                unsupported_sentences=[],
                support_score=0.0,
                coverage_score=0.0,
                numeric_support_rate=0.0,
                overall_grounded=False,
            )
            logger.debug("select_best returning empty answer, no grounded candidate found")
            return PostProcessResult(
                answer="",
                confidence=best_confidence,
                verification=best_verification or fallback_verification,
                candidate_scores=scored,
            )

        best_answer = self._rewrite_banned_opener(best_answer, opener=opener)

        logger.debug("select_best returning confidence=%.3f, grounded=%s, scored_count=%s",
                     best_confidence, best_verification.overall_grounded if best_verification else None, len(scored))
        return PostProcessResult(
            answer=best_answer,
            confidence=best_confidence,
            verification=best_verification or scored[0].verification,
            candidate_scores=scored,
        )

    @staticmethod
    def _strip_unsupported(text: str, verification: VerificationReport) -> str:
        if not verification.unsupported_sentences:
            return text
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        supported = [s for s in sentences if s.strip() and s not in verification.unsupported_sentences]
        return " ".join(supported).strip()

    @staticmethod
    def _auto_cite(text: str, sources: List[Dict[str, Any]]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        cited: List[str] = []
        source_texts = [str(src.get("excerpt") or "") for src in sources]

        def _tokenize(value: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]{3,}", value.lower()))

        source_tokens = [_tokenize(t) for t in source_texts]
        for sentence in sentences:
            if not sentence.strip():
                continue
            if re.search(r"\[SOURCE-\d+\]", sentence) or re.search(r"\(.*?Page.*?Section.*?\)", sentence):
                cited.append(sentence)
                continue
            sent_tokens = _tokenize(sentence)
            if not sent_tokens:
                cited.append(sentence)
                continue
            best_idx = None
            best_score = 0.0
            for idx, tokens in enumerate(source_tokens):
                if not tokens:
                    continue
                overlap = len(sent_tokens & tokens) / max(len(sent_tokens), 1)
                if overlap > best_score:
                    best_score = overlap
                    best_idx = idx
            if best_idx is not None and best_score >= 0.15:
                label = PostProcessor._format_label(sources[best_idx])
                cited.append(f"{sentence} {label}")
            else:
                cited.append(sentence)
        return " ".join(cited).strip()

    @staticmethod
    def _format_label(source: Dict[str, Any]) -> str:
        citation = str(source.get("citation") or "").strip()
        if citation:
            return f"({citation})"
        name = str(source.get("source_name") or "").strip()
        if not name:
            return ""
        return f"(Source: {name})"

    @staticmethod
    def _rewrite_banned_opener(text: str, *, opener: Optional[str] = None) -> str:
        if not text:
            return text
        cleaned = text.strip()
        if not contains_banned_opener(cleaned):
            return text
        safe_opener = opener or "Here's what I found."
        remainder = PostProcessor._strip_first_sentence(cleaned)
        if not remainder:
            return safe_opener
        return f"{safe_opener}\n\n{remainder}".strip()

    @staticmethod
    def _strip_first_sentence(text: str) -> str:
        parts = re.split(r"(?<=[.!?])\\s+", text.strip(), maxsplit=1)
        if len(parts) == 1:
            return ""
        return parts[1].strip()
