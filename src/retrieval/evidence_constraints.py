from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_NUMBER_RE = re.compile(r"\b\d+[\d,\.]*\b")
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"\b(usd|eur|gbp|inr|\$|%|percent|percentage|years?|months?|days?|hours?|mins?|minutes?)\b",
    re.IGNORECASE,
)


@dataclass
class EvidenceRequirements:
    requires_number: bool = False
    requires_date_range: bool = False
    requires_units: bool = False
    required_keywords: List[str] = field(default_factory=list)
    required_entities: List[str] = field(default_factory=list)
    required_phrases: List[str] = field(default_factory=list)
    error_tokens: List[str] = field(default_factory=list)
    comparison_entities: List[str] = field(default_factory=list)

    @staticmethod
    def from_required_evidence(required: Dict[str, object]) -> "EvidenceRequirements":
        return EvidenceRequirements(
            requires_number=bool(required.get("contains_number")),
            requires_date_range=bool(required.get("contains_date_range")),
            requires_units=bool(required.get("contains_units")),
            required_keywords=list(required.get("must_include_keywords") or []),
            required_entities=list(required.get("contains_named_entities") or []),
            required_phrases=list(required.get("must_include_phrases") or []),
            error_tokens=list(required.get("error_tokens") or []),
            comparison_entities=list(required.get("comparison_entities") or []),
        )


@dataclass
class EvidenceCoverage:
    satisfied: bool
    coverage_ratio: float
    missing: List[str]
    breakdown: Dict[str, float]


class EvidenceConstraints:
    """Evaluate evidence constraints against retrieved chunks."""

    def score_chunk(self, text: str, requirements: EvidenceRequirements) -> float:
        if not text:
            return 0.0
        logger.debug("score_chunk: text_len=%d", len(text))
        score = 0.0
        text_lower = text.lower()
        if requirements.requires_number:
            score += 0.35 if _NUMBER_RE.search(text_lower) else -0.2
        if requirements.requires_date_range:
            score += 0.25 if _DATE_RE.search(text_lower) else -0.15
        if requirements.requires_units:
            score += 0.2 if _UNIT_RE.search(text_lower) else -0.1
        if requirements.required_keywords:
            keyword_hits = self._keyword_ratio(text_lower, requirements.required_keywords)
            score += 0.2 * keyword_hits
        if requirements.required_entities:
            entity_hits = self._keyword_ratio(text_lower, requirements.required_entities)
            score += 0.2 * entity_hits
        if requirements.required_phrases:
            phrase_hits = self._phrase_ratio(text_lower, requirements.required_phrases)
            score += 0.15 * phrase_hits
        if requirements.error_tokens:
            error_hits = self._keyword_ratio(text, requirements.error_tokens)
            score += 0.2 * error_hits
        if requirements.comparison_entities:
            comp_hits = self._keyword_ratio(text_lower, requirements.comparison_entities)
            score += 0.2 * comp_hits
        return score

    def evaluate(self, chunks: Iterable[object], requirements: EvidenceRequirements, min_ratio: float = 0.6) -> EvidenceCoverage:
        chunks_list = list(chunks)
        logger.debug("evaluate: chunks=%d, min_ratio=%.2f", len(chunks_list), min_ratio)
        if not chunks_list:
            return EvidenceCoverage(False, 0.0, ["no_chunks"], {})

        has_number = any(self._has_number(self._text(c)) for c in chunks_list) if requirements.requires_number else True
        has_date = any(self._has_date(self._text(c)) for c in chunks_list) if requirements.requires_date_range else True
        has_units = any(self._has_units(self._text(c)) for c in chunks_list) if requirements.requires_units else True
        keyword_ratio = self._coverage_ratio(chunks_list, requirements.required_keywords)
        entity_ratio = self._coverage_ratio(chunks_list, requirements.required_entities)
        phrase_ratio = self._phrase_coverage_ratio(chunks_list, requirements.required_phrases)
        error_ratio = self._coverage_ratio(chunks_list, requirements.error_tokens)
        comparison_ratio = self._coverage_ratio(chunks_list, requirements.comparison_entities)

        coverage_items: List[Tuple[str, float]] = []
        if requirements.requires_number:
            coverage_items.append(("number", 1.0 if has_number else 0.0))
        if requirements.requires_date_range:
            coverage_items.append(("date", 1.0 if has_date else 0.0))
        if requirements.requires_units:
            coverage_items.append(("units", 1.0 if has_units else 0.0))
        if requirements.required_keywords:
            coverage_items.append(("keywords", keyword_ratio))
        if requirements.required_entities:
            coverage_items.append(("entities", entity_ratio))
        if requirements.required_phrases:
            coverage_items.append(("phrases", phrase_ratio))
        if requirements.error_tokens:
            coverage_items.append(("errors", error_ratio))
        if requirements.comparison_entities:
            coverage_items.append(("comparison", comparison_ratio))

        if not coverage_items:
            return EvidenceCoverage(True, 1.0, [], {})

        total = sum(value for _, value in coverage_items)
        ratio = total / max(len(coverage_items), 1)
        missing = [name for name, value in coverage_items if value < 0.6]
        breakdown = {name: round(value, 3) for name, value in coverage_items}
        satisfied = ratio >= min_ratio
        logger.debug("evaluate: satisfied=%s, ratio=%.3f, missing=%s", satisfied, ratio, missing)
        return EvidenceCoverage(satisfied, ratio, missing, breakdown)

    @staticmethod
    def _text(chunk: object) -> str:
        return (getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else "") or "")

    @staticmethod
    def _has_number(text: str) -> bool:
        return bool(_NUMBER_RE.search(text))

    @staticmethod
    def _has_date(text: str) -> bool:
        return bool(_DATE_RE.search(text))

    @staticmethod
    def _has_units(text: str) -> bool:
        return bool(_UNIT_RE.search(text))

    @staticmethod
    def _keyword_ratio(text: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        text_lower = text.lower()
        hits = 0
        for term in keywords:
            term_lower = term.lower()
            if term_lower and term_lower in text_lower:
                hits += 1
        return hits / max(len(keywords), 1)

    def _coverage_ratio(self, chunks: List[object], keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        text_combined = " ".join(self._text(c) for c in chunks).lower()
        return self._keyword_ratio(text_combined, keywords)

    @staticmethod
    def _phrase_ratio(text: str, phrases: List[str]) -> float:
        if not phrases:
            return 1.0
        text_lower = text.lower()
        hits = 0
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower and phrase_lower in text_lower:
                hits += 1
        return hits / max(len(phrases), 1)

    def _phrase_coverage_ratio(self, chunks: List[object], phrases: List[str]) -> float:
        if not phrases:
            return 1.0
        text_combined = " ".join(self._text(c) for c in chunks).lower()
        return self._phrase_ratio(text_combined, phrases)
