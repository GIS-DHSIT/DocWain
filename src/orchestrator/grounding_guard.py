from __future__ import annotations

import re
from typing import Iterable


_NUMERIC_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
_LABELLED_ENTITY_RE = re.compile(r"(?i)\b(brand|model|manufacturer|vendor|supplier)\b\s*[:\-]\s*([^\n\r\.]+)")


def _extract_numbers(text: str) -> Iterable[str]:
    return set(_NUMERIC_RE.findall(text or ""))


def _sanitize_numbers(answer: str, evidence: str) -> str:
    evidence_numbers = _extract_numbers(evidence)
    if not evidence_numbers:
        return _NUMERIC_RE.sub("[not explicitly stated]", answer)

    def replace(match: re.Match) -> str:
        token = match.group(0)
        return token if token in evidence_numbers else "[not explicitly stated]"

    return _NUMERIC_RE.sub(replace, answer)


def _sanitize_labelled_entities(answer: str, evidence: str) -> str:
    evidence_lower = (evidence or "").lower()

    def replace(match: re.Match) -> str:
        label = match.group(1)
        value = match.group(2).strip()
        if value and value.lower() in evidence_lower:
            return match.group(0)
        return ""

    return _LABELLED_ENTITY_RE.sub(replace, answer)


def apply_grounding_guard(answer: str, evidence: str) -> str:
    guarded = _sanitize_numbers(answer, evidence)
    guarded = _sanitize_labelled_entities(guarded, evidence)
    return guarded


__all__ = ["apply_grounding_guard"]
