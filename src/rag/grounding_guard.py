from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence


_CURRENCY_RE = re.compile(r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_INLINE_CITATION_RE = re.compile(r"\[(?:SOURCE-\d+(?:,\s*)?)+\]", re.IGNORECASE)


@dataclass
class GroundingResult:
    passed: bool
    missing_values: List[str]
    missing_items: List[str]


def _chunk_texts(chunks: Sequence[Any]) -> List[str]:
    texts: List[str] = []
    for chunk in chunks or []:
        if isinstance(chunk, dict):
            text = chunk.get("text") or ""
        else:
            text = getattr(chunk, "text", "") or ""
        if text:
            texts.append(text)
    return texts


def _strip_citations(text: str) -> str:
    if not text:
        return ""
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith("citations:"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    return _INLINE_CITATION_RE.sub("", cleaned)


def _strip_doc_names(text: str, doc_names: Iterable[str]) -> str:
    cleaned = text
    for name in doc_names or []:
        if not name:
            continue
        cleaned = cleaned.replace(name, "")
    return cleaned


def _extract_numeric_values(text: str) -> List[str]:
    currency_values = _CURRENCY_RE.findall(text)
    text_wo_currency = _CURRENCY_RE.sub(" ", text)
    number_values = _NUMBER_RE.findall(text_wo_currency)
    values = currency_values + number_values
    return [v.strip() for v in values if v.strip()]


def _normalize_number(value: str) -> str:
    return re.sub(r"[^\d.]", "", value or "")


def _value_in_chunks(value: str, chunk_texts: Sequence[str]) -> bool:
    if not value:
        return False
    compact_value = " ".join(value.split())
    for text in chunk_texts:
        if value in text or compact_value in text:
            return True
        normalized_text = " ".join(text.split())
        if compact_value and compact_value in normalized_text:
            return True
    return False


def verify_grounding(
    answer: str,
    selected_chunks: Sequence[Any],
    *,
    intent: str,
    extracted_items: Optional[Sequence[str]] = None,
    computed_values: Optional[Sequence[str]] = None,
    doc_names: Optional[Sequence[str]] = None,
) -> GroundingResult:
    cleaned = _strip_citations(answer or "")
    cleaned = _strip_doc_names(cleaned, doc_names or [])
    numeric_values = _extract_numeric_values(cleaned)
    chunk_texts = _chunk_texts(selected_chunks)

    missing_values: List[str] = []
    computed_norm = {_normalize_number(v) for v in (computed_values or []) if v}
    allow_compute = intent == "TOTALS"

    for value in numeric_values:
        if _value_in_chunks(value, chunk_texts):
            continue
        if allow_compute and _normalize_number(value) in computed_norm:
            continue
        missing_values.append(value)

    missing_items: List[str] = []
    if intent == "PRODUCTS_SERVICES":
        items = [i for i in (extracted_items or []) if i]
        if items:
            answer_lower = cleaned.lower()
            if not any(item.lower() in answer_lower for item in items):
                missing_items = items[:3]

    return GroundingResult(passed=not missing_values and not missing_items, missing_values=missing_values, missing_items=missing_items)


__all__ = ["GroundingResult", "verify_grounding"]
