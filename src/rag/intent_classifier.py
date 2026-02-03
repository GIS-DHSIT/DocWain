from __future__ import annotations

from typing import Iterable


INTENT_COMPARE = "COMPARE"
INTENT_SUMMARIZE = "SUMMARIZE"
INTENT_PRODUCTS_SERVICES = "PRODUCTS_SERVICES"
INTENT_TOTALS = "TOTALS"
INTENT_LOOKUP = "LOOKUP"

_COMPARE_CUES = {"compare", "difference", "vs", "versus", "across", "between", "all"}
_SUMMARIZE_CUES = {"summarize", "summary", "overview"}
_PRODUCTS_SERVICES_CUES = {
    "products",
    "services",
    "covered",
    "included",
    "line items",
    "items",
}
_TOTALS_CUES = {"total", "subtotal", "amount due", "balance"}

_MULTI_DOC_CUES = {
    "documents",
    "invoices",
    "files",
    "across",
    "compare",
}
_MULTI_DOC_PHRASES = {
    "these invoices",
    "these documents",
    "all documents",
    "all invoices",
    "across documents",
    "across invoices",
}


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    return any(cue in text for cue in cues)


def classify_intent(query: str) -> str:
    text = (query or "").lower()
    if _contains_any(text, _COMPARE_CUES):
        return INTENT_COMPARE
    if _contains_any(text, _SUMMARIZE_CUES):
        return INTENT_SUMMARIZE
    if _contains_any(text, _PRODUCTS_SERVICES_CUES):
        return INTENT_PRODUCTS_SERVICES
    if _contains_any(text, _TOTALS_CUES):
        return INTENT_TOTALS
    return INTENT_LOOKUP


def has_multi_doc_cues(query: str) -> bool:
    text = (query or "").lower()
    if _contains_any(text, _MULTI_DOC_CUES):
        return True
    return _contains_any(text, _MULTI_DOC_PHRASES)


__all__ = [
    "INTENT_COMPARE",
    "INTENT_SUMMARIZE",
    "INTENT_PRODUCTS_SERVICES",
    "INTENT_TOTALS",
    "INTENT_LOOKUP",
    "classify_intent",
    "has_multi_doc_cues",
]
