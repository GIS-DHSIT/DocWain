from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from src.rag.doc_inventory import DocInventoryItem


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "about",
    "across",
    "between",
    "compare",
    "show",
    "list",
    "find",
    "table",
    "tabular",
    "indexed",
    "document",
    "documents",
    "report",
    "reports",
}

_TITLE_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
_ALL_CAPS_RE = re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b")
_PRODUCT_TOKEN_RE = re.compile(r"\b[A-Za-z]+[0-9][A-Za-z0-9\-]*\b")
_PRODUCT_HYPHEN_RE = re.compile(r"\b[A-Za-z0-9]+-[A-Za-z0-9\-]+\b")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,60})['\"]")


@dataclass(frozen=True)
class EntityDetectionResult:
    people: List[str]
    products: List[str]
    documents: List[str]
    raw_matches: List[str]


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s._\-]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        value = (item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _strip_extension(name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name or "").strip()


def _match_docs(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> List[str]:
    if not query_text or not doc_inventory:
        return []
    q_norm = _normalize(query_text)
    matches: List[str] = []
    for item in doc_inventory:
        candidates = [item.source_file, item.document_name, _strip_extension(item.source_file)]
        for candidate in candidates:
            cand_norm = _normalize(candidate)
            if not cand_norm:
                continue
            if cand_norm in q_norm:
                matches.append(candidate)
                break
    return _dedupe(matches)


def _is_valid_name(match: str) -> bool:
    tokens = match.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    lowered = [t.lower() for t in tokens]
    if any(token in _STOPWORDS for token in lowered):
        return False
    return True


def _is_product_candidate(value: str) -> bool:
    if not value or len(value) < 2:
        return False
    lowered = value.lower()
    if lowered in _STOPWORDS:
        return False
    if any(char.isdigit() for char in value):
        return True
    if any(char.isupper() for char in value) and " " in value:
        return True
    return False


def detect_entities(query_text: str, doc_inventory: Sequence[DocInventoryItem] | None = None) -> EntityDetectionResult:
    text = query_text or ""
    raw_matches: List[str] = []

    documents = _match_docs(text, doc_inventory or [])

    people: List[str] = []
    for match in _TITLE_CASE_RE.findall(text):
        raw_matches.append(match)
        if _is_valid_name(match):
            people.append(match)
    for match in _ALL_CAPS_RE.findall(text):
        raw_matches.append(match)
        if _is_valid_name(match):
            people.append(match)

    products: List[str] = []
    for token in _PRODUCT_TOKEN_RE.findall(text):
        raw_matches.append(token)
        if _is_product_candidate(token):
            products.append(token)
    for token in _PRODUCT_HYPHEN_RE.findall(text):
        raw_matches.append(token)
        if _is_product_candidate(token):
            products.append(token)
    for match in _QUOTED_RE.findall(text):
        raw_matches.append(match)
        if _is_product_candidate(match):
            products.append(match)

    return EntityDetectionResult(
        people=_dedupe(people),
        products=_dedupe(products),
        documents=_dedupe(documents),
        raw_matches=_dedupe(raw_matches),
    )


__all__ = ["EntityDetectionResult", "detect_entities"]
