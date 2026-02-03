import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ScopeType(str, Enum):
    SINGLE_DOC = "SINGLE_DOC"
    MULTI_DOC = "MULTI_DOC"
    SUBSET = "SUBSET"


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    filename: str
    title: str
    doc_domain: str = "general"
    doc_kind: str = "general_doc"


@dataclass(frozen=True)
class Scope:
    scope_type: ScopeType
    matched_docs: List[DocMeta]


_MULTI_DOC_HINTS = [
    "top 5", "top five", "compare", "rank", "across", "between",
    "from given documents", "all documents", "multiple documents", "list",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_docs_by_name(prompt: str, known_docs: List[DocMeta]) -> List[DocMeta]:
    matches: List[DocMeta] = []
    for doc in known_docs:
        for token in (doc.filename, doc.title):
            name = _normalize(token)
            if name and name in prompt:
                matches.append(doc)
                break
    return matches


def _match_invoice_number(prompt: str, known_docs: List[DocMeta]) -> List[DocMeta]:
    match = re.search(r"invoice\s*(?:#|no\.?|number)\s*([\w-]+)", prompt)
    if not match:
        return []
    invoice_no = match.group(1).lower()
    return [doc for doc in known_docs if invoice_no in _normalize(doc.filename)]


def _extract_named_entity(prompt: str) -> Optional[str]:
    match = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", prompt)
    if match:
        return match.group(1).lower()
    return None


def resolve_scope(prompt: str, known_docs: List[DocMeta]) -> Scope:
    normalized = _normalize(prompt)
    if not normalized:
        return Scope(scope_type=ScopeType.MULTI_DOC, matched_docs=[])

    for hint in _MULTI_DOC_HINTS:
        if hint in normalized:
            return Scope(scope_type=ScopeType.MULTI_DOC, matched_docs=[])

    matches = _match_docs_by_name(normalized, known_docs)
    if not matches:
        matches = _match_invoice_number(normalized, known_docs)
    if matches:
        return Scope(scope_type=ScopeType.SINGLE_DOC, matched_docs=matches)

    entity = _extract_named_entity(prompt)
    if entity:
        subset = [doc for doc in known_docs if entity in _normalize(doc.filename) or entity in _normalize(doc.title)]
        if subset:
            return Scope(scope_type=ScopeType.SUBSET, matched_docs=subset)

    return Scope(scope_type=ScopeType.MULTI_DOC, matched_docs=[])


__all__ = ["DocMeta", "Scope", "ScopeType", "resolve_scope"]
