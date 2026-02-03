from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from src.rag.doc_inventory import DocInventoryItem
from src.rag.entity_detector import detect_entities


_INTENT_TABLE_CUES = {"table", "tabular", "spreadsheet", "matrix", "grid"}
_INTENT_COMPARE_CUES = {"compare", "difference", "vs", "versus"}
_INTENT_RANK_CUES = {"rank", "ranking", "shortlist", "top", "best", "highest"}
_INTENT_SUMMARIZE_CUES = {"summarize", "summary", "overview", "recap", "brief"}
_INTENT_EXTRACT_CUES = {"extract", "list", "pull", "find", "show", "fields", "details"}
_INTENT_FILTER_CUES = {"filter", "only", "exclude", "without"}
_INTENT_CALCULATE_CUES = {"calculate", "compute", "sum", "total", "subtotal"}

_PRONOUN_ONLY_RE = re.compile(r"\b(this|that|it|these|those)\b", re.IGNORECASE)
_DOC_REF_PATTERNS = [
    re.compile(r"\bfrom\s+the\s+document\s+(.+)", re.IGNORECASE),
    re.compile(r"\bfrom\s+(.+)", re.IGNORECASE),
    re.compile(r"\bin\s+(.+)", re.IGNORECASE),
]


@dataclass(frozen=True)
class IntentClassification:
    intent_type: str
    ambiguity: str
    mentions_single_doc: bool
    mentions_entity: List[str]
    is_vague: bool
    doc_mentions: List[str]


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text or "")


def _extract_doc_mentions(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> List[str]:
    if not query_text or not doc_inventory:
        return []
    lowered = query_text.lower()
    matches: List[str] = []
    for doc in doc_inventory:
        for candidate in (doc.source_file, doc.document_name):
            if not candidate:
                continue
            cand_lower = candidate.lower()
            if cand_lower and cand_lower in lowered:
                matches.append(candidate)
    return list(dict.fromkeys(matches))


def _has_explicit_doc_pattern(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> bool:
    if not query_text:
        return False
    lowered = query_text.lower()
    for pattern in _DOC_REF_PATTERNS:
        match = pattern.search(lowered)
        if match and match.group(1):
            return True
    return bool(_extract_doc_mentions(query_text, doc_inventory))


def _is_vague_query(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> bool:
    tokens = _tokenize(query_text)
    if len(tokens) < 5:
        return True
    lowered = query_text.lower().strip()
    if _PRONOUN_ONLY_RE.fullmatch(lowered) and not _extract_doc_mentions(query_text, doc_inventory):
        return True
    if _PRONOUN_ONLY_RE.search(lowered) and not _extract_doc_mentions(query_text, doc_inventory):
        # Pronoun-based query without explicit entity/doc mention.
        entities = detect_entities(query_text, doc_inventory)
        if not (entities.people or entities.products or entities.documents):
            return True
    return False


def _ambiguity_level(is_vague: bool, mentions_single_doc: bool, entity_count: int) -> str:
    if is_vague:
        return "high"
    if mentions_single_doc:
        return "low"
    if entity_count >= 2:
        return "medium"
    return "medium"


def classify(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> IntentClassification:
    lowered = (query_text or "").lower()

    if _contains_any(lowered, _INTENT_COMPARE_CUES):
        intent_type = "compare"
    elif _contains_any(lowered, _INTENT_RANK_CUES):
        intent_type = "rank"
    elif _contains_any(lowered, _INTENT_SUMMARIZE_CUES):
        intent_type = "summarize"
    elif _contains_any(lowered, _INTENT_EXTRACT_CUES):
        intent_type = "extract_fields"
    elif _contains_any(lowered, _INTENT_FILTER_CUES):
        intent_type = "filter"
    elif _contains_any(lowered, _INTENT_CALCULATE_CUES):
        intent_type = "calculate"
    elif _contains_any(lowered, _INTENT_TABLE_CUES):
        intent_type = "request_table"
    else:
        intent_type = "lookup_fact"

    entities = detect_entities(query_text, doc_inventory)
    mentions = list(dict.fromkeys(entities.people + entities.products))
    doc_mentions = _extract_doc_mentions(query_text, doc_inventory)
    mentions_single_doc = bool(doc_mentions) or _has_explicit_doc_pattern(query_text, doc_inventory)
    if not mentions_single_doc and intent_type == "summarize" and entities.people:
        lowered_people = [p.lower() for p in entities.people]
        for doc in doc_inventory:
            haystack = " ".join([doc.source_file or "", doc.document_name or ""]).lower()
            if any(person in haystack for person in lowered_people):
                mentions_single_doc = True
                doc_mentions = list(dict.fromkeys(doc_mentions + [doc.source_file or doc.document_name]))
                break

    is_vague = _is_vague_query(query_text, doc_inventory)
    ambiguity = _ambiguity_level(is_vague, mentions_single_doc, len(mentions))

    return IntentClassification(
        intent_type=intent_type,
        ambiguity=ambiguity,
        mentions_single_doc=mentions_single_doc,
        mentions_entity=mentions,
        is_vague=is_vague,
        doc_mentions=doc_mentions,
    )


__all__ = ["IntentClassification", "classify"]
