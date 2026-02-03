from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from src.rag.doc_inventory import DocInventoryItem
from src.rag.entity_detector import EntityDetectionResult
from src.rag.intent_router_v3 import IntentClassification


@dataclass(frozen=True)
class ScopeDecision:
    scope_type: str
    target_docs: List[DocInventoryItem]
    reason: str
    dominant_doc_id: Optional[str] = None


def _dedupe_docs(items: Iterable[DocInventoryItem]) -> List[DocInventoryItem]:
    seen = set()
    output: List[DocInventoryItem] = []
    for item in items:
        key = item.doc_id or item.source_file or item.document_name
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _match_docs_by_entities(
    entities: Iterable[str],
    doc_inventory: Sequence[DocInventoryItem],
) -> List[DocInventoryItem]:
    matches: List[DocInventoryItem] = []
    for entity in entities:
        lowered = (entity or "").lower()
        if not lowered:
            continue
        for doc in doc_inventory:
            haystack = " ".join(
                [doc.source_file or "", doc.document_name or "", doc.doc_type or ""]
            ).lower()
            if lowered in haystack:
                matches.append(doc)
    return _dedupe_docs(matches)


def resolve(
    *,
    intent: IntentClassification,
    entities: EntityDetectionResult,
    doc_inventory: Sequence[DocInventoryItem],
) -> ScopeDecision:
    doc_inventory = list(doc_inventory or [])

    if intent.mentions_single_doc and doc_inventory:
        matched = [
            doc for doc in doc_inventory
            if any(
                (name or "").lower() in (doc.source_file or "").lower()
                or (name or "").lower() in (doc.document_name or "").lower()
                for name in intent.doc_mentions or []
            )
        ]
        matched = _dedupe_docs(matched) or doc_inventory[:1]
        return ScopeDecision(
            scope_type="single_doc",
            target_docs=matched[:1],
            reason="explicit_single_doc_request",
        )

    entity_matches = _match_docs_by_entities(entities.people + entities.products, doc_inventory)
    if entity_matches:
        return ScopeDecision(
            scope_type="targeted_docs",
            target_docs=entity_matches,
            reason="entity_doc_match",
        )

    if intent.intent_type in {"compare", "rank", "filter"}:
        return ScopeDecision(
            scope_type="multi_doc",
            target_docs=list(doc_inventory),
            reason="multi_doc_intent",
        )

    if len(doc_inventory) == 1:
        return ScopeDecision(
            scope_type="single_doc",
            target_docs=list(doc_inventory),
            reason="single_document_available",
        )

    if intent.intent_type == "summarize":
        return ScopeDecision(
            scope_type="multi_doc",
            target_docs=list(doc_inventory),
            reason="summarize_defaults_multi",
        )

    return ScopeDecision(
        scope_type="multi_doc",
        target_docs=list(doc_inventory),
        reason="lookup_defaults_multi_until_dominant",
    )


def select_dominant_doc(chunks: Sequence[object], threshold: float = 0.6) -> Optional[str]:
    if not chunks:
        return None
    counts = {}
    total = 0
    for chunk in chunks:
        meta = getattr(chunk, "metadata", {}) or {}
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or "")
        if not doc_id:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        total += 1
    if not counts or total == 0:
        return None
    dominant_doc, dominant_count = max(counts.items(), key=lambda item: item[1])
    if dominant_count / total >= threshold:
        return dominant_doc
    return None


__all__ = ["ScopeDecision", "resolve", "select_dominant_doc"]
