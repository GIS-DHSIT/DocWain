from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from src.rag.doc_inventory import DocInventoryItem
from src.rag.entity_detector import EntityDetectionResult


@dataclass(frozen=True)
class ScopeResolution:
    scope: str
    target_docs: List[DocInventoryItem]
    matched_docs: List[DocInventoryItem]
    reason: str


def _dedupe_docs(items: Iterable[DocInventoryItem]) -> List[DocInventoryItem]:
    seen = set()
    output = []
    for item in items:
        key = item.doc_id or item.source_file or item.document_name
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _match_docs_by_name(doc_names: Sequence[str], doc_inventory: Sequence[DocInventoryItem]) -> List[DocInventoryItem]:
    if not doc_names or not doc_inventory:
        return []
    lowered = {name.lower() for name in doc_names if name}
    matches: List[DocInventoryItem] = []
    for item in doc_inventory:
        candidates = [item.source_file, item.document_name]
        if any(candidate and candidate.lower() in lowered for candidate in candidates):
            matches.append(item)
    return _dedupe_docs(matches)


def _match_docs_by_entity(
    entities: Iterable[str],
    doc_inventory: Sequence[DocInventoryItem],
    doc_matcher: Optional[Callable[[str, Sequence[DocInventoryItem]], List[DocInventoryItem]]],
) -> List[DocInventoryItem]:
    matches: List[DocInventoryItem] = []
    for entity in entities:
        if not entity:
            continue
        if doc_matcher:
            matches.extend(doc_matcher(entity, doc_inventory))
        else:
            lowered = entity.lower()
            for item in doc_inventory:
                if lowered in (item.source_file or "").lower() or lowered in (item.document_name or "").lower():
                    matches.append(item)
    return _dedupe_docs(matches)


def resolve_scope(
    *,
    query_text: str,
    intent_type: str,
    wants_all_docs: bool,
    entities: EntityDetectionResult,
    doc_inventory: Sequence[DocInventoryItem],
    doc_matcher: Optional[Callable[[str, Sequence[DocInventoryItem]], List[DocInventoryItem]]] = None,
) -> ScopeResolution:
    _ = query_text
    doc_inventory = list(doc_inventory or [])

    explicit_doc_matches = _match_docs_by_name(entities.documents, doc_inventory)
    if explicit_doc_matches:
        scope = "single_doc" if len(explicit_doc_matches) == 1 else "targeted_docs"
        return ScopeResolution(
            scope=scope,
            target_docs=explicit_doc_matches,
            matched_docs=explicit_doc_matches,
            reason="explicit_document_match",
        )

    entity_matches: List[DocInventoryItem] = []
    if entities.people or entities.products:
        entity_matches = _match_docs_by_entity(
            list(entities.people) + list(entities.products),
            doc_inventory,
            doc_matcher,
        )
        if entity_matches:
            scope = "single_doc" if len(entity_matches) == 1 else "targeted_docs"
            return ScopeResolution(
                scope=scope,
                target_docs=entity_matches,
                matched_docs=entity_matches,
                reason="entity_doc_match",
            )

    if wants_all_docs or intent_type in {"compare", "rank"}:
        return ScopeResolution(
            scope="multi_doc",
            target_docs=list(doc_inventory),
            matched_docs=list(doc_inventory),
            reason="explicit_multi_doc_intent",
        )

    if len(doc_inventory) == 1:
        return ScopeResolution(
            scope="single_doc",
            target_docs=list(doc_inventory),
            matched_docs=list(doc_inventory),
            reason="single_document_available",
        )

    if intent_type in {"summarize", "compare", "rank", "filter", "extract_fields", "tabular_request"}:
        return ScopeResolution(
            scope="multi_doc",
            target_docs=list(doc_inventory),
            matched_docs=list(doc_inventory),
            reason="multi_doc_default_for_intent",
        )

    # Default for lookup/calculate: prefer multi-doc when multiple docs exist to avoid silent bias.
    return ScopeResolution(
        scope="multi_doc",
        target_docs=list(doc_inventory),
        matched_docs=list(doc_inventory),
        reason="multi_doc_default_for_ambiguity",
    )


__all__ = ["ScopeResolution", "resolve_scope"]
