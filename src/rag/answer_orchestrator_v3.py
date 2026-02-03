from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.rag.doc_inventory import DocInventoryItem
from src.rag.entity_detector import EntityDetectionResult, detect_entities
from src.rag.intent_router_v3 import IntentClassification, classify as classify_intent
from src.rag.qdrant_profile_digest import build_profile_digest
from src.rag.scope_resolver_v3 import ScopeDecision, resolve as resolve_scope, select_dominant_doc


_CONTACT_RE = re.compile(r"\b(email|e-mail|phone|tel|mobile|fax|address|street|zip|postal)\b", re.IGNORECASE)


@dataclass(frozen=True)
class OrchestratorState:
    digest: Dict[str, Any]
    intent: IntentClassification
    entities: EntityDetectionResult
    scope: ScopeDecision


def build_state(
    *,
    qdrant_client: Any,
    collection_name: str,
    profile_id: str,
    subscription_id: Optional[str],
    redis_client: Optional[Any],
    doc_inventory: Sequence[DocInventoryItem],
    query_text: str,
) -> OrchestratorState:
    digest = build_profile_digest(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        profile_id=profile_id,
        subscription_id=subscription_id,
        redis_client=redis_client,
        doc_inventory=doc_inventory,
    )
    intent = classify_intent(query_text, doc_inventory)
    entities = detect_entities(query_text, doc_inventory)
    scope = resolve_scope(intent=intent, entities=entities, doc_inventory=doc_inventory)
    return OrchestratorState(
        digest=digest,
        intent=intent,
        entities=entities,
        scope=scope,
    )


def _doc_label(doc: Dict[str, Any]) -> str:
    name = doc.get("source_file") or doc.get("document_id") or "Document"
    signal = doc.get("doc_signal")
    if signal and signal != "unknown":
        return f"{name} ({signal})"
    return name


def _collect_top_entities(digest: Dict[str, Any], limit: int = 6) -> List[str]:
    entities: List[str] = []
    for doc in digest.get("documents") or []:
        for ent in doc.get("top_entities") or []:
            if ent and ent not in entities:
                entities.append(ent)
            if len(entities) >= limit:
                return entities
    return entities


def build_clarification_response(query_text: str, digest: Dict[str, Any], intent: IntentClassification) -> str:
    doc_count = digest.get("doc_count") or 0
    documents = digest.get("documents") or []
    doc_labels = [_doc_label(doc) for doc in documents[:3]]
    topics = digest.get("dominant_topics") or []
    top_entities = _collect_top_entities(digest, limit=5)

    sentences: List[str] = []
    sentences.append("I can help, but the request is a bit broad for this profile.")
    if doc_labels:
        sentences.append(f"This profile has {doc_count} document(s), including {', '.join(doc_labels)}.")
    else:
        sentences.append(f"This profile has {doc_count} document(s) indexed in Qdrant.")
    if topics:
        sentences.append(f"Common topics include {', '.join(topics[:4])}.")
    if top_entities:
        sentences.append(f"Notable entities include {', '.join(top_entities[:4])}.")

    if intent.intent_type in {"compare", "rank", "filter"}:
        sentences.append("You asked for a multi-document analysis, so I will use multiple files unless you narrow the fields.")
        sentences.append("Which fields should I compare or rank (for example totals, dates, or key sections)?")
    else:
        options = []
        if doc_labels:
            options.append(f"Focus on {doc_labels[0]}")
        if len(doc_labels) > 1:
            options.append(f"Focus on {doc_labels[1]}")
        options.append("Give an overview across all documents")
        sentences.append("Choose one of these options so I can proceed.")
        sentences.append("Options: " + "; ".join(options) + ".")

    response = " ".join(sentences)
    return response.strip()


def enforce_single_doc_filter(
    chunks: Sequence[Any],
    target_doc: Optional[DocInventoryItem],
) -> List[Any]:
    if not target_doc or not chunks:
        return list(chunks or [])
    target_doc_id = str(target_doc.doc_id or "")
    target_source = (target_doc.source_file or target_doc.document_name or "").lower()
    filtered: List[Any] = []
    for chunk in chunks:
        meta = getattr(chunk, "metadata", {}) or {}
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or "")
        source_file = str(meta.get("source_file") or meta.get("source") or meta.get("file_name") or "").lower()
        if target_doc_id and doc_id == target_doc_id:
            filtered.append(chunk)
            continue
        if target_source and target_source in source_file:
            filtered.append(chunk)
    return filtered


def remove_junk_sections(chunks: Sequence[Any], query_text: str) -> List[Any]:
    if not chunks:
        return []
    lowered = (query_text or "").lower()
    if any(token in lowered for token in ["email", "phone", "address", "contact"]):
        return list(chunks)
    cleaned: List[Any] = []
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        if _CONTACT_RE.search(text) and len(text.split()) < 40:
            continue
        cleaned.append(chunk)
    return cleaned


def dominant_doc_for_lookup(
    intent: IntentClassification,
    chunks: Sequence[Any],
    threshold: float = 0.6,
) -> Optional[str]:
    if intent.intent_type != "lookup_fact":
        return None
    return select_dominant_doc(chunks, threshold=threshold)


__all__ = [
    "OrchestratorState",
    "build_state",
    "build_clarification_response",
    "enforce_single_doc_filter",
    "remove_junk_sections",
    "dominant_doc_for_lookup",
]
