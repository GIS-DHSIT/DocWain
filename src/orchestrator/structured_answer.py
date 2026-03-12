from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from src.orchestrator.response_validator import validate_response_payload
from src.retrieval.profile_evidence import DocumentEvidence, ProfileEvidenceGraph

logger = get_logger(__name__)

def select_output_schema(intent: str) -> Dict[str, Any]:
    if intent == "extract":
        return {"schema": "extract", "required_fields": ["schema", "documents"]}
    if intent == "list":
        return {"schema": "list", "required_fields": ["schema", "items", "documents"]}
    if intent == "compare":
        return {"schema": "compare", "required_fields": ["schema", "comparisons", "documents"]}
    if intent == "rank":
        return {"schema": "rank", "required_fields": ["schema", "ranking", "documents"]}
    return {"schema": "answer", "required_fields": ["schema", "answer", "documents"]}

def generate_structured_answer(
    *,
    user_query: str,
    intent: str,
    retrieval_scope: str,
    target_document_ids: List[str],
    evidence_graph: ProfileEvidenceGraph,
    model_name: Optional[str],
    llm_client=None,
) -> str:
    schema = select_output_schema(intent)
    base_payload = build_payload(
        user_query=user_query,
        schema_name=schema["schema"],
        retrieval_scope=retrieval_scope,
        target_document_ids=target_document_ids,
        evidence_graph=evidence_graph,
    )

    if not model_name and llm_client is None:
        return json.dumps(base_payload, indent=2)

    prompt = _build_prompt(user_query, schema["schema"], base_payload)
    try:
        if llm_client is not None:
            text = llm_client.generate(prompt)
        else:
            from src.llm.gateway import get_llm_gateway
            text = get_llm_gateway().generate(prompt)
        text = (text or "").strip()
        payload = _extract_json(text)
        if payload and validate_response_payload(payload, schema, evidence_graph):
            return json.dumps(payload, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Structured answer generation failed: %s", exc)
    return json.dumps(base_payload, indent=2)

def build_payload(
    *,
    user_query: str,
    schema_name: str,
    retrieval_scope: str,
    target_document_ids: List[str],
    evidence_graph: ProfileEvidenceGraph,
) -> Dict[str, Any]:
    documents = [_document_payload(doc) for doc in evidence_graph.documents.values()]
    payload: Dict[str, Any] = {
        "schema": schema_name,
        "query": user_query,
        "retrieval_scope": retrieval_scope,
        "target_document_ids": target_document_ids,
        "documents": documents,
    }
    if _should_merge(user_query):
        payload["merged_entities"] = _merge_documents(evidence_graph)

    if schema_name == "extract":
        return payload
    if schema_name == "list":
        payload["items"] = _list_items_from_evidence(evidence_graph)
        return payload
    if schema_name == "compare":
        payload["comparisons"] = _compare_documents(evidence_graph)
        return payload
    if schema_name == "rank":
        payload["ranking"] = _rank_documents(evidence_graph)
        return payload
    payload["answer"] = _summarize_documents(evidence_graph)
    return payload

def _document_payload(doc: DocumentEvidence) -> Dict[str, Any]:
    contacts = doc.contacts
    return {
        "document_id": doc.document_id,
        "source_name": doc.source_name,
        "contacts": {
            "phones": _items_or_not_mentioned(contacts.get("phones", [])),
            "emails": _items_or_not_mentioned(contacts.get("emails", [])),
            "urls": _items_or_not_mentioned(contacts.get("urls", [])),
        },
        "dates": _items_or_not_mentioned(doc.dates),
        "identifiers": _items_or_not_mentioned(doc.identifiers),
        "entities": _items_or_not_mentioned(doc.entities),
        "sections": _items_or_not_mentioned(doc.sections),
        "tables": _items_or_not_mentioned(doc.tables),
    }

def _items_or_not_mentioned(items: List[Any]) -> Any:
    if not items:
        return "Not Mentioned"
    return [_item_payload(item) for item in items]

def _item_payload(item: Any) -> Dict[str, Any]:
    return {
        "value": item.value,
        "snippet": item.snippet,
        "document_id": item.document_id,
        "source_name": item.source_name,
        "chunk_id": item.chunk_id,
        "section_title": item.section_title,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "meta": item.meta,
    }

def _summarize_documents(graph: ProfileEvidenceGraph) -> str:
    parts: List[str] = []
    for doc in graph.documents.values():
        contact_count = sum(len(doc.contacts.get(key, [])) for key in ("phones", "emails", "urls"))
        identifier_count = len(doc.identifiers)
        date_count = len(doc.dates)
        parts.append(
            f"{doc.source_name or doc.document_id}: {contact_count} contacts, "
            f"{identifier_count} identifiers, {date_count} dates."
        )
    return " | ".join(parts) if parts else "Not Mentioned"

def _list_items_from_evidence(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for doc in graph.documents.values():
        for ident in doc.identifiers:
            items.append({"item": ident.value, "document_id": doc.document_id, "source_name": doc.source_name})
    return items

def _compare_documents(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    doc_list = list(graph.documents.values())
    for idx, doc in enumerate(doc_list):
        for other in doc_list[idx + 1 :]:
            comparisons.append(
                {
                    "document_a": doc.document_id,
                    "document_b": other.document_id,
                    "similarities": _shared_values(doc, other),
                    "differences": _diff_values(doc, other),
                    "notes": "Not Mentioned" if not (doc.identifiers or other.identifiers) else "See identifiers.",
                }
            )
    return comparisons

def _rank_documents(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    ranked = sorted(
        graph.documents.values(),
        key=lambda doc: len(doc.identifiers) + len(doc.dates) + sum(len(doc.contacts.get(k, [])) for k in ("phones", "emails", "urls")),
        reverse=True,
    )
    ranking: List[Dict[str, Any]] = []
    for idx, doc in enumerate(ranked, start=1):
        ranking.append(
            {
                "rank": idx,
                "document_id": doc.document_id,
                "source_name": doc.source_name,
                "reason": "Based on evidence density in extracted fields.",
            }
        )
    return ranking

def _shared_values(doc: DocumentEvidence, other: DocumentEvidence) -> List[str]:
    left = {item.value.lower() for item in doc.identifiers}
    right = {item.value.lower() for item in other.identifiers}
    return sorted(left.intersection(right))

def _diff_values(doc: DocumentEvidence, other: DocumentEvidence) -> List[str]:
    left = {item.value.lower() for item in doc.identifiers}
    right = {item.value.lower() for item in other.identifiers}
    return sorted(left.symmetric_difference(right))

def _build_prompt(user_query: str, schema_name: str, base_payload: Dict[str, Any]) -> str:
    schema_json = json.dumps(base_payload, indent=2)
    return (
        "You are an evidence-first synthesis engine. "
        "Use only the provided structured evidence. "
        "Do not add new facts. "
        "Return ONLY valid JSON following the given schema.\n\n"
        f"Schema name: {schema_name}\n"
        f"User query: {user_query}\n\n"
        "Structured evidence (fill in narrative fields as needed):\n"
        f"{schema_json}\n"
    )

def _should_merge(user_query: str) -> bool:
    lowered = (user_query or "").lower()
    return any(keyword in lowered for keyword in ("merge", "combine", "together"))

def _merge_documents(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for doc in graph.documents.values():
        keys = _join_keys(doc)
        if not keys:
            continue
        entities.append(
            {
                "document_id": doc.document_id,
                "source_name": doc.source_name,
                "join_keys": sorted(keys),
                "contacts": _document_payload(doc)["contacts"],
            }
        )
    merged: List[Dict[str, Any]] = []
    used = set()
    for idx, entity in enumerate(entities):
        if idx in used:
            continue
        group = [entity]
        keys = set(entity["join_keys"])
        for jdx, other in enumerate(entities[idx + 1 :], start=idx + 1):
            if keys.intersection(other["join_keys"]):
                group.append(other)
                used.add(jdx)
        merged.append({"group": group, "merge_key": sorted(keys)})
    return merged

def _join_keys(doc: DocumentEvidence) -> set[str]:
    keys = set()
    for item in doc.contacts.get("emails", []):
        keys.add(item.value.lower())
    for item in doc.contacts.get("phones", []):
        keys.add(re.sub(r"[^\d+]", "", item.value))
    for item in doc.contacts.get("urls", []):
        url = item.value.lower()
        if "linkedin.com" in url:
            keys.add(url)
    return keys

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

__all__ = ["select_output_schema", "generate_structured_answer", "build_payload"]
