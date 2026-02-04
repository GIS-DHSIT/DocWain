from __future__ import annotations

import re
from typing import Any, Dict

from src.retrieval.profile_evidence import ProfileEvidenceGraph


SELF_INTRO_RE = re.compile(r"\b(i am|i'm|docwain|i help)\b", re.IGNORECASE)


def validate_response_payload(payload: Dict[str, Any], schema: Dict[str, Any], evidence_graph: ProfileEvidenceGraph) -> bool:
    if not _has_required_fields(payload, schema):
        return False
    if _contains_self_intro(payload):
        return False
    if not _not_mentioned_consistent(payload, evidence_graph):
        return False
    if not _per_document_consistent(payload, evidence_graph):
        return False
    return True


def _has_required_fields(payload: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    for field in schema.get("required_fields") or []:
        if field not in payload:
            return False
    return True


def _contains_self_intro(payload: Dict[str, Any]) -> bool:
    for value in _walk_strings(payload):
        if SELF_INTRO_RE.search(value):
            return True
    return False


def _not_mentioned_consistent(payload: Dict[str, Any], evidence_graph: ProfileEvidenceGraph) -> bool:
    documents = {doc.document_id: doc for doc in evidence_graph.documents.values()}
    for doc in payload.get("documents", []):
        doc_id = doc.get("document_id")
        evidence = documents.get(doc_id)
        if not evidence:
            continue
        contacts = doc.get("contacts", {})
        for key in ("phones", "emails", "urls"):
            if contacts.get(key) == "Not Mentioned" and evidence.contacts.get(key):
                return False
        if doc.get("dates") == "Not Mentioned" and evidence.dates:
            return False
        if doc.get("identifiers") == "Not Mentioned" and evidence.identifiers:
            return False
        if doc.get("entities") == "Not Mentioned" and evidence.entities:
            return False
    return True


def _per_document_consistent(payload: Dict[str, Any], evidence_graph: ProfileEvidenceGraph) -> bool:
    doc_ids = set(evidence_graph.documents.keys())
    for doc in payload.get("documents", []):
        doc_id = doc.get("document_id")
        if doc_id not in doc_ids:
            return False
        for field in ("dates", "identifiers", "entities", "sections", "tables"):
            items = doc.get(field)
            if isinstance(items, list):
                for item in items:
                    if item.get("document_id") != doc_id:
                        return False
        contacts = doc.get("contacts", {})
        if isinstance(contacts, dict):
            for key in ("phones", "emails", "urls"):
                entries = contacts.get(key)
                if isinstance(entries, list):
                    for item in entries:
                        if item.get("document_id") != doc_id:
                            return False
    return True


def _walk_strings(payload: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(payload, str):
        strings.append(payload)
    elif isinstance(payload, dict):
        for value in payload.values():
            strings.extend(_walk_strings(value))
    elif isinstance(payload, list):
        for value in payload:
            strings.extend(_walk_strings(value))
    return strings


__all__ = ["validate_response_payload"]
