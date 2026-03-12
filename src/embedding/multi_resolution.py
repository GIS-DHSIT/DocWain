"""Multi-resolution vector builder for the Document Intelligence Graph.

Creates document-level and section-level vectors that complement chunk-level
vectors.  Short fragments rescued during chunking are folded into their parent
section vectors so no information is lost.
"""
from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

def _stable_id(prefix: str, *parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

def build_doc_vector_text(
    *,
    doc_summary: str,
    key_entities: Optional[List[Any]] = None,
    intent_tags: Optional[List[str]] = None,
    doc_domain: str = "generic",
    doc_type: str = "other",
) -> str:
    """Build a single text string that represents the entire document."""
    parts: List[str] = []
    if doc_summary:
        parts.append(f"Document Summary: {doc_summary.strip()}")
    if doc_type and doc_type != "other":
        parts.append(f"Document Type: {doc_type}")
    if doc_domain and doc_domain != "generic":
        parts.append(f"Domain: {doc_domain}")
    if key_entities:
        entity_texts = []
        for e in key_entities[:30]:
            if isinstance(e, dict):
                entity_texts.append(f"{e.get('type', 'ENTITY')}: {e.get('text', '')}")
            elif isinstance(e, str):
                entity_texts.append(e)
        if entity_texts:
            parts.append(f"Key Entities: {', '.join(entity_texts)}")
    if intent_tags:
        parts.append(f"Topics: {', '.join(str(t) for t in intent_tags[:10])}")
    return "\n".join(parts) if parts else ""

def build_section_vector_text(
    *,
    section_title: str,
    section_summary: str = "",
    rescued_fragments: Optional[List[str]] = None,
    section_role: str = "",
) -> str:
    """Build text for a section-level vector, including rescued short fragments."""
    parts: List[str] = []
    if section_title and section_title.lower() not in ("untitled section", "section", "document"):
        parts.append(f"Section: {section_title}")
    if section_role:
        parts.append(f"Role: {section_role.replace('_', ' ').title()}")
    if section_summary:
        parts.append(section_summary.strip())
    if rescued_fragments:
        unique = list(dict.fromkeys(rescued_fragments))
        parts.append(f"Details: {', '.join(unique)}")
    return "\n".join(parts) if parts else ""

def build_multi_resolution_extras(
    *,
    subscription_id: str,
    profile_id: str,
    document_id: str,
    doc_name: str,
    doc_domain: str = "generic",
    doc_type: str = "other",
    understanding: Optional[Dict[str, Any]] = None,
    rescued_fragments: Optional[Dict[str, List[str]]] = None,
    section_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
    answerability: Optional[List[str]] = None,
    schema_completeness: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build extra vectors (doc-level + section-level) for multi-resolution embedding.

    Returns a list of dicts, each with keys: text, metadata.
    The caller is responsible for encoding these texts and upserting to Qdrant.
    """
    extras: List[Dict[str, Any]] = []
    understanding = understanding or {}

    doc_summary = str(understanding.get("document_summary") or "")[:500]
    key_entities = understanding.get("key_entities") or []
    intent_tags = understanding.get("intent_tags") or []
    section_summaries = understanding.get("section_summaries") or {}

    # ── Document-level vector ──
    doc_text = build_doc_vector_text(
        doc_summary=doc_summary,
        key_entities=key_entities,
        intent_tags=intent_tags,
        doc_domain=doc_domain,
        doc_type=doc_type,
    )
    if doc_text and len(doc_text.strip()) >= 30:
        doc_meta = {
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "document_id": document_id,
            "source_name": doc_name,
            "doc_domain": doc_domain,
            "resolution": "doc",
            "chunk_kind": "doc_context",
            "chunk_id": _stable_id("doc", document_id),
            "chunk_index": 0,
            "section_id": "doc_root",
            "section_title": "Document Overview",
            "section_kind": "doc_summary",
            "page": 0,
            "content": doc_text,
            "embedding_text": doc_text,
        }
        if answerability:
            doc_meta["answerability"] = answerability
        if schema_completeness is not None:
            doc_meta["schema_completeness"] = schema_completeness
        if doc_summary:
            doc_meta["doc_summary"] = doc_summary[:500]
        if key_entities:
            doc_meta["doc_key_entities"] = key_entities
        if intent_tags:
            doc_meta["doc_intent_tags"] = intent_tags
        extras.append({"text": doc_text, "metadata": doc_meta})
        logger.info(
            "Multi-resolution: doc vector for %s (%d chars)", doc_name, len(doc_text),
        )

    # ── Section-level vectors ──
    section_meta_map = section_meta_map or {}
    rescued_fragments = rescued_fragments or {}

    # Collect all section IDs from summaries and rescued fragments
    all_section_ids = set()
    for sec_id in rescued_fragments:
        all_section_ids.add(sec_id)
    for title in section_summaries:
        # Try to find matching section_id from metadata
        for sid, smeta in section_meta_map.items():
            if smeta.get("section_title") == title:
                all_section_ids.add(sid)
                break
        else:
            # Generate a section ID from the title
            sid = hashlib.sha1(f"{document_id}|{title}".encode()).hexdigest()[:12]
            all_section_ids.add(sid)
            section_meta_map.setdefault(sid, {"section_title": title})

    sec_idx = 0
    for sec_id in all_section_ids:
        smeta = section_meta_map.get(sec_id, {})
        title = smeta.get("section_title") or "Section"
        summary = section_summaries.get(title, "")
        frags = rescued_fragments.get(sec_id, [])
        role = smeta.get("section_role") or smeta.get("inferred_section_role") or ""

        sec_text = build_section_vector_text(
            section_title=title,
            section_summary=summary,
            rescued_fragments=frags,
            section_role=role,
        )
        if not sec_text or len(sec_text.strip()) < 20:
            continue

        sec_meta = {
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "document_id": document_id,
            "source_name": doc_name,
            "doc_domain": doc_domain,
            "resolution": "section",
            "chunk_kind": "section_summary",
            "chunk_id": _stable_id("sec", document_id, sec_id),
            "chunk_index": sec_idx,
            "section_id": sec_id,
            "section_title": title,
            "section_kind": smeta.get("section_kind", "misc"),
            "page": smeta.get("page_start") or smeta.get("page") or 0,
            "content": sec_text,
            "embedding_text": sec_text,
        }
        if answerability:
            sec_meta["answerability"] = answerability
        if doc_summary:
            sec_meta["doc_summary"] = doc_summary[:500]
        extras.append({"text": sec_text, "metadata": sec_meta})
        sec_idx += 1

    if sec_idx:
        logger.info(
            "Multi-resolution: %d section vectors for %s (%d rescued fragment groups)",
            sec_idx, doc_name, len(rescued_fragments),
        )

    return extras

__all__ = [
    "build_doc_vector_text",
    "build_section_vector_text",
    "build_multi_resolution_extras",
]
