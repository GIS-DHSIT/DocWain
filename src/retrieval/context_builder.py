"""Build prompt context from ranked evidence chunks."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.retrieval.retriever import EvidenceChunk


def build_context(
    chunks: List[EvidenceChunk],
    doc_intelligence: Dict[str, dict],
) -> Tuple[List[Dict], Dict]:
    """Convert ranked chunks + doc intelligence into prompt-ready context.

    Returns:
        (evidence, doc_context) where evidence is a numbered list of dicts
        and doc_context aggregates summaries, entities, and key_facts from
        doc_intelligence keyed by document_id.
    """
    if not chunks:
        return [], {}

    evidence: List[Dict] = []
    seen_doc_ids: List[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        evidence.append({
            "source_index": idx,
            "source_name": chunk.source_name,
            "section": chunk.section,
            "page": chunk.page_start,
            "text": chunk.text,
            "score": chunk.score,
            "document_id": chunk.document_id,
            "profile_id": chunk.profile_id,
            "chunk_id": chunk.chunk_id,
        })
        if chunk.document_id and chunk.document_id not in seen_doc_ids:
            seen_doc_ids.append(chunk.document_id)

    # Aggregate document intelligence
    if not doc_intelligence:
        return evidence, {}

    summaries: List[str] = []
    entities: List[str] = []
    key_facts: List[str] = []

    for doc_id in seen_doc_ids:
        intel = doc_intelligence.get(doc_id)
        if not intel:
            continue
        s = intel.get("summary", "")
        if s and s not in summaries:
            summaries.append(s)
        for e in intel.get("entities", []):
            if e not in entities:
                entities.append(e)
        for f in intel.get("key_facts", []):
            if f not in key_facts:
                key_facts.append(f)

    doc_context: Dict = {}
    if summaries:
        doc_context["summary"] = " ".join(summaries)
    if entities:
        doc_context["entities"] = entities
    if key_facts:
        doc_context["key_facts"] = key_facts

    return evidence, doc_context
