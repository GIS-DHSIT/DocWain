"""Build prompt context from ranked evidence chunks."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.retrieval.retriever import EvidenceChunk


def _merge_adjacent_text(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
    """Merge chunks that are consecutive in the same section/document.

    When two high-scoring chunks come from adjacent positions in the same
    document (consecutive page_start or section), merge their text for
    better context continuity. This helps with contracts, legal docs, and
    multi-page tables where meaning spans chunk boundaries.
    """
    if len(chunks) <= 1:
        return chunks

    # Group by (document_id, section)
    groups: Dict[str, List[EvidenceChunk]] = {}
    for c in chunks:
        key = f"{c.document_id}::{c.section}"
        groups.setdefault(key, []).append(c)

    merged: List[EvidenceChunk] = []
    merged_ids = set()

    for key, group in groups.items():
        if len(group) < 2:
            merged.extend(group)
            continue

        # Sort by page_start to find adjacencies
        group.sort(key=lambda c: (c.page_start, c.chunk_id))

        i = 0
        while i < len(group):
            current = group[i]
            # Check if next chunk is adjacent (same or next page)
            if (i + 1 < len(group)
                    and abs(group[i + 1].page_start - current.page_end) <= 1
                    and current.chunk_id not in merged_ids
                    and group[i + 1].chunk_id not in merged_ids):
                # Merge: combine text, keep higher score, span pages
                next_chunk = group[i + 1]
                merged_text = current.text.rstrip() + "\n" + next_chunk.text.lstrip()
                merged_chunk = EvidenceChunk(
                    text=merged_text,
                    source_name=current.source_name,
                    document_id=current.document_id,
                    profile_id=current.profile_id,
                    section=current.section,
                    page_start=current.page_start,
                    page_end=next_chunk.page_end,
                    score=max(current.score, next_chunk.score),
                    chunk_id=current.chunk_id,
                    chunk_type=current.chunk_type,
                )
                merged.append(merged_chunk)
                merged_ids.add(current.chunk_id)
                merged_ids.add(next_chunk.chunk_id)
                i += 2
            else:
                if current.chunk_id not in merged_ids:
                    merged.append(current)
                i += 1

    # Re-sort by score
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


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

    # Merge adjacent chunks for better context continuity
    merged_chunks = _merge_adjacent_text(chunks)

    evidence: List[Dict] = []
    seen_doc_ids: List[str] = []

    for idx, chunk in enumerate(merged_chunks, start=1):
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
    entities: List = []
    key_facts: List = []
    doc_types: List[str] = []

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
        for f in intel.get("key_facts") or intel.get("facts") or []:
            if f not in key_facts:
                key_facts.append(f)
        dt = intel.get("document_type", "")
        if dt and dt not in doc_types:
            doc_types.append(dt)

    doc_context: Dict = {}
    if summaries:
        doc_context["summary"] = " ".join(summaries)
    if entities:
        doc_context["entities"] = entities[:20]
    if key_facts:
        doc_context["key_facts"] = key_facts[:15]
    if doc_types:
        doc_context["document_types"] = doc_types

    return evidence, doc_context
