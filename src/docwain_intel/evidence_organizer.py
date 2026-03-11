"""Evidence organizer — restructures retrieved chunks and graph facts before LLM consumption."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .models import ExtractionResult


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EvidenceGroup(BaseModel):
    entity_id: Optional[str] = None
    entity_text: Optional[str] = None
    entity_label: Optional[str] = None
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    relevance_score: float = 0.0


class EvidenceGap(BaseModel):
    field_name: str
    description: str


class ProvenanceRecord(BaseModel):
    source_document: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None


class OrganizedEvidence(BaseModel):
    entity_groups: List[EvidenceGroup] = Field(default_factory=list)
    ungrouped_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[EvidenceGap] = Field(default_factory=list)
    provenance: List[ProvenanceRecord] = Field(default_factory=list)
    total_facts: int = 0
    total_chunks: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return s.strip().lower()


def _chunk_score(chunk: Dict[str, Any]) -> float:
    """Return a numeric relevance score from a chunk dict."""
    score = chunk.get("score")
    if score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            pass
    return 0.0


def _dedup_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep highest-confidence fact for each (subject, predicate, value) triple."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for f in facts:
        key = (
            _norm(str(f.get("subject", f.get("subject_id", "")))),
            _norm(str(f.get("predicate", ""))),
            _norm(str(f.get("value", f.get("object_value", "")))),
        )
        existing = best.get(key)
        new_conf = float(f.get("confidence", 0.0))
        if existing is None or new_conf > float(existing.get("confidence", 0.0)):
            best[key] = f
    return list(best.values())


def _extract_provenance(chunk: Dict[str, Any]) -> ProvenanceRecord:
    """Build a ProvenanceRecord from chunk metadata."""
    payload = chunk.get("payload", {})
    meta = chunk.get("metadata", payload)

    source_doc = (
        meta.get("source_document")
        or meta.get("document_name")
        or meta.get("filename")
        or payload.get("source_document")
        or payload.get("document_name")
        or payload.get("filename")
        or "unknown"
    )

    page = meta.get("page") or meta.get("page_number") or payload.get("page") or payload.get("page_number")
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None

    section = meta.get("section") or meta.get("section_title") or payload.get("section") or payload.get("section_title")

    chunk_id = chunk.get("id") or chunk.get("chunk_id") or meta.get("chunk_id") or payload.get("chunk_id")

    return ProvenanceRecord(
        source_document=source_doc,
        page=page,
        section=section,
        chunk_id=chunk_id,
    )


def _get_entity_ids(chunk: Dict[str, Any]) -> List[str]:
    """Return the list of entity ids associated with this chunk."""
    payload = chunk.get("payload", {})
    ids = payload.get("intel_entity_ids") or chunk.get("intel_entity_ids")
    if ids and isinstance(ids, list):
        return [str(i) for i in ids]
    return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def organize_evidence(
    *,
    chunks: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    query_entities: List[str],
    extraction: Optional[ExtractionResult] = None,
) -> OrganizedEvidence:
    """Organize retrieved chunks and facts into entity-grouped evidence."""

    # --- Fact deduplication ---
    deduped_facts = _dedup_facts(facts)

    # --- Build entity text/label lookup from extraction ---
    entity_info: Dict[str, Dict[str, str]] = {}
    if extraction:
        for e in extraction.entities:
            entity_info[e.entity_id] = {"text": e.text, "label": e.label}

    # --- Entity grouping of chunks ---
    group_map: Dict[str, EvidenceGroup] = {}
    ungrouped: List[Dict[str, Any]] = []

    for chunk in chunks:
        eids = _get_entity_ids(chunk)
        if not eids:
            ungrouped.append(chunk)
        else:
            for eid in eids:
                if eid not in group_map:
                    info = entity_info.get(eid, {})
                    group_map[eid] = EvidenceGroup(
                        entity_id=eid,
                        entity_text=info.get("text"),
                        entity_label=info.get("label"),
                    )
                group_map[eid].chunks.append(chunk)

    # --- Assign facts to groups by subject ---
    for f in deduped_facts:
        subject = str(f.get("subject", f.get("subject_id", "")))
        if subject in group_map:
            group_map[subject].facts.append(f)
        else:
            # Try matching by normalized entity text
            matched = False
            for eid, grp in group_map.items():
                if grp.entity_text and _norm(grp.entity_text) == _norm(subject):
                    grp.facts.append(f)
                    matched = True
                    break
            if not matched:
                # Fact doesn't map to any group — still counted
                pass

    # --- Compute relevance scores and sort within groups ---
    for grp in group_map.values():
        scores = [_chunk_score(c) for c in grp.chunks]
        grp.relevance_score = sum(scores) / len(scores) if scores else 0.0
        grp.chunks.sort(key=_chunk_score, reverse=True)

    # Sort groups by average relevance descending
    entity_groups = sorted(group_map.values(), key=lambda g: g.relevance_score, reverse=True)

    # --- Gap detection ---
    gaps: List[EvidenceGap] = []
    covered_texts = set()
    for grp in entity_groups:
        if grp.entity_text:
            covered_texts.add(_norm(grp.entity_text))
        if grp.entity_id:
            covered_texts.add(_norm(grp.entity_id))

    for qe in query_entities:
        if _norm(qe) not in covered_texts:
            gaps.append(EvidenceGap(
                field_name=qe,
                description=f"No evidence found for entity '{qe}'",
            ))

    # --- Provenance ---
    provenance = [_extract_provenance(c) for c in chunks]

    return OrganizedEvidence(
        entity_groups=entity_groups,
        ungrouped_chunks=ungrouped,
        gaps=gaps,
        provenance=provenance,
        total_facts=len(deduped_facts),
        total_chunks=len(chunks),
    )
