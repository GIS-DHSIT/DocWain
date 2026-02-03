from __future__ import annotations

import re
from typing import Iterable, List

from src.rag.context_reasoning import WorkingContext


_FIELD_CUES = {
    "name": {"name", "candidate"},
    "years": {"years", "experience"},
    "skills": {"skills", "technologies", "tools"},
    "education": {"education", "degree", "university"},
    "certifications": {"certification", "certificate", "certified"},
    "awards": {"award", "achievement", "honor"},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _requested_fields(query: str) -> List[str]:
    lowered = (query or "").lower()
    fields = []
    for field, cues in _FIELD_CUES.items():
        if any(cue in lowered for cue in cues):
            fields.append(field)
    return fields


def select_evidence_lines(
    *,
    query: str,
    context: WorkingContext,
    max_lines_per_doc: int = 3,
) -> List[str]:
    requested = _requested_fields(query)
    include_doc_name = len(context.resolved_scope) > 1

    candidates = []
    for fact in context.key_facts:
        label_norm = _normalize(fact.label)
        if requested and not any(token in label_norm for token in requested):
            continue
        line = f"{fact.label}: {fact.value}"
        if include_doc_name:
            line = f"{fact.doc_name}: {line}"
        candidates.append((fact.doc_name, line, fact.chunk_id))
    for claim in context.numeric_claims:
        label_norm = _normalize(claim.label)
        if requested and not any(token in label_norm for token in requested):
            continue
        line = f"{claim.label}: {claim.value}"
        if include_doc_name:
            line = f"{claim.doc_name}: {line}"
        candidates.append((claim.doc_name, line, claim.chunk_id))

    if not candidates:
        for fact in context.key_facts:
            line = f"{fact.label}: {fact.value}"
            if include_doc_name:
                line = f"{fact.doc_name}: {line}"
            candidates.append((fact.doc_name, line, fact.chunk_id))

    selected: List[str] = []
    per_doc_counts = {}
    seen = set()
    for doc_name, line, _chunk_id in candidates:
        if not line:
            continue
        key = (doc_name or "", _normalize(line))
        if key in seen:
            continue
        seen.add(key)
        per_doc_counts.setdefault(doc_name, 0)
        if per_doc_counts[doc_name] >= max_lines_per_doc:
            continue
        selected.append(f"- {line}.")
        per_doc_counts[doc_name] += 1
    return selected


__all__ = ["select_evidence_lines"]
