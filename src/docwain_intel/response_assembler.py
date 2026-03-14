"""Template-based response assembler for rendering answers without LLM calls.

Renders structured answers from pre-computed knowledge graph facts and vector
search chunks using deterministic templates (bullet lists, markdown tables,
aggregation summaries, evidence excerpts).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.docwain_intel.query_router import QueryRoute


class AssembledResponse(BaseModel):
    """Rendered response with source attribution and confidence."""

    text: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    route_used: str = ""
    fact_count: int = 0
    chunk_count: int = 0


def assemble_response(
    *,
    query: str,
    route: QueryRoute,
    facts: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    is_comparison: bool = False,
    is_aggregation: bool = False,
) -> AssembledResponse:
    """Assemble a rendered response from facts and/or chunks.

    Routing logic:
    1. If *is_comparison* and facts reference 2+ subjects -> markdown table.
    2. If *is_aggregation* and facts present -> count unique subjects.
    3. If facts present (GRAPH_DIRECT / HYBRID_SEARCH) -> bullet list grouped
       by subject.
    4. If chunks present (FULL_SEARCH) -> numbered evidence excerpts.
    5. Otherwise -> no-results message.
    """
    facts = facts or []
    chunks = chunks or []

    sources = _extract_sources(facts, chunks)
    confidence = _compute_confidence(facts, chunks)

    # --- pick rendering strategy ---
    if is_comparison and facts:
        text = _render_comparison_table(facts)
    elif is_aggregation and facts:
        text = _render_aggregation(facts, query)
    elif facts:
        text = _render_entity_bullets(facts)
    elif chunks:
        text = _render_chunk_evidence(chunks)
    else:
        text = _render_no_results(query)
        confidence = 0.0

    return AssembledResponse(
        text=text,
        sources=sources,
        confidence=confidence,
        route_used=route.value,
        fact_count=len(facts),
        chunk_count=len(chunks),
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_entity_bullets(facts: List[Dict[str, Any]]) -> str:
    """Render facts as bullet list grouped by subject."""
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in facts:
        by_subject[f.get("subject", "Unknown")].append(f)

    lines: list[str] = []
    for subject, group in by_subject.items():
        if len(by_subject) > 1:
            lines.append(f"**{subject}**")
        for f in group:
            predicate = _humanize_predicate(f.get("predicate", ""))
            value = f.get("value", "")
            lines.append(f"- **{predicate}**: {value}")
        if len(by_subject) > 1:
            lines.append("")  # blank line between subjects

    return "\n".join(lines).rstrip()


def _render_comparison_table(facts: List[Dict[str, Any]]) -> str:
    """Render a markdown comparison table with subjects as rows."""
    by_subject: Dict[str, Dict[str, str]] = defaultdict(dict)
    all_predicates: list[str] = []

    for f in facts:
        subj = f.get("subject", "Unknown")
        pred = f.get("predicate", "")
        val = f.get("value", "")
        by_subject[subj][pred] = val
        if pred not in all_predicates:
            all_predicates.append(pred)

    # Header row
    header = "| Subject | " + " | ".join(_humanize_predicate(p) for p in all_predicates) + " |"
    separator = "| --- | " + " | ".join("---" for _ in all_predicates) + " |"

    rows: list[str] = [header, separator]
    for subj in by_subject:
        cells = " | ".join(by_subject[subj].get(p, "-") for p in all_predicates)
        rows.append(f"| {subj} | {cells} |")

    return "\n".join(rows)


def _render_aggregation(facts: List[Dict[str, Any]], query: str) -> str:
    """Render aggregation result with count and subject list."""
    unique_subjects = list(dict.fromkeys(f.get("subject", "") for f in facts))
    count = len(unique_subjects)
    subject_list = ", ".join(unique_subjects)
    return f"Found {count} matches: {subject_list}."


def _render_chunk_evidence(chunks: List[Dict[str, Any]]) -> str:
    """Render chunk excerpts as numbered evidence list."""
    lines: list[str] = []
    for i, ch in enumerate(chunks, 1):
        text = ch.get("text", "").strip()
        source = ch.get("source", "")
        page = ch.get("page", "")
        score = ch.get("score", 0.0)
        loc = f" (p. {page})" if page else ""
        lines.append(f"{i}. {text}")
        lines.append(f"   *Source: {source}{loc} | relevance: {score:.2f}*")
    return "\n".join(lines)


def _render_no_results(query: str) -> str:
    """Produce a helpful no-results message."""
    q = query.strip() or "your query"
    return f"I couldn't find relevant information for '{q}' in the available documents."


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _humanize_predicate(predicate: str) -> str:
    """Convert predicate keys like HAS_SKILL to Title Case."""
    return predicate.replace("_", " ").title()


def _extract_sources(
    facts: List[Dict[str, Any]], chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Collect unique (source, page) pairs from facts and chunks."""
    seen: set[tuple[str, Any]] = set()
    sources: list[Dict[str, Any]] = []
    for item in [*facts, *chunks]:
        src = item.get("source", "")
        page = item.get("page")
        if not src:
            continue
        key = (src, page)
        if key not in seen:
            seen.add(key)
            entry: Dict[str, Any] = {"source": src}
            if page is not None:
                entry["page"] = page
            sources.append(entry)
    return sources


def _compute_confidence(
    facts: List[Dict[str, Any]], chunks: List[Dict[str, Any]]
) -> float:
    """Average confidence from facts/chunks, falling back to a heuristic."""
    values: list[float] = []
    for f in facts:
        if "confidence" in f:
            values.append(float(f["confidence"]))
    for c in chunks:
        if "score" in c:
            values.append(float(c["score"]))

    if values:
        return sum(values) / len(values)

    # Heuristic: if we have data but no explicit confidence, assign moderate.
    if facts or chunks:
        return 0.7
    return 0.0
