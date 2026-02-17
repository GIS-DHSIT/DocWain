"""Cross-document comparison engine.

Provides structured field-aligned comparison across documents with
overlap/difference detection and strength analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import Candidate


@dataclass
class FieldComparison:
    field_name: str
    values: Dict[str, Any] = field(default_factory=dict)
    comparison_type: str = "text"  # "overlap", "numeric", "text", "list"
    overlap: Optional[List[str]] = None
    differences: Optional[Dict[str, List[str]]] = None


@dataclass
class ComparisonResult:
    documents: List[str] = field(default_factory=list)
    field_comparisons: List[FieldComparison] = field(default_factory=list)
    summary: str = ""
    strengths: Dict[str, List[str]] = field(default_factory=dict)


# Fields that are lists (set overlap/difference analysis)
_LIST_FIELDS = {"technical_skills", "functional_skills", "certifications", "education", "achievements"}

# Fields that are numeric
_NUMERIC_FIELDS = {"total_years_experience"}

# Query keyword → focus field mapping
_FOCUS_MAP = {
    "skill": ["technical_skills", "functional_skills"],
    "skills": ["technical_skills", "functional_skills"],
    "technical": ["technical_skills"],
    "certification": ["certifications"],
    "certifications": ["certifications"],
    "education": ["education"],
    "experience": ["total_years_experience", "experience_summary"],
    "achievement": ["achievements"],
    "achievements": ["achievements"],
}


def _infer_focus_fields(query: str) -> Optional[List[str]]:
    """Infer which fields to focus on from the query."""
    query_lower = query.lower()
    focused: List[str] = []
    for keyword, fields in _FOCUS_MAP.items():
        if keyword in query_lower:
            for f in fields:
                if f not in focused:
                    focused.append(f)
    return focused if focused else None


def _parse_years(value: str) -> Optional[float]:
    """Extract numeric years from a string like '5 years'."""
    if not value:
        return None
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def _compare_list_field(
    field_name: str,
    values: Dict[str, Any],
) -> FieldComparison:
    """Compare list fields by computing set overlap and differences."""
    sets: Dict[str, set] = {}
    for doc_name, val in values.items():
        items = val if isinstance(val, list) else []
        sets[doc_name] = {item.lower().strip() for item in items if item}

    all_sets = list(sets.values())
    if all_sets:
        shared = set.intersection(*all_sets) if len(all_sets) > 1 else set()
    else:
        shared = set()

    diffs: Dict[str, List[str]] = {}
    for doc_name, s in sets.items():
        unique = s - shared
        # Restore original casing from the values
        original_items = values.get(doc_name) or []
        diffs[doc_name] = [
            item for item in original_items
            if item and item.lower().strip() in unique
        ]

    # Restore casing for overlap
    overlap_list: List[str] = []
    if shared:
        # Use first doc's items for casing
        first_doc_items = list(values.values())[0] or []
        overlap_list = [
            item for item in first_doc_items
            if item and item.lower().strip() in shared
        ]

    return FieldComparison(
        field_name=field_name,
        values=values,
        comparison_type="overlap",
        overlap=overlap_list,
        differences=diffs,
    )


def _compare_numeric_field(
    field_name: str,
    values: Dict[str, Any],
) -> FieldComparison:
    """Compare numeric fields."""
    return FieldComparison(
        field_name=field_name,
        values=values,
        comparison_type="numeric",
    )


def _compare_text_field(
    field_name: str,
    values: Dict[str, Any],
) -> FieldComparison:
    """Compare text fields side-by-side."""
    return FieldComparison(
        field_name=field_name,
        values=values,
        comparison_type="text",
    )


def _compute_strengths(
    comparisons: List[FieldComparison],
    documents: List[str],
) -> Dict[str, List[str]]:
    """Identify strengths for each document based on comparisons."""
    strengths: Dict[str, List[str]] = {doc: [] for doc in documents}

    for comp in comparisons:
        if comp.comparison_type == "overlap" and comp.differences:
            # Whoever has more unique items has an advantage
            max_unique = 0
            for doc, diff in comp.differences.items():
                max_unique = max(max_unique, len(diff))
            for doc, diff in comp.differences.items():
                if len(diff) == max_unique and max_unique > 0:
                    strengths[doc].append(
                        f"More unique {comp.field_name.replace('_', ' ')} ({len(diff)} unique)"
                    )

        elif comp.comparison_type == "numeric":
            max_val = None
            max_doc = None
            for doc, val in comp.values.items():
                num = _parse_years(str(val)) if val else None
                if num is not None and (max_val is None or num > max_val):
                    max_val = num
                    max_doc = doc
            if max_doc and max_val is not None:
                strengths[max_doc].append(
                    f"Higher {comp.field_name.replace('_', ' ')} ({max_val})"
                )

        elif comp.comparison_type == "text":
            # Text fields: longer/more detailed = strength
            max_len = 0
            max_doc = None
            for doc, val in comp.values.items():
                length = len(str(val or ""))
                if length > max_len:
                    max_len = length
                    max_doc = doc
            if max_doc and max_len > 20:
                strengths[max_doc].append(
                    f"More detailed {comp.field_name.replace('_', ' ')}"
                )

    return strengths


def compare_documents(
    contexts: list,
    query: str,
    focus_fields: Optional[List[str]] = None,
) -> ComparisonResult:
    """Build structured comparison across DocumentContexts.

    Args:
        contexts: List of DocumentContext objects (from document_context.py)
        query: User query for focus field inference
        focus_fields: Optional explicit list of fields to compare
    """
    if not contexts:
        return ComparisonResult()

    if len(contexts) == 1:
        doc_name = contexts[0].document_name
        return ComparisonResult(
            documents=[doc_name],
            summary=f"Only one document available: {doc_name}",
        )

    documents = [ctx.document_name for ctx in contexts]

    # Determine focus fields
    effective_focus = focus_fields or _infer_focus_fields(query)

    # Build per-field values from context fields
    all_fields: Dict[str, Dict[str, Any]] = {}
    for ctx in contexts:
        for label, value in ctx.fields.items():
            normalized = label.lower().replace(" ", "_")
            all_fields.setdefault(normalized, {})[ctx.document_name] = value

    # Compare each field
    comparisons: List[FieldComparison] = []
    compared_fields = set()

    for field_name, values in all_fields.items():
        if effective_focus and field_name not in effective_focus:
            continue
        if field_name in _LIST_FIELDS:
            comparisons.append(_compare_list_field(field_name, values))
        elif field_name in _NUMERIC_FIELDS:
            comparisons.append(_compare_numeric_field(field_name, values))
        else:
            comparisons.append(_compare_text_field(field_name, values))
        compared_fields.add(field_name)

    strengths = _compute_strengths(comparisons, documents)

    summary_parts = [f"Comparison of {len(documents)} documents: {', '.join(documents)}."]
    if comparisons:
        summary_parts.append(f"{len(comparisons)} field(s) compared.")
    for doc, strs in strengths.items():
        if strs:
            summary_parts.append(f"{doc}: {'; '.join(strs)}.")

    return ComparisonResult(
        documents=documents,
        field_comparisons=comparisons,
        summary=" ".join(summary_parts),
        strengths=strengths,
    )


def compare_candidates_from_schema(
    candidates: list,
    query: str,
) -> ComparisonResult:
    """Build comparison from pre-extracted HRSchema Candidate objects."""
    if not candidates:
        return ComparisonResult()

    documents = [c.name or f"Candidate {i+1}" for i, c in enumerate(candidates)]

    if len(candidates) == 1:
        return ComparisonResult(
            documents=documents,
            summary=f"Only one candidate: {documents[0]}",
        )

    focus = _infer_focus_fields(query)

    # Build comparisons from Candidate attributes
    candidate_fields = {
        "technical_skills": lambda c: c.technical_skills,
        "functional_skills": lambda c: c.functional_skills,
        "certifications": lambda c: c.certifications,
        "education": lambda c: c.education,
        "achievements": lambda c: c.achievements,
        "total_years_experience": lambda c: c.total_years_experience,
        "experience_summary": lambda c: c.experience_summary,
        "role": lambda c: c.role,
    }

    comparisons: List[FieldComparison] = []
    for field_name, getter in candidate_fields.items():
        if focus and field_name not in focus:
            continue
        values: Dict[str, Any] = {}
        for i, cand in enumerate(candidates):
            doc_name = cand.name or f"Candidate {i+1}"
            val = getter(cand)
            if val:
                values[doc_name] = val

        if not values:
            continue

        if field_name in _LIST_FIELDS:
            comparisons.append(_compare_list_field(field_name, values))
        elif field_name in _NUMERIC_FIELDS:
            comparisons.append(_compare_numeric_field(field_name, values))
        else:
            comparisons.append(_compare_text_field(field_name, values))

    strengths = _compute_strengths(comparisons, documents)

    summary_parts = [f"Comparison of {len(documents)} candidates: {', '.join(documents)}."]
    if comparisons:
        summary_parts.append(f"{len(comparisons)} field(s) compared.")

    return ComparisonResult(
        documents=documents,
        field_comparisons=comparisons,
        summary=" ".join(summary_parts),
        strengths=strengths,
    )


def render_comparison(result: ComparisonResult, intent: str = "") -> str:
    """Render comparison as formatted markdown text."""
    if not result.documents:
        return ""

    if len(result.documents) == 1:
        return result.summary

    # No meaningful comparisons — return empty so caller falls through to detail view
    if not result.field_comparisons:
        return ""

    lines: List[str] = []

    if len(result.documents) == 2:
        # Side-by-side format for 2 documents
        doc_a, doc_b = result.documents[:2]
        lines.append(f"**Comparison: {doc_a} vs {doc_b}**")
        lines.append("")

        for comp in result.field_comparisons:
            pretty_name = comp.field_name.replace("_", " ").title()
            lines.append(f"**{pretty_name}:**")

            if comp.comparison_type == "overlap":
                if comp.overlap:
                    lines.append(f"- Shared: {', '.join(comp.overlap)}")
                if comp.differences:
                    for doc, diff in comp.differences.items():
                        if diff:
                            lines.append(f"- Only {doc}: {', '.join(diff)}")

            elif comp.comparison_type == "numeric":
                for doc, val in comp.values.items():
                    lines.append(f"- {doc}: {val}")

            elif comp.comparison_type == "text":
                for doc, val in comp.values.items():
                    text = str(val or "")[:200]
                    lines.append(f"- {doc}: {text}")

            lines.append("")
    else:
        # Tabular format for 3+ documents
        lines.append(f"**Comparison of {len(result.documents)} documents**")
        lines.append("")

        for comp in result.field_comparisons:
            pretty_name = comp.field_name.replace("_", " ").title()
            lines.append(f"**{pretty_name}:**")
            for doc in result.documents:
                val = comp.values.get(doc)
                if val is None:
                    lines.append(f"- {doc}: N/A")
                elif isinstance(val, list):
                    lines.append(f"- {doc}: {', '.join(str(v) for v in val)}")
                else:
                    lines.append(f"- {doc}: {val}")
            lines.append("")

    # Strengths summary
    has_strengths = any(strs for strs in result.strengths.values())
    if has_strengths:
        lines.append("**Strengths:**")
        for doc, strs in result.strengths.items():
            if strs:
                lines.append(f"- {doc}: {'; '.join(strs)}")

    return "\n".join(lines).strip()
