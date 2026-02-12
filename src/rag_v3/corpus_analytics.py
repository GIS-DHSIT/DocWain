"""Corpus analytics engine for aggregate document queries.

Handles questions like "how many invoices?", "total amount across all invoices?",
"average years of experience?" by computing corpus-level statistics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ANALYTICS_PATTERNS = [
    re.compile(r"\bhow many\b", re.I),
    re.compile(r"\btotal (?:amount|number|count)\b", re.I),
    re.compile(r"\baverage\b", re.I),
    re.compile(r"\bsum of\b", re.I),
    re.compile(r"\bacross all\b", re.I),
    re.compile(r"\bin total\b", re.I),
    re.compile(r"\bcount of\b", re.I),
    re.compile(r"\bdistribution\b", re.I),
]

# Patterns to detect what type of analytic the user wants
_COUNT_RE = re.compile(r"\bhow many\b", re.I)
_TOTAL_RE = re.compile(r"\btotal\b", re.I)
_AVERAGE_RE = re.compile(r"\baverage\b", re.I)
_LIST_RE = re.compile(r"\b(?:list|enumerate|show)\s+(?:all|every)\b", re.I)

# Domain keywords for filtering
_DOMAIN_KEYWORDS = {
    "resume": ["resume", "resumes", "candidate", "candidates"],
    "invoice": ["invoice", "invoices"],
    "contract": ["contract", "contracts"],
    "report": ["report", "reports"],
}

# Numeric field patterns in chunk text
_NUMERIC_KV_RE = re.compile(
    r"^([A-Z][A-Za-z /&-]{1,40}):\s+\$?([\d,]+\.?\d*)",
    re.MULTILINE,
)


@dataclass
class CorpusStats:
    total_documents: int = 0
    documents_by_domain: Dict[str, int] = field(default_factory=dict)
    document_names: List[str] = field(default_factory=list)
    numeric_aggregations: Dict[str, Dict[str, float]] = field(default_factory=dict)


def is_analytics_query(query: str) -> bool:
    """Detect if query asks for corpus-level aggregation."""
    for pattern in _ANALYTICS_PATTERNS:
        if pattern.search(query):
            return True
    return False


def compute_corpus_stats(contexts: list) -> CorpusStats:
    """Compute aggregate statistics across all document contexts.

    Args:
        contexts: List of DocumentContext objects
    """
    if not contexts:
        return CorpusStats()

    domain_counts: Dict[str, int] = {}
    doc_names: List[str] = []
    numeric_values: Dict[str, List[float]] = {}

    for ctx in contexts:
        domain = getattr(ctx, "doc_domain", "generic")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        doc_names.append(getattr(ctx, "document_name", "Unknown"))

        # Extract numeric fields from per-doc fields
        fields = getattr(ctx, "fields", {}) or {}
        for label, value in fields.items():
            num = _try_parse_number(str(value))
            if num is not None:
                numeric_values.setdefault(label, []).append(num)

        # Also scan chunk text for numeric KV pairs
        for chunk in getattr(ctx, "chunks", []):
            text = getattr(chunk, "text", "") or ""
            for match in _NUMERIC_KV_RE.finditer(text):
                label = match.group(1).strip()
                raw_val = match.group(2).replace(",", "")
                try:
                    num = float(raw_val)
                    numeric_values.setdefault(label, []).append(num)
                except ValueError:
                    pass

    # Compute aggregations
    aggregations: Dict[str, Dict[str, float]] = {}
    for label, values in numeric_values.items():
        if not values:
            continue
        aggregations[label] = {
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": float(len(values)),
        }

    return CorpusStats(
        total_documents=len(contexts),
        documents_by_domain=domain_counts,
        document_names=doc_names,
        numeric_aggregations=aggregations,
    )


def _try_parse_number(value: str) -> Optional[float]:
    """Try to extract a number from a string value."""
    if not value:
        return None
    cleaned = value.replace(",", "").replace("$", "").strip()
    # Try the first token
    token = cleaned.split()[0] if cleaned else ""
    try:
        return float(token)
    except (ValueError, IndexError):
        return None


def _detect_domain_filter(query: str) -> Optional[str]:
    """Detect if query targets a specific document domain."""
    query_lower = query.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return domain
    return None


def answer_analytics_query(
    query: str,
    stats: CorpusStats,
    contexts: list,
) -> str:
    """Generate a natural language answer for an analytics query.

    Args:
        query: User's query
        stats: Pre-computed CorpusStats
        contexts: List of DocumentContext objects (for additional detail)
    """
    if stats.total_documents == 0:
        return "No documents found in the profile."

    domain_filter = _detect_domain_filter(query)
    parts: List[str] = []

    # Count queries: "How many resumes?"
    if _COUNT_RE.search(query):
        if domain_filter:
            count = stats.documents_by_domain.get(domain_filter, 0)
            names = [
                getattr(ctx, "document_name", "Unknown")
                for ctx in contexts
                if getattr(ctx, "doc_domain", "") == domain_filter
            ]
            parts.append(
                f"There {'is' if count == 1 else 'are'} {count} {domain_filter}"
                f"{'s' if count != 1 and not domain_filter.endswith('s') else ''}"
                f" in the profile."
            )
            if names:
                parts.append("Documents: " + ", ".join(names) + ".")
        else:
            parts.append(f"There are {stats.total_documents} document(s) in the profile.")
            if stats.documents_by_domain:
                breakdown = [f"{count} {domain}" for domain, count in stats.documents_by_domain.items()]
                parts.append("Breakdown: " + ", ".join(breakdown) + ".")

    # Total queries: "Total invoice amount?"
    elif _TOTAL_RE.search(query):
        if stats.numeric_aggregations:
            for label, agg in stats.numeric_aggregations.items():
                parts.append(f"{label}: Total = {agg['sum']:.2f} (across {int(agg['count'])} documents).")
        if not parts:
            parts.append(f"Found {stats.total_documents} document(s) but no numeric totals to aggregate.")

    # Average queries: "Average years of experience?"
    elif _AVERAGE_RE.search(query):
        if stats.numeric_aggregations:
            for label, agg in stats.numeric_aggregations.items():
                parts.append(f"{label}: Average = {agg['avg']:.2f} (range: {agg['min']:.1f} - {agg['max']:.1f}).")
        if not parts:
            parts.append(f"Found {stats.total_documents} document(s) but no numeric fields to average.")

    # List queries: "List all candidates"
    elif _LIST_RE.search(query):
        if domain_filter:
            names = [
                getattr(ctx, "document_name", "Unknown")
                for ctx in contexts
                if getattr(ctx, "doc_domain", "") == domain_filter
            ]
        else:
            names = stats.document_names
        if names:
            parts.append(f"Documents ({len(names)}):")
            for i, name in enumerate(names, 1):
                parts.append(f"{i}. {name}")
        else:
            parts.append("No matching documents found.")

    # General analytics fallback
    else:
        parts.append(f"Profile contains {stats.total_documents} document(s).")
        if stats.documents_by_domain:
            breakdown = [f"{count} {domain}" for domain, count in stats.documents_by_domain.items()]
            parts.append("Breakdown: " + ", ".join(breakdown) + ".")
        if stats.numeric_aggregations:
            for label, agg in stats.numeric_aggregations.items():
                parts.append(f"{label}: {agg['sum']:.2f} total, {agg['avg']:.2f} average.")

    return "\n".join(parts)
