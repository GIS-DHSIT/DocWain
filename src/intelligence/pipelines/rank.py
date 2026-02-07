from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from src.intelligence.pipelines.common import build_sources, group_chunks_by_document
from src.utils.payload_utils import get_content_text


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower())


def _score_doc(chunks: Iterable[Any], query_terms: List[str]) -> Tuple[float, str, str, str]:
    evidence_hits = 0
    sample = ""
    citation = ""
    label = "Document"
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        text = getattr(chunk, "text", "") or ""
        content_text = get_content_text(meta) or text
        if not sample:
            sample = content_text[:160].strip()
        if not citation:
            citation = meta.get("source_name") or (meta.get("source") or {}).get("name") or ""
            page = meta.get("page") or meta.get("page_start") or meta.get("page_number")
            if citation and page:
                citation = f"{citation}, p. {page}"
        if not label:
            label = meta.get("source_name") or (meta.get("source") or {}).get("name") or "Document"
        tokens = _tokenize(text)
        if not tokens:
            continue
        for term in query_terms:
            if term in tokens:
                evidence_hits += 1
    if not label:
        label = "Document"
    return float(evidence_hits), sample, citation, label


def rank_task(
    *,
    query: str,
    chunks: List[Any],
) -> Dict[str, Any]:
    if not chunks:
        return {
            "response": "No grounded ranking available.",
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {"task": "compare_rank", "query": query},
        }

    grouped = group_chunks_by_document(chunks)
    query_terms = _tokenize(query)
    rows: List[Tuple[str, float, str, str]] = []
    for _doc_id, doc_chunks in grouped.items():
        score, sample, citation, label = _score_doc(doc_chunks, query_terms)
        rows.append((label, score, sample, citation))
    rows.sort(key=lambda r: r[1], reverse=True)

    lines = [
        "| Rank | Document | Evidence | Citations |",
        "| --- | --- | --- | --- |",
    ]
    for idx, (label, _score, sample, citation) in enumerate(rows, start=1):
        evidence = sample or "Relevant evidence found in document."
        lines.append(f"| {idx} | {label} | {evidence} | {citation} |")

    criteria = "Criteria: alignment of skills/experience with the query."
    response = "\n".join(lines + [criteria])
    return {
        "response": response,
        "sources": build_sources(chunks),
        "grounded": True,
        "context_found": True,
        "metadata": {"task": "compare_rank", "query": query, "acknowledged": True},
    }


__all__ = ["rank_task"]
