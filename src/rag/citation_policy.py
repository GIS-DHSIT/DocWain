from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from src.rag.citations import build_citations, filter_inline_citations


_EVIDENCE_BLOCK_RE = re.compile(r"\n?Evidence:\n(?:.*\n?)*?(?=\n\n|\Z)", re.IGNORECASE)
_CITATIONS_LINE_RE = re.compile(r"^\s*citations:\s*.*$", re.IGNORECASE | re.MULTILINE)


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        value = (item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _group_by_doc(citations: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for cite in citations:
        doc_name = cite.split("|")[0].strip() if "|" in cite else cite.strip()
        grouped.setdefault(doc_name, []).append(cite)
    return grouped


def _limit_citations(
    citations: List[str],
    *,
    max_total: int,
    max_per_doc: int,
) -> List[str]:
    if not citations:
        return []
    grouped = _group_by_doc(citations)
    ordered_docs = list(grouped.keys())
    limited: List[str] = []
    for doc in ordered_docs:
        limited.extend(grouped[doc][:max_per_doc])
    if len(limited) > max_total:
        limited = limited[:max_total]
    return _dedupe_preserve(limited)


def _strip_evidence_blocks(text: str) -> str:
    if not text:
        return text
    return _EVIDENCE_BLOCK_RE.sub("", text).strip()


def _strip_existing_citations(text: str) -> str:
    if not text:
        return text
    return _CITATIONS_LINE_RE.sub("", text).strip()


def _append_citations(text: str, citations_line: str) -> str:
    if not text:
        return citations_line
    base = text.rstrip()
    return base + "\n\n" + citations_line


def apply(
    answer_text: str,
    sources: List[Dict[str, object]],
    *,
    user_requested_evidence: bool = False,
    scope_type: Optional[str] = None,
    max_total: int = 6,
) -> str:
    text = answer_text or ""
    if not user_requested_evidence:
        text = _strip_evidence_blocks(text)
    text = filter_inline_citations(text, valid_source_ids=[])
    text = _strip_existing_citations(text)

    citations_line = build_citations(sources)
    citations_payload = citations_line.split(":", 1)[1].strip()
    citations = [c.strip() for c in citations_payload.split(";") if c.strip()]

    per_doc_cap = 2 if scope_type in {"multi_doc", "targeted_docs"} else 4
    limited = _limit_citations(citations, max_total=max_total, max_per_doc=per_doc_cap)
    if limited:
        citations_line = "Citations: " + "; ".join(limited)
    else:
        citations_line = "Citations:"

    return _append_citations(text, citations_line).strip()


__all__ = ["apply"]
