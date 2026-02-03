from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

from src.rag.doc_inventory import DocInventoryItem


_RESUME_SECTION_TITLES = {
    "experience",
    "work experience",
    "education",
    "skills",
    "certifications",
    "summary",
    "projects",
}

_RESUME_KEYWORDS = {
    "experience",
    "skills",
    "education",
    "certifications",
    "projects",
    "professional summary",
}

_RESUME_TOKENS = {"responsibilities", "achievements", "role", "client"}

_DATE_RANGE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{1,2})[\s/]+\d{4}\s*[-\u2013\u2014]\s*"
    r"(?:present|current|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{1,2})?\s*\d{0,4}\b",
    re.IGNORECASE,
)

_YEAR_RANGE_RE = re.compile(r"\b\d{4}\s*[-\u2013\u2014]\s*(?:\d{4}|present|current)\b", re.IGNORECASE)

_MULTI_PROFILE_CUES = {
    "all profiles",
    "all candidates",
    "all documents",
    "all the documents",
    "all docs",
    "all resumes",
    "all resume",
    "across all documents",
    "do this for all",
    "for all profiles",
    "for all documents",
    "documents present",
}

_RESUME_QUERY_CUES = {
    "candidate",
    "candidates",
    "resume",
    "resumes",
    "cv",
    "curriculum vitae",
    "linkedin",
    "experience",
    "skills",
    "certifications",
    "education",
    "awards",
    "rank",
    "compare",
    "shortlist",
}


def _chunk_text(chunk: Any) -> str:
    return getattr(chunk, "text", "") or ""


def _chunk_meta(chunk: Any) -> dict:
    return getattr(chunk, "metadata", {}) or {}


def is_multi_profile_request(query_text: str) -> bool:
    lowered = (query_text or "").lower()
    if any(cue in lowered for cue in _MULTI_PROFILE_CUES):
        return True
    if "all" in lowered and ("documents" in lowered or "profiles" in lowered or "candidates" in lowered):
        return True
    return False


def is_resume_query(query_text: str) -> bool:
    lowered = (query_text or "").lower()
    return any(cue in lowered for cue in _RESUME_QUERY_CUES)


def _resume_signal_from_metadata(meta: dict) -> bool:
    if not meta:
        return False
    doc_type = (meta.get("doc_type") or meta.get("document_type") or meta.get("source_type") or "").lower()
    if doc_type in {"resume", "cv", "curriculum vitae", "linkedin"}:
        return True
    return False


def _resume_section_from_metadata(meta: dict) -> bool:
    section = (meta.get("section_title") or meta.get("section_path") or "").lower()
    return any(title in section for title in _RESUME_SECTION_TITLES)


def _resume_content_hits(text: str) -> int:
    lowered = (text or "").lower()
    hits = 0
    if any(keyword in lowered for keyword in _RESUME_KEYWORDS):
        hits += 1
    if _DATE_RANGE_RE.search(text) or _YEAR_RANGE_RE.search(text):
        hits += 1
    if any(token in lowered for token in _RESUME_TOKENS):
        hits += 1
    return hits


def is_resume_like_chunk(chunk: Any) -> bool:
    meta = _chunk_meta(chunk)
    if _resume_signal_from_metadata(meta) or _resume_section_from_metadata(meta):
        return True
    return _resume_content_hits(_chunk_text(chunk)) >= 2


def has_resume_like_content(chunks: Sequence[Any]) -> bool:
    return any(is_resume_like_chunk(chunk) for chunk in chunks or [])


def should_route_resume_analyzer(
    *,
    query_text: str,
    chunks: Sequence[Any],
    doc_inventory: Sequence[DocInventoryItem],
) -> bool:
    if is_resume_query(query_text):
        return True
    if has_resume_like_content(chunks):
        return True
    _ = doc_inventory
    return False


def select_resume_docs(
    *,
    doc_inventory: Sequence[DocInventoryItem],
    chunks_by_doc: dict | None = None,
) -> List[DocInventoryItem]:
    selected: List[DocInventoryItem] = []
    for doc in doc_inventory or []:
        doc_type = (doc.doc_type or "").lower()
        name = (doc.source_file or doc.document_name or "").lower()
        if doc_type in {"resume", "cv", "linkedin"} or "resume" in name or "cv" in name:
            selected.append(doc)
            continue
        if chunks_by_doc:
            chunks = chunks_by_doc.get(doc.source_file or doc.document_name or doc.doc_id, [])
            if chunks and has_resume_like_content(chunks):
                selected.append(doc)
    return selected


def should_bypass_clarification(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> bool:
    if not is_multi_profile_request(query_text):
        matched = match_resume_docs_by_name(query_text, doc_inventory)
        return bool(matched)
    for doc in doc_inventory or []:
        doc_type = (doc.doc_type or "").lower()
        name = (doc.source_file or doc.document_name or "").lower()
        if doc_type in {"resume", "cv", "linkedin"} or "resume" in name or "cv" in name:
            return True
    return False


def has_resume_docs(doc_inventory: Sequence[DocInventoryItem]) -> bool:
    for doc in doc_inventory or []:
        doc_type = (doc.doc_type or "").lower()
        name = (doc.source_file or doc.document_name or "").lower()
        if doc_type in {"resume", "cv", "linkedin"} or "resume" in name or "cv" in name or "linkedin" in name:
            return True
    return False


def match_resume_docs_by_name(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> List[DocInventoryItem]:
    if not query_text:
        return []
    match = re.search(r"([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2})'s\s+resume", query_text, re.IGNORECASE)
    if not match:
        match = re.search(r"resume\s+of\s+([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2})", query_text, re.IGNORECASE)
    if not match:
        return []
    name = match.group(1).strip().lower()
    if not name:
        return []
    matches: List[DocInventoryItem] = []
    for doc in doc_inventory or []:
        haystack = " ".join([doc.source_file or "", doc.document_name or ""]).lower()
        if name in haystack:
            matches.append(doc)
    return matches


__all__ = [
    "is_multi_profile_request",
    "is_resume_query",
    "is_resume_like_chunk",
    "has_resume_like_content",
    "has_resume_docs",
    "match_resume_docs_by_name",
    "should_route_resume_analyzer",
    "select_resume_docs",
    "should_bypass_clarification",
]
