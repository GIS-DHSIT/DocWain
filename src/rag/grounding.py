from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b")
_NUMBER_RE = re.compile(r"\b\d[\d,./:-]*\b")
_CITATION_RE = re.compile(r"\[SOURCE-\d+(?:,\s*SOURCE-\d+)*\]")
_HEADER_RE = re.compile(r"^(DOCUMENT\s*/\s*INFORMATION|META|INTENT)\b", re.IGNORECASE)

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "about",
    "which",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "could",
    "would",
    "should",
    "into",
    "onto",
    "over",
    "under",
}

_NAME_STOPWORDS = {
    "resume",
    "cv",
    "profile",
    "document",
    "documents",
    "certifications",
    "skills",
    "experience",
    "education",
    "about",
    "for",
    "of",
    "on",
    "the",
}

_FALLBACK_TEXT = "Not explicitly mentioned in the provided documents."


def extract_claims(answer_text: str) -> List[str]:
    if not answer_text:
        return []
    claims: List[str] = []
    for line in answer_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("citations:"):
            continue
        if _HEADER_RE.match(stripped):
            continue
        claims.append(stripped)
    return claims


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text or "").strip()


def _extract_numbers(text: str) -> List[str]:
    return _NUMBER_RE.findall(text or "")


def _extract_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.%-]*", text or "")
    normalized = []
    for token in tokens:
        lower = token.lower()
        if len(lower) <= 3 or lower in _STOPWORDS:
            continue
        normalized.append(lower)
    return normalized


def is_claim_supported(claim: str, evidence_text: str) -> bool:
    if not claim:
        return True
    lowered = claim.lower()
    if "not explicitly mentioned" in lowered or "not explicitly stated" in lowered:
        return True

    evidence = evidence_text or ""
    evidence_lower = evidence.lower()

    for token in _EMAIL_RE.findall(claim):
        if token.lower() not in evidence_lower:
            return False

    for token in _PHONE_RE.findall(claim):
        if token.lower() not in evidence_lower:
            return False

    numbers = _extract_numbers(claim)
    for number in numbers:
        if number not in evidence:
            return False

    tokens = _extract_tokens(claim)
    if tokens:
        overlap = sum(1 for token in set(tokens) if token in evidence_lower)
        if overlap == 0:
            return False
    return True


def _line_fallback(line: str) -> str:
    prefix_match = re.match(r"^(\s*(?:[-*•]|\d+[\).])\s*)?(.*)$", line)
    if not prefix_match:
        return _FALLBACK_TEXT
    bullet_prefix = prefix_match.group(1) or ""
    rest = prefix_match.group(2) or ""
    rest = _strip_citations(rest)
    if ":" in rest:
        label, _value = rest.split(":", 1)
        label = label.strip()
        if label:
            return f"{bullet_prefix}{label}: {_FALLBACK_TEXT}"
    return f"{bullet_prefix}{_FALLBACK_TEXT}"


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("excerpt") or "")
    return str(getattr(chunk, "text", "") or "")


def enforce_grounding(answer_text: str, evidence_chunks: List[Any]) -> Tuple[str, List[Any]]:
    evidence_text = "\n\n".join(_chunk_text(chunk) for chunk in evidence_chunks)
    if not answer_text:
        return answer_text, evidence_chunks
    lines = answer_text.splitlines()
    updated_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            updated_lines.append(line)
            continue
        if stripped.lower().startswith("citations:") or _HEADER_RE.match(stripped):
            updated_lines.append(line)
            continue
        claim = _strip_citations(stripped)
        if not is_claim_supported(claim, evidence_text):
            updated_lines.append(_line_fallback(line))
            continue
        updated_lines.append(line)
    return "\n".join(updated_lines), evidence_chunks


def _extract_person_name(query: str) -> Optional[str]:
    if not query:
        return None
    match = re.search(r"\b(?:of|for|about|on|regarding)\s+([A-Za-z][A-Za-z\-. ]+)", query, re.IGNORECASE)
    candidate = ""
    if match:
        candidate = match.group(1)
    else:
        tokens = query.strip().split()
        if len(tokens) >= 1:
            candidate = tokens[-1]
    candidate = re.sub(r"[^\w\s\-.]", "", candidate)
    words = [w for w in candidate.split() if w and w.lower() not in _NAME_STOPWORDS]
    if not words:
        return None
    name = " ".join(words).strip()
    return name if len(name) >= 2 else None


def _chunk_matches_name(chunk: Any, name: str) -> bool:
    if not name:
        return True
    name_lower = name.lower()
    meta: Dict[str, Any] = {}
    text_val = ""
    if isinstance(chunk, dict):
        meta = chunk.get("metadata") or chunk
        text_val = chunk.get("text") or ""
    else:
        meta = getattr(chunk, "metadata", {}) or {}
        text_val = getattr(chunk, "text", "") or ""

    for key in ("profile_name", "person_name", "document_name", "source_file", "file_name"):
        value = meta.get(key)
        if value and name_lower in str(value).lower():
            return True
    if name_lower in str(text_val).lower():
        return True
    return False


def filter_chunks_by_query_entity(query: str, chunks: List[Any]) -> List[Any]:
    name = _extract_person_name(query)
    if not name:
        return chunks
    matches = [chunk for chunk in chunks if _chunk_matches_name(chunk, name)]
    return matches or chunks


__all__ = [
    "extract_claims",
    "is_claim_supported",
    "enforce_grounding",
    "filter_chunks_by_query_entity",
]
