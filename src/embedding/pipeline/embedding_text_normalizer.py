from __future__ import annotations

import re
from typing import Optional


_PAGE_RE = re.compile(
    r"^\s*(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$",
    re.IGNORECASE,
)
_HEADER_FOOTER_RE = re.compile(r"^(?:confidential|internal use only|copyright)\b", re.IGNORECASE)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+")
_MULTISPACE_RE = re.compile(r"\s{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_TABLE_SPLIT_RE = re.compile(r"\s{2,}|\t+")


def _should_drop_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _PAGE_RE.match(stripped):
        return True
    if _HEADER_FOOTER_RE.match(stripped):
        return True
    return False


def _normalize_table_line(line: str) -> str:
    if _MULTISPACE_RE.search(line) or "\t" in line:
        parts = [part.strip() for part in _TABLE_SPLIT_RE.split(line) if part.strip()]
        if len(parts) >= 2:
            return " | ".join(parts)
    return line


def normalize_for_embedding(
    content: str,
    doc_domain: Optional[str] = None,
    section_kind: Optional[str] = None,
    *,
    force: bool = False,
) -> str:
    _ = (doc_domain, section_kind)
    text = content or ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines = []
    seen = set()
    for line in lines:
        if _should_drop_line(line):
            continue
        cleaned = _normalize_table_line(line.strip())
        if not cleaned:
            continue
        if _BULLET_PREFIX_RE.match(cleaned):
            cleaned = _BULLET_PREFIX_RE.sub("", cleaned).strip()
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized_lines.append(cleaned)

    joined = " ".join(normalized_lines)
    joined = _WHITESPACE_RE.sub(" ", joined).strip()
    if not joined:
        return ""

    if force:
        joined = re.sub(r"[^\w\s\-\.,:/%$]", " ", joined)
        joined = _WHITESPACE_RE.sub(" ", joined).strip()
    return joined


def ensure_embedding_text(
    content: str,
    doc_domain: Optional[str] = None,
    section_kind: Optional[str] = None,
) -> str:
    raw = (content or "").strip()
    if not raw:
        return ""
    normalized = normalize_for_embedding(content, doc_domain, section_kind)
    if normalized == raw:
        normalized = normalize_for_embedding(content, doc_domain, section_kind, force=True)
    if normalized == raw:
        normalized = raw.lower()
    if normalized == raw:
        normalized = f"{raw} .".strip()
    return normalized


__all__ = ["normalize_for_embedding", "ensure_embedding_text"]
