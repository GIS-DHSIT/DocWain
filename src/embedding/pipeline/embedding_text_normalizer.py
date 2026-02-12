from __future__ import annotations

import re
from typing import Optional


_PAGE_RE = re.compile(
    r"^\s*(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$",
    re.IGNORECASE,
)
_HEADER_FOOTER_RE = re.compile(r"^(?:confidential|internal use only|copyright)\b", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s{2,}")
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
    """Normalize text for embedding while preserving structure.

    - Preserves line breaks (newlines) for structural awareness
    - Deduplicates identical lines
    - Drops page numbers and boilerplate headers/footers
    - Normalizes table lines with pipe separators
    - Does NOT strip bullet prefixes (preserves list structure)
    - Does NOT force-lowercase or append punctuation
    """
    _ = (doc_domain, section_kind, force)
    text = (content or "").strip()
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned = []
    seen = set()
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or _should_drop_line(line):
            continue
        line = _normalize_table_line(line)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def ensure_embedding_text(
    content: str,
    doc_domain: Optional[str] = None,
    section_kind: Optional[str] = None,
) -> str:
    """Return normalized embedding text. No forced mutations."""
    raw = (content or "").strip()
    if not raw:
        return ""
    normalized = normalize_for_embedding(raw, doc_domain, section_kind)
    return normalized if normalized else raw


__all__ = ["normalize_for_embedding", "ensure_embedding_text"]
