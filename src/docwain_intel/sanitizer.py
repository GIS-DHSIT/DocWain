import os
import re
from typing import Iterable

_REDACTED = "[redacted]"

_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{10,}\b")
_ID_LABEL_RE = re.compile(r"\b(subscription_id|profile_id|document_id|doc_id|chunk_id|point_id|uuid)\b\s*[:=]\s*\S+", re.IGNORECASE)
_INTERNAL_URL_RE = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)[^\s]+")

_META_LINE_RE = re.compile(
    r"^\s*(i\s+(?:have\s+)?(?:analyzed|reviewed|looked\s+at)|documents?\s+used|sources?|citations?|evidence:|retrieved\s+sections|the\s+retrieved\s+sections|i\s+searched|i\s+couldn'?t\s+find|i\s+cannot\s+find)\b",
    re.IGNORECASE,
)

_BLOCK_START_RE = re.compile(r"^\s*(citations?|sources?|evidence)\s*:", re.IGNORECASE)


def _strip_meta_lines(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    skip_block = False
    for line in lines:
        if _BLOCK_START_RE.match(line):
            skip_block = True
            continue
        if skip_block:
            if not line.strip():
                skip_block = False
                continue
            if re.match(r"^\s*[-•\\d\\[]", line):
                continue
            skip_block = False
        if _META_LINE_RE.match(line):
            continue
        cleaned.append(line)
    return cleaned


def _dedupe_blocks(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    if len(lines) >= 2 and len(lines) % 2 == 0:
        half = len(lines) // 2
        if lines[:half] == lines[half:]:
            return "\n".join(lines[:half]).strip()
    deduped: list[str] = []
    for line in lines:
        if deduped and line.strip() and line == deduped[-1]:
            continue
        deduped.append(line)
    return "\n".join(deduped).strip()


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\b[A-Z]\)\s*", "", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _redact_ids(text: str) -> str:
    text = _ID_LABEL_RE.sub(lambda m: m.group(1) + ": " + _REDACTED, text)
    text = _UUID_RE.sub(_REDACTED, text)
    text = _HASH_RE.sub(_REDACTED, text)

    def _clean_url(match: re.Match) -> str:
        url = match.group(0)
        if any(token in url.lower() for token in ("internal", "localhost", "127.0.0.1", "docwain")):
            return _REDACTED
        return url

    text = _INTERNAL_URL_RE.sub(_clean_url, text)
    text = re.sub(r"\binternal\s+url\s*:\s*\[redacted\]", _REDACTED, text, flags=re.IGNORECASE)
    text = _PATH_RE.sub(_REDACTED, text)
    return text


def sanitize_output(text: str) -> str:
    if not text:
        return ""
    text = _dedupe_blocks(text)
    lines = text.splitlines()
    lines = _strip_meta_lines(lines)
    cleaned = "\n".join(line for line in lines if line.strip() and line.strip() != _REDACTED)
    cleaned = _redact_ids(cleaned)
    cleaned = "\n".join(
        line for line in cleaned.splitlines() if line.strip() and line.strip() != _REDACTED
    )
    cleaned = _normalize_spacing(cleaned)
    return cleaned


__all__ = ["sanitize_output"]
