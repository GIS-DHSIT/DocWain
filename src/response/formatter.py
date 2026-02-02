from __future__ import annotations

import re
from typing import Iterable

_STRUCTURE_HEADERS: Iterable[str] = (
    "DOCUMENT / INFORMATION",
    "DOCUMENT/INFORMATION",
    "META",
    "INTENT",
)

_LEADING_TOKEN_RE = re.compile(
    r"^\s*[A-Z]\)\s+(?=(" + "|".join(re.escape(h) for h in _STRUCTURE_HEADERS) + r"))",
    re.IGNORECASE,
)

_SOFT_HYPHENS = "\u00ad\u2010\u2011\u2212"
_HYPHEN_ARTIFACT_RE = re.compile(rf"([A-Za-z])(?:[{_SOFT_HYPHENS}])\s*([A-Za-z])")


def sanitize_leading_tokens(text: str) -> str:
    if not text:
        return text
    return _LEADING_TOKEN_RE.sub("", text, count=1)


def strip_internal_meta_blocks(text: str) -> str:
    if not text:
        return text
    blocks = re.split(r"\n\s*\n", text)
    kept = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        first = lines[0].lstrip()
        if first.startswith("Intent:"):
            lowered = block.lower()
            if (
                "meta / persona" in lowered
                or "meta/persona" in lowered
                or "i'm docwain" in lowered
                or "i will" in lowered
                or "i will not" in lowered
            ):
                continue
        kept.append(block)
    return "\n\n".join(kept)


def normalize_bullets_and_glyphs(text: str) -> str:
    if not text:
        return text
    normalized = text
    while True:
        updated = _HYPHEN_ARTIFACT_RE.sub(r"\1\2", normalized)
        if updated == normalized:
            break
        normalized = updated
    return normalized


def format_response_text(text: str) -> str:
    text = strip_internal_meta_blocks(text)
    text = sanitize_leading_tokens(text)
    text = normalize_bullets_and_glyphs(text)
    return text


__all__ = [
    "sanitize_leading_tokens",
    "strip_internal_meta_blocks",
    "normalize_bullets_and_glyphs",
    "format_response_text",
]
