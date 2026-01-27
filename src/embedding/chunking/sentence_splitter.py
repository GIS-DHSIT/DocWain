"""Lightweight sentence segmentation utilities.

The goal is pragmatic sentence boundaries without heavy dependencies.
We keep common abbreviations intact and split on terminal punctuation
followed by whitespace and an uppercase letter/number or a new bullet.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Common abbreviations that should not trigger a sentence split.
_ABBREVIATIONS = {
    "e.g",
    "i.e",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "fig",
    "eq",
    "no",
    "inc",
    "ltd",
    "co",
    "st",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

# Bullets and list markers that often start a new thought.
_BULLET_START_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+")

# Terminal punctuation candidates.
_TERMINAL_RE = re.compile(r"[.!?]+")


def _looks_like_abbreviation(token: str) -> bool:
    """Return True when a token likely represents an abbreviation."""
    if not token:
        return False
    clean = token.strip().strip("()[]{}\"'").rstrip(".")
    if not clean:
        return False
    lower = clean.lower()
    if lower in _ABBREVIATIONS:
        return True
    # Single capital letter like "A.".
    if len(clean) == 1 and clean.isalpha():
        return True
    # Multi-letter uppercase acronyms like "U.S." or "DHS.".
    if clean.isupper() and len(clean) <= 5:
        return True
    return False


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like units with abbreviation awareness."""
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    sentences: List[str] = []
    start = 0

    for match in _TERMINAL_RE.finditer(normalized):
        end = match.end()
        punct = match.group(0)

        # Look behind for the preceding token.
        prev_segment = normalized[start:end]
        prev_tokens = prev_segment.split()
        prev_token = prev_tokens[-1] if prev_tokens else ""

        # Avoid splitting on abbreviations like "e.g.".
        if punct.endswith(".") and _looks_like_abbreviation(prev_token):
            continue

        # Decide whether the following text looks like a new sentence.
        following = normalized[end:].lstrip()
        if not following:
            sentences.append(prev_segment.strip())
            start = end
            continue

        next_char = following[0]
        should_split = next_char.isupper() or next_char.isdigit() or _BULLET_START_RE.match(following) is not None
        if should_split:
            sentences.append(prev_segment.strip())
            start = end

    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)

    return [s for s in sentences if s]


def join_sentences(sentences: Iterable[str], max_chars: int) -> List[str]:
    """Join sentences into chunks that do not exceed max_chars when possible."""
    max_chars = max(1, int(max_chars))
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    def _flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append(" ".join(buffer).strip())
            buffer = []
            buffer_len = 0

    for sentence in sentences:
        if not sentence:
            continue
        sent = sentence.strip()
        if not sent:
            continue
        sent_len = len(sent) + (1 if buffer else 0)
        if buffer and buffer_len + sent_len > max_chars:
            _flush()
        buffer.append(sent)
        buffer_len += sent_len
        if buffer_len >= max_chars:
            _flush()

    _flush()
    return [chunk for chunk in chunks if chunk]
