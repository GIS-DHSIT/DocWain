from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

from .section_builder import Section


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    chunk_count: int
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    chunk_kind: str
    canonical_text: str
    raw_text: str


def build_chunks(sections: List[Section], *, max_tokens: int = 220) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_index = 0
    for section in sections:
        canonical = canonicalize_text(section.text)
        tokens = canonical.split()
        if not tokens:
            continue
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + max_tokens)
            piece = " ".join(tokens[start:end])
            chunk_id = _chunk_hash(section.section_id, chunk_index, piece)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    chunk_count=0,
                    section_id=section.section_id,
                    section_title=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_kind="section_text",
                    canonical_text=piece,
                    raw_text=section.text,
                )
            )
            chunk_index += 1
            start = end

    for idx, chunk in enumerate(chunks):
        chunk.chunk_index = idx
        chunk.chunk_count = len(chunks)
    return chunks


def canonicalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    deduped: List[str] = []
    recent: List[str] = []
    for line in lines:
        if not line:
            if deduped and deduped[-1] != "":
                deduped.append("")
            continue
        norm = " ".join(line.lower().split())
        if norm in recent:
            continue
        deduped.append(line)
        recent.append(norm)
        if len(recent) > 6:
            recent.pop(0)
    cleaned_lines = []
    for line in deduped:
        if not line:
            cleaned_lines.append("")
            continue
        cleaned_lines.append(" ".join(line.split()))
    merged = "\n".join(cleaned_lines).strip()
    merged = _repair_token_boundaries(merged)
    return " ".join(merged.split())


def _repair_token_boundaries(text: str) -> str:
    if not text:
        return ""
    chars = list(text)
    fixed: List[str] = []
    for idx, ch in enumerate(chars):
        prev = chars[idx - 1] if idx > 0 else ""
        nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
        if prev and ch.isupper() and prev.islower() and (nxt.islower() or nxt.isupper()):
            fixed.append(" ")
        if prev and ch.isdigit() and prev.isalpha():
            fixed.append(" ")
        if prev and ch.isalpha() and prev.isdigit():
            fixed.append(" ")
        fixed.append(ch)
    return "".join(fixed)


def _chunk_hash(section_id: str, idx: int, text: str) -> str:
    digest = hashlib.sha1(f"{section_id}:{idx}:{text[:80]}".encode("utf-8")).hexdigest()
    return digest[:12]


__all__ = ["Chunk", "build_chunks", "canonicalize_text"]
