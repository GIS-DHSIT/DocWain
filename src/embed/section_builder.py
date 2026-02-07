from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

from .layout_graph import LayoutGraph


@dataclass
class Section:
    section_id: str
    title: str
    page_start: int
    page_end: int
    kind_structural: bool
    text: str


def build_sections(layout: LayoutGraph) -> List[Section]:
    blocks = layout.blocks
    if not blocks:
        return []

    sections: List[Section] = []
    current_title = "Section 1"
    current_lines: List[str] = []
    current_start = blocks[0].page
    current_end = blocks[0].page
    section_index = 1

    def _flush() -> None:
        nonlocal section_index, current_title, current_lines, current_start, current_end
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return
        sec_id = _section_hash(layout.document_id, section_index, current_title)
        sections.append(
            Section(
                section_id=sec_id,
                title=current_title,
                page_start=current_start,
                page_end=current_end,
                kind_structural=True,
                text=text,
            )
        )
        section_index += 1
        current_lines = []

    for block in blocks:
        block_text = (block.text or "").strip()
        if not block_text:
            continue
        if _is_heading(block_text) and current_lines:
            _flush()
            current_title = block_text
            current_start = block.page
            current_end = block.page
            continue
        if _is_heading(block_text) and not current_lines:
            current_title = block_text
            current_start = block.page
            current_end = block.page
            continue
        current_lines.append(block_text)
        current_end = max(current_end, block.page)

    if current_lines:
        _flush()

    return sections


def _is_heading(text: str) -> bool:
    words = [w for w in text.split() if w]
    if not words:
        return False
    if len(words) > 8:
        return False
    title_like = 0
    for word in words:
        if word and word[0].isupper():
            title_like += 1
    if title_like == len(words):
        return True
    if text.isupper() and len(text) <= 60:
        return True
    if len(text) <= 40 and title_like >= max(1, len(words) - 1):
        return True
    return False


def _section_hash(document_id: str, idx: int, title: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{idx}:{title[:40]}".encode("utf-8")).hexdigest()
    return digest[:12]


__all__ = ["Section", "build_sections"]
