from __future__ import annotations

import re
from typing import Iterable, List

from src.embedding.chunking.section_chunker import (
    Block,
    SectionChunker as _BaseSectionChunker,
)

_SOFT_HYPHENS = "\u00ad\u2010\u2011\u2212"
_HYPHEN_ARTIFACT_RE = re.compile(rf"([A-Za-z])(?:[{_SOFT_HYPHENS}])\s*([A-Za-z])")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = text
    while True:
        updated = _HYPHEN_ARTIFACT_RE.sub(r"\1\2", normalized)
        if updated == normalized:
            break
        normalized = updated
    return normalized


class SectionChunker(_BaseSectionChunker):
    """Section-aware chunker with conservative de-hyphenation."""

    def _coalesce_blocks(self, blocks: Iterable[Block]) -> List[Block]:
        cleaned: List[Block] = []
        for block in blocks:
            text = normalize_text(block.text)
            if not text:
                continue
            block_type = block.block_type or self._classify_block(text)
            cleaned.append(Block(text, block_type, block.page_start, block.page_end))
        return cleaned


__all__ = ["SectionChunker", "normalize_text"]
