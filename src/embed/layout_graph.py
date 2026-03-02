from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.api.pipeline_models import ExtractedDocument


@dataclass
class LayoutBlock:
    block_id: str
    text: str
    page: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutGraph:
    document_id: str
    file_name: str
    blocks: List[LayoutBlock]


def build_layout_graph(content: Any, *, document_id: str, file_name: str) -> LayoutGraph:
    text, page_map = _extract_text(content)
    blocks = _split_blocks(text, page_map)
    return LayoutGraph(document_id=str(document_id), file_name=file_name, blocks=blocks)


def _extract_text(content: Any) -> tuple[str, Dict[int, int]]:
    if isinstance(content, ExtractedDocument):
        if content.sections:
            text_parts: List[str] = []
            page_map: Dict[int, int] = {}
            for idx, sec in enumerate(content.sections):
                text_parts.append(sec.text or "")
                page_map[idx] = int(sec.start_page or sec.end_page or 1)
            return "\n\n".join(text_parts), page_map
        return content.full_text or "", {}
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content.get("text") or "", {}
        if isinstance(content.get("content"), str):
            return content.get("content") or "", {}
    if isinstance(content, list):
        parts = [str(item) for item in content if item is not None]
        return "\n\n".join(parts), {}
    if isinstance(content, str):
        return content, {}
    return str(content or ""), {}


def _split_blocks(text: str, page_map: Dict[int, int]) -> List[LayoutBlock]:
    lines = (text or "").replace("\r", "\n").split("\n")
    blocks: List[LayoutBlock] = []
    current: List[str] = []
    block_index = 0

    def _flush() -> None:
        nonlocal block_index
        if not current:
            return
        block_text = "\n".join(current).strip()
        if not block_text:
            current.clear()
            return
        page = page_map.get(block_index, 1)
        block_id = _block_hash(block_text, block_index)
        blocks.append(LayoutBlock(block_id=block_id, text=block_text, page=page))
        block_index += 1
        current.clear()

    for line in lines:
        if line.strip():
            current.append(line.strip())
        else:
            _flush()
    _flush()
    return blocks


def _block_hash(text: str, idx: int) -> str:
    digest = hashlib.sha1(f"{idx}:{text[:80]}".encode("utf-8")).hexdigest()
    return digest[:12]


__all__ = ["LayoutGraph", "LayoutBlock", "build_layout_graph"]
