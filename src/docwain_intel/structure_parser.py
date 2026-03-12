"""Stage 2: Structural parsing — convert raw blocks into semantic units.

Discovers document structure from visual/typographic signals, not domain
knowledge. A heading is a heading whether it says 'Work Experience' or
'Clause 7.2'. The structure is discovered from block types and adjacency.
"""
from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

from .models import (
    Block, ExtractedDocumentJSON, SemanticUnit, StructuredDocument,
    Section, Table, UnitType,
)

logger = get_logger(__name__)

MAX_UNIT_CHARS = 2000
MIN_UNIT_CHARS = 30

def _unit_id(document_id: str, index: int, text: str) -> str:
    raw = f"{document_id}|{index}|{text[:64]}"
    return f"su_{hashlib.sha1(raw.encode()).hexdigest()[:12]}"

def _build_section_map(sections: List[Section]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for section in sections:
        for block_id in section.content_refs:
            mapping[block_id] = section.section_path
    return mapping

def _collect_ordered_blocks(doc: ExtractedDocumentJSON) -> List[Block]:
    blocks: List[Block] = []
    for page in sorted(doc.pages, key=lambda p: p.page_number):
        sorted_blocks = sorted(page.blocks, key=lambda b: b.reading_order or 0)
        blocks.extend(sorted_blocks)
    return blocks

def _split_text_at_sentence_boundary(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    import re
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]

def parse_structure(
    doc: ExtractedDocumentJSON,
    *,
    document_id: str,
) -> StructuredDocument:
    """Convert an ExtractedDocumentJSON into a StructuredDocument of semantic units."""
    blocks = _collect_ordered_blocks(doc)
    section_map = _build_section_map(doc.sections)
    units: List[SemanticUnit] = []
    unit_index = 0
    current_heading_path: List[str] = []

    i = 0
    while i < len(blocks):
        block = blocks[i]
        heading_path = section_map.get(block.block_id, current_heading_path)

        if block.type == "heading":
            heading_text = (block.text or "").strip()
            if heading_text:
                current_heading_path = heading_path if heading_path else [heading_text]
            i += 1
            continue

        if block.type == "key_value":
            kv_blocks = [block]
            j = i + 1
            while j < len(blocks) and blocks[j].type == "key_value":
                kv_blocks.append(blocks[j])
                j += 1
            kv_pairs: Dict[str, str] = {}
            texts: List[str] = []
            for kvb in kv_blocks:
                if kvb.key and kvb.value:
                    kv_pairs[kvb.key] = kvb.value
                texts.append(kvb.text or "")
            text = "\n".join(t for t in texts if t)
            page_numbers = [b.page_number or 1 for b in kv_blocks]
            units.append(SemanticUnit(
                unit_id=_unit_id(document_id, unit_index, text),
                unit_type=UnitType.KV_GROUP, text=text,
                page_start=min(page_numbers), page_end=max(page_numbers),
                heading_path=heading_path, kv_pairs=kv_pairs,
                raw_spans=[{"block_id": b.block_id, "page": b.page_number} for b in kv_blocks],
                confidence=0.95,
            ))
            unit_index += 1
            i = j
            continue

        if block.type == "list_item":
            list_blocks = [block]
            j = i + 1
            while j < len(blocks) and blocks[j].type == "list_item":
                list_blocks.append(blocks[j])
                j += 1
            text = "\n".join(b.text or "" for b in list_blocks if b.text)
            page_numbers = [b.page_number or 1 for b in list_blocks]
            for chunk_text in _split_text_at_sentence_boundary(text, MAX_UNIT_CHARS):
                units.append(SemanticUnit(
                    unit_id=_unit_id(document_id, unit_index, chunk_text),
                    unit_type=UnitType.LIST, text=chunk_text,
                    page_start=min(page_numbers), page_end=max(page_numbers),
                    heading_path=heading_path,
                    raw_spans=[{"block_id": b.block_id, "page": b.page_number} for b in list_blocks],
                    confidence=0.90,
                ))
                unit_index += 1
            i = j
            continue

        if block.type in ("paragraph", "other"):
            para_blocks = [block]
            j = i + 1
            while j < len(blocks) and blocks[j].type in ("paragraph", "other"):
                para_blocks.append(blocks[j])
                j += 1
            combined_text = "\n".join(b.text or "" for b in para_blocks if b.text)
            page_numbers = [b.page_number or 1 for b in para_blocks]
            for chunk_text in _split_text_at_sentence_boundary(combined_text, MAX_UNIT_CHARS):
                if len(chunk_text.strip()) < MIN_UNIT_CHARS:
                    unit_type = UnitType.FRAGMENT
                else:
                    unit_type = UnitType.PARAGRAPH
                units.append(SemanticUnit(
                    unit_id=_unit_id(document_id, unit_index, chunk_text),
                    unit_type=unit_type, text=chunk_text,
                    page_start=min(page_numbers), page_end=max(page_numbers),
                    heading_path=heading_path,
                    raw_spans=[{"block_id": b.block_id, "page": b.page_number} for b in para_blocks],
                    confidence=0.90,
                ))
                unit_index += 1
            i = j
            continue

        text = (block.text or "").strip()
        if text:
            units.append(SemanticUnit(
                unit_id=_unit_id(document_id, unit_index, text),
                unit_type=UnitType.PARAGRAPH if len(text) >= MIN_UNIT_CHARS else UnitType.FRAGMENT,
                text=text, page_start=block.page_number or 1, page_end=block.page_number or 1,
                heading_path=heading_path,
                raw_spans=[{"block_id": block.block_id, "page": block.page_number}],
                confidence=0.85,
            ))
            unit_index += 1
        i += 1

    for table in (doc.tables or []):
        headers = table.headers or []
        rows_dicts: List[Dict[str, Any]] = []
        for row in (table.rows or []):
            row_dict = {}
            for idx, cell in enumerate(row):
                col_name = headers[idx] if idx < len(headers) else f"col_{idx + 1}"
                row_dict[col_name] = cell
            rows_dicts.append(row_dict)
        text_parts = []
        if headers:
            text_parts.append(" | ".join(headers))
        for row in (table.rows or []):
            text_parts.append(" | ".join(str(c) for c in row))
        text = "\n".join(text_parts)
        if text.strip():
            units.append(SemanticUnit(
                unit_id=_unit_id(document_id, unit_index, text),
                unit_type=UnitType.TABLE, text=text,
                page_start=table.page_number or 1, page_end=table.page_number or 1,
                heading_path=current_heading_path,
                table_headers=headers, table_rows=rows_dicts,
                raw_spans=[{"table_id": table.table_id, "page": table.page_number}],
                confidence=0.95,
            ))
            unit_index += 1

    total_chars = sum(len(u.text) for u in units)
    return StructuredDocument(
        document_id=document_id, units=units,
        unit_count=len(units), total_chars=total_chars,
    )

__all__ = ["parse_structure"]
