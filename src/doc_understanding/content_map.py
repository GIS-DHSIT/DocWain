from __future__ import annotations

from typing import Any, Dict, List, Optional


def _section_offsets(sections: List[Any]) -> Dict[str, Dict[str, int]]:
    offsets: Dict[str, Dict[str, int]] = {}
    cursor = 0
    for sec in sections:
        text = getattr(sec, "text", "") or ""
        start = cursor
        cursor += len(text)
        offsets[getattr(sec, "section_id", "") or getattr(sec, "title", "") or str(len(offsets))] = {
            "start": start,
            "end": cursor,
        }
        cursor += 1  # spacer
    return offsets


def build_content_map(extracted: Any) -> Dict[str, Any]:
    sections_payload: List[Dict[str, Any]] = []
    tables_payload: List[Dict[str, Any]] = []
    images_payload: List[Dict[str, Any]] = []

    sections = list(getattr(extracted, "sections", []) or [])
    offsets = _section_offsets(sections)

    for idx, sec in enumerate(sections):
        sec_id = getattr(sec, "section_id", None) or str(idx)
        offsets_entry = offsets.get(sec_id) or offsets.get(getattr(sec, "title", ""))
        sections_payload.append(
            {
                "title": getattr(sec, "title", "Untitled Section"),
                "start_page": getattr(sec, "start_page", None),
                "end_page": getattr(sec, "end_page", None),
                "text_span_offsets": offsets_entry or {},
                "confidence": 0.9,
            }
        )

    for idx, table in enumerate(getattr(extracted, "tables", []) or []):
        tables_payload.append(
            {
                "page": getattr(table, "page", None),
                "bbox": None,
                "table_index": idx,
                "csv": getattr(table, "csv", None),
                "json": None,
                "text": getattr(table, "text", None),
                "confidence": 0.85,
            }
        )

    for idx, figure in enumerate(getattr(extracted, "figures", []) or []):
        images_payload.append(
            {
                "page": getattr(figure, "page", None),
                "bbox": None,
                "image_index": idx,
                "caption": getattr(figure, "caption", None),
                "confidence": 0.8,
            }
        )

    return {
        "sections": sections_payload,
        "tables": tables_payload,
        "images": images_payload,
    }


__all__ = ["build_content_map"]
