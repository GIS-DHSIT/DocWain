from __future__ import annotations

from typing import Any, Dict, List


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Access attribute or dict key — supports both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _section_offsets(sections: List[Any]) -> Dict[str, Dict[str, int]]:
    offsets: Dict[str, Dict[str, int]] = {}
    cursor = 0
    for sec in sections:
        text = _get(sec, "text", "") or ""
        start = cursor
        cursor += len(text)
        offsets[_get(sec, "section_id", "") or _get(sec, "title", "") or str(len(offsets))] = {
            "start": start,
            "end": cursor,
        }
        cursor += 1  # spacer
    return offsets


def build_content_map(extracted: Any) -> Dict[str, Any]:
    sections_payload: List[Dict[str, Any]] = []
    tables_payload: List[Dict[str, Any]] = []
    images_payload: List[Dict[str, Any]] = []

    sections = list(_get(extracted, "sections", []) or [])
    offsets = _section_offsets(sections)

    for idx, sec in enumerate(sections):
        sec_id = _get(sec, "section_id") or str(idx)
        offsets_entry = offsets.get(sec_id) or offsets.get(_get(sec, "title", ""))
        sections_payload.append(
            {
                "title": _get(sec, "title", "Untitled Section"),
                "start_page": _get(sec, "start_page"),
                "end_page": _get(sec, "end_page"),
                "text_span_offsets": offsets_entry or {},
                "confidence": 0.9,
            }
        )

    for idx, table in enumerate(_get(extracted, "tables", []) or []):
        tables_payload.append(
            {
                "page": _get(table, "page"),
                "bbox": None,
                "table_index": idx,
                "csv": _get(table, "csv"),
                "json": None,
                "text": _get(table, "text"),
                "confidence": 0.85,
            }
        )

    for idx, figure in enumerate(_get(extracted, "figures", []) or []):
        images_payload.append(
            {
                "page": _get(figure, "page"),
                "bbox": None,
                "image_index": idx,
                "caption": _get(figure, "caption"),
                "confidence": 0.8,
            }
        )

    return {
        "sections": sections_payload,
        "tables": tables_payload,
        "images": images_payload,
    }


__all__ = ["build_content_map"]
