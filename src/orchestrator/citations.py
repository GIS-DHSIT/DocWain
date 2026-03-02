from __future__ import annotations

from typing import Any, Dict, List

from src.utils.payload_utils import get_source_name


def _page_range(page_start: Any, page_end: Any) -> str:
    if page_start is None:
        return "N/A"
    if page_end is None:
        return str(page_start)
    return f"{page_start}-{page_end}"


def format_citation(hit: Dict[str, Any]) -> str:
    file_name = get_source_name(hit) or hit.get("file_name") or "Unknown"
    section_title = hit.get("section_title") or "Section"
    page_range = _page_range(hit.get("page_start"), hit.get("page_end"))
    return f"[Source: {file_name}, Section: {section_title}, Page: {page_range}]"


def build_citations(hits: List[Dict[str, Any]]) -> List[str]:
    return [format_citation(hit) for hit in hits]


__all__ = ["format_citation", "build_citations"]
