from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


def _page_range(meta: Dict[str, Any], fallback: Optional[Any] = None) -> str:
    page_start = meta.get("page_start")
    page_end = meta.get("page_end")
    if page_start is None and fallback is not None:
        page_start = fallback
    if page_start is None:
        return "N/A"
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _meta_from_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return chunk.get("metadata") or chunk


def _source_id(chunk: Dict[str, Any], index: int) -> int:
    value = chunk.get("source_id")
    if value is None:
        return index + 1
    try:
        return int(value)
    except Exception:
        return index + 1


def _safe_value(meta: Dict[str, Any], keys: Iterable[str], default: str) -> str:
    for key in keys:
        value = meta.get(key)
        if value:
            return str(value)
    return default


def build_citations(used_chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, chunk in enumerate(used_chunks):
        meta = _meta_from_chunk(chunk)
        file_name = chunk.get("source_name") or _safe_value(
            meta,
            ("source_file", "file_name", "filename", "document_name", "title"),
            "Unknown",
        )
        section = chunk.get("section") or _safe_value(
            meta,
            ("section_title", "section_path", "section"),
            "Section",
        )
        page = chunk.get("page") or meta.get("page")
        page_display = _page_range(meta, fallback=page)
        parts.append(f"{file_name} | {section} | {page_display}")
    if not parts:
        return "Citations:"
    return "Citations: " + "; ".join(parts)


def replace_citations_line(answer_text: str, citations_line: str) -> str:
    if not answer_text:
        return answer_text
    lines = answer_text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("citations:"):
            lines[idx] = citations_line
            return "\n".join(lines)
    if lines and lines[-1].strip():
        lines.append(citations_line)
    else:
        lines.append(citations_line)
    return "\n".join(lines)


def filter_inline_citations(answer_text: str, valid_source_ids: Iterable[int]) -> str:
    if not answer_text:
        return answer_text
    return re.sub(r"\[(?:SOURCE-\d+(?:,\s*)?)+\]", "", answer_text)


__all__ = ["build_citations", "replace_citations_line", "filter_inline_citations"]
