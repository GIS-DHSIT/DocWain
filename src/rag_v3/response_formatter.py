from __future__ import annotations

from typing import Any, Dict, List, Optional

from .sanitize import sanitize_text


def format_rag_v3_response(
    *,
    response_text: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> str:
    cleaned = sanitize_text(response_text or "")
    source_line = _source_line(sources or [])
    if source_line and "source:" not in cleaned.lower():
        return f"{cleaned}\n{source_line}".strip()
    return cleaned


def _source_line(sources: List[Dict[str, Any]]) -> Optional[str]:
    if not sources:
        return None
    for src in sources:
        name = src.get("file_name") or src.get("source_name") or src.get("document_name")
        if name:
            return f"Source: {name}"
    return None


__all__ = ["format_rag_v3_response"]
