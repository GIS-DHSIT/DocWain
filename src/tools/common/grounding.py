from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def build_source_record(
    source_type: str,
    identifier: str,
    *,
    title: Optional[str] = None,
    page: Optional[int] = None,
    chunk_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "type": source_type,
        "id": identifier,
        "title": title or identifier,
        "page": page,
        "chunk_id": chunk_id,
        "metadata": metadata or {},
    }


def merge_sources(*sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for group in sources:
        if not group:
            continue
        for src in group:
            merged.append(src)
    return merged

