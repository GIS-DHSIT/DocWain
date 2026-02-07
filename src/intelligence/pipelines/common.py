from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


def extract_sentences(text: str, max_sentences: int = 2) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences[:max_sentences]


def dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value.strip())
    return output


def build_sources(chunks: Iterable[Any], *, max_sources: int = 12) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        source_name = meta.get("source_name") or (meta.get("source") or {}).get("name") or meta.get("file_name")
        page = meta.get("page") or meta.get("page_start") or meta.get("page_number")
        sources.append(
            {
                "source_name": source_name,
                "doc_domain": meta.get("doc_domain"),
                "page": page,
            }
        )
    return sources[:max_sources]


def group_chunks_by_document(chunks: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        doc_id = meta.get("document_id") or "unknown"
        grouped.setdefault(str(doc_id), []).append(chunk)
    return grouped


def format_bullets(bullets: List[str]) -> str:
    return "\n".join([f"- {b}" for b in bullets if b])
