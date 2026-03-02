from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.intelligence.pipelines.common import (
    build_sources,
    dedupe_strings,
    extract_sentences,
    format_bullets,
    group_chunks_by_document,
)
from src.utils.payload_utils import get_content_text


def _summarize_chunks(chunks: Iterable[Any], max_bullets: int) -> List[str]:
    bullets: List[str] = []
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        text = get_content_text(meta) or getattr(chunk, "text", "") or ""
        sentences = extract_sentences(text, max_sentences=1)
        bullets.extend(sentences)
        if len(bullets) >= max_bullets:
            break
    bullets = dedupe_strings(bullets)
    return bullets[:max_bullets]


def _doc_label(chunks: List[Any], fallback: str) -> str:
    if not chunks:
        return fallback
    meta = getattr(chunks[0], "metadata", None) or getattr(chunks[0], "payload", None) or {}
    return meta.get("source_name") or (meta.get("source") or {}).get("name") or fallback


def _doc_citation(chunks: List[Any], fallback: str) -> str:
    if not chunks:
        return fallback
    meta = getattr(chunks[0], "metadata", None) or getattr(chunks[0], "payload", None) or {}
    name = meta.get("source_name") or (meta.get("source") or {}).get("name") or fallback
    page = meta.get("page") or meta.get("page_start") or meta.get("page_number")
    if page:
        return f"{name}, p. {page}"
    return f"{name}"


def summarize_task(
    *,
    query: str,
    chunks: List[Any],
    scope: str,
) -> Dict[str, Any]:
    if not chunks:
        return {
            "response": "No grounded summary available.",
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {"task": "summarize", "query": query},
        }

    grouped = group_chunks_by_document(chunks)
    bullets: List[str] = []
    if scope == "current_document":
        doc_label = _doc_label(chunks, "Document")
        citation = _doc_citation(chunks, doc_label)
        doc_bullets = _summarize_chunks(chunks, max_bullets=6)
        for sentence in doc_bullets:
            bullets.append(f"{sentence} ({citation})")
    else:
        for idx, (_doc_id, doc_chunks) in enumerate(grouped.items(), start=1):
            doc_label = _doc_label(doc_chunks, f"Document {idx}")
            citation = _doc_citation(doc_chunks, doc_label)
            doc_bullets = _summarize_chunks(doc_chunks, max_bullets=2)
            if doc_bullets:
                bullets.append(f"{doc_label}: {doc_bullets[0]} ({citation})")
            if len(bullets) >= 6:
                break
    bullets = bullets[:8]
    response = format_bullets(bullets)
    if response:
        response = f"Here’s a synthesized summary from the documents:\n{response}"

    return {
        "response": response,
        "sources": build_sources(chunks),
        "grounded": True,
        "context_found": True,
        "metadata": {"task": "summarize", "query": query},
    }


__all__ = ["summarize_task"]
