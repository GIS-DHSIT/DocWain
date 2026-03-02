from __future__ import annotations

import re
from typing import Any, Dict, Optional


def get_source_name(payload: Dict[str, Any]) -> Optional[str]:
    return (
        (payload.get("source") or {}).get("name")
        or payload.get("source_name")
        or payload.get("source_file")
        or payload.get("filename")
        or payload.get("file_name")
        or payload.get("document_name")
    )


def token_count(text: Optional[str]) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def is_valid_text(
    text: Optional[str],
    *,
    min_chars: Optional[int] = None,
    min_tokens: Optional[int] = None,
) -> bool:
    if text is None:
        return False
    stripped = str(text).strip()
    if not stripped:
        return False
    if min_chars is None or min_tokens is None:
        try:
            from src.api.config import Config

            if min_chars is None:
                min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 80))
            if min_tokens is None:
                min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 15))
        except Exception:
            min_chars = min_chars or 80
            min_tokens = min_tokens or 15
    if len(stripped) < int(min_chars or 0):
        return False
    if token_count(stripped) < int(min_tokens or 0):
        return False
    return True


def get_canonical_text(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    canonical = payload.get("canonical_text")
    if is_valid_text(canonical):
        return str(canonical)
    content_value = payload.get("content")
    if is_valid_text(content_value):
        return str(content_value)
    text_value = payload.get("text")
    if is_valid_text(text_value):
        return str(text_value)
    return ""


def get_embedding_text(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    embedding_text = payload.get("embedding_text") or payload.get("text_clean") or payload.get("text_embedding")
    if is_valid_text(embedding_text, min_chars=10, min_tokens=2):
        return str(embedding_text)
    canonical = get_canonical_text(payload)
    if canonical:
        return canonical
    content = payload.get("content")
    return str(content) if content else ""


def get_content_text(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    content = payload.get("content")
    if is_valid_text(content):
        return str(content)
    canonical = payload.get("canonical_text")
    if is_valid_text(canonical):
        return str(canonical)
    text_value = payload.get("text")
    if is_valid_text(text_value):
        return str(text_value)
    return ""


def get_document_type(payload: Dict[str, Any]) -> Optional[str]:
    return (
        (payload.get("document") or {}).get("type")
        or payload.get("doc_type")
        or payload.get("document_type")
        or payload.get("file_type")
    )


def get_chunk_hash(payload: Dict[str, Any]) -> Optional[str]:
    return (
        (payload.get("chunk") or {}).get("hash")
        or payload.get("chunk_hash")
        or payload.get("text_hash")
    )


def get_chunk_sentence_complete(payload: Dict[str, Any], default: bool = True) -> bool:
    value = (payload.get("chunk") or {}).get("sentence_complete")
    if value is None:
        value = payload.get("chunk_sentence_complete")
    if value is None:
        return default
    return bool(value)


__all__ = [
    "get_chunk_hash",
    "get_chunk_sentence_complete",
    "get_canonical_text",
    "get_content_text",
    "get_embedding_text",
    "get_document_type",
    "get_source_name",
    "is_valid_text",
    "token_count",
]
