from __future__ import annotations

from typing import Any, Dict, Optional


def get_source_name(payload: Dict[str, Any]) -> Optional[str]:
    return (
        (payload.get("source") or {}).get("name")
        or payload.get("source_file")
        or payload.get("filename")
        or payload.get("file_name")
        or payload.get("document_name")
    )


def get_document_type(payload: Dict[str, Any]) -> Optional[str]:
    return (
        (payload.get("document") or {}).get("type")
        or payload.get("doc_type")
        or payload.get("document_type")
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
    "get_document_type",
    "get_source_name",
]
