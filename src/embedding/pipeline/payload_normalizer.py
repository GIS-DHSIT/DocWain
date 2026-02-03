from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, Optional

from src.embedding.pipeline.chunk_integrity import clean_text_for_embedding

_EVIDENCE_RE = re.compile(r"Section:\s*(.*?)(?:,|$)\s*Page:\s*(.*)$", re.IGNORECASE)


def _drop_empty(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        cleaned = [v for v in (_drop_empty(v) for v in value) if v is not None]
        return cleaned or None
    if isinstance(value, dict):
        cleaned = {k: v for k, v in ((k, _drop_empty(v)) for k, v in value.items()) if v is not None}
        return cleaned or None
    return value


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _intify(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_evidence_pointer(pointer: Optional[str]) -> Dict[str, Any]:
    if not pointer:
        return {}
    match = _EVIDENCE_RE.search(pointer)
    if not match:
        return {}
    section_title = match.group(1).strip() or None
    page_range = match.group(2).strip() if match.group(2) else ""
    page_start = None
    page_end = None
    if page_range and page_range.upper() != "N/A":
        if "-" in page_range:
            start, end = page_range.split("-", 1)
            page_start = _intify(start.strip())
            page_end = _intify(end.strip())
        else:
            page_start = _intify(page_range)
            page_end = page_start
    result: Dict[str, Any] = {}
    if section_title:
        result["section_title"] = section_title
    if page_start is not None:
        result["page_start"] = page_start
    if page_end is not None:
        result["page_end"] = page_end
    return result


def _section_path_list(section_path: Optional[str]) -> Optional[list[str]]:
    if not section_path:
        return None
    parts = [part.strip() for part in section_path.split(">")]
    parts = [part for part in parts if part]
    return parts or None


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean_text = clean_text_for_embedding(payload.get("text") or "")
    raw_text = payload.get("text_raw") or payload.get("text") or ""
    chunk_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest() if clean_text else None

    section_title = _stringify(payload.get("section_title") or payload.get("section"))
    section_path = _stringify(payload.get("section_path") or section_title)
    section_id = _stringify(payload.get("section_id"))

    page_start = _intify(payload.get("page_start") or payload.get("page"))
    page_end = _intify(payload.get("page_end") or payload.get("page"))

    evidence = _parse_evidence_pointer(_stringify(payload.get("evidence_pointer")))
    if evidence.get("section_title") and not section_title:
        section_title = evidence.get("section_title")
    if evidence.get("page_start") is not None:
        page_start = evidence.get("page_start")
    if evidence.get("page_end") is not None:
        page_end = evidence.get("page_end")

    chunk_index = _intify(payload.get("chunk_index"))
    chunk_count = _intify(payload.get("chunk_count"))

    chunk_type = _stringify(payload.get("chunk_type")) or "text"
    chunk_role = _stringify(payload.get("chunk_kind")) or "section_text"

    canonical: Dict[str, Any] = {
        "subscription_id": _stringify(payload.get("subscription_id")),
        "profile_id": _stringify(payload.get("profile_id")),
        "document_id": _stringify(payload.get("document_id")),
        "profile_name": _stringify(payload.get("profile_name")),
        "text": clean_text,
        "text_data": {
            "clean": clean_text,
            "raw": raw_text if raw_text and raw_text != clean_text else None,
        },
        "source": {
            "name": _stringify(payload.get("source_name") or payload.get("source_file") or payload.get("filename")),
            "uri": _stringify(payload.get("source_uri")),
        },
        "document": {
            "type": _stringify(payload.get("doc_type") or payload.get("document_type")),
        },
        "section": {
            "id": section_id,
            "title": section_title,
            "path": _section_path_list(section_path),
        },
        "chunk": {
            "id": _stringify(payload.get("chunk_id")),
            "index": chunk_index,
            "count": chunk_count,
            "type": chunk_type,
            "role": chunk_role,
            "hash": chunk_hash,
            "size": {"chars": _intify(payload.get("chunk_char_len") or len(clean_text or ""))},
            "links": {
                "prev": _stringify(payload.get("prev_chunk_id")),
                "next": _stringify(payload.get("next_chunk_id")),
            },
            "sentence_complete": bool(payload.get("chunk_sentence_complete", payload.get("sentence_complete", False))),
        },
        "provenance": {
            "page_start": page_start,
            "page_end": page_end,
            "section_title": section_title,
        },
    }

    flattened: Dict[str, Any] = {
        "subscription_id": canonical["subscription_id"],
        "profile_id": canonical["profile_id"],
        "document_id": canonical["document_id"],
        "profile_name": canonical["profile_name"],
        "text": canonical["text"],
        "text_clean": clean_text,
        "text_raw": raw_text if raw_text and raw_text != clean_text else None,
        "section_title": section_title,
        "section_path": section_path,
        "section_id": section_id,
        "page_start": page_start,
        "page_end": page_end,
        "page": page_start,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_id": canonical["chunk"]["id"],
        "prev_chunk_id": canonical["chunk"]["links"]["prev"],
        "next_chunk_id": canonical["chunk"]["links"]["next"],
        "chunk_type": chunk_type,
        "chunk_kind": chunk_role,
        "chunk_sentence_complete": canonical["chunk"]["sentence_complete"],
    }

    canonical.update(flattened)
    cleaned = _drop_empty(canonical) or {}
    return cleaned


__all__ = ["normalize_payload"]
