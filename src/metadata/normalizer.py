from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

ALLOWED_CHUNK_KINDS = {
    "section_text",
    "table_text",
    "image_caption",
    "doc_summary",
    "section_summary",
    "structured_field",
}


class MetadataNormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizedMetadata:
    subscription_id: Optional[str]
    profile_id: Optional[str]
    profile_name: Optional[str]
    document_id: Optional[str]
    file_name: Optional[str]
    document_type: Optional[str]
    chunk_kind: Optional[str]
    section_title: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "chunk_kind": self.chunk_kind,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


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
    except Exception as exc:  # noqa: BLE001
        raise MetadataNormalizationError(f"Invalid integer value: {value}") from exc


def _resolve_field(
    metadata: Dict[str, Any],
    canonical_key: str,
    aliases: Iterable[str] = (),
    *,
    required: bool = False,
) -> Optional[str]:
    candidates = []
    for key in (canonical_key, *aliases):
        if key in metadata:
            value = _stringify(metadata.get(key))
            if value is not None:
                candidates.append((key, value))
    if not candidates:
        if required:
            raise MetadataNormalizationError(f"Missing required metadata field: {canonical_key}")
        return None
    distinct = {val for _key, val in candidates}
    if len(distinct) > 1:
        logger.error("Metadata mismatch for %s: %s", canonical_key, candidates)
        raise MetadataNormalizationError(f"Conflicting values for {canonical_key}: {distinct}")
    return candidates[0][1]


def _resolve_int_field(
    metadata: Dict[str, Any],
    canonical_key: str,
    aliases: Iterable[str] = (),
) -> Optional[int]:
    candidates = []
    for key in (canonical_key, *aliases):
        if key in metadata:
            value = metadata.get(key)
            if value is not None and value != "":
                candidates.append((key, value))
    if not candidates:
        return None
    distinct_raw = {str(val) for _key, val in candidates}
    if len(distinct_raw) > 1:
        logger.error("Metadata mismatch for %s: %s", canonical_key, candidates)
        raise MetadataNormalizationError(f"Conflicting values for {canonical_key}: {distinct_raw}")
    return _intify(candidates[0][1])


def _infer_chunk_kind(metadata: Dict[str, Any], *, fallback: str = "section_text") -> str:
    chunk_kind = _stringify(metadata.get("chunk_kind"))
    if chunk_kind:
        return chunk_kind
    chunk_type = _stringify(metadata.get("chunk_type")) or "text"
    if chunk_type in {"table", "table_row", "table_header"}:
        return "table_text"
    if chunk_type == "image_caption":
        return "image_caption"
    if chunk_type == "summary":
        return "section_summary"
    return fallback


def normalize_chunk_metadata(metadata: Dict[str, Any], *, strict: bool = True) -> NormalizedMetadata:
    document_type = _resolve_field(metadata, "document_type", aliases=("doc_type", "type"))
    doc_type = _stringify(metadata.get("doc_type"))
    if doc_type and document_type and doc_type != document_type:
        logger.error("doc_type/document_type mismatch: %s vs %s", doc_type, document_type)
        raise MetadataNormalizationError("doc_type/document_type mismatch")
    chunk_kind = _infer_chunk_kind(metadata)
    if chunk_kind not in ALLOWED_CHUNK_KINDS:
        logger.error("Invalid chunk_kind: %s", chunk_kind)
        raise MetadataNormalizationError(f"Invalid chunk_kind: {chunk_kind}")

    page_start = _resolve_int_field(metadata, "page_start", aliases=("page", "page_number"))
    page_end = _resolve_int_field(metadata, "page_end")
    if page_end is None:
        page_end = page_start
    if page_start is not None and page_end is not None and page_end < page_start:
        logger.error("page_end < page_start (%s < %s)", page_end, page_start)
        raise MetadataNormalizationError("page_end cannot be less than page_start")

    normalized = NormalizedMetadata(
        subscription_id=_resolve_field(metadata, "subscription_id", aliases=("subscriptionId",)),
        profile_id=_resolve_field(metadata, "profile_id", aliases=("profileId",)),
        profile_name=_resolve_field(metadata, "profile_name", aliases=("profileName",)),
        document_id=_resolve_field(metadata, "document_id", aliases=("documentId", "doc_id", "id", "_id")),
        file_name=_resolve_field(
            metadata,
            "file_name",
            aliases=("filename", "source_file", "source_filename", "name"),
        ),
        document_type=document_type or doc_type,
        chunk_kind=chunk_kind,
        section_title=_resolve_field(metadata, "section_title", aliases=("section", "section_path")),
        page_start=page_start,
        page_end=page_end,
    )

    if strict:
        if normalized.subscription_id is None:
            raise MetadataNormalizationError("subscription_id is required for chunk metadata")
        if normalized.profile_id is None:
            raise MetadataNormalizationError("profile_id is required for chunk metadata")
        if normalized.document_id is None:
            raise MetadataNormalizationError("document_id is required for chunk metadata")

    return normalized


def normalize_document_metadata(metadata: Dict[str, Any], *, strict: bool = False) -> NormalizedMetadata:
    document_type = _resolve_field(metadata, "document_type", aliases=("doc_type", "type"))
    doc_type = _stringify(metadata.get("doc_type"))
    if doc_type and document_type and doc_type != document_type:
        logger.error("doc_type/document_type mismatch: %s vs %s", doc_type, document_type)
        raise MetadataNormalizationError("doc_type/document_type mismatch")

    normalized = NormalizedMetadata(
        subscription_id=_resolve_field(metadata, "subscription_id", aliases=("subscriptionId",)),
        profile_id=_resolve_field(metadata, "profile_id", aliases=("profileId",)),
        profile_name=_resolve_field(metadata, "profile_name", aliases=("profileName",)),
        document_id=_resolve_field(metadata, "document_id", aliases=("documentId", "doc_id", "id", "_id")),
        file_name=_resolve_field(metadata, "file_name", aliases=("filename", "name")),
        document_type=document_type or doc_type,
        chunk_kind=None,
        section_title=None,
        page_start=None,
        page_end=None,
    )

    if strict and normalized.document_id is None:
        raise MetadataNormalizationError("document_id is required for document metadata")

    return normalized


def normalize_payload_metadata(payload: Dict[str, Any], *, strict: bool = True) -> Dict[str, Any]:
    normalized = normalize_chunk_metadata(payload, strict=strict)
    normalized_dict = normalized.to_dict()
    merged = dict(payload)
    for key, value in normalized_dict.items():
        if value is not None:
            merged[key] = value
    if normalized.document_type and _stringify(merged.get("doc_type")) in {None, ""}:
        merged["doc_type"] = normalized.document_type
    if normalized.file_name and _stringify(merged.get("filename")) in {None, ""}:
        merged["filename"] = normalized.file_name
    if normalized.file_name and _stringify(merged.get("source_file")) in {None, ""}:
        merged["source_file"] = normalized.file_name
    return merged


__all__ = [
    "ALLOWED_CHUNK_KINDS",
    "MetadataNormalizationError",
    "NormalizedMetadata",
    "normalize_chunk_metadata",
    "normalize_document_metadata",
    "normalize_payload_metadata",
]
