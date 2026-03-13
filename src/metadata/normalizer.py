from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from src.utils.payload_utils import get_canonical_text, token_count

logger = get_logger(__name__)

ALLOWED_CHUNK_KINDS = {
    "section_text",
    "table_text",
    "table",
    "image_caption",
    "doc_summary",
    "section_summary",
    "structured_field",
    "fact",
    "entity_fact",
    "diagram",
    "doc_context",
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
    ingestion_source: Optional[str] = None
    metadata_warnings: Optional[Tuple[str, ...]] = None

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
            "ingestion_source": self.ingestion_source,
            "metadata_warnings": list(self.metadata_warnings or ()),
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

def infer_chunk_kind(metadata: Dict[str, Any], *, fallback: str = "section_text") -> str:
    return _infer_chunk_kind(metadata, fallback=fallback)

def normalize_chunk_kind(
    metadata: Dict[str, Any],
    *,
    fallback: str = "section_text",
    strict: bool = True,
) -> str:
    chunk_kind = _infer_chunk_kind(metadata, fallback=fallback)
    if chunk_kind not in ALLOWED_CHUNK_KINDS:
        if strict:
            logger.error("Invalid chunk_kind: %s", chunk_kind)
            raise MetadataNormalizationError(f"Invalid chunk_kind: {chunk_kind}")
        logger.debug("Unknown chunk_kind '%s'; falling back to %s", chunk_kind, fallback)
        return fallback
    return chunk_kind

def normalize_chunk_metadata(metadata: Dict[str, Any], *, strict: bool = True) -> NormalizedMetadata:
    document_type = _resolve_field(metadata, "document_type", aliases=("type",), required=False)
    doc_type = _stringify(metadata.get("doc_type"))
    warnings: list[str] = []
    if doc_type and document_type and doc_type != document_type:
        warnings.append(f"document_type_conflict:{document_type} vs {doc_type}")
    chunk_kind = normalize_chunk_kind(metadata, strict=strict)

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
        ingestion_source=None,
        metadata_warnings=tuple(warnings) if warnings else None,
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
    document_type = _resolve_field(metadata, "document_type", aliases=("type",), required=False)
    doc_type = _stringify(metadata.get("doc_type"))
    warnings: list[str] = []
    if doc_type and document_type and doc_type != document_type:
        warnings.append(f"document_type_conflict:{document_type} vs {doc_type}")

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
        ingestion_source=None,
        metadata_warnings=tuple(warnings) if warnings else None,
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

def normalize_chunk_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ingestion chunk payloads using the canonical metadata normalizer."""
    canonical_text = get_canonical_text({"text": raw.get("text"), "content": raw.get("content")})
    chunk_hash = hashlib.sha256((canonical_text or "").encode("utf-8")).hexdigest() if canonical_text else None

    normalized_meta = normalize_payload_metadata(raw, strict=False)
    chunk_kind = normalize_chunk_kind(
        {"chunk_kind": normalized_meta.get("chunk_kind"), "chunk_type": normalized_meta.get("chunk_type")},
        strict=False,
    )

    payload: Dict[str, Any] = {
        "subscription_id": _stringify(normalized_meta.get("subscription_id") or raw.get("subscription_id")),
        "profile_id": _stringify(normalized_meta.get("profile_id") or raw.get("profile_id")),
        "document_id": _stringify(normalized_meta.get("document_id") or raw.get("document_id")),
        "filename": _stringify(
            normalized_meta.get("file_name")
            or raw.get("filename")
            or raw.get("source_name")
            or raw.get("source_file")
        ),
        "source_type": _stringify(raw.get("source_type")),
        "domain": _stringify(raw.get("domain")) or "generic",
        "chunk_kind": chunk_kind,
        "section_path": raw.get("section_path") or [],
        "page_range": raw.get("page_range") or [raw.get("page_start"), raw.get("page_end")],
        "anchors": raw.get("anchors") or [],
        "entity_types_present": raw.get("entity_types_present") or [],
        "checksum_sha256": _stringify(raw.get("checksum_sha256")),
        "source_version": _stringify(raw.get("source_version")),
        "ingest_timestamp": raw.get("ingest_timestamp") or time.time(),
        "chunk_id": _stringify(raw.get("chunk_id")),
        "chunk_hash": chunk_hash,
        "text": canonical_text,
        "canonical_text": canonical_text,
        "canonical_text_len": len(canonical_text) if canonical_text else 0,
        "canonical_token_count": token_count(canonical_text) if canonical_text else 0,
        "text_data": {"clean": canonical_text, "raw": raw.get("text") if raw.get("text") != canonical_text else None},
    }

    payload["source_name"] = payload.get("filename")
    payload["doc_domain"] = payload.get("domain")
    payload["section_path"] = payload.get("section_path")
    if payload.get("page_range"):
        payload["page_start"] = _intify(payload["page_range"][0]) if isinstance(payload["page_range"], list) else None
        payload["page_end"] = _intify(payload["page_range"][1]) if isinstance(payload["page_range"], list) else None
        payload["page"] = payload["page_start"]

    return payload

def normalize_ingestion_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metadata for ingestion/embedding without hard-failing on conflicts."""
    warnings: list[str] = []
    doc_type_raw = _stringify(metadata.get("doc_type"))
    document_type_raw = _stringify(metadata.get("document_type"))

    connector_sources = {"LOCAL", "S3", "AZURE", "BLOB", "URL"}
    ingestion_source = doc_type_raw if doc_type_raw and doc_type_raw.upper() in connector_sources else None

    if document_type_raw:
        resolved_document_type = document_type_raw
    else:
        resolved_document_type = None
        if doc_type_raw and doc_type_raw.upper() not in connector_sources:
            resolved_document_type = doc_type_raw

    if document_type_raw and doc_type_raw and doc_type_raw != document_type_raw:
        warnings.append(f"document_type_conflict:{document_type_raw} vs {doc_type_raw}")

    normalized = {
        "subscription_id": _stringify(metadata.get("subscription_id") or metadata.get("subscriptionId")),
        "profile_id": _stringify(metadata.get("profile_id") or metadata.get("profileId")),
        "document_id": _stringify(metadata.get("document_id") or metadata.get("documentId") or metadata.get("doc_id")),
        "document_name": _stringify(
            metadata.get("document_name")
            or metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("source_file")
            or metadata.get("source_name")
        ),
        "chunk_id": _stringify(metadata.get("chunk_id")),
        "chunk_kind": _stringify(metadata.get("chunk_kind") or metadata.get("chunk_type")),
        "document_type": resolved_document_type or "generic",
        "ingestion_source": ingestion_source,
        "source_type": ingestion_source,
        "metadata_warnings": warnings or None,
    }
    if doc_type_raw:
        normalized["doc_type"] = doc_type_raw
    return normalized

__all__ = [
    "ALLOWED_CHUNK_KINDS",
    "infer_chunk_kind",
    "normalize_chunk_kind",
    "MetadataNormalizationError",
    "NormalizedMetadata",
    "normalize_chunk_metadata",
    "normalize_chunk_payload",
    "normalize_document_metadata",
    "normalize_ingestion_metadata",
    "normalize_payload_metadata",
]
