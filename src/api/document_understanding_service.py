from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.api.content_store import load_extracted_pickle
from src.api.document_status import set_error, update_document_fields, update_stage
from src.api.embedding_service import embed_documents
from src.api.extraction_service import extract_uploaded_document
from src.doc_understanding import build_content_map, identify_document, understand_document
from src.metadata.normalizer import MetadataNormalizationError, normalize_document_metadata
from src.profiles.profile_store import resolve_profile_name

logger = logging.getLogger(__name__)


class UnderstandingError(Exception):
    pass


def _select_extracted(extracted_payload: Any) -> Tuple[str, Any]:
    if isinstance(extracted_payload, dict) and extracted_payload:
        filename, content = next(iter(extracted_payload.items()))
        return filename, content
    raise UnderstandingError("No extracted content available")


def _update_metadata(document_id: str, metadata: Dict[str, Any]) -> None:
    update_fields: Dict[str, Any] = {}
    try:
        normalized = normalize_document_metadata(metadata, strict=False)
    except MetadataNormalizationError as exc:
        logger.error("Metadata normalization failed for %s: %s", document_id, exc)
        raise

    normalized_dict = normalized.to_dict()
    for key, value in metadata.items():
        if value is None:
            continue
        update_fields[key] = value
        update_fields[f"metadata.{key}"] = value
    for key, value in normalized_dict.items():
        if value is None:
            continue
        update_fields[key] = value
        update_fields[f"metadata.{key}"] = value
    update_document_fields(document_id, update_fields)


def run_document_understanding(
    *,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    profile_name: Optional[str] = None,
    model_name: Optional[str] = None,
    embed_after: bool = False,
) -> Dict[str, Any]:
    update_stage(
        document_id,
        "understanding",
        {"status": "IN_PROGRESS", "started_at": time.time(), "error": None},
    )

    if not profile_name:
        profile_name = resolve_profile_name(subscription_id=subscription_id, profile_id=profile_id)

    extracted_payload = load_extracted_pickle(document_id)
    filename, extracted = _select_extracted(extracted_payload)

    identification = identify_document(
        extracted=extracted,
        filename=filename,
        profile_name=profile_name,
        model_name=model_name,
    )

    metadata_update = {
        "profile_id": profile_id,
        "profile_name": profile_name,
        "subscription_id": subscription_id,
        "document_type": identification.document_type,
        "doc_type": identification.document_type,
        "doc_title": identification.doc_name,
        "doc_type_confidence": identification.confidence,
        "file_format": identification.file_format,
        "page_count": identification.page_count,
    }
    _update_metadata(document_id, metadata_update)

    content_map = build_content_map(extracted)
    update_document_fields(document_id, {"content_map": content_map})

    understanding = understand_document(extracted=extracted, doc_type=identification.document_type, model_name=model_name)

    understanding_update = {
        "document_summary": understanding.get("document_summary"),
        "section_summaries": understanding.get("section_summaries"),
        "key_entities": understanding.get("key_entities"),
        "key_facts": understanding.get("key_facts"),
        "doc_intent_tags": understanding.get("intent_tags"),
        "understanding_json": understanding,
    }
    update_document_fields(document_id, understanding_update)

    update_stage(
        document_id,
        "understanding",
        {
            "status": "COMPLETED",
            "completed_at": time.time(),
            "error": None,
            "doc_type": identification.document_type,
            "doc_type_confidence": identification.confidence,
        },
    )

    embed_result = None
    if embed_after:
        embed_result = embed_documents(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            doc_type=identification.document_type,
        )

    return {
        "document_id": document_id,
        "document_type": identification.document_type,
        "doc_type_confidence": identification.confidence,
        "content_map": content_map,
        "understanding": understanding,
        "embedding": embed_result,
    }


def extract_and_understand(
    *,
    document_id: str,
    file_bytes: bytes,
    filename: str,
    subscription_id: str,
    profile_id: str,
    profile_name: Optional[str] = None,
    content_type: Optional[str] = None,
    content_size: Optional[int] = None,
    model_name: Optional[str] = None,
    embed_after: bool = False,
) -> Dict[str, Any]:
    try:
        extract_uploaded_document(
            document_id=document_id,
            file_bytes=file_bytes,
            filename=filename,
            subscription_id=subscription_id,
            profile_id=profile_id,
            profile_name=profile_name,
            doc_type=None,
            content_type=content_type,
            content_size=content_size,
        )
    except Exception as exc:  # noqa: BLE001
        set_error(document_id, "extraction", exc)
        raise

    return run_document_understanding(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        model_name=model_name,
        embed_after=embed_after,
    )
