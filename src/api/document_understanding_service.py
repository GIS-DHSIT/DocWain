from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, Optional, Tuple

from src.api.content_store import load_extracted_pickle
from src.api.document_status import set_error, update_document_fields, update_stage
from src.api.embedding_service import embed_documents
from src.api.screening_service import promote_to_screening_completed
from src.api.extraction_service import extract_uploaded_document
from src.doc_understanding import build_content_map, identify_document
from src.metadata.normalizer import MetadataNormalizationError, normalize_document_metadata
from src.profiles.profile_store import resolve_profile_name

logger = get_logger(__name__)

class UnderstandingError(Exception):
    pass

def _select_extracted(extracted_payload: Any) -> Tuple[str, Any]:
    if isinstance(extracted_payload, dict) and extracted_payload:
        # Enriched pickle format: {raw: {filename: content}, structured: ..., ...}
        # Navigate into 'raw' to find the actual extracted document.
        raw = extracted_payload.get("raw")
        if isinstance(raw, dict) and raw:
            filename, content = next(iter(raw.items()))
            return filename, content
        # Legacy format: {filename: content} directly
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

    # Deep intelligence analysis (replaces legacy understand_document)
    from src.intelligence_v2.analyzer import DocumentAnalyzer
    from src.llm.gateway import get_llm_gateway
    from src.api.config import Config
    from pymongo import MongoClient

    llm = get_llm_gateway()
    mongo_client = MongoClient(Config.MongoDB.URI)
    mongodb = mongo_client[Config.MongoDB.DB][Config.MongoDB.DOCUMENTS]

    try:
        from src.kg.neo4j_store import Neo4jStore
        neo4j = Neo4jStore()
    except Exception:
        neo4j = None

    if neo4j is not None:
        analyzer = DocumentAnalyzer(llm_gateway=llm, neo4j_store=neo4j, mongodb=mongodb)
        intel_result = analyzer.analyze(
            document_id=document_id,
            extracted=extracted,
            subscription_id=subscription_id,
            profile_id=profile_id,
            filename=filename,
            doc_type=identification.document_type,
        )
        understanding = intel_result["intelligence"]
    else:
        # Fallback: LLM analysis without KG
        from src.intelligence_v2.summarizer import DocumentSummarizer
        from src.intelligence_v2.analyzer import _get_text
        logger.info("[UNDERSTANDING] Neo4j unavailable, using fallback LLM analysis for doc=%s", document_id)
        summarizer = DocumentSummarizer(llm_gateway=llm)
        text = _get_text(extracted)
        logger.info("[UNDERSTANDING] Extracted text length=%d for doc=%s", len(text), document_id)
        try:
            analysis = summarizer.analyze(text=text, filename=filename, doc_type=identification.document_type)
            understanding = analysis.to_dict()
            logger.info(
                "[UNDERSTANDING] Analysis complete for doc=%s: entities=%d, facts=%d, summary_len=%d",
                document_id, len(analysis.entities), len(analysis.facts), len(analysis.summary),
            )
        except Exception:
            logger.error("[UNDERSTANDING] Summarizer failed for doc=%s", document_id, exc_info=True)
            understanding = {"document_type": identification.document_type, "summary": text[:500]}
        mongodb.update_one(
            {"document_id": document_id},
            {"$set": {"intelligence": understanding, "intelligence_ready": True}},
        )

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
        promote_to_screening_completed(document_id)
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
