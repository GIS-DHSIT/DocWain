"""Standalone extraction pipeline endpoint.

POST /api/extraction/extract -- extracts content, entities, temporal spans,
domain assignment, quality grading, and optionally stores to configured databases.
"""

from src.utils.logging_utils import get_logger
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = get_logger(__name__)

extraction_router = APIRouter(prefix="/extraction", tags=["Extraction Pipeline"])

class ExtractionPipelineResponse(BaseModel):
    """Response from the standalone extraction pipeline."""
    document_id: str
    filename: str
    document_domain: str = "generic"
    document_type: str = "GENERIC"
    classification_confidence: float = 0.0
    full_text: str = ""
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    temporal_spans: List[Dict[str, Any]] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)
    document_summary: str = ""
    quality_grade: str = ""
    quality_score: float = 0.0
    page_count: int = 0
    extraction_time_ms: float = 0.0
    stored: bool = False
    storage_targets: List[str] = Field(default_factory=list)

@extraction_router.post("/extract", response_model=ExtractionPipelineResponse)
async def extract_document_endpoint(
    file: UploadFile = File(...),
    subscription_id: str = Form(...),
    profile_id: str = Form(...),
    document_id: Optional[str] = Form(None),
    enable_deep_analysis: bool = Form(True),
    enable_thinking: bool = Form(False),
    store_to_db: bool = Form(True),
    target_db: str = Form("all"),
) -> ExtractionPipelineResponse:
    """Standalone extraction pipeline.

    Extracts text, tables, images, entities, temporal spans, domain,
    quality grade from any document. Optionally stores to MongoDB + Qdrant.
    """
    try:
        from src.api.config import Config
        cfg = getattr(Config, "ExtractionPipeline", None)
        if cfg and not getattr(cfg, "ENABLED", True):
            raise HTTPException(status_code=503, detail="Extraction pipeline is disabled")
    except HTTPException:
        raise
    except Exception:
        pass

    start_time = time.time()
    doc_id = document_id or str(uuid.uuid4())
    fname = file.filename or "uploaded_document"

    # Read file bytes
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    # Check file size
    try:
        from src.api.config import Config
        max_mb = getattr(Config.ExtractionPipeline, "MAX_FILE_SIZE_MB", 100)
        if len(file_bytes) > max_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {max_mb}MB limit")
    except HTTPException:
        raise
    except Exception:
        pass

    response = ExtractionPipelineResponse(document_id=doc_id, filename=fname)

    # Step 1: Raw extraction
    try:
        from src.api.dataHandler import fileProcessor
        extracted = fileProcessor(file_bytes, fname, content_type=file.content_type or "")
        if not extracted:
            raise HTTPException(status_code=422, detail="No content could be extracted from file")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}")

    # Normalize extracted content
    try:
        from src.api.extraction_service import _sanitize_raw_text_fields
        extracted = _sanitize_raw_text_fields(extracted) if isinstance(extracted, dict) else extracted
    except Exception:
        pass

    # Extract full text
    full_text = ""
    sections_list: List[Dict[str, Any]] = []
    tables_list: List[Dict[str, Any]] = []
    figures_list: List[Dict[str, Any]] = []
    page_count = 0

    if isinstance(extracted, dict):
        for _fn, content in extracted.items():
            if isinstance(content, dict):
                full_text = content.get("full_text") or content.get("text") or ""
                for sec in content.get("sections") or []:
                    if isinstance(sec, dict):
                        sections_list.append(sec)
                        page_count = max(page_count, sec.get("end_page") or sec.get("start_page") or 0)
                for tbl in content.get("tables") or []:
                    if isinstance(tbl, dict):
                        tables_list.append(tbl)
                    elif hasattr(tbl, "text"):
                        tables_list.append({"text": tbl.text, "page": getattr(tbl, "page", None)})
                for fig in content.get("figures") or []:
                    fig_dict: Dict[str, Any] = {}
                    if isinstance(fig, dict):
                        fig_dict = fig
                    elif hasattr(fig, "caption"):
                        fig_dict = {
                            "page": getattr(fig, "page", None),
                            "caption": getattr(fig, "caption", ""),
                            "is_diagram": getattr(fig, "is_diagram", False),
                            "diagram_type": getattr(fig, "diagram_type", None),
                        }
                    if fig_dict:
                        figures_list.append(fig_dict)
                break

    response.full_text = full_text[:50000]  # Cap at 50k chars in response
    response.sections = sections_list
    response.tables = tables_list
    response.figures = figures_list
    response.page_count = max(page_count, 1)

    # Step 2: Document identification
    doc_type = "GENERIC"
    try:
        from src.doc_understanding.identify import identify_document
        docs_for_id = extracted if isinstance(extracted, dict) else {fname: extracted}
        for _fn, _content in docs_for_id.items():
            identification = identify_document(extracted=_content, filename=_fn)
            doc_type = identification.document_type
            response.document_type = doc_type
            break
    except Exception as exc:
        logger.debug("Document identification failed: %s", exc)

    # Step 3: Deep analysis (entities, temporal, quality, domain signals)
    deep_result = None
    if enable_deep_analysis:
        try:
            from src.doc_understanding.deep_analyzer import deep_analyze
            docs_for_deep = extracted if isinstance(extracted, dict) else {fname: extracted}
            for _fn, _content in docs_for_deep.items():
                deep_result = deep_analyze(_content)
                response.entities = [e.to_dict() for e in deep_result.entities[:200]]
                response.temporal_spans = deep_result.temporal_spans[:50]
                response.quality_grade = deep_result.quality_grade
                response.quality_score = deep_result.quality_score
                break
        except Exception as exc:
            logger.debug("Deep analysis failed: %s", exc)

    # Step 4: Document understanding (summary, key facts, intent tags)
    try:
        from src.doc_understanding.understand import understand_document
        docs_for_understand = extracted if isinstance(extracted, dict) else {fname: extracted}
        for _fn, _content in docs_for_understand.items():
            understanding = understand_document(extracted=_content, doc_type=doc_type)
            response.document_summary = understanding.get("document_summary", "")
            response.key_facts = understanding.get("key_facts", [])
            break
    except Exception as exc:
        logger.debug("Document understanding failed: %s", exc)

    # Step 5: Domain assignment
    try:
        from src.api.extraction_service import _resolve_authoritative_domain
        domain_signals = deep_result.domain_signals if deep_result else {}
        structured_domain = ""
        domain_result = _resolve_authoritative_domain(
            domain_signals=domain_signals,
            doc_type=doc_type,
            structured_domain=structured_domain,
        )
        response.document_domain = domain_result.get("domain", "generic")
        response.classification_confidence = domain_result.get("confidence", 0.0)
    except Exception as exc:
        logger.debug("Domain assignment failed: %s", exc)

    # Step 6: Optional storage
    storage_targets: List[str] = []
    if store_to_db and target_db != "none":
        # MongoDB storage
        if target_db in ("mongodb", "all"):
            try:
                from src.api.document_status import init_document_record, update_document_fields
                init_document_record(
                    document_id=doc_id,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    doc_type=doc_type,
                    filename=fname,
                    content_type=file.content_type,
                    size=len(file_bytes),
                )
                # Store key fields
                update_fields: Dict[str, Any] = {
                    "document_domain": response.document_domain,
                    "document_summary": response.document_summary[:2000] if response.document_summary else "",
                    "quality_grade": response.quality_grade,
                    "quality_score": response.quality_score,
                    "extraction_pipeline": True,
                }
                update_document_fields(doc_id, update_fields)
                storage_targets.append("mongodb")
            except Exception as exc:
                logger.warning("MongoDB storage failed: %s", exc)

        # HITL: Embedding requires screening first. Do not auto-embed.
        # User must manually trigger screening (POST /api/gateway/screen)
        # then embedding (POST /api/documents/embed) after verification.
        if target_db in ("qdrant", "all"):
            logger.info("HITL: skipping auto-embed for %s; user must trigger screening then embedding", doc_id)

        # Profile domain refresh
        try:
            from src.profiles.profile_domain_tagger import refresh_profile_domain_on_document_change
            if subscription_id and profile_id:
                refresh_profile_domain_on_document_change(subscription_id, profile_id)
        except Exception:
            pass

    response.stored = bool(storage_targets)
    response.storage_targets = storage_targets
    response.extraction_time_ms = round((time.time() - start_time) * 1000, 1)

    return response
