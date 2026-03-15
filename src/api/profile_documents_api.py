from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.document_understanding_service import run_document_understanding
from src.api.document_status import init_document_record
from src.profiles.profile_store import resolve_profile_name
from src.retrieval.profile_query import query_profile
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

profile_docs_router = APIRouter(prefix="/profiles", tags=["Profiles"])


class UnderstandRequest(BaseModel):
    subscription_id: str = Field(..., description="Subscription identifier")
    profile_name: Optional[str] = Field(None, description="Profile display name")
    model_name: Optional[str] = Field(None, description="Ollama model name override")
    embed_after: bool = Field(False, description="Whether to embed after understanding (HITL: must be False by default)")


class QueryRequest(BaseModel):
    subscription_id: str = Field(..., description="Subscription identifier")
    query: str = Field(..., description="User query")
    model_name: Optional[str] = Field(None, description="Ollama model name override")
    top_k: int = Field(6, ge=1, le=20)


@profile_docs_router.post("/{profile_id}/documents/upload", summary="Upload a document for a profile")
async def upload_document_for_profile(
    profile_id: str,
    subscription_id: str = Form(...),
    profile_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    document_id = str(uuid.uuid4())
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail={"error": {"code": "empty_file", "message": "Empty file"}})

    filename = file.filename or "document"
    content_type = file.content_type or "application/octet-stream"
    file_type = os.path.splitext(filename)[1].lstrip(".").lower() or "bin"

    # ── 1. Store raw file to Azure Blob ──────────────────────────────
    blob_url: Optional[str] = None
    try:
        from src.api.blob_content_store import get_blob_client
        from azure.storage.blob import ContentSettings

        container = get_blob_client()
        blob_name = f"raw/{document_id}/{filename}"
        blob_client = container.get_blob_client(blob_name)
        blob_client.upload_blob(
            file_bytes,
            overwrite=True,
            metadata={"document_id": document_id, "type": "raw_upload"},
            content_settings=ContentSettings(content_type=content_type),
        )
        blob_url = blob_client.url
        logger.info("Stored raw upload to blob: doc=%s blob=%s size=%d", document_id, blob_name, len(file_bytes))
    except Exception:
        logger.warning("Blob storage unavailable for doc=%s; proceeding without blob_url", document_id, exc_info=True)

    # ── 2. Create MongoDB document record ────────────────────────────
    profile_name = profile_name or resolve_profile_name(subscription_id=subscription_id, profile_id=profile_id)

    init_document_record(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        source_file=filename,
        file_type=file_type,
        content_type=content_type,
        content_size=len(file_bytes),
        blob_url=blob_url,
        created_by="user",
    )

    # ── 3. Auto-dispatch extraction via Celery (fallback: synchronous) ─
    celery_task_id: Optional[str] = None
    extraction_mode = "queued"
    try:
        from src.tasks.extraction import extract_document
        task = extract_document.delay(document_id, subscription_id, profile_id)
        celery_task_id = task.id
        logger.info("Dispatched extraction task: doc=%s task_id=%s", document_id, celery_task_id)
    except Exception:
        logger.warning("Celery unavailable for doc=%s; falling back to synchronous extraction", document_id, exc_info=True)
        extraction_mode = "sync"
        try:
            from src.api.extraction_service import extract_uploaded_document
            extract_uploaded_document(
                document_id=document_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
                profile_name=profile_name,
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
                content_size=len(file_bytes),
                doc_type=file_type,
            )
        except Exception:
            logger.error("Synchronous extraction also failed for doc=%s", document_id, exc_info=True)
            extraction_mode = "failed"

    return {
        "document_id": document_id,
        "status": "EXTRACTION_QUEUED" if extraction_mode == "queued" else "EXTRACTION_IN_PROGRESS" if extraction_mode == "sync" else "UPLOADED",
        "celery_task_id": celery_task_id,
        "blob_url": blob_url,
        "extraction_mode": extraction_mode,
    }


@profile_docs_router.post("/{profile_id}/documents/{document_id}/understand", summary="Run understanding on a document")
def understand_document(
    profile_id: str,
    document_id: str,
    payload: UnderstandRequest = Body(...),
):
    profile_name = payload.profile_name or resolve_profile_name(
        subscription_id=payload.subscription_id, profile_id=profile_id
    )
    result = run_document_understanding(
        document_id=document_id,
        subscription_id=payload.subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        model_name=payload.model_name,
        embed_after=payload.embed_after,
    )
    return result


@profile_docs_router.post("/{profile_id}/query", summary="Query a profile")
def query_profile_endpoint(profile_id: str, payload: QueryRequest = Body(...)):
    response = query_profile(
        subscription_id=payload.subscription_id,
        profile_id=profile_id,
        query=payload.query,
        model_name=payload.model_name,
        top_k=payload.top_k,
    )
    return response
