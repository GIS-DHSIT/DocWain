import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import DocumentListResponse
from src.documents.document_store_adapter import list_documents
from src.api.blob_store import BlobConfigurationError
from src.api.document_status import get_document_record, get_training_progress, get_documents_collection
from src.api.embedding_service import embed_documents, embedding_integrity_report
from src.api.statuses import STATUS_EXTRACTION_OR_CHUNKING_FAILED
from src.security.response_sanitizer import sanitize_user_payload

documents_router = APIRouter()


class DocumentEmbedRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="Document identifier")
    document_ids: Optional[List[str]] = Field(None, description="Multiple document identifiers")
    subscription_id: Optional[str] = Field(None, description="Optional subscription id override")
    profile_id: Optional[str] = Field(None, description="Optional profile id override")
    doc_type: Optional[str] = Field(None, description="Optional document type override")
    max_blobs: Optional[int] = Field(None, description="Optional max blobs to process per request")


class DocumentIntegrityRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="Document identifier")
    document_ids: Optional[List[str]] = Field(None, description="Multiple document identifiers")
    subscription_id: Optional[str] = Field(None, description="Optional subscription id override")
    profile_id: Optional[str] = Field(None, description="Optional profile id override")
    limit: Optional[int] = Field(None, description="Max documents to inspect when ids are not provided")


@documents_router.get("/documents", response_model=DocumentListResponse)
def get_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, description="Optional name search query"),
    doc_type: str | None = Query(None, description="Optional document type filter"),
    profile_id: str | None = Query(None, description="Optional profile id filter"),
    created_after: str | None = Query(None, description="ISO timestamp filter"),
    created_before: str | None = Query(None, description="ISO timestamp filter"),
):
    """
    List documents with stable pagination and optional filters.
    Returns doc_id, document_name, doc_type, profile_id, subscription_id, created_at, updated_at when available.
    """
    try:
        items, total = list_documents(
            limit=limit,
            offset=offset,
            q=q,
            doc_type=doc_type,
            profile_id=profile_id,
            created_after=created_after,
            created_before=created_before,
        )
        return {
            "items": items,
            "limit": limit,
            "offset": offset,
            "total": total,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "documents_list_failed", "message": str(exc)}},
        )


@documents_router.post("/documents/embed")
def embed_document(request: DocumentEmbedRequest = Body(...)):
    try:
        result = embed_documents(
            document_id=request.document_id,
            document_ids=request.document_ids,
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
            doc_type=request.doc_type,
            max_blobs=request.max_blobs,
        )
        sanitized = sanitize_user_payload(result)
        if request.document_id and not request.document_ids:
            documents = sanitized.get("documents") or []
            if documents:
                doc_result = documents[0]
                if doc_result.get("failed_reason") == "extraction_or_chunking_failed":
                    diagnostics = doc_result.get("diagnostics") or {}
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "status": STATUS_EXTRACTION_OR_CHUNKING_FAILED,
                            "document_id": doc_result.get("document_id"),
                            "error": {
                                "code": "extraction_or_chunking_failed",
                                "message": doc_result.get("error_message") or "chunking failed",
                            },
                            "diagnostics": diagnostics,
                        },
                    )
        return sanitized
    except HTTPException:
        raise
    except BlobConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@documents_router.post("/documents/embed/report")
def embed_report(request: DocumentIntegrityRequest = Body(...)):
    try:
        result = embedding_integrity_report(
            document_id=request.document_id,
            document_ids=request.document_ids,
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
            limit=request.limit,
        )
        return sanitize_user_payload(result)
    except BlobConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Training status: polling + SSE streaming
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = {
    "TRAINING_COMPLETED", "TRAINING_FAILED",
    "TRAINING_BLOCKED_SECURITY", "TRAINING_BLOCKED_CONFIDENTIAL",
    "EXTRACTION_OR_CHUNKING_FAILED",
}


@documents_router.get("/documents/{document_id}/training-status")
def get_training_status(document_id: str):
    """Poll single document training status + real-time progress from Redis."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    progress = get_training_progress(document_id)

    return {
        "document_id": document_id,
        "status": record.get("status"),
        "extraction": record.get("extraction", {}),
        "embedding": record.get("embedding", {}),
        "training_started_at": record.get("training_started_at"),
        "trained_at": record.get("trained_at"),
        "training_error": record.get("training_error"),
        "progress": progress,
    }


@documents_router.get("/documents/training-status/batch")
def get_batch_training_status(
    subscription_id: str | None = Query(None),
    profile_id: str | None = Query(None),
    status_filter: str | None = Query(None, description="Filter by status"),
):
    """Batch training status for all documents matching filters."""
    collection = get_documents_collection()
    query: dict = {}
    if subscription_id:
        query["subscription_id"] = subscription_id
    if profile_id:
        query["profile_id"] = profile_id
    if status_filter:
        query["status"] = status_filter
    else:
        query["status"] = {"$in": [
            "UNDER_REVIEW", "EXTRACTION_COMPLETED", "SCREENING_COMPLETED",
            "TRAINING_STARTED", "TRAINING_COMPLETED", "TRAINING_FAILED",
            "TRAINING_PARTIALLY_COMPLETED", "EXTRACTION_OR_CHUNKING_FAILED",
        ]}

    docs = list(collection.find(query, {
        "document_id": 1, "status": 1, "source_file": 1,
        "extraction": 1, "embedding": 1,
        "training_started_at": 1, "trained_at": 1, "training_error": 1,
        "updated_at": 1, "created_at": 1,
    }).sort("updated_at", -1).limit(100))

    for doc in docs:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        if doc.get("status") == "TRAINING_STARTED":
            doc["progress"] = get_training_progress(doc_id)
        doc.pop("_id", None)

    return {"documents": docs, "total": len(docs)}


@documents_router.get("/documents/{document_id}/training-status/stream")
async def stream_training_status(document_id: str):
    """SSE endpoint for real-time training progress updates for a single document."""

    async def event_generator():
        from src.api.dw_newron import get_redis_client
        sync_client = get_redis_client()
        if not sync_client:
            yield {"event": "error", "data": json.dumps({"error": "Redis unavailable"})}
            return

        pubsub = sync_client.pubsub()
        pubsub.subscribe("dw:training:events")

        # Send initial state
        record = get_document_record(document_id)
        if record:
            progress = get_training_progress(document_id)
            initial = {
                "document_id": document_id,
                "status": record.get("status"),
                "progress": progress,
            }
            yield {"event": "status", "data": json.dumps(initial, default=str)}

        try:
            while True:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        if data.get("document_id") == document_id:
                            yield {"event": "progress", "data": json.dumps(data)}
                            if data.get("stage") in ("completed", "failed"):
                                yield {"event": "done", "data": json.dumps(data)}
                                return
                    except (json.JSONDecodeError, TypeError):
                        pass
                else:
                    yield {"event": "heartbeat", "data": ""}

                # Safety: check MongoDB for terminal status
                record = get_document_record(document_id)
                if record and record.get("status") in _TERMINAL_STATUSES:
                    final = {
                        "document_id": document_id,
                        "status": record.get("status"),
                        "stage": "completed" if record.get("status") == "TRAINING_COMPLETED" else "failed",
                        "progress": 1.0 if record.get("status") == "TRAINING_COMPLETED" else 0.0,
                    }
                    yield {"event": "done", "data": json.dumps(final, default=str)}
                    return

                await asyncio.sleep(0.5)
        finally:
            pubsub.unsubscribe("dw:training:events")
            pubsub.close()

    return EventSourceResponse(event_generator())


@documents_router.get("/documents/training-status/stream")
async def stream_all_training_status(
    subscription_id: str | None = Query(None),
    profile_id: str | None = Query(None),
):
    """SSE endpoint for all active training progress (dashboard use case)."""

    async def event_generator():
        from src.api.dw_newron import get_redis_client
        sync_client = get_redis_client()
        if not sync_client:
            yield {"event": "error", "data": json.dumps({"error": "Redis unavailable"})}
            return

        pubsub = sync_client.pubsub()
        pubsub.subscribe("dw:training:events")

        try:
            while True:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        yield {"event": "progress", "data": json.dumps(data)}
                    except (json.JSONDecodeError, TypeError):
                        pass
                else:
                    yield {"event": "heartbeat", "data": ""}
                await asyncio.sleep(0.5)
        finally:
            pubsub.unsubscribe("dw:training:events")
            pubsub.close()

    return EventSourceResponse(event_generator())
