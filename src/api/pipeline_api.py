"""Pipeline API endpoints — UI-triggered document processing stages."""

from fastapi import APIRouter, HTTPException
from src.api.document_status import (
    get_document_record, append_audit_log
)
from src.api.statuses import (
    PIPELINE_EXTRACTION_COMPLETED, PIPELINE_SCREENING_COMPLETED
)
from src.tasks.screening import screen_document
from src.tasks.embedding import embed_document
import logging

logger = logging.getLogger(__name__)

pipeline_router = APIRouter(prefix="/documents", tags=["pipeline"])


@pipeline_router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    """Get pipeline status and per-stage summaries for a document."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": record.get("document_id"),
        "pipeline_status": record.get("pipeline_status"),
        "extraction": {
            "status": record.get("extraction", {}).get("status"),
            "summary": record.get("extraction", {}).get("summary"),
        },
        "screening": {
            "status": record.get("screening", {}).get("status"),
            "summary": record.get("screening", {}).get("summary"),
        },
        "knowledge_graph": {
            "status": record.get("knowledge_graph", {}).get("status"),
            "node_count": record.get("knowledge_graph", {}).get("node_count", 0),
            "edge_count": record.get("knowledge_graph", {}).get("edge_count", 0),
        },
        "embedding": {
            "status": record.get("embedding", {}).get("status"),
            "summary": record.get("embedding", {}).get("summary"),
        }
    }


@pipeline_router.get("/{document_id}/extraction")
async def get_extraction_summary(document_id: str):
    """Get extraction summary from MongoDB (for UI review)."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "extraction": record.get("extraction", {})
    }


@pipeline_router.get("/{document_id}/extraction/detail")
async def get_extraction_detail(document_id: str):
    """Fetch full extraction JSON from Azure Blob."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    blob_path = (record.get("extraction", {}).get("summary") or {}).get("blob_path")
    if not blob_path:
        raise HTTPException(status_code=404, detail="Extraction data not available")

    # TODO: Load from Azure Blob using blob_path
    raise HTTPException(status_code=501, detail="Azure Blob fetch not yet implemented")


@pipeline_router.post("/{document_id}/screen")
async def trigger_screening(document_id: str):
    """HITL trigger: user approved extraction, start screening."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    # Accept either new pipeline_status or legacy status field
    pipeline_status = record.get("pipeline_status", "")
    legacy_status = record.get("status", "")
    extraction_stage = record.get("extraction", {}).get("status", "")
    extraction_done = (
        pipeline_status == PIPELINE_EXTRACTION_COMPLETED
        or legacy_status == "EXTRACTION_COMPLETED"
        or extraction_stage == "COMPLETED"
    )
    if not extraction_done:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot screen: extraction not complete "
                   f"(pipeline_status={pipeline_status}, status={legacy_status}, "
                   f"extraction.status={extraction_stage})"
        )

    subscription_id = record["subscription_id"]
    profile_id = record["profile_id"]

    # Try Celery, fallback to sync screening via existing gateway
    task_id = None
    mode = "queued"
    try:
        task = screen_document.delay(document_id, subscription_id, profile_id)
        task_id = task.id
    except Exception:
        mode = "sync"
        try:
            from src.api.extraction_service import _run_auto_screening
            _run_auto_screening(document_id, doc_type=record.get("doc_type"))
        except Exception as exc:
            logger.error("Sync screening failed for doc=%s: %s", document_id, exc)
            mode = "failed"

    append_audit_log(document_id, "SCREENING_TRIGGERED", by="user",
                    celery_task_id=task_id)

    return {"document_id": document_id, "status": "SCREENING_IN_PROGRESS",
            "task_id": task_id, "mode": mode}


@pipeline_router.get("/{document_id}/screening")
async def get_screening_summary(document_id: str):
    """Get screening summary from MongoDB (for UI review)."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "screening": record.get("screening", {})
    }


@pipeline_router.get("/{document_id}/screening/detail")
async def get_screening_detail(document_id: str):
    """Fetch full screening report from Azure Blob."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    blob_path = (record.get("screening", {}).get("summary") or {}).get("blob_path")
    if not blob_path:
        raise HTTPException(status_code=404, detail="Screening data not available")

    raise HTTPException(status_code=501, detail="Azure Blob fetch not yet implemented")


@pipeline_router.post("/{document_id}/embed")
async def trigger_embedding(document_id: str):
    """HITL trigger: user approved screening, start embedding."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    # Accept either new pipeline_status or legacy status field
    pipeline_status = record.get("pipeline_status", "")
    legacy_status = record.get("status", "")
    screening_stage = record.get("screening", {}).get("status", "")
    screening_done = (
        pipeline_status == PIPELINE_SCREENING_COMPLETED
        or legacy_status == "SCREENING_COMPLETED"
        or screening_stage == "COMPLETED"
    )
    if not screening_done:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot embed: screening not complete "
                   f"(pipeline_status={pipeline_status}, screening.status={screening_stage})"
        )

    subscription_id = record["subscription_id"]
    profile_id = record["profile_id"]

    task = embed_document.delay(document_id, subscription_id, profile_id)
    append_audit_log(document_id, "EMBEDDING_TRIGGERED", by="user",
                    celery_task_id=task.id)

    return {"document_id": document_id, "status": "EMBEDDING_IN_PROGRESS",
            "task_id": task.id}


@pipeline_router.get("/{document_id}/kg/status")
async def get_kg_status(document_id: str):
    """Get KG build status (independent of pipeline)."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "knowledge_graph": record.get("knowledge_graph", {})
    }
