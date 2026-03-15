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

    if record.get("pipeline_status") != PIPELINE_EXTRACTION_COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot screen: pipeline_status is {record.get('pipeline_status')}, "
                   f"expected EXTRACTION_COMPLETED"
        )

    subscription_id = record["subscription_id"]
    profile_id = record["profile_id"]

    task = screen_document.delay(document_id, subscription_id, profile_id)
    append_audit_log(document_id, "SCREENING_TRIGGERED", by="user",
                    celery_task_id=task.id)

    return {"document_id": document_id, "status": "SCREENING_IN_PROGRESS",
            "task_id": task.id}


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

    if record.get("pipeline_status") != PIPELINE_SCREENING_COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot embed: pipeline_status is {record.get('pipeline_status')}, "
                   f"expected SCREENING_COMPLETED"
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
