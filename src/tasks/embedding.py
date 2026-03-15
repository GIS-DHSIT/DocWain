"""Embedding pipeline Celery task."""

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import (
    update_stage, update_pipeline_status, append_audit_log
)
from src.api.statuses import (
    STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED,
    PIPELINE_EMBEDDING_IN_PROGRESS, PIPELINE_TRAINING_COMPLETED,
    PIPELINE_EMBEDDING_FAILED
)
import logging

logger = logging.getLogger(__name__)


@app.task(bind=True, name="src.tasks.embedding.embed_document",
          max_retries=3, soft_time_limit=1500)
def embed_document(self, document_id: str, subscription_id: str,
                   profile_id: str):
    """Generate embeddings for validated document content."""
    try:
        update_stage(document_id, "embedding", STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        update_pipeline_status(document_id, PIPELINE_EMBEDDING_IN_PROGRESS)
        append_audit_log(document_id, "EMBEDDING_STARTED", by="user",
                        celery_task_id=self.request.id)

        # TODO: Phase 5 implementation
        logger.info(f"Embedding task stub called for document {document_id}")

    except SoftTimeLimitExceeded:
        error = {"message": "Embedding timed out", "code": "TIMEOUT"}
        update_stage(document_id, "embedding", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EMBEDDING_FAILED)
        append_audit_log(document_id, "EMBEDDING_FAILED", error="timeout")

    except Exception as exc:
        error = {"message": str(exc), "code": "EMBEDDING_ERROR"}
        update_stage(document_id, "embedding", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EMBEDDING_FAILED)
        append_audit_log(document_id, "EMBEDDING_FAILED", error=str(exc))
        self.retry(exc=exc)
