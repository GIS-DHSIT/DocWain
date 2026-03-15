"""Screening pipeline Celery task."""

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import (
    update_stage, update_pipeline_status, append_audit_log
)
from src.api.statuses import (
    STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED,
    PIPELINE_SCREENING_IN_PROGRESS, PIPELINE_SCREENING_COMPLETED,
    PIPELINE_SCREENING_FAILED
)
import logging

logger = logging.getLogger(__name__)


@app.task(bind=True, name="src.tasks.screening.screen_document",
          max_retries=3, soft_time_limit=1500)
def screen_document(self, document_id: str, subscription_id: str,
                    profile_id: str):
    """Run plugin-based screening on extracted document.

    1. Load extraction from Azure Blob
    2. Run mandatory security plugins (PII, secrets, legality)
    3. Run profile-configured plugins
    4. Store full report to Azure Blob, summary to MongoDB
    5. Auto-dispatch KG building (async, independent)
    """
    try:
        update_stage(document_id, "screening", STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        update_pipeline_status(document_id, PIPELINE_SCREENING_IN_PROGRESS)
        append_audit_log(document_id, "SCREENING_STARTED", by="user",
                        celery_task_id=self.request.id)

        # TODO: Phase 3 implementation
        logger.info(f"Screening task stub called for document {document_id}")

    except SoftTimeLimitExceeded:
        error = {"message": "Screening timed out", "code": "TIMEOUT"}
        update_stage(document_id, "screening", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_SCREENING_FAILED)
        append_audit_log(document_id, "SCREENING_FAILED", error="timeout")

    except Exception as exc:
        error = {"message": str(exc), "code": "SCREENING_ERROR"}
        update_stage(document_id, "screening", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_SCREENING_FAILED)
        append_audit_log(document_id, "SCREENING_FAILED", error=str(exc))
        self.retry(exc=exc)
