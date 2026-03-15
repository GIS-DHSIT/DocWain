"""Extraction pipeline Celery task."""

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import (
    update_stage, update_pipeline_status, append_audit_log
)
from src.api.statuses import (
    STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED,
    PIPELINE_EXTRACTION_IN_PROGRESS, PIPELINE_EXTRACTION_COMPLETED,
    PIPELINE_EXTRACTION_FAILED
)
import logging

logger = logging.getLogger(__name__)


@app.task(bind=True, name="src.tasks.extraction.extract_document",
          max_retries=3, soft_time_limit=1500)
def extract_document(self, document_id: str, subscription_id: str,
                     profile_id: str):
    """Run three-model parallel extraction on an uploaded document.

    Models:
    - Triton: LayoutLM/DocFormer (structural)
    - Ollama qwen3:14b (semantic)
    - Ollama glm-ocr (vision)

    Results merged, stored to Azure Blob. Summary to MongoDB.
    """
    try:
        update_stage(document_id, "extraction", STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_IN_PROGRESS)
        append_audit_log(document_id, "EXTRACTION_STARTED",
                        celery_task_id=self.request.id)

        # TODO: Phase 2 implementation
        # 1. Load document from Azure Blob
        # 2. Run structural pipeline (Triton: LayoutLM/DocFormer)
        # 3. Run semantic pipeline (Ollama: qwen3:14b)
        # 4. Run vision pipeline (Ollama: glm-ocr)
        # 5. Merge and reconcile outputs
        # 6. Store extraction result to Azure Blob
        # 7. Store summary to MongoDB

        logger.info(f"Extraction task stub called for document {document_id}")

    except SoftTimeLimitExceeded:
        error = {"message": "Extraction timed out", "code": "TIMEOUT"}
        update_stage(document_id, "extraction", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_FAILED)
        append_audit_log(document_id, "EXTRACTION_FAILED", error="timeout")

    except Exception as exc:
        error = {"message": str(exc), "code": "EXTRACTION_ERROR"}
        update_stage(document_id, "extraction", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_FAILED)
        append_audit_log(document_id, "EXTRACTION_FAILED", error=str(exc))
        self.retry(exc=exc)
