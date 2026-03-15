"""Knowledge graph building Celery task -- runs independently, never blocks pipeline."""

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import update_stage, append_audit_log
from src.api.statuses import STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED
import logging

logger = logging.getLogger(__name__)


@app.task(bind=True, name="src.tasks.kg.build_knowledge_graph",
          max_retries=2, soft_time_limit=1500)
def build_knowledge_graph(self, document_id: str, subscription_id: str,
                          profile_id: str):
    """Build knowledge graph from extraction + screening data.

    Runs async after screening. Never blocks the main pipeline.
    """
    try:
        update_stage(document_id, "knowledge_graph", STAGE_IN_PROGRESS)
        append_audit_log(document_id, "KG_BUILD_STARTED",
                        celery_task_id=self.request.id)

        # TODO: Phase 4 implementation
        logger.info(f"KG build task stub called for document {document_id}")

    except SoftTimeLimitExceeded:
        error = {"message": "KG build timed out", "code": "TIMEOUT"}
        update_stage(document_id, "knowledge_graph", STAGE_FAILED, error=error)
        append_audit_log(document_id, "KG_BUILD_FAILED", error="timeout")

    except Exception as exc:
        error = {"message": str(exc), "code": "KG_ERROR"}
        update_stage(document_id, "knowledge_graph", STAGE_FAILED, error=error)
        append_audit_log(document_id, "KG_BUILD_FAILED", error=str(exc))
        self.retry(exc=exc)
