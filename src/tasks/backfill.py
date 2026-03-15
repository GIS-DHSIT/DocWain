"""Backfill task -- updates Qdrant chunk payloads with KG node IDs after KG completes."""

from src.celery_app import app
from src.api.document_status import append_audit_log
import logging

logger = logging.getLogger(__name__)


@app.task(bind=True, name="src.tasks.backfill.backfill_kg_refs",
          max_retries=2, soft_time_limit=600)
def backfill_kg_refs(self, document_id: str, subscription_id: str,
                     profile_id: str):
    """Backfill KG node IDs into Qdrant chunk payloads.

    Called when KG building completes AFTER embedding was already done.
    """
    try:
        append_audit_log(document_id, "KG_BACKFILL_STARTED",
                        celery_task_id=self.request.id)

        # TODO: Phase 5 implementation
        logger.info(f"KG backfill task stub called for document {document_id}")

    except Exception as exc:
        append_audit_log(document_id, "KG_BACKFILL_FAILED", error=str(exc))
        self.retry(exc=exc)
