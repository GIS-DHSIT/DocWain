"""Screening pipeline Celery task."""

import json
import time

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import (
    get_document_record, update_stage, update_pipeline_status, append_audit_log
)
from src.api.statuses import (
    STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED,
    PIPELINE_SCREENING_IN_PROGRESS, PIPELINE_SCREENING_COMPLETED,
    PIPELINE_SCREENING_FAILED
)
import logging

logger = logging.getLogger(__name__)


def _load_extraction_json(blob_path: str) -> dict:
    """Download and parse extraction JSON from Azure Blob."""
    from src.api.blob_content_store import get_blob_client

    container = get_blob_client()
    blob_client = container.get_blob_client(blob_path)
    raw = blob_client.download_blob().readall()
    return json.loads(raw)


def _upload_screening_json(subscription_id: str, profile_id: str,
                           document_id: str, report: dict) -> str:
    """Upload full screening report JSON to Azure Blob.

    Blob path: {subscription_id}/{profile_id}/{document_id}/screening.json
    Returns the blob path.
    """
    from src.api.blob_content_store import get_blob_client
    from azure.storage.blob import ContentSettings

    container = get_blob_client()
    blob_path = f"{subscription_id}/{profile_id}/{document_id}/screening.json"
    blob_client = container.get_blob_client(blob_path)

    payload = json.dumps(report, default=str, ensure_ascii=False).encode("utf-8")
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata={
            "docwain_artifact": "true",
            "document_id": document_id,
            "type": "screening_report",
            "version": "v1",
        },
        content_settings=ContentSettings(content_type="application/json"),
    )
    logger.info("Uploaded screening JSON: %s (%d bytes)", blob_path, len(payload))
    return blob_path


def _get_screening_config(profile_id: str) -> dict:
    """Load profile's screening_config from MongoDB."""
    from pymongo import MongoClient
    from src.api.config import Config

    client = MongoClient(Config.MongoDB.URI)
    db = client[Config.MongoDB.DB]
    profile = db["profiles"].find_one({"profile_id": profile_id})
    return (profile or {}).get("screening_config", {})


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
    start_time = time.time()

    try:
        update_stage(document_id, "screening", STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        update_pipeline_status(document_id, PIPELINE_SCREENING_IN_PROGRESS)
        append_audit_log(document_id, "SCREENING_STARTED", by="user",
                        celery_task_id=self.request.id)

        # 1. Get document record from MongoDB
        doc_record = get_document_record(document_id)
        if not doc_record:
            raise ValueError(f"Document record not found for {document_id}")

        # 2. Load extraction JSON from Azure Blob
        extraction_summary = (doc_record.get("extraction") or {}).get("summary") or {}
        blob_path = extraction_summary.get("blob_path")
        if not blob_path:
            raise ValueError(
                f"No extraction blob_path found for document {document_id}. "
                "Extraction must complete before screening."
            )
        extraction_data = _load_extraction_json(blob_path)
        logger.info("Loaded extraction data for %s from %s", document_id, blob_path)

        # 3. Load profile's screening_config from MongoDB
        screening_config = _get_screening_config(profile_id)

        # 4. Run ScreeningOrchestrator
        from src.screening.orchestrator import ScreeningOrchestrator

        orchestrator = ScreeningOrchestrator()
        report = orchestrator.run(
            document_id=document_id,
            extraction_data=extraction_data,
            document_meta=doc_record,
            profile_config=screening_config,
        )

        # 5. Store full screening report to Azure Blob
        screening_blob_path = _upload_screening_json(
            subscription_id, profile_id, document_id, report
        )

        # 6. Store summary to MongoDB via update_stage()
        summary = {
            "domain_tags": report.get("domain_tags", []),
            "doc_category": report.get("doc_category", "unknown"),
            "risk_level": report.get("risk_level", "low"),
            "entity_scores": report.get("entity_scores", {}),
            "flags": report.get("flags", []),
            "plugins_run": report.get("plugins_run", []),
            "blob_path": screening_blob_path,
        }

        duration_seconds = round(time.time() - start_time, 2)
        summary["duration_seconds"] = duration_seconds

        update_stage(document_id, "screening", status=STAGE_COMPLETED,
                     summary=summary, blob_path=screening_blob_path, error=None)

        # 7. Update pipeline status to SCREENING_COMPLETED
        update_pipeline_status(document_id, PIPELINE_SCREENING_COMPLETED)
        append_audit_log(document_id, "SCREENING_COMPLETED",
                         duration_seconds=duration_seconds,
                         risk_level=summary["risk_level"],
                         plugins_run=len(summary["plugins_run"]),
                         flags_count=len(summary["flags"]),
                         blob_path=screening_blob_path)

        logger.info(
            "Screening completed for %s in %.2fs: risk=%s, %d plugins, %d flags",
            document_id, duration_seconds,
            summary["risk_level"],
            len(summary["plugins_run"]),
            len(summary["flags"]),
        )

        # 8. Auto-dispatch KG building task (async, independent)
        from src.tasks.kg import build_knowledge_graph

        build_knowledge_graph.delay(document_id, subscription_id, profile_id)
        logger.info("Dispatched KG build task for %s", document_id)

    except SoftTimeLimitExceeded:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": "Screening timed out", "code": "TIMEOUT"}
        update_stage(document_id, "screening", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_SCREENING_FAILED)
        append_audit_log(document_id, "SCREENING_FAILED",
                         error="timeout", duration_seconds=duration_seconds)

    except Exception as exc:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": str(exc), "code": "SCREENING_ERROR"}
        update_stage(document_id, "screening", STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_SCREENING_FAILED)
        append_audit_log(document_id, "SCREENING_FAILED",
                         error=str(exc), duration_seconds=duration_seconds)
        self.retry(exc=exc)
