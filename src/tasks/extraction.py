"""Extraction pipeline Celery task."""

import json
import time

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import (
    get_document_record, update_stage, update_pipeline_status, append_audit_log
)
from src.api.statuses import (
    STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED,
    PIPELINE_EXTRACTION_IN_PROGRESS, PIPELINE_EXTRACTION_COMPLETED,
    PIPELINE_EXTRACTION_FAILED
)
import logging

logger = logging.getLogger(__name__)


def _download_document_bytes(document_id: str, source_file: str) -> bytes:
    """Download raw document bytes from Azure Blob storage."""
    from src.api.blob_content_store import get_blob_client

    container = get_blob_client()
    blob_name = f"raw/{document_id}/{source_file}"
    blob_client = container.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def _extract_text_content(document_bytes: bytes, source_file: str,
                          content_type: str = "") -> str:
    """Use fileProcessor to extract basic text for the semantic pipeline."""
    from src.api.dataHandler import fileProcessor

    try:
        extracted = fileProcessor(document_bytes, source_file,
                                  content_type=content_type)
        # fileProcessor returns a dict keyed by filename/sheet
        # Combine all text values into a single string
        texts = []
        for key, value in extracted.items():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                # ExtractedDocument or dict with text fields
                text = value.get("full_text") or value.get("text") or ""
                if text:
                    texts.append(str(text))
                else:
                    texts.append(str(value))
            elif isinstance(value, list):
                texts.append("\n".join(str(item) for item in value))
            else:
                # Dataclass or object with text attribute
                text = getattr(value, "full_text", None) or getattr(value, "text", None)
                if text:
                    texts.append(str(text))
                else:
                    texts.append(str(value))
        return "\n\n".join(texts)
    except Exception as exc:
        logger.warning("fileProcessor failed for %s: %s — proceeding with empty text",
                       source_file, exc)
        return ""


def _upload_extraction_json(subscription_id: str, profile_id: str,
                            document_id: str, result_dict: dict) -> str:
    """Upload full extraction result JSON to Azure Blob.

    Blob path: {subscription_id}/{profile_id}/{document_id}/extraction.json
    Returns the blob path.
    """
    from src.api.blob_content_store import get_blob_client
    from azure.storage.blob import ContentSettings

    container = get_blob_client()
    blob_path = f"{subscription_id}/{profile_id}/{document_id}/extraction.json"
    blob_client = container.get_blob_client(blob_path)

    payload = json.dumps(result_dict, default=str, ensure_ascii=False).encode("utf-8")
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata={
            "docwain_artifact": "true",
            "document_id": document_id,
            "type": "extraction_result",
            "version": "v1",
        },
        content_settings=ContentSettings(content_type="application/json"),
    )
    logger.info("Uploaded extraction JSON: %s (%d bytes)", blob_path, len(payload))
    return blob_path


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
    start_time = time.time()

    try:
        update_stage(document_id, "extraction", status=STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_IN_PROGRESS)
        append_audit_log(document_id, "EXTRACTION_STARTED",
                         celery_task_id=self.request.id)

        # 1. Get document record from MongoDB
        doc_record = get_document_record(document_id)
        if not doc_record:
            raise ValueError(f"Document record not found for {document_id}")

        source_file = doc_record.get("source_file") or doc_record.get("filename") or "document"
        file_type = doc_record.get("doc_type") or doc_record.get("file_type") or "pdf"
        content_type = doc_record.get("content_type") or ""

        # 2. Download document bytes from Azure Blob
        try:
            document_bytes = _download_document_bytes(document_id, source_file)
            logger.info("Downloaded document %s (%d bytes)", document_id, len(document_bytes))
        except Exception as exc:
            raise RuntimeError(f"Failed to download document from blob: {exc}") from exc

        # 3. Extract basic text using fileProcessor for the semantic pipeline
        text_content = _extract_text_content(document_bytes, source_file,
                                             content_type=content_type)
        logger.info("Text extraction for %s: %d chars", document_id, len(text_content))

        # 4. Run ExtractionEngine.extract() with all three pipelines
        from src.extraction import ExtractionEngine

        engine = ExtractionEngine()
        result = engine.extract(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_bytes=document_bytes,
            file_type=file_type,
            text_content=text_content,
        )

        # 5. Store full extraction result JSON to Azure Blob
        result_dict = result.to_dict()
        blob_path = _upload_extraction_json(
            subscription_id, profile_id, document_id, result_dict
        )

        # 6. Store summary to MongoDB via update_stage()
        summary = result.to_summary()
        summary["blob_path"] = blob_path

        duration_seconds = round(time.time() - start_time, 2)
        summary["duration_seconds"] = duration_seconds

        update_stage(document_id, "extraction", status=STAGE_COMPLETED,
                     summary=summary, blob_path=blob_path, error=None)

        # 7. Update pipeline status to EXTRACTION_COMPLETED
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_COMPLETED)
        append_audit_log(document_id, "EXTRACTION_COMPLETED",
                         duration_seconds=duration_seconds,
                         entity_count=summary.get("entity_count", 0),
                         table_count=summary.get("table_count", 0),
                         blob_path=blob_path)

        logger.info(
            "Extraction completed for %s in %.2fs: %d entities, %d tables, confidence=%.2f",
            document_id, duration_seconds,
            summary.get("entity_count", 0),
            summary.get("table_count", 0),
            summary.get("extraction_confidence", 0.0),
        )

    except SoftTimeLimitExceeded:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": "Extraction timed out", "code": "TIMEOUT"}
        update_stage(document_id, "extraction", status=STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_FAILED)
        append_audit_log(document_id, "EXTRACTION_FAILED",
                         error="timeout", duration_seconds=duration_seconds)

    except Exception as exc:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": str(exc), "code": "EXTRACTION_ERROR"}
        update_stage(document_id, "extraction", status=STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_FAILED)
        append_audit_log(document_id, "EXTRACTION_FAILED",
                         error=str(exc), duration_seconds=duration_seconds)
        self.retry(exc=exc)
