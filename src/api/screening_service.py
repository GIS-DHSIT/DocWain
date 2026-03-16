from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.dataHandler import update_security_screening
from src.api.document_status import get_document_record, update_document_fields, update_stage
from src.api.statuses import (
    STATUS_EMBEDDING_COMPLETED,
    STATUS_EXTRACTION_COMPLETED,
    STATUS_SCREENING_COMPLETED,
    STATUS_UNDER_REVIEW,
    STATUS_TRAINING_COMPLETED,
    STATUS_TRAINING_PARTIALLY_COMPLETED,
    STATUS_TRAINING_BLOCKED_SECURITY,
    STATUS_TRAINING_FAILED,
    STATUS_TRAINING_STARTED,
)

logger = get_logger(__name__)

DONT_DOWNGRADE_STATUSES = {
    STATUS_EMBEDDING_COMPLETED,
    STATUS_TRAINING_STARTED,
    STATUS_TRAINING_COMPLETED,
    STATUS_TRAINING_PARTIALLY_COMPLETED,
    STATUS_TRAINING_FAILED,
}

# Only allow screening AFTER extraction completes — never bypass extraction.
# UNDER_REVIEW excluded: extraction must run first to create the pickle.
SCREENING_ELIGIBLE_STATUSES = {
    STATUS_EXTRACTION_COMPLETED,
    STATUS_SCREENING_COMPLETED,
}

def _embedding_already_completed(record: Dict[str, Any]) -> bool:
    """Detect embedding completion even if the top-level status is stale."""
    if not isinstance(record, dict):
        return False
    if record.get("embedding_status") == STATUS_EMBEDDING_COMPLETED:
        return True
    if record.get("trained_at"):
        return True
    embedding = record.get("embedding") or {}
    if isinstance(embedding, dict) and str(embedding.get("status") or "").upper() == "COMPLETED":
        return True
    return False

def filter_doc_ids_by_status(
    doc_ids: List[str],
    required_status: str = STATUS_EXTRACTION_COMPLETED,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    eligible: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        record = get_document_record(doc_id) or {}
        status = record.get("status")
        if status in SCREENING_ELIGIBLE_STATUSES:
            eligible.append(doc_id)
        else:
            skipped.append({"document_id": doc_id, "status": status})
    return eligible, skipped

def promote_to_screening_completed(document_id: str) -> None:
    """Advance document status to SCREENING_COMPLETED if currently eligible.

    Called after any successful screening endpoint (security, integrity, etc.).
    Respects the status hierarchy: never downgrades from embedding/training states.
    """
    record = get_document_record(document_id) or {}
    current_status = record.get("status")
    if current_status in DONT_DOWNGRADE_STATUSES:
        logger.debug(
            "Skipping status promotion for %s; already at %s",
            document_id, current_status,
        )
        return
    if current_status in SCREENING_ELIGIBLE_STATUSES:
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)
        update_stage(document_id, "screening", {"status": "COMPLETED", "completed_at": time.time(), "error": None})
        logger.info(
            "Document %s promoted from %s to SCREENING_COMPLETED",
            document_id, current_status,
        )
    else:
        logger.warning(
            "Cannot promote document %s to SCREENING_COMPLETED; current status '%s' is not eligible. "
            "Eligible statuses: %s",
            document_id, current_status, SCREENING_ELIGIBLE_STATUSES,
        )

def _set_document_status(
    document_id: str,
    status: str,
    error_msg: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    fields: Dict[str, Any] = {"status": status, "updated_at": time.time()}
    if error_msg:
        fields["training_error"] = error_msg
        fields["training_failed_at"] = time.time()
    else:
        fields["training_error"] = None
    if extra_fields:
        fields.update(extra_fields)
    update_document_fields(document_id, fields)
    logger.info("Document %s status updated to %s", document_id, status)

def _status_from_security_report(report: Dict[str, Any]) -> str:
    risk_level = str(report.get("overall_risk_level") or report.get("risk_level") or "").upper()
    return "passed" if risk_level and risk_level not in {"HIGH", "CRITICAL"} else "failed"

def _update_pickle_with_screening(document_id: str, screening_report: Dict[str, Any]) -> None:
    """Load existing pickle, add screening results, re-save."""
    try:
        from src.api.content_store import load_extracted_pickle, save_extracted_pickle

        existing = load_extracted_pickle(document_id)
        if isinstance(existing, dict):
            existing["screening"] = screening_report
        else:
            existing = {"raw": existing, "screening": screening_report}
        save_extracted_pickle(document_id, existing)
        logger.info("Updated pickle with screening results for %s", document_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to update pickle with screening for %s: %s", document_id, exc)

def apply_security_result(document_id: str, report: Dict[str, Any]) -> None:
    status_text = _status_from_security_report(report)
    update_stage(document_id, "screening", {"status": "IN_PROGRESS", "started_at": time.time(), "error": None})
    update_security_screening(document_id, report, status_text)
    _update_pickle_with_screening(document_id, {
        "status": status_text,
        "risk_level": str(report.get("overall_risk_level") or report.get("risk_level") or ""),
        "report": report,
    })
    update_stage(document_id, "screening", {
        "status": "COMPLETED",
        "completed_at": time.time(),
        "result": status_text,
        "error": None,
    })
    record = get_document_record(document_id) or {}
    current_status = record.get("status")
    if status_text == "passed":
        if current_status in DONT_DOWNGRADE_STATUSES:
            logger.info(
                "Security screening passed for %s; keeping existing status %s",
                document_id,
                current_status,
            )
            return
        if current_status not in SCREENING_ELIGIBLE_STATUSES:
            logger.warning(
                "Security screening passed for %s but status '%s' is not eligible for promotion. "
                "Forcing promotion to SCREENING_COMPLETED. Eligible: %s",
                document_id, current_status, SCREENING_ELIGIBLE_STATUSES,
            )
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)
        # HITL: Screening only sets SCREENING_COMPLETED.
        # User must manually trigger embedding (POST /api/documents/embed).
    else:
        _set_document_status(document_id, STATUS_TRAINING_BLOCKED_SECURITY, "Security screening failed")

def apply_security_results_for_endpoint(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        if entry.get("status") != "succeeded":
            continue
        result = entry.get("result") or {}
        if not isinstance(result, dict):
            continue
        apply_security_result(entry.get("doc_id", ""), result)

def apply_security_results_for_run(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        if entry.get("errors"):
            continue
        results = entry.get("results") or {}
        if not isinstance(results, dict):
            continue
        security_payload = results.get("security")
        if isinstance(security_payload, dict):
            apply_security_result(entry.get("doc_id", ""), security_payload)
