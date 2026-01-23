import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.dataHandler import update_security_screening
from src.api.document_status import get_document_record, update_document_fields
from src.api.statuses import (
    STATUS_EXTRACTION_COMPLETED,
    STATUS_SCREENING_COMPLETED,
    STATUS_TRAINING_BLOCKED_SECURITY,
)

logger = logging.getLogger(__name__)


def filter_doc_ids_by_status(
    doc_ids: List[str],
    required_status: str = STATUS_EXTRACTION_COMPLETED,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    eligible: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        record = get_document_record(doc_id) or {}
        status = record.get("status")
        if status == required_status:
            eligible.append(doc_id)
        else:
            skipped.append({"document_id": doc_id, "status": status})
    return eligible, skipped


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


def apply_security_result(document_id: str, report: Dict[str, Any]) -> None:
    status_text = _status_from_security_report(report)
    update_security_screening(document_id, report, status_text)
    if status_text == "passed":
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)
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
