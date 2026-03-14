from __future__ import annotations

import base64
from src.utils.logging_utils import get_logger
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Union

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field, conint, constr, field_validator
from pymongo import UpdateOne

from src.api.config import Config
from src.api.dataHandler import db
from src.api.screening_service import (
    apply_security_results_for_endpoint,
    apply_security_results_for_run,
    filter_doc_ids_by_status,
    promote_to_screening_completed,
)
from src.api.statuses import STATUS_EXTRACTION_COMPLETED
from .engine import ScreeningEngine
from .security_service import SecurityScreeningService
from . import storage_adapter
from .models import ToolResult
from .resume.models import ResumeScreeningDetailedResponse

logger = get_logger(__name__)

screening_router = APIRouter(prefix="/screening", tags=["Screening"])
_engine = ScreeningEngine()

def get_screening_engine() -> ScreeningEngine:
    return _engine

def _decode_raw_bytes(raw_bytes_base64: Optional[str]) -> Optional[bytes]:
    if not raw_bytes_base64:
        return None
    try:
        return base64.b64decode(raw_bytes_base64.encode("utf-8"))
    except Exception:
        return None

def _require_domain_specific() -> None:
    if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Domain-specific screening is deprecated. Enable DOCWAIN_DOMAIN_SPECIFIC_ENABLED to use.",
        )

def _error_detail(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}

def _format_results(doc_id: str, results: Sequence[ToolResult], engine: Optional[ScreeningEngine] = None) -> Dict[str, Any]:
    active_engine = engine or _engine
    response = {"doc_id": doc_id, "results": [res.to_dict() for res in results]}
    if len(results) > 1:
        overall = active_engine._blend_score(results)  # type: ignore[attr-defined]
        response["overall_score_0_100"] = round(overall, 2)
        response["risk_level"] = active_engine._risk_level(overall)  # type: ignore[attr-defined]
    elif results:
        response["risk_level"] = results[0].risk_level
    return response

def _handle_error(exc: ValueError):
    detail = str(exc)
    status_code = status.HTTP_400_BAD_REQUEST
    if "not found" in detail.lower():
        status_code = status.HTTP_404_NOT_FOUND
    raise HTTPException(status_code=status_code, detail=_error_detail("screening_error", detail))

def _get_documents_collection():
    try:
        return db[Config.MongoDB.DOCUMENTS]
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Unable to access documents collection: %s", exc, exc_info=True)
        return None

def _get_screening_collection():
    try:
        return db["screening"]
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Unable to access screening collection: %s", exc, exc_info=True)
        return None

class ScreenDocumentRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1)
    doc_id: Optional[str] = Field(None, description="Optional document identifier")
    chunk_id: Optional[str] = Field(None, description="Optional chunk identifier")
    doc_type: Optional[str] = Field(None, description="Document type hint (e.g., RESUME, POLICY)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    internet_enabled: Optional[bool] = Field(None, description="Override config to enable/disable web checks")
    previous_version_text: Optional[str] = Field(None, description="Optional previous version text for diffing")
    raw_bytes_base64: Optional[str] = Field(None, description="Optional base64-encoded raw bytes")

class ScreenResumeRequest(ScreenDocumentRequest):
    candidate_name: Optional[str] = Field(None, description="Candidate name (optional)")

class ScreeningRunOptions(BaseModel):
    doc_type: Optional[str] = Field(None, description="Optional document type override")
    internet_enabled: Optional[bool] = Field(None, description="Override config to enable/disable web checks")

class MultiDocOptions(BaseModel):
    doc_ids: List[constr(strip_whitespace=True, min_length=1)] = Field(
        ...,
        description="Array of document IDs to screen in this request",
    )
    doc_type: Optional[str] = Field(None, description="Optional document type override")
    internet_enabled: Optional[bool] = Field(None, description="Override config to enable/disable web checks")

    model_config = {"extra": "forbid"}

    @field_validator("doc_ids", mode="before")
    @classmethod
    def _normalize_doc_ids(cls, value):
        if not value:
            return value
        cleaned = [str(doc_id).strip() for doc_id in value if str(doc_id).strip()]
        return list(dict.fromkeys(cleaned))

class MultiDocLegalityOptions(MultiDocOptions):
    region: Optional[str] = Field(None, description="Target region for legality checks")
    jurisdiction: Optional[str] = Field(None, description="Specific jurisdiction or venue")

    model_config = {"extra": "forbid"}

class ScreeningDocumentResult(BaseModel):
    doc_id: str
    results: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)

class ScreeningProfileSummary(BaseModel):
    processed: int = 0
    succeeded: int = 0
    failed: int = 0

class ScreeningProfileResult(BaseModel):
    profile_id: str
    status: str
    summary: ScreeningProfileSummary
    documents: List[ScreeningDocumentResult]
    message: Optional[str] = None

class ScreeningEndpointDocumentResult(BaseModel):
    doc_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)

class ScreeningEndpointResponse(BaseModel):
    run_id: str
    endpoint: str
    persisted: bool
    persisted_count: int = 0
    persist_error: Optional[str] = None
    documents: List[ScreeningEndpointDocumentResult]
    summary: ScreeningProfileSummary

class ScreeningRunRequest(BaseModel):
    profile_ids: List[constr(min_length=1)]
    categories: Optional[List[str]] = Field(default=None, description="Defaults to ['all']")
    doc_type: Optional[str] = None
    internet_enabled: Optional[bool] = None
    fail_fast: bool = False
    max_docs_per_profile: Optional[conint(gt=0)] = None

    @field_validator("categories", mode="before")
    @classmethod
    def _default_categories(cls, value):
        if value is None:
            return ["all"]
        return value

class ScreeningRunResponse(BaseModel):
    status: str
    profiles: List[ScreeningProfileResult]

_ALLOWED_CATEGORIES = {
    "integrity",
    "compliance",
    "quality",
    "language",
    "security",
    "ai_authorship",
    "ai-authorship",
    "resume",
    "legality",
    "all",
}

def _normalize_categories(raw_categories: Optional[List[str]]) -> List[str]:
    if not raw_categories:
        return ["all"]
    normalized = []
    for cat in raw_categories:
        key = cat.strip().lower().replace(" ", "_")
        key = key.replace("-", "_")
        if key == "ai_authorship":
            normalized.append("ai_authorship")
        elif key in _ALLOWED_CATEGORIES:
            normalized.append(key.replace("_", "_"))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_error_detail("invalid_category", f"Unsupported category '{cat}'"),
            )
    if "all" in normalized:
        return ["all"]
    return normalized

def _screen_document_task(task: Dict[str, Any]) -> Dict[str, Any]:
    engine = ScreeningEngine()
    doc_id = task["doc_id"]
    categories: List[str] = task["categories"]
    doc_type = task.get("doc_type")
    internet_enabled = task.get("internet_enabled")
    results: Dict[str, Any] = {}
    errors: List[str] = []
    try:
        for category in categories:
            key = category.replace("-", "_")
            if category == "all":
                report = engine.run_all(
                    doc_id,
                    doc_type=doc_type,
                    internet_enabled_override=internet_enabled,
                )
                payload = report.to_dict()
                payload["doc_id"] = doc_id
                results["all"] = payload
                continue
            if category in {"ai_authorship", "ai-authorship"}:
                run_result = engine.run_one(
                    "ai_authorship",
                    doc_id,
                    doc_type=doc_type,
                    internet_enabled_override=internet_enabled,
                )
                results["ai_authorship"] = _format_results(doc_id, [run_result], engine)
                continue
            if category == "security":
                results["security"] = SecurityScreeningService().screen_document(
                    doc_id,
                    doc_type=doc_type,
                    internet_enabled_override=internet_enabled,
                )
                continue
            run_results = engine.run_category(
                category,
                doc_id,
                doc_type=doc_type,
                internet_enabled_override=internet_enabled,
                region=task.get("region") if category == "legality" else None,
                jurisdiction=task.get("jurisdiction") if category == "legality" else None,
            )
            results[key] = _format_results(doc_id, run_results, engine)
    except Exception as exc:  # pragma: no cover - worker safety net
        logger.error("Screening failed for doc_id=%s: %s", doc_id, exc, exc_info=True)
        errors.append(str(exc))
    return {"doc_id": doc_id, "results": results or None, "errors": errors}

def _run_parallel_screening(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(tasks) <= 1:
        return [_screen_document_task(task) for task in tasks]

    results: List[Dict[str, Any]] = []
    max_workers = min(len(tasks), max(os.cpu_count() or 2, 2))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_screen_document_task, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Parallel screening failed for doc_id=%s: %s", task.get("doc_id"), exc, exc_info=True)
                results.append({"doc_id": task.get("doc_id", ""), "results": None, "errors": [str(exc)]})
    return results

def _extract_doc_id(doc: Dict[str, Any]) -> Optional[str]:
    for key in ("_id", "document_id", "documentId", "doc_id", "id"):
        value = doc.get(key)
        if value:
            return str(value)
    return None

def _summary_from_results(doc_results: List[ScreeningDocumentResult]) -> ScreeningProfileSummary:
    processed = len(doc_results)
    failed = len([d for d in doc_results if d.errors])
    succeeded = processed - failed
    return ScreeningProfileSummary(processed=processed, succeeded=succeeded, failed=failed)

def _normalize_doc_ids(options: MultiDocOptions) -> List[str]:
    doc_ids = [doc_id.strip() for doc_id in options.doc_ids if doc_id and doc_id.strip()]
    if not doc_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail("invalid_doc_ids", "doc_ids must be a non-empty array"),
        )
    normalized = list(dict.fromkeys(doc_ids))
    invalid = [
        doc_id
        for doc_id in normalized
        if doc_id.lower() == "string" or len(doc_id) < 6
    ]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail("invalid_doc_ids", "Invalid doc_id placeholder. Pass real document ids."),
        )
    return normalized

def _serialize_results(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value

def _summary_for_endpoint_results(doc_results: List[Dict[str, Any]]) -> ScreeningProfileSummary:
    processed = len(doc_results)
    failed = len([doc for doc in doc_results if doc.get("status") != "succeeded"])
    succeeded = processed - failed
    return ScreeningProfileSummary(processed=processed, succeeded=succeeded, failed=failed)

def _screen_doc_task(task: Dict[str, Any]) -> Dict[str, Any]:
    engine = ScreeningEngine()
    doc_id = task["doc_id"]
    endpoint = task["endpoint"]
    doc_type = task.get("doc_type")
    internet_enabled = task.get("internet_enabled")
    warnings: List[str] = []
    subscription_id = None
    try:
        subscription_id = storage_adapter.get_document_subscription_id(doc_id)
        if not subscription_id:
            warnings.append("subscription_id_unavailable_for_document")
    except Exception as exc:
        warnings.append(str(exc))
    try:
        if endpoint in {"integrity", "compliance", "quality", "language", "legality"}:
            results = engine.run_category(
                endpoint,
                doc_id,
                doc_type=doc_type,
                internet_enabled_override=internet_enabled,
                region=task.get("region") if endpoint == "legality" else None,
                jurisdiction=task.get("jurisdiction") if endpoint == "legality" else None,
            )
            payload = _format_results(doc_id, results, engine)
        elif endpoint == "security":
            payload = SecurityScreeningService().screen_document(
                doc_id,
                doc_type=doc_type,
                internet_enabled_override=internet_enabled,
            )
        elif endpoint == "ai_authorship":
            result = engine.run_one(
                "ai_authorship",
                doc_id,
                doc_type=doc_type,
                internet_enabled_override=internet_enabled,
            )
            payload = _format_results(doc_id, [result], engine)
        elif endpoint == "resume":
            result = engine.resume_analysis_from_doc(
                doc_id,
                doc_type=doc_type or "RESUME",
                internet_enabled_override=internet_enabled,
            )
            payload = _serialize_results(result)
        elif endpoint == "all":
            report = engine.run_all(
                doc_id,
                doc_type=doc_type,
                internet_enabled_override=internet_enabled,
            )
            payload = report.to_dict()
            payload["doc_id"] = doc_id
        else:
            raise ValueError(f"Unsupported screening endpoint '{endpoint}'")
        return {
            "doc_id": doc_id,
            "status": "succeeded",
            "result": payload,
            "errors": [],
            "warnings": warnings,
            "subscription_id": subscription_id,
            "duration_seconds": time.time() - task["started_at"],
        }
    except Exception as exc:  # pragma: no cover - worker safety net
        logger.error("Screening failed for doc_id=%s endpoint=%s: %s", doc_id, endpoint, exc, exc_info=True)
        return {
            "doc_id": doc_id,
            "status": "failed",
            "result": None,
            "errors": [str(exc)],
            "warnings": warnings,
            "subscription_id": subscription_id,
            "duration_seconds": time.time() - task["started_at"],
        }

def _run_parallel_doc_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(tasks) <= 1:
        return [_screen_doc_task(task) for task in tasks]

    results: Dict[int, Dict[str, Any]] = {}
    max_workers = min(len(tasks), max(os.cpu_count() or 2, 2))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_screen_doc_task, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            index = int(task.get("index", len(results)))
            try:
                results[index] = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Parallel screening failed for doc_id=%s: %s", task.get("doc_id"), exc, exc_info=True)
                results[index] = {
                    "doc_id": task.get("doc_id", ""),
                    "status": "failed",
                    "result": None,
                    "errors": [str(exc)],
                }
    return [results[index] for index in sorted(results)]

def _persist_screening_reports(
    run_id: str,
    endpoint: str,
    options: Dict[str, Any],
    doc_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    collection = _get_screening_collection()
    if collection is None:
        return {
            "persisted": False,
            "persisted_count": 0,
            "persist_error": "Screening results store is unavailable.",
        }

    now = time.time()
    operations: List[UpdateOne] = []
    for entry in doc_entries:
        doc_id = entry["doc_id"]
        update_doc = {
            "doc_id": doc_id,
            "endpoint": endpoint,
            "run_id": run_id,
            "status": entry["status"],
            "errors": list(entry.get("errors") or []),
            "warnings": list(entry.get("warnings") or []),
            "subscription_id": entry.get("subscription_id"),
            "options": options,
            "result": entry.get("result"),
            "duration_seconds": entry.get("duration_seconds"),
            "updated_at": now,
        }
        operations.append(
            UpdateOne(
                {"doc_id": doc_id, "endpoint": endpoint, "run_id": run_id},
                {"$set": update_doc, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
        )
    try:
        result = collection.bulk_write(operations, ordered=False)
        persisted_count = (result.upserted_count or 0) + (result.modified_count or 0)
        return {"persisted": True, "persisted_count": persisted_count}
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to persist screening results run_id=%s: %s", run_id, exc, exc_info=True)
        return {"persisted": False, "persisted_count": 0, "persist_error": str(exc)}

def _run_doc_based_screening(endpoint: str, options: MultiDocOptions) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    doc_ids = _normalize_doc_ids(options)
    eligible_doc_ids, skipped_docs = filter_doc_ids_by_status(doc_ids, STATUS_EXTRACTION_COMPLETED)
    tasks: List[Dict[str, Any]] = []
    for index, doc_id in enumerate(eligible_doc_ids):
        tasks.append(
            {
                "index": index,
                "endpoint": endpoint,
                "doc_id": doc_id,
                "doc_type": options.doc_type,
                "internet_enabled": options.internet_enabled,
                "region": getattr(options, "region", None) if endpoint == "legality" else None,
                "jurisdiction": getattr(options, "jurisdiction", None) if endpoint == "legality" else None,
                "started_at": time.time(),
            }
        )
    doc_entries = _run_parallel_doc_tasks(tasks)
    for skipped in skipped_docs:
        doc_entries.append(
            {
                "doc_id": skipped.get("document_id"),
                "status": "skipped",
                "result": None,
                "errors": [f"document_status_not_eligible:{skipped.get('status')}"] if skipped.get("status") else ["document_status_missing"],
                "warnings": [],
                "subscription_id": None,
                "duration_seconds": 0,
            }
        )
    summary = _summary_for_endpoint_results(doc_entries)
    options_payload = {
        "doc_type": options.doc_type,
        "internet_enabled": options.internet_enabled,
    }
    if endpoint == "legality":
        options_payload["region"] = getattr(options, "region", None)
        options_payload["jurisdiction"] = getattr(options, "jurisdiction", None)
    documents = [
        {"doc_id": doc["doc_id"], "status": doc["status"], "result": doc["result"], "errors": doc["errors"]}
        for doc in doc_entries
    ]
    response: Dict[str, Any] = {
        "run_id": run_id,
        "endpoint": endpoint,
        "documents": documents,
        "summary": summary,
    }
    if endpoint == "security":
        apply_security_results_for_endpoint(doc_entries)
    # Promote successfully-screened docs to SCREENING_COMPLETED for all endpoints
    for entry in doc_entries:
        if entry.get("status") == "succeeded" and entry.get("doc_id"):
            promote_to_screening_completed(entry["doc_id"])
    persist_result = _persist_screening_reports(run_id, endpoint, options_payload, doc_entries)
    response.update(persist_result)
    logger.info(
        "Screening run persisted",
        extra={
            "run_id": run_id,
            "endpoint": endpoint,
            "doc_ids_count": len(doc_ids),
            "persisted": persist_result.get("persisted"),
            "persisted_count": persist_result.get("persisted_count"),
        },
    )
    return response

@screening_router.get("/health")
def screening_health(engine: ScreeningEngine = Depends(get_screening_engine)):
    return {"status": "ok", "enabled_tools": engine.config.enabled_tools}

@screening_router.post("/document")
def screen_document(request: ScreenDocumentRequest, engine: ScreeningEngine = Depends(get_screening_engine)):
    raw_bytes = _decode_raw_bytes(request.raw_bytes_base64)
    result = engine.evaluate(
        text=request.text,
        doc_id=request.doc_id,
        doc_type=request.doc_type,
        metadata=request.metadata,
        raw_bytes=raw_bytes,
        previous_version_text=request.previous_version_text,
        internet_enabled_override=request.internet_enabled,
    )
    return result

@screening_router.post("/resume", response_model=Union[ResumeScreeningDetailedResponse, ScreeningEndpointResponse])
def screen_resume(
    request: Union[MultiDocOptions, ScreenResumeRequest] = Body(...),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    _require_domain_specific()
    if isinstance(request, MultiDocOptions):
        return _run_doc_based_screening("resume", request)

    raw_bytes = _decode_raw_bytes(request.raw_bytes_base64)
    metadata = dict(request.metadata or {})
    if request.candidate_name:
        metadata["candidate_name"] = request.candidate_name
    result = engine.resume_analysis_from_text(
        text=request.text,
        doc_id=request.doc_id,
        doc_type=request.doc_type or "RESUME",
        metadata=metadata,
        raw_bytes=raw_bytes,
        previous_version_text=request.previous_version_text,
        internet_enabled_override=request.internet_enabled,
    )
    return result

@screening_router.post("/chunk")
def screen_chunk(request: ScreenDocumentRequest, engine: ScreeningEngine = Depends(get_screening_engine)):
    raw_bytes = _decode_raw_bytes(request.raw_bytes_base64)
    result = engine.evaluate(
        text=request.text,
        doc_id=request.doc_id or request.chunk_id,
        doc_type=request.doc_type,
        metadata=request.metadata,
        raw_bytes=raw_bytes,
        previous_version_text=request.previous_version_text,
        internet_enabled_override=request.internet_enabled,
    )
    result["chunk_id"] = request.chunk_id or request.doc_id
    return result

@screening_router.post("/integrity", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_integrity(options: MultiDocOptions = Body(...)):
    return _run_doc_based_screening("integrity", options)

@screening_router.post("/compliance", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_compliance(options: MultiDocOptions = Body(...)):
    return _run_doc_based_screening("compliance", options)

@screening_router.post("/quality", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_quality(options: MultiDocOptions = Body(...)):
    return _run_doc_based_screening("quality", options)

@screening_router.post("/language", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_language(options: MultiDocOptions = Body(...)):
    return _run_doc_based_screening("language", options)

@screening_router.post("/security", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_security(options: MultiDocOptions = Body(...)):
    return _run_doc_based_screening("security", options)

@screening_router.post("/legality", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_legality(options: MultiDocLegalityOptions = Body(...)):
    _require_domain_specific()
    return _run_doc_based_screening("legality", options)

@screening_router.post("/ai-authorship", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_ai_authorship(options: MultiDocOptions = Body(...)):
    return _run_doc_based_screening("ai_authorship", options)

@screening_router.post("/all", response_model=ScreeningEndpointResponse, response_model_exclude_none=True)
def screen_all(options: MultiDocLegalityOptions = Body(...)):
    return _run_doc_based_screening("all", options)

@screening_router.post("/run", response_model=ScreeningRunResponse)
def run_screening(
    request: ScreeningRunRequest,
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    categories = _normalize_categories(request.categories)
    collection = _get_documents_collection()
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_detail("documents_unavailable", "Document store is not accessible."),
        )

    profiles: List[ScreeningProfileResult] = []
    overall_status = "success"

    for profile_id in request.profile_ids:
        query = {
            "$and": [
                {"$or": [{"profile": profile_id}, {"profile_id": profile_id}, {"profileId": profile_id}]},
                {"status": STATUS_EXTRACTION_COMPLETED},
            ]
        }
        cursor = collection.find(query, projection={"_id": 1, "doc_type": 1, "document_type": 1, "type": 1})
        if request.max_docs_per_profile:
            cursor = cursor.limit(int(request.max_docs_per_profile))

        docs = list(cursor)
        if not docs:
            profiles.append(
                ScreeningProfileResult(
                    profile_id=profile_id,
                    status="not_found",
                    summary=ScreeningProfileSummary(processed=0, succeeded=0, failed=0),
                    documents=[],
                    message="No documents found for profile",
                )
            )
            overall_status = "partial" if overall_status == "success" else overall_status
            continue

        tasks: List[Dict[str, Any]] = []
        for doc in docs:
            doc_id = _extract_doc_id(doc)
            if not doc_id:
                continue
            doc_type = (
                request.doc_type
                or doc.get("doc_type")
                or doc.get("document_type")
                or doc.get("type")
            )
            tasks.append(
                {
                    "doc_id": doc_id,
                    "categories": categories,
                    "doc_type": doc_type,
                    "internet_enabled": request.internet_enabled,
                }
            )

        doc_results: List[Dict[str, Any]] = []
        if request.fail_fast:
            for task in tasks:
                result = _screen_document_task(task)
                doc_results.append(result)
                if result.get("errors"):
                    break
        else:
            doc_results = _run_parallel_screening(tasks)

        documents = [ScreeningDocumentResult(**res) for res in doc_results]
        if "security" in categories:
            apply_security_results_for_run(doc_results)
        # Promote successfully-screened docs to SCREENING_COMPLETED
        # so embedding can proceed regardless of which categories were run
        for task_item in tasks:
            _doc_id = task_item.get("doc_id")
            if _doc_id:
                matching = [r for r in doc_results if r.get("doc_id") == _doc_id and not r.get("errors")]
                if matching:
                    promote_to_screening_completed(_doc_id)
        summary = _summary_from_results(documents)
        status_text = "success"
        if summary.failed and summary.succeeded == 0:
            status_text = "failed"
        elif summary.failed:
            status_text = "partial"

        if status_text == "failed":
            overall_status = "failed"
        elif status_text == "partial" and overall_status == "success":
            overall_status = "partial"

        profiles.append(
            ScreeningProfileResult(
                profile_id=profile_id,
                status=status_text,
                summary=summary,
                documents=documents,
                message=None,
            )
        )

        if request.fail_fast and any(doc.errors for doc in documents):
            overall_status = "failed"
            break

    return ScreeningRunResponse(status=overall_status, profiles=profiles)
