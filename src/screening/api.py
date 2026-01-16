from __future__ import annotations

import base64
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field, conint, constr, field_validator

from src.api.config import Config
from src.api.dataHandler import db
from .engine import ScreeningEngine
from .models import ToolResult
from .resume.models import ResumeScreeningDetailedResponse

logger = logging.getLogger(__name__)

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


class ScreenDocumentRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1)
    doc_id: Optional[str] = Field(None, description="Optional document identifier")
    chunk_id: Optional[str] = Field(None, description="Optional chunk identifier")
    doc_type: Optional[str] = Field(None, description="Document type hint (e.g., RESUME, POLICY)")
    region: Optional[str] = Field(None, description="Target region for legality checks (e.g., US, EU)")
    jurisdiction: Optional[str] = Field(None, description="Specific jurisdiction or venue")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    internet_enabled: Optional[bool] = Field(None, description="Override config to enable/disable web checks")
    previous_version_text: Optional[str] = Field(None, description="Optional previous version text for diffing")
    raw_bytes_base64: Optional[str] = Field(None, description="Optional base64-encoded raw bytes")


class ScreenResumeRequest(ScreenDocumentRequest):
    candidate_name: Optional[str] = Field(None, description="Candidate name (optional)")


class ScreeningRunOptions(BaseModel):
    doc_type: Optional[str] = Field(None, description="Optional document type override")
    internet_enabled: Optional[bool] = Field(None, description="Override config to enable/disable web checks")
    region: Optional[str] = Field(None, description="Target region for legality checks")
    jurisdiction: Optional[str] = Field(None, description="Specific venue or state for legality checks")


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


class ScreeningRunRequest(BaseModel):
    profile_ids: List[constr(min_length=1)]
    categories: Optional[List[str]] = Field(default=None, description="Defaults to ['all']")
    doc_type: Optional[str] = None
    internet_enabled: Optional[bool] = None
    region: Optional[str] = None
    jurisdiction: Optional[str] = None
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
    region = task.get("region")
    jurisdiction = task.get("jurisdiction")
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
                    region=region,
                    jurisdiction=jurisdiction,
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
            run_results = engine.run_category(
                category,
                doc_id,
                doc_type=doc_type,
                internet_enabled_override=internet_enabled,
                region=region,
                jurisdiction=jurisdiction,
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
        region=request.region,
        jurisdiction=request.jurisdiction,
    )
    return result


@screening_router.post("/resume", response_model=ResumeScreeningDetailedResponse)
def screen_resume(request: ScreenResumeRequest, engine: ScreeningEngine = Depends(get_screening_engine)):
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
        region=request.region,
        jurisdiction=request.jurisdiction,
    )
    result["chunk_id"] = request.chunk_id or request.doc_id
    return result


@screening_router.post("/{doc_id}/integrity")
def screen_integrity(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        results = engine.run_category(
            "integrity",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results, engine)
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/compliance")
def screen_compliance(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        results = engine.run_category(
            "compliance",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results, engine)
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/quality")
def screen_quality(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        results = engine.run_category(
            "quality",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results, engine)
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/language")
def screen_language(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        results = engine.run_category(
            "language",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results, engine)
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/security")
def screen_security(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        results = engine.run_category(
            "security",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results, engine)
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/legality")
def screen_legality(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        results = engine.run_category(
            "legality",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
            region=options.region,
            jurisdiction=options.jurisdiction,
        )
        return _format_results(doc_id, results, engine)
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/ai-authorship")
def screen_ai_authorship(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        result = engine.run_one(
            "ai_authorship",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, [result], engine)
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/resume", response_model=ResumeScreeningDetailedResponse)
def screen_resume_entities(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        result = engine.resume_analysis_from_doc(
            doc_id,
            doc_type=options.doc_type or "RESUME",
            internet_enabled_override=options.internet_enabled,
        )
        return result
    except ValueError as exc:
        _handle_error(exc)


@screening_router.post("/{doc_id}/all")
def screen_all(
    doc_id: str,
    options: ScreeningRunOptions = Body(default_factory=ScreeningRunOptions),
    engine: ScreeningEngine = Depends(get_screening_engine),
):
    try:
        report = engine.run_all(
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
            region=options.region,
            jurisdiction=options.jurisdiction,
        )
        payload = report.to_dict()
        payload["doc_id"] = doc_id
        return payload
    except ValueError as exc:
        _handle_error(exc)


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
        query = {"$or": [{"profile": profile_id}, {"profile_id": profile_id}, {"profileId": profile_id}]}
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
                    "region": request.region,
                    "jurisdiction": request.jurisdiction,
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
