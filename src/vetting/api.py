from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from .engine import VettingEngine

router = APIRouter()
guardrails_router = APIRouter(tags=["Guardrails"])
_engine = VettingEngine()


class VetDocumentRequest(BaseModel):
    doc_id: Optional[str] = Field(None, description="Optional document identifier")
    chunk_id: Optional[str] = Field(None, description="Optional chunk identifier")
    doc_type: Optional[str] = Field(None, description="Document type hint (e.g., RESUME, POLICY)")
    text: str = Field(..., description="Full document text to vet")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    internet_enabled: Optional[bool] = Field(None, description="Override config to enable/disable web checks")
    previous_version_text: Optional[str] = Field(None, description="Optional previous version text for diffing")
    raw_bytes_base64: Optional[str] = Field(
        None, description="Optional base64-encoded raw bytes to hash instead of text"
    )


class VetResumeRequest(VetDocumentRequest):
    candidate_name: Optional[str] = Field(None, description="Candidate name (optional)")


class VettingRunOptions(BaseModel):
    doc_type: Optional[str] = Field(None, description="Optional document type override")
    internet_enabled: Optional[bool] = Field(None, description="Override config to enable/disable web checks")


def _decode_raw_bytes(raw_bytes_base64: Optional[str]) -> Optional[bytes]:
    if not raw_bytes_base64:
        return None
    try:
        return base64.b64decode(raw_bytes_base64.encode("utf-8"))
    except Exception:
        return None


def _format_results(doc_id: str, results):
    response = {"doc_id": doc_id, "results": [res.to_dict() for res in results]}
    if len(results) > 1:
        overall = _engine._blend_score(results)
        response["overall_score_0_100"] = round(overall, 2)
        response["risk_level"] = _engine._risk_level(overall)
    elif results:
        response["risk_level"] = results[0].risk_level
    return response


def _handle_error(exc: ValueError):
    detail = str(exc)
    status = 404 if "not found" in detail.lower() else 400
    raise HTTPException(status_code=status, detail=detail)


@guardrails_router.post("/document")
def vet_document(request: VetDocumentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required for guardrails evaluation")

    raw_bytes = _decode_raw_bytes(request.raw_bytes_base64)
    result = _engine.evaluate(
        text=request.text,
        doc_id=request.doc_id,
        doc_type=request.doc_type,
        metadata=request.metadata,
        raw_bytes=raw_bytes,
        previous_version_text=request.previous_version_text,
        internet_enabled_override=request.internet_enabled,
    )
    return result


@guardrails_router.post("/resume")
def vet_resume(request: VetResumeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required for guardrails evaluation")

    raw_bytes = _decode_raw_bytes(request.raw_bytes_base64)
    metadata = dict(request.metadata or {})
    if request.candidate_name:
        metadata["candidate_name"] = request.candidate_name
    result = _engine.evaluate(
        text=request.text,
        doc_id=request.doc_id,
        doc_type=request.doc_type or "RESUME",
        metadata=metadata,
        raw_bytes=raw_bytes,
        previous_version_text=request.previous_version_text,
        internet_enabled_override=request.internet_enabled,
    )
    return result


@guardrails_router.post("/chunk")
def vet_chunk(request: VetDocumentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required for guardrails evaluation")
    raw_bytes = _decode_raw_bytes(request.raw_bytes_base64)
    result = _engine.evaluate(
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


@guardrails_router.get("/health")
def guardrails_health():
    return {"status": "ok", "enabled_tools": _engine.config.enabled_tools}


@guardrails_router.post("/{doc_id}/integrity")
def vet_integrity(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        results = _engine.run_category(
            "integrity",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results)
    except ValueError as exc:
        _handle_error(exc)


@guardrails_router.post("/{doc_id}/compliance")
def vet_compliance(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        results = _engine.run_category(
            "compliance",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results)
    except ValueError as exc:
        _handle_error(exc)


@guardrails_router.post("/{doc_id}/quality")
def vet_quality(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        results = _engine.run_category(
            "quality",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results)
    except ValueError as exc:
        _handle_error(exc)


@guardrails_router.post("/{doc_id}/language")
def vet_language(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        results = _engine.run_category(
            "language",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results)
    except ValueError as exc:
        _handle_error(exc)


@guardrails_router.post("/{doc_id}/security")
def vet_security(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        results = _engine.run_category(
            "security",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results)
    except ValueError as exc:
        _handle_error(exc)


@guardrails_router.post("/{doc_id}/ai-authorship")
def vet_ai_authorship(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        result = _engine.run_one(
            "ai_authorship",
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, [result])
    except ValueError as exc:
        _handle_error(exc)


@guardrails_router.post("/{doc_id}/resume")
def vet_resume_entities(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        results = _engine.run_category(
            "resume",
            doc_id,
            doc_type=options.doc_type or "RESUME",
            internet_enabled_override=options.internet_enabled,
        )
        return _format_results(doc_id, results)
    except ValueError as exc:
        _handle_error(exc)


@guardrails_router.post("/{doc_id}/all")
def vet_all(doc_id: str, options: VettingRunOptions = Body(default_factory=VettingRunOptions)):
    try:
        report = _engine.run_all(
            doc_id,
            doc_type=options.doc_type,
            internet_enabled_override=options.internet_enabled,
        )
        payload = report.to_dict()
        payload["doc_id"] = doc_id
        return payload
    except ValueError as exc:
        _handle_error(exc)


# Expose Guardrails endpoints under the final prefix only.
router.include_router(guardrails_router, prefix="/guardrails")
