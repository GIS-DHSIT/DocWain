from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, Optional, Tuple

from src.api.rag_state import AppState

logger = get_logger(__name__)

SERVICE_UNAVAILABLE_CODES = {
    "RETRIEVAL_INDEX_MISSING",
    "RETRIEVAL_QDRANT_UNAVAILABLE",
    "RETRIEVAL_INDEX_BOOTSTRAP_FAILED",
}

def _extract_error_code(answer: Dict[str, Any]) -> Optional[str]:
    if not isinstance(answer, dict):
        return None
    if answer.get("error_code"):
        return str(answer.get("error_code"))
    meta = answer.get("metadata") or {}
    err = meta.get("error") or {}
    code = err.get("code")
    return str(code) if code else None

def _ensure_ok_flag(answer: Dict[str, Any], ok: bool) -> Dict[str, Any]:
    if not isinstance(answer, dict):
        return answer
    answer.setdefault("metadata", {})
    answer["metadata"].setdefault("ok", ok)
    answer["ok"] = ok
    return answer

def attach_instance_ids(answer: Dict[str, Any], state: Optional[AppState]) -> Dict[str, Any]:
    if not isinstance(answer, dict) or not state:
        return answer
    meta = answer.get("metadata") or {}
    meta.setdefault("instance_ids", state.instance_ids or {})
    answer["metadata"] = meta
    return answer

def apply_error_contract(
    answer: Dict[str, Any],
    *,
    correlation_id: Optional[str],
    state: Optional[AppState],
) -> Tuple[Dict[str, Any], Optional[str]]:
    code = _extract_error_code(answer)
    if code:
        _ensure_ok_flag(answer, False)
        meta = answer.get("metadata") or {}
        if not meta.get("error"):
            meta["error"] = {"code": code, "details": answer.get("details", "")}
        if correlation_id and not meta.get("correlation_id"):
            meta["correlation_id"] = correlation_id
        if answer.get("documents_searched") and not meta.get("documents_searched"):
            meta["documents_searched"] = answer.get("documents_searched")
        answer["metadata"] = meta
    answer = attach_instance_ids(answer, state)
    return answer, code

def should_return_503(error_code: Optional[str]) -> bool:
    return bool(error_code and error_code in SERVICE_UNAVAILABLE_CODES)

def get_rag_system_from_app(app) -> Any:  # noqa: ANN401
    rag_system = getattr(app.state, "rag_system", None)
    if rag_system is None:
        logger.error("RAG system missing in app.state; request will fall back to lazy init")
    return rag_system

__all__ = [
    "apply_error_contract",
    "attach_instance_ids",
    "get_rag_system_from_app",
    "should_return_503",
]
