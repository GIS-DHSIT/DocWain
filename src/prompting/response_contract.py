from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional

from src.rag_v3.response_formatter import format_rag_v3_response


_DOMAIN_LABELS = {
    "resume": "Resume",
    "medical": "Medical",
    "invoice": "Invoice",
    "tax": "Tax",
    "bank_statement": "Bank Statement",
    "purchase_order": "Purchase Order",
    "generic": "Document",
}

_DOMAIN_ALIASES = {
    "bank": "bank_statement",
    "bankstatement": "bank_statement",
    "bank-statement": "bank_statement",
    "purchaseorder": "purchase_order",
    "purchase-order": "purchase_order",
    "po": "purchase_order",
    "cv": "resume",
}

_RETRIEVAL_FAILURE_CODES = {
    "RETRIEVAL_FILTER_FAILED",
    "RETRIEVAL_INDEX_MISSING",
    "RETRIEVAL_QDRANT_UNAVAILABLE",
    "RETRIEVAL_INDEX_BOOTSTRAP_FAILED",
}

_MISSING_SCOPE_CODES = {"MISSING_PROFILE_SCOPE"}

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
    re.IGNORECASE,
)
_TASK_LABELS = {
    "qa": "answer",
    "summarize": "summarize",
    "compare": "compare",
    "rank": "rank",
    "generate": "generate",
    "extract": "extract",
    "list": "list",
    "greet": "greet",
    "info": "info",
    "meta": "info",
}


def _normalize_domain(value: Optional[str]) -> str:
    if not value:
        return "generic"
    cleaned = str(value).strip().lower().replace(" ", "_")
    cleaned = _DOMAIN_ALIASES.get(cleaned, cleaned)
    return cleaned if cleaned in _DOMAIN_LABELS else "generic"


def _infer_domain(metadata: Dict[str, Any], response_text: str, sources: List[Dict[str, Any]]) -> str:
    route_plan = metadata.get("route_plan") or {}
    for key in ("domain_hint", "doc_domain", "document_category", "document_type", "domain"):
        candidate = route_plan.get(key) or metadata.get(key)
        normalized = _normalize_domain(candidate)
        if normalized != "generic":
            return normalized
    blob_parts: List[str] = [response_text]
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        name = src.get("source_name") or src.get("source")
        if name:
            blob_parts.append(str(name))
    blob = " ".join(blob_parts).lower()
    if any(token in blob for token in ("resume", "cv")):
        return "resume"
    if "invoice" in blob:
        return "invoice"
    if "purchase order" in blob or "purchase_order" in blob or "po " in blob:
        return "purchase_order"
    if "bank" in blob and "statement" in blob:
        return "bank_statement"
    if "tax" in blob:
        return "tax"
    if "medical" in blob or "patient" in blob or "diagnosis" in blob:
        return "medical"
    return "generic"


def _format_file_list(values: Iterable[Any]) -> List[str]:
    entries: List[str] = []
    seen = set()
    for value in values:
        if not value:
            continue
        if isinstance(value, dict):
            value = value.get("source_name") or value.get("name") or value.get("source")
        name = os.path.basename(str(value))
        if not name or name in seen:
            continue
        seen.add(name)
        entries.append(name)
    return entries


def _format_sources(sources: Iterable[Dict[str, Any]]) -> List[str]:
    entries: List[str] = []
    seen = set()
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        name = src.get("source_name") or src.get("source")
        if not name:
            continue
        base = os.path.basename(str(name))
        page = src.get("page")
        label = f"{base} (p. {page})" if page is not None else base
        if label in seen:
            continue
        seen.add(label)
        entries.append(label)
    return entries


def _detect_task_type(query: str, metadata: Dict[str, Any]) -> str:
    route_plan = metadata.get("route_plan") or {}
    for key in ("task_type", "task"):
        value = route_plan.get(key) or metadata.get(key)
        if value:
            return str(value).lower()
    lowered = (query or "").lower()
    if _GREETING_RE.match(lowered):
        return "greet"
    if any(token in lowered for token in ("thank you", "thanks", "appreciate")):
        return "greet"
    if any(token in lowered for token in ("bye", "goodbye", "see you", "farewell")):
        return "greet"
    if any(token in lowered for token in ("compare", "comparison", "vs", "versus")):
        return "compare"
    if any(token in lowered for token in ("rank", "ranking", "top", "best", "most")):
        return "rank"
    if any(token in lowered for token in ("summarize", "summarise", "summary", "overview", "recap", "highlights")):
        return "summarize"
    if any(token in lowered for token in ("generate", "draft", "create", "write", "compose")):
        return "generate"
    if any(token in lowered for token in ("extract", "identify", "find", "pull")):
        return "extract"
    if any(token in lowered for token in ("list", "show", "which", "filter")):
        return "list"
    return "qa"


def _extract_documents_searched(metadata: Dict[str, Any], explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return _format_file_list(explicit)
    meta = metadata or {}
    if meta.get("documents_searched"):
        return _format_file_list(meta.get("documents_searched") or [])
    trace = meta.get("execution_trace") or {}
    if trace.get("documents_searched"):
        return _format_file_list(trace.get("documents_searched") or [])
    return []


def _already_formatted(text: str) -> bool:
    if not text:
        return False
    has_understanding = re.search(r"(?im)^understanding\s*(?:&|and)\s*scope", text) is not None
    has_answer = re.search(r"(?im)^answer\s*:?", text) is not None
    has_evidence = re.search(r"(?im)^evidence\s*(?:&|and)\s*gaps", text) is not None
    return has_understanding and has_answer and has_evidence


def format_docwain_response(
    *,
    response_text: str,
    query: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    context_found: Optional[bool] = None,
    grounded: Optional[bool] = None,
    documents_searched: Optional[List[str]] = None,
) -> str:
    """Format the response for the UI.

    Returns clean expert prose — no template wrapper. Error states
    (missing scope, retrieval failures, no evidence) are handled with
    clear, professional messages.
    """
    text = (response_text or "").strip()
    meta = metadata or {}

    # Check for error states that need specific messaging
    error_code = None
    err = meta.get("error") or {}
    if err.get("code"):
        error_code = str(err.get("code"))
    elif meta.get("error_code"):
        error_code = str(meta.get("error_code"))

    if error_code in _MISSING_SCOPE_CODES:
        return "A profile must be selected to answer this question."

    if error_code in _RETRIEVAL_FAILURE_CODES:
        return "Unable to retrieve evidence due to a system issue. Please try again."

    # No evidence found
    ctx_found = bool(context_found) if context_found is not None else bool(sources)
    task_type = _detect_task_type(query, meta)
    is_greeting = task_type == "greet"
    is_meta = task_type in {"info", "meta"}

    if not ctx_found and not is_greeting and not is_meta:
        searched = _extract_documents_searched(meta, documents_searched)
        if searched:
            files_text = ", ".join(searched)
            return f"I couldn't find relevant information for this query. Documents searched: {files_text}."
        return "I couldn't find relevant information for this query in the current profile documents."

    # Normal response — return the text as-is (clean expert prose from the LLM)
    if not text:
        return "No answer available from the current profile documents."

    return text


__all__ = ["format_docwain_response"]
