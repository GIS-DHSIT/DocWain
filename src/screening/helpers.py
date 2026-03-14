"""Screening helper functions extracted from api.py for reuse by the gateway.

These functions handle result formatting, category normalization, document ID
validation, and result persistence — shared between the screening API endpoints
and the unified gateway.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .engine import ScreeningEngine
from .models import ToolResult

logger = get_logger(__name__)

def format_results(
    doc_id: str,
    results: Sequence[ToolResult],
    engine: Optional[ScreeningEngine] = None,
) -> Dict[str, Any]:
    """Format screening tool results into a standard response dict."""
    active_engine = engine or ScreeningEngine()
    response: Dict[str, Any] = {"doc_id": doc_id, "results": [res.to_dict() for res in results]}
    if len(results) > 1:
        overall = active_engine._blend_score(results)
        response["overall_score_0_100"] = round(overall, 2)
        response["risk_level"] = active_engine._risk_level(overall)
    elif results:
        response["risk_level"] = results[0].risk_level
    return response

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

def normalize_categories(raw_categories: Optional[List[str]]) -> List[str]:
    """Normalize and validate screening category names."""
    if not raw_categories:
        return ["all"]
    normalized = []
    for cat in raw_categories:
        key = cat.strip().lower().replace(" ", "_").replace("-", "_")
        if key == "ai_authorship":
            normalized.append("ai_authorship")
        elif key in _ALLOWED_CATEGORIES:
            normalized.append(key)
        else:
            raise ValueError(f"Unsupported category '{cat}'")
    if "all" in normalized:
        return ["all"]
    return normalized

def normalize_doc_ids(doc_ids: List[str]) -> List[str]:
    """Validate and deduplicate document IDs."""
    cleaned = [str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()]
    if not cleaned:
        raise ValueError("doc_ids must be a non-empty array")
    normalized = list(dict.fromkeys(cleaned))
    invalid = [
        doc_id
        for doc_id in normalized
        if doc_id.lower() == "string" or len(doc_id) < 6
    ]
    if invalid:
        raise ValueError("Invalid doc_id placeholder. Pass real document ids.")
    return normalized

def persist_screening_reports(
    run_id: str,
    endpoint: str,
    options: Dict[str, Any],
    doc_entries: List[Dict[str, Any]],
    collection=None,
) -> Dict[str, Any]:
    """Persist screening reports to MongoDB."""
    if collection is None:
        try:
            from src.api.dataHandler import db
            collection = db["screening"] if db is not None else None
        except Exception:  # noqa: BLE001
            pass

    if collection is None:
        return {
            "persisted": False,
            "persisted_count": 0,
            "persist_error": "Screening results store is unavailable.",
        }

    from pymongo import UpdateOne

    now = time.time()
    operations = []
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
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist screening results run_id=%s: %s", run_id, exc, exc_info=True)
        return {"persisted": False, "persisted_count": 0, "persist_error": str(exc)}
