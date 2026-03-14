import json
from src.utils.logging_utils import get_logger
import time
import traceback
from typing import Any, Dict, Optional

from pymongo import ReturnDocument
from bson import ObjectId

from src.api.config import Config
from src.api.statuses import STATUS_UNDER_REVIEW

logger = get_logger(__name__)

_PROGRESS_TTL = 3600  # 1 hour
_PROGRESS_CHANNEL = "dw:training:events"

def emit_progress(
    document_id: str,
    stage: str,
    progress: float,
    detail: str = "",
    extra: dict = None,
):
    """Emit a training progress event to Redis for real-time SSE streaming."""
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return

        event = {
            "document_id": str(document_id),
            "stage": stage,
            "progress": round(min(max(progress, 0.0), 1.0), 3),
            "detail": detail,
            "timestamp": time.time(),
        }
        if extra:
            event["extra"] = extra

        payload = json.dumps(event)
        client.setex(f"dw:training:progress:{document_id}", _PROGRESS_TTL, payload)
        client.publish(_PROGRESS_CHANNEL, payload)
    except Exception:
        pass  # Best-effort, never block the pipeline

def get_training_progress(document_id: str) -> Optional[dict]:
    """Get the latest training progress from Redis (for polling)."""
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return None
        raw = client.get(f"dw:training:progress:{document_id}")
        return json.loads(raw) if raw else None
    except Exception:
        return None

_ZOMBIE_TIMEOUT_SECONDS = 1800  # 30 minutes
_EXTRACTION_ZOMBIE_TIMEOUT_SECONDS = 600  # 10 minutes

def recover_zombie_documents(timeout_seconds: int = _ZOMBIE_TIMEOUT_SECONDS) -> int:
    """Auto-fail documents stuck in TRAINING_STARTED beyond timeout."""
    collection = get_documents_collection()
    cutoff = time.time() - timeout_seconds
    zombies = list(collection.find(
        {"status": "TRAINING_STARTED", "training_started_at": {"$lt": cutoff}},
        {"_id": 1, "document_id": 1, "training_started_at": 1},
    ))
    recovered = 0
    for doc in zombies:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        hours = (time.time() - (doc.get("training_started_at") or 0)) / 3600
        try:
            update_document_fields(doc_id, {
                "status": "TRAINING_FAILED",
                "training_error": f"Zombie recovery: stuck for {hours:.1f}h",
                "training_failed_at": time.time(),
                "error_summary": "zombie_timeout",
            })
            update_stage(doc_id, "embedding", {
                "status": "FAILED", "completed_at": time.time(),
                "reason": "zombie_timeout",
                "error": {"message": f"Process died — stuck for {hours:.1f}h, auto-recovered"},
            })
            emit_progress(doc_id, "failed", 0.0, f"Auto-recovered: stuck for {hours:.1f}h")
            recovered += 1
            logger.info("Recovered zombie document %s (stuck %.1fh)", doc_id, hours)
        except Exception:
            logger.warning("Failed to recover zombie %s", doc_id, exc_info=True)
    return recovered


def recover_zombie_extractions(timeout_seconds: int = _EXTRACTION_ZOMBIE_TIMEOUT_SECONDS) -> int:
    """Reset documents stuck in extraction IN_PROGRESS back to UNDER_REVIEW.

    When the server is killed during extraction, documents are left with
    ``status=UNDER_REVIEW`` and ``extraction.status=IN_PROGRESS`` forever.
    This function detects those zombies and resets extraction state so the
    next ``extract_documents()`` call will retry them.
    """
    collection = get_documents_collection()
    cutoff = time.time() - timeout_seconds
    zombies = list(collection.find(
        {
            "status": STATUS_UNDER_REVIEW,
            "extraction.status": "IN_PROGRESS",
            "extraction.started_at": {"$lt": cutoff},
        },
        {"_id": 1, "document_id": 1, "extraction.started_at": 1},
    ))
    recovered = 0
    for doc in zombies:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        started = (doc.get("extraction") or {}).get("started_at", 0)
        minutes = (time.time() - started) / 60 if started else 0
        try:
            update_stage(doc_id, "extraction", {
                "status": "PENDING",
                "completed_at": None,
                "error": None,
                "recovery_reason": f"zombie_reset: stuck in IN_PROGRESS for {minutes:.0f}min",
            })
            recovered += 1
            logger.info("Reset zombie extraction %s (stuck %.0fmin)", doc_id, minutes)
        except Exception:
            logger.warning("Failed to reset zombie extraction %s", doc_id, exc_info=True)
    return recovered

_MISSING = object()

def get_documents_collection():
    from src.api.dataHandler import db
    return db[Config.MongoDB.DOCUMENTS]

def get_screening_collection():
    from src.api.dataHandler import db
    return db["screening"]

def _doc_id_value(document_id: str):
    if ObjectId.is_valid(str(document_id)):
        return ObjectId(str(document_id))
    return str(document_id)

def _doc_filter(document_id: str) -> Dict[str, Any]:
    doc_id_str = str(document_id)
    candidates = [
        {"_id": doc_id_str},
        {"document_id": doc_id_str},
        {"documentId": doc_id_str},
        {"doc_id": doc_id_str},
        {"id": doc_id_str},
    ]
    if ObjectId.is_valid(str(document_id)):
        candidates.insert(0, {"_id": ObjectId(str(document_id))})
    return {"$or": candidates}

def init_document_record(
    document_id: str,
    subscription_id: Optional[str],
    profile_id: Optional[str],
    doc_type: Optional[str],
    filename: Optional[str],
    content_type: Optional[str],
    size: Optional[int],
):
    now = time.time()
    update: Dict[str, Any] = {"document_id": str(document_id), "updated_at": now}
    if subscription_id:
        update["subscription_id"] = str(subscription_id)
    if profile_id:
        update["profile_id"] = str(profile_id)
    if doc_type:
        update["doc_type"] = str(doc_type)
    if filename:
        update["source_file"] = filename
    if content_type:
        update["content_type"] = content_type
    if size is not None:
        update["content_size"] = int(size)

    collection = get_documents_collection()
    return collection.find_one_and_update(
        _doc_filter(document_id),
        {
            "$set": update,
            "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id), "status": STATUS_UNDER_REVIEW},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

def _flatten(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        target = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(target, value))
        else:
            flat[target] = value
    return flat

def update_document_fields(document_id: str, fields: Dict[str, Any]):
    now = time.time()
    update = dict(fields)
    update["updated_at"] = now
    unset_fields: Dict[str, Any] = {}

    # Enforce: error is either missing or an object (never null).
    error_value = update.pop("error", _MISSING)
    if error_value is not _MISSING:
        if error_value is None:
            unset_fields["error"] = ""
        elif isinstance(error_value, dict):
            update["error"] = error_value
        else:
            update["error"] = {"message": str(error_value)}

    for key, value in list(update.items()):
        if not key.endswith(".error"):
            continue
        update.pop(key)
        if value is None:
            unset_fields[key] = ""
        elif isinstance(value, dict):
            update[key] = value
        else:
            update[key] = {"message": str(value)}

    collection = get_documents_collection()
    update_ops: Dict[str, Any] = {
        "$set": update,
        "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id)},
    }
    if unset_fields:
        update_ops["$unset"] = unset_fields
    return collection.find_one_and_update(
        _doc_filter(document_id),
        update_ops,
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

def update_stage(document_id: str, stage: str, patch: Dict[str, Any]):
    now = time.time()
    patch_copy = dict(patch)
    error_value = patch_copy.pop("error", _MISSING)
    flat = _flatten(stage, patch_copy)
    flat["updated_at"] = now

    set_error: Dict[str, Any] = {}
    unset_error: Dict[str, Any] = {}
    if error_value is not _MISSING:
        error_path = f"{stage}.error"
        if error_value is None:
            unset_error[error_path] = ""
        elif isinstance(error_value, dict):
            set_error[error_path] = error_value
        else:
            set_error[error_path] = {"message": str(error_value)}

    collection = get_documents_collection()
    update_ops: Dict[str, Any] = {
        "$set": {**flat, **set_error},
        "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id)},
    }
    if unset_error:
        update_ops["$unset"] = unset_error
    return collection.find_one_and_update(
        _doc_filter(document_id),
        update_ops,
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

def set_error(document_id: str, stage: str, exc: Exception):
    message = str(exc)
    trace = traceback.format_exc()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return update_stage(
        document_id,
        stage,
        {
            "status": "FAILED",
            "completed_at": time.time(),
            "error": {"message": message, "trace": trace, "code": code},
        },
    )

def upsert_screening_report(
    run_id: str,
    document_id: str,
    endpoint: str,
    status: str,
    result: Optional[Dict[str, Any]],
    errors: Optional[list],
    warnings: Optional[list],
    options: Optional[Dict[str, Any]] = None,
    subscription_id: Optional[str] = None,
):
    now = time.time()
    update = {
        "run_id": run_id,
        "doc_id": str(document_id),
        "endpoint": endpoint,
        "status": status,
        "result": result,
        "errors": errors or [],
        "warnings": warnings or [],
        "options": options or {},
        "subscription_id": subscription_id,
        "updated_at": now,
    }
    collection = get_screening_collection()
    return collection.update_one(
        {"run_id": run_id, "doc_id": str(document_id), "endpoint": endpoint},
        {"$set": update, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )

def get_document_record(document_id: str) -> Optional[Dict[str, Any]]:
    collection = get_documents_collection()
    return collection.find_one(_doc_filter(document_id))

_ERROR_NULL_PATHS = [
    "error",
    "embedding.error",
    "extraction.error",
    "understanding.error",
    "screening.error",
    "screening.security.error",
    "cleanup.error",
]

def normalize_error_fields(collection=None) -> Dict[str, Any]:
    """
    Ensure error fields are either missing or objects (never null).

    Choice: unset null error fields (do not replace with {}).
    """
    collection = collection or get_documents_collection()
    if collection is None:
        return {"updated": 0, "paths": list(_ERROR_NULL_PATHS), "skipped": True}

    updated = 0
    for path in _ERROR_NULL_PATHS:
        result = collection.update_many({path: None}, {"$unset": {path: ""}})
        updated += int(getattr(result, "modified_count", 0) or 0)

    return {"updated": updated, "paths": list(_ERROR_NULL_PATHS), "skipped": False}
