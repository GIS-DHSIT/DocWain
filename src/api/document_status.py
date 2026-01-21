import logging
import time
import traceback
from typing import Any, Dict, Optional

from pymongo import ReturnDocument
from bson import ObjectId

from src.api.config import Config

logger = logging.getLogger(__name__)


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
    candidates = [{"_id": str(document_id)}, {"document_id": str(document_id)}]
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
            "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id)},
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
    collection = get_documents_collection()
    return collection.find_one_and_update(
        _doc_filter(document_id),
        {
            "$set": update,
            "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id)},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )


def update_stage(document_id: str, stage: str, patch: Dict[str, Any]):
    now = time.time()
    flat = _flatten(stage, patch)
    flat["updated_at"] = now
    collection = get_documents_collection()
    return collection.find_one_and_update(
        _doc_filter(document_id),
        {
            "$set": flat,
            "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id)},
        },
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
