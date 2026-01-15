import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.api.dataHandler import db

logger = logging.getLogger(__name__)


def _parse_dt(value: Optional[str]) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        # Accept ISO strings; fallback to timestamp in seconds
        return datetime.datetime.fromisoformat(value)
    except Exception:
        try:
            return datetime.datetime.fromtimestamp(float(value))
        except Exception:
            logger.debug("Could not parse datetime value: %s", value)
            return None


def _resolve_name(doc: Dict[str, Any]) -> str:
    return (
        doc.get("name")
        or doc.get("docName")
        or doc.get("title")
        or doc.get("file_name")
        or doc.get("filename")
        or doc.get("document_name")
        or "Untitled Document"
    )


def _resolve_type(doc: Dict[str, Any]) -> Optional[str]:
    return (
        doc.get("doc_type")
        or doc.get("document_type")
        or doc.get("type")
        or doc.get("doctype")
    )


def _resolve_created(doc: Dict[str, Any]) -> Optional[Any]:
    return (
        doc.get("created_at")
        or doc.get("createdAt")
        or doc.get("created")
        or doc.get("createdDate")
        or doc.get("creationTime")
        or doc.get("timestamp")
    )


def _resolve_updated(doc: Dict[str, Any]) -> Optional[Any]:
    return (
        doc.get("updated_at")
        or doc.get("updatedAt")
        or doc.get("modified_at")
        or doc.get("modifiedAt")
        or doc.get("lastModified")
        or doc.get("lastUpdated")
    )


def list_documents(
    limit: int = 50,
    offset: int = 0,
    q: Optional[str] = None,
    doc_type: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    List documents from the canonical documents collection with pagination and optional filters.
    """
    collection = db[Config.MongoDB.DOCUMENTS]

    filters: Dict[str, Any] = {}
    and_filters: List[Dict[str, Any]] = []

    if q:
        regex = {"$regex": q, "$options": "i"}
        and_filters.append({"$or": [{"name": regex}, {"docName": regex}, {"title": regex}, {"file_name": regex}]})

    if doc_type:
        and_filters.append({"$or": [{"doc_type": doc_type}, {"document_type": doc_type}, {"type": doc_type}]})

    created_after_dt = _parse_dt(created_after)
    if created_after_dt:
        and_filters.append(
            {
                "$or": [
                    {"created_at": {"$gte": created_after_dt}},
                    {"createdAt": {"$gte": created_after_dt}},
                    {"created": {"$gte": created_after_dt}},
                    {"createdDate": {"$gte": created_after_dt}},
                    {"creationTime": {"$gte": created_after_dt}},
                    {"timestamp": {"$gte": created_after_dt}},
                ]
            }
        )

    created_before_dt = _parse_dt(created_before)
    if created_before_dt:
        and_filters.append(
            {
                "$or": [
                    {"created_at": {"$lte": created_before_dt}},
                    {"createdAt": {"$lte": created_before_dt}},
                    {"created": {"$lte": created_before_dt}},
                    {"createdDate": {"$lte": created_before_dt}},
                    {"creationTime": {"$lte": created_before_dt}},
                    {"timestamp": {"$lte": created_before_dt}},
                ]
            }
        )

    if and_filters:
        filters["$and"] = and_filters

    # Choose best-effort sort order
    sort_fields = []
    probe = collection.find_one(filters, projection={"updated_at": 1, "updatedAt": 1, "created_at": 1, "createdAt": 1})
    if probe:
        if _resolve_updated(probe) is not None:
            sort_fields.append(("updated_at", -1))
            sort_fields.append(("updatedAt", -1))
        if _resolve_created(probe) is not None:
            sort_fields.append(("created_at", -1))
            sort_fields.append(("createdAt", -1))
    if not sort_fields:
        sort_fields.append(("name", 1))

    cursor = collection.find(filters)
    for field, direction in sort_fields:
        try:
            cursor = cursor.sort(field, direction)
            break
        except Exception:
            continue

    total = 0
    try:
        total = cursor.count()  # type: ignore[attr-defined]
    except Exception:
        try:
            total = collection.count_documents(filters)
        except Exception:
            total = 0

    cursor = cursor.skip(max(0, offset)).limit(max(1, min(limit, 200)))

    items: List[Dict[str, Any]] = []
    for doc in cursor:
        doc_id = str(doc.get("_id"))
        document_name = _resolve_name(doc)
        item = {
            "doc_id": doc_id,
            "document_name": document_name,
        }

        dtype = _resolve_type(doc)
        if dtype:
            item["doc_type"] = dtype

        created_val = _resolve_created(doc)
        if created_val:
            item["created_at"] = created_val

        updated_val = _resolve_updated(doc)
        if updated_val:
            item["updated_at"] = updated_val

        items.append(item)

    return items, total
