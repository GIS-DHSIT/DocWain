from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient

logger = get_logger(__name__)

REQUIRED_PAYLOAD_INDEX_FIELDS: List[str] = [
    "subscription_id",
    "profile_id",
    "document_id",
    "source_name",
    "doc_domain",
    "section_kind",
    "section_id",
    "chunk_kind",
    "chunk_id",
    "page",
    "embed_pipeline_version",
    "resolution",
]

_INDEX_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_INDEX_CACHE_TTL_SEC = 300

_MISSING_INDEX_RE = re.compile(r'Index required but not found for "([^"]+)"')

def _coerce_str(value: Any) -> str:
    return str(value) if value is not None else ""

def _cache_get(collection_name: str) -> Optional[List[str]]:
    cached = _INDEX_CACHE.get(collection_name)
    if not cached:
        return None
    ts, fields = cached
    if (time.time() - ts) > _INDEX_CACHE_TTL_SEC:
        return None
    return list(fields)

def _cache_set(collection_name: str, fields: List[str]) -> None:
    _INDEX_CACHE[collection_name] = (time.time(), list(fields))

def list_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    *,
    refresh: bool = False,
) -> List[str]:
    if not refresh:
        cached = _cache_get(collection_name)
        if cached is not None:
            return cached

    info = client.get_collection(collection_name)
    payload_schema = getattr(info, "payload_schema", None) or {}
    fields: List[str] = []
    if isinstance(payload_schema, dict):
        fields = sorted(payload_schema.keys())
    _cache_set(collection_name, fields)
    return fields

def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    required_fields: Iterable[str],
    *,
    create_missing: bool = True,
) -> Dict[str, Any]:
    fields = [str(f) for f in required_fields if f]
    existing = set(list_payload_indexes(client, collection_name, refresh=create_missing))
    missing = [field for field in fields if field not in existing]

    created: List[str] = []
    failures: List[str] = []
    if create_missing and missing:
        for field in missing:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
                created.append(field)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{field}: {exc}")
                logger.error("Failed to create payload index for %s in %s: %s", field, collection_name, exc)

        # Re-fetch to verify indexes actually exist before marking validated.
        existing = set(list_payload_indexes(client, collection_name, refresh=True))

    still_missing = [field for field in fields if field not in existing]
    if still_missing:
        raise ValueError(
            "Missing required Qdrant payload indexes: "
            + ", ".join(still_missing)
            + ("; failures=" + "; ".join(failures) if failures else "")
        )

    logger.info(
        "Qdrant payload indexes validated for %s (created=%s)",
        collection_name,
        created,
    )
    return {
        "collection": collection_name,
        "existing": sorted(existing),
        "missing": still_missing,
        "created": created,
        "failures": failures,
    }

def parse_missing_index_error(detail: Any) -> Optional[str]:
    if not detail:
        return None
    text = detail.decode("utf-8", errors="ignore") if isinstance(detail, (bytes, bytearray)) else _coerce_str(detail)
    match = _MISSING_INDEX_RE.search(text)
    if not match:
        return None
    return match.group(1).strip() if match.group(1) else None

def autoheal_missing_index(
    client: QdrantClient,
    collection_name: str,
    detail: Any,
) -> Optional[str]:
    field = parse_missing_index_error(detail)
    if not field:
        return None
    try:
        ensure_payload_indexes(client, collection_name, [field], create_missing=True)
        logger.warning("Auto-healed missing payload index for %s in %s", field, collection_name)
        return field
    except Exception as exc:  # noqa: BLE001
        logger.error("Auto-heal failed for payload index %s in %s: %s", field, collection_name, exc)
        return None

__all__ = [
    "REQUIRED_PAYLOAD_INDEX_FIELDS",
    "autoheal_missing_index",
    "ensure_payload_indexes",
    "list_payload_indexes",
    "parse_missing_index_error",
]
