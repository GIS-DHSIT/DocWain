from src.utils.logging_utils import get_logger
from typing import Dict, List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config

logger = get_logger(__name__)

def _ensure_profile_indexes(client: QdrantClient, collection_name: str) -> None:
    for field in ("profile_id", "profileId"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword",
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "already exists" in msg or "index exists" in msg:
                continue
            logger.debug("Ensure payload index %s on %s failed: %s", field, collection_name, exc)

def list_collections(client: Optional[QdrantClient] = None) -> List[str]:
    client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    resp = client.get_collections()
    return [col.name for col in getattr(resp, "collections", [])]

def list_profile_ids(
    collection_name: str,
    client: Optional[QdrantClient] = None,
    scan_limit: int = 1024,
    batch_size: int = 64,
) -> Dict[str, object]:
    client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    _ensure_profile_indexes(client, collection_name)
    profiles: Dict[str, int] = {}
    offset = None
    scanned = 0

    while True:
        limit = min(batch_size, scan_limit - scanned) if scan_limit else batch_size
        if limit <= 0:
            break
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_vectors=False,
            with_payload=True,
            offset=offset,
        )
        if not batch:
            break
        scanned += len(batch)
        for point in batch:
            payload = point.payload or {}
            profile = payload.get("profile_id") or payload.get("profileId")
            if profile:
                key = str(profile)
                profiles[key] = profiles.get(key, 0) + 1
        if offset is None or (scan_limit and scanned >= scan_limit):
            break

    return {
        "profile_ids": sorted(profiles.keys()),
        "counts": profiles,
        "scanned": scanned,
    }
