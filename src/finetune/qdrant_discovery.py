import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config

logger = logging.getLogger(__name__)


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
