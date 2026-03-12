from src.utils.logging_utils import get_logger
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient

logger = get_logger(__name__)

# Default payload paths to inspect for a profile id.
default_profile_paths = [
    "profile_id",
    "profileId",
    "profile",
    "metadata.profile_id",
    "metadata.profileId",
]

def _get_nested(payload: Dict[str, Any], path: str) -> Optional[Any]:
    """Resolve dotted keys from nested payload dictionaries."""
    if not payload or not path:
        return None
    parts = path.split(".")
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current

def _iter_profile_ids(payload: Dict[str, Any], payload_paths: Iterable[str]) -> Iterable[str]:
    """Yield all matching profile ids from the payload based on configured paths."""
    for path in payload_paths:
        value = _get_nested(payload, path)
        if value:
            yield str(value)

def discover_profile_ids_from_collection(
    client: QdrantClient,
    collection_name: str,
    max_points: Optional[int] = None,
    payload_paths: List[str] = default_profile_paths,
) -> List[str]:
    """
    Scan a Qdrant collection to discover unique profile ids without loading vectors.

    - Validates the collection exists up front.
    - Uses scroll with payloads only to avoid high memory usage.
    - Respects max_points if provided to cap scanned points.
    """
    if not collection_name:
        raise ValueError("collection_name is required for profile discovery")

    # Ensure profile indexes exist to prevent Qdrant 400 errors during filtered operations elsewhere.
    for field in ("profile_id", "profileId"):
        try:
            client.create_payload_index(collection_name=collection_name, field_name=field, field_schema="keyword")
        except Exception as exc:  # noqa: BLE001
            if "already exists" not in str(exc).lower():
                logger.debug("Ensure index %s failed: %s", field, exc)

    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")

    discovered = set()
    offset = None
    scanned = 0
    batch_limit = 200

    while True:
        if max_points is not None and scanned >= max_points:
            break
        limit = batch_limit
        if max_points is not None:
            limit = min(batch_limit, max_points - scanned)
            if limit <= 0:
                break

        points, offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        scanned += len(points)
        for point in points:
            payload = point.payload or {}
            for pid in _iter_profile_ids(payload, payload_paths):
                discovered.add(pid)

        if offset is None:
            break

    if discovered:
        logger.info(
            "Discovered %d profile ids in collection %s (scanned=%d, max_points=%s)",
            len(discovered),
            collection_name,
            scanned,
            max_points,
        )
    else:
        logger.warning(
            "No profile ids found in collection %s (scanned=%d, max_points=%s)",
            collection_name,
            scanned,
            max_points,
        )

    return sorted(discovered)
