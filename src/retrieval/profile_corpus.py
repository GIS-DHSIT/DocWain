from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from src.api.vector_store import build_qdrant_filter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


_PROFILE_ID_KEYS: Tuple[str, ...] = ("profile_id", "profileId", "profile", "candidate_id", "candidateId")


def _extract_profile_id(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in _PROFILE_ID_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def get_all_profile_ids(
    *,
    client: QdrantClient,
    subscription_id: str,
    collection: str,
    batch_limit: int = 256,
) -> List[str]:
    """
    Cross-profile discovery is disabled to enforce strict profile isolation.

    Use the catalog index or explicit profile_ids instead.
    """
    logger.debug("get_all_profile_ids: called but disabled, subscription_id=%s, collection=%s", subscription_id, collection)
    _ = (client, subscription_id, collection, batch_limit)
    raise ValueError("Profile discovery across Qdrant is disabled; use catalog-indexed profile ids.")


def get_profile_points(
    *,
    client: QdrantClient,
    subscription_id: str,
    profile_id: str,
    collection: str,
    batch_limit: int = 256,
) -> List[Dict[str, Any]]:
    """
    Retrieve the full corpus for a single profile via scroll pagination.

    Returns all point payloads for that (subscription_id, profile_id) scope.
    """
    logger.debug("get_profile_points: subscription_id=%s, profile_id=%s, collection=%s", subscription_id, profile_id, collection)
    if not subscription_id:
        raise ValueError("subscription_id is required")
    if not profile_id:
        raise ValueError("profile_id is required")
    if not collection:
        raise ValueError("collection is required")

    scroll_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
    )
    payloads: List[Dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=int(batch_limit),
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for pt in points or []:
            payload = getattr(pt, "payload", None) or {}
            if isinstance(payload, dict):
                payloads.append(payload)
        if not offset:
            break
    logger.debug("get_profile_points: returning %d payloads", len(payloads))
    return payloads


__all__ = ["get_all_profile_ids", "get_profile_points"]
