from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


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
    Discover the candidate universe for a subscription by scrolling Qdrant with a subscription filter.

    Returns sorted, unique profile ids.
    """
    if not subscription_id:
        raise ValueError("subscription_id is required")
    if not collection:
        raise ValueError("collection is required")

    scroll_filter = Filter(
        must=[FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id)))]
    )

    discovered: Set[str] = set()
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
            profile_id = _extract_profile_id(payload)
            if profile_id:
                discovered.add(profile_id)
        if not offset:
            break

    return sorted(discovered)


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
    if not subscription_id:
        raise ValueError("subscription_id is required")
    if not profile_id:
        raise ValueError("profile_id is required")
    if not collection:
        raise ValueError("collection is required")

    def _scroll_all(profile_key: str) -> List[Dict[str, Any]]:
        scroll_filter = Filter(
            must=[
                FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
                FieldCondition(key=profile_key, match=MatchValue(value=str(profile_id))),
            ]
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
        return payloads

    # Prefer canonical payload key; fall back to legacy keys if necessary.
    payloads = _scroll_all("profile_id")
    if not payloads:
        payloads = _scroll_all("profileId")
    return payloads


__all__ = ["get_all_profile_ids", "get_profile_points"]
