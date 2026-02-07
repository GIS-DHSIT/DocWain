from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict


class FakeQueryResult:
    def __init__(self, points: List[FakePoint]):
        self.points = points


class FakeQdrant:
    def __init__(self, points: List[FakePoint]):
        self._points = points
        self.last_filter = None

    def query_points(self, **kwargs):  # noqa: ANN003
        self.last_filter = kwargs.get("query_filter")
        points = _apply_filter(self._points, self.last_filter)
        limit = kwargs.get("limit")
        if isinstance(limit, int) and limit > 0:
            points = points[:limit]
        return FakeQueryResult(points)

    def scroll(self, **kwargs):  # noqa: ANN003
        query_filter = kwargs.get("query_filter") or kwargs.get("scroll_filter")
        if query_filter is not None:
            self.last_filter = query_filter
        points = _apply_filter(self._points, query_filter)
        limit = kwargs.get("limit")
        if isinstance(limit, int) and limit > 0:
            points = points[:limit]
        return points, None


class FakeEmbedder:
    def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):  # noqa: ANN001
        _ = (text, convert_to_numpy, normalize_embeddings)
        return [[0.1, 0.1, 0.1, 0.1]]


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):  # noqa: ANN001
        return self.store.get(key)

    def setex(self, key, ttl, value):  # noqa: ANN001, ARG002
        self.store[key] = value

    def set(self, key, value):  # noqa: ANN001
        self.store[key] = value

    def delete(self, key):  # noqa: ANN001
        self.store.pop(key, None)


def _payload_lookup(payload: dict, key: str):  # noqa: ANN001
    parts = (key or "").split(".")
    current = payload
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _match_condition(payload: dict, cond: Any) -> bool:  # noqa: ANN001
    if isinstance(cond, Filter):
        must = getattr(cond, "must", []) or []
        should = getattr(cond, "should", []) or []
        if must and not _match_all(payload, must):
            return False
        if should:
            matched = sum(1 for item in should if _match_condition(payload, item))
            required = getattr(getattr(cond, "min_should", None), "min_count", None)
            required = int(required) if required is not None else 1
            return matched >= required
        return True
    key = getattr(cond, "key", None)
    match = getattr(cond, "match", None)
    if not key or match is None:
        return True
    value = _payload_lookup(payload, key)
    if isinstance(match, MatchValue):
        return str(value) == str(match.value)
    if isinstance(match, MatchAny):
        values = [str(v) for v in (match.any or [])]
        return str(value) in values
    return True


def _match_all(payload: dict, conditions: List[Any]) -> bool:
    for cond in conditions:
        if not _match_condition(payload, cond):
            return False
    return True


def _apply_filter(points: List[FakePoint], query_filter: Filter | None) -> List[FakePoint]:
    if not query_filter:
        return points
    must = getattr(query_filter, "must", []) or []
    filtered = []
    for pt in points:
        payload = pt.payload or {}
        if _match_all(payload, must):
            filtered.append(pt)
    return filtered


def make_point(
    *,
    pid: str,
    profile_id: str,
    document_id: str,
    file_name: str,
    text: str,
    page: int,
    score: float = 0.9,
    section_id: str = "sec-1",
    chunk_kind: str = "section_text",
    doc_domain: str = "generic",
):
    return FakePoint(
        id=pid,
        score=score,
        payload={
            "subscription_id": "sub-1",
            "profile_id": profile_id,
            "document_id": document_id,
            "source_name": file_name,
            "section_id": section_id,
            "section_title": "Section",
            "page": page,
            "chunk_kind": chunk_kind,
            "doc_domain": doc_domain,
            "canonical_text": text,
        },
    )


__all__ = [
    "FakeQdrant",
    "FakeEmbedder",
    "FakeRedis",
    "make_point",
    "FakePoint",
    "FakeQueryResult",
]
