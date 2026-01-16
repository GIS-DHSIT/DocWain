from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional

from ..search import NullSearchClient, SearchClient, SearchHit
from .models import EvidenceBundle, EvidenceSource

DEFAULT_CACHE_TTL = 7 * 24 * 3600  # 7 days
MAX_RATE_PER_SEC = 5
MAX_TOTAL_QUERIES = 60


class CachedSearchClient:
    """Lightweight cached + rate-limited search wrapper."""

    def __init__(
        self,
        search_client: Optional[SearchClient],
        *,
        internet_enabled: bool,
        ttl_seconds: int = DEFAULT_CACHE_TTL,
        max_rate_per_sec: int = MAX_RATE_PER_SEC,
        max_total_queries: int = MAX_TOTAL_QUERIES,
    ) -> None:
        self.search_client = search_client or NullSearchClient()
        self.internet_enabled = internet_enabled
        self.ttl_seconds = ttl_seconds
        self.max_rate_per_sec = max_rate_per_sec
        self.max_total_queries = max_total_queries
        self._cache: Dict[tuple[str, int], tuple[float, List[SearchHit]]] = {}
        self._timestamps: deque[float] = deque()
        self.stats: Dict[str, int] = {"total_queries": 0, "skipped_due_to_rate_limit": 0}

    def _prune(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] > 1.0:
            self._timestamps.popleft()

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        if not self.internet_enabled or isinstance(self.search_client, NullSearchClient):
            return []

        key = (query, k)
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return cached[1]

        self._prune(now)
        if self.stats["total_queries"] >= self.max_total_queries:
            self.stats["skipped_due_to_rate_limit"] += 1
            return []
        if len(self._timestamps) >= self.max_rate_per_sec:
            self.stats["skipped_due_to_rate_limit"] += 1
            return []

        hits = self.search_client.search(query, k=k) or []
        self._timestamps.append(now)
        self.stats["total_queries"] += 1
        self._cache[key] = (now + self.ttl_seconds, hits)
        return hits


class OrganizationValidator:
    """Existence-only organization validation (no identity checks)."""

    def __init__(self, search: CachedSearchClient) -> None:
        self.search = search

    def validate(self, name: str, org_type: str) -> EvidenceBundle:
        normalized = (name or "").strip()
        if not normalized:
            return EvidenceBundle(
                name="",
                type=org_type,
                exists=False,
                confidence_0_100=0.0,
                sources=[],
                notes=["No organization name provided"],
                status="unknown",
            )

        query = f"{normalized} {org_type} official site"
        hits = self.search.search(query, k=3)
        sources = [EvidenceSource(title=h.title or "", url=h.url or "", snippet=h.snippet or "", source=h.source, score=h.score) for h in hits[:3]]

        exists = bool(hits)
        strong_match = any(normalized.lower() in (h.title or "").lower() or normalized.lower() in (h.url or "").lower() for h in hits)
        confidence = 80.0 if strong_match else 55.0 if hits else 20.0
        status = "verified" if strong_match else "likely_valid" if hits else "uncertain"
        notes: List[str] = []
        if not hits and not self.search.internet_enabled:
            notes.append("Internet validation disabled; organization existence not checked.")
        if self.search.stats["skipped_due_to_rate_limit"]:
            notes.append("Some lookups skipped due to rate limiting.")

        return EvidenceBundle(
            name=normalized,
            type=org_type,
            exists=exists,
            confidence_0_100=confidence,
            sources=sources,
            notes=notes,
            status=status,
        )

    def validate_many(self, names: Iterable[str], org_type: str) -> List[EvidenceBundle]:
        bundles: List[EvidenceBundle] = []
        seen = set()
        for name in names:
            normalized = (name or "").strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            bundles.append(self.validate(normalized, org_type))
        return bundles

