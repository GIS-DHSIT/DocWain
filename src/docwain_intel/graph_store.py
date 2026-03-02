from __future__ import annotations

import threading
from typing import Dict, Optional

from .models import EntityFactBundle


class GraphStore:
    def save(self, *, subscription_id: str, profile_id: str, document_id: str, bundle: EntityFactBundle) -> None:
        raise NotImplementedError

    def get(self, *, subscription_id: str, profile_id: str, document_id: Optional[str] = None) -> Optional[EntityFactBundle]:
        raise NotImplementedError


class InMemoryGraphStore(GraphStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, EntityFactBundle] = {}

    def _key(self, subscription_id: str, profile_id: str, document_id: Optional[str]) -> str:
        return "|".join([str(subscription_id), str(profile_id), str(document_id or "profile")])

    def save(self, *, subscription_id: str, profile_id: str, document_id: str, bundle: EntityFactBundle) -> None:
        key = self._key(subscription_id, profile_id, document_id)
        with self._lock:
            self._store[key] = bundle

    def get(self, *, subscription_id: str, profile_id: str, document_id: Optional[str] = None) -> Optional[EntityFactBundle]:
        key = self._key(subscription_id, profile_id, document_id)
        with self._lock:
            return self._store.get(key)


__all__ = ["GraphStore", "InMemoryGraphStore"]
