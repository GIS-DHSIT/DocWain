from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class RedisKeys:
    subscription_id: str
    profile_id: str

    def __post_init__(self) -> None:
        if not (self.subscription_id and str(self.subscription_id).strip()):
            raise ValueError("subscription_id is required for Redis keys")
        if not (self.profile_id and str(self.profile_id).strip()):
            raise ValueError("profile_id is required for Redis keys")

    def catalog(self) -> str:
        return f"dw:catalog:{self.subscription_id}:{self.profile_id}"

    def layout_ref(self, document_id: str) -> str:
        _guard_doc_id(document_id)
        return f"dw:doc:layout_ref:{self.subscription_id}:{self.profile_id}:{document_id}"

    def section_index(self, document_id: str) -> str:
        _guard_doc_id(document_id)
        return f"dw:doc:section_index:{self.subscription_id}:{self.profile_id}:{document_id}"

    def entity_index(self) -> str:
        return f"dw:profile:entity_index:{self.subscription_id}:{self.profile_id}"

    def kg_anchor_stats(self) -> str:
        return f"dw:profile:kg_anchor_stats:{self.subscription_id}:{self.profile_id}"

    def plan_cache(self, session_id: str, query: str) -> str:
        session = _safe_part(session_id) or "default"
        digest = _hash_query(query)
        return f"dw:plan:{self.subscription_id}:{self.profile_id}:{session}:{digest}"

    @staticmethod
    def hash_query(query: str) -> str:
        return _hash_query(query)


def _guard_doc_id(document_id: str) -> None:
    if not (document_id and str(document_id).strip()):
        raise ValueError("document_id is required for Redis keys")


def _safe_part(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _hash_query(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return "empty"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return digest[:16]


__all__ = ["RedisKeys"]
