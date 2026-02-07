from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RedisSchema:
    """Centralized Redis key schema for DocWain intelligence caches."""

    prefix: str = "dwx"

    def key(self, *parts: Optional[str]) -> str:
        safe_parts = [str(part).strip() for part in parts if part is not None]
        return ":".join([self.prefix, *safe_parts])

    def session_key(self, subscription_id: str, session_id: str) -> str:
        return self.key("session", subscription_id, session_id)

    def catalog_key(self, subscription_id: str, profile_id: str) -> str:
        return self.key("catalog", subscription_id, profile_id)

    def sections_key(self, subscription_id: str, profile_id: str, document_id: str) -> str:
        return self.key("sections", subscription_id, profile_id, document_id)

    def layout_key(self, subscription_id: str, profile_id: str, document_id: str) -> str:
        return self.key("layout", subscription_id, profile_id, document_id)

    def facts_doc_key(self, subscription_id: str, profile_id: str, document_id: str) -> str:
        return self.key("facts", "doc", subscription_id, profile_id, document_id)

    def facts_profile_key(self, subscription_id: str, profile_id: str) -> str:
        return self.key("facts", "profile", subscription_id, profile_id)

    def entities_key(self, subscription_id: str, profile_id: str) -> str:
        return self.key("entities", subscription_id, profile_id)

    def docsum_key(self, document_id: str) -> str:
        return self.key("docsum", document_id)

    def profilesum_key(self, subscription_id: str, profile_id: str) -> str:
        return self.key("profilesum", subscription_id, profile_id)


__all__ = ["RedisSchema"]
