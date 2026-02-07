from __future__ import annotations

from typing import Optional

from src.api.config import Config


class DomainRouter:
    @staticmethod
    def resolve(
        query: str,
        tool_hint: Optional[str],
        retrieved_metadata: Optional[dict],
    ) -> str:
        if (
            Config.Features.DOMAIN_SPECIFIC_ENABLED
            and tool_hint
            and str(tool_hint).strip().lower() == "resume"
        ):
            return "resume"
        _ = (query, retrieved_metadata)
        return "generic"


__all__ = ["DomainRouter"]
