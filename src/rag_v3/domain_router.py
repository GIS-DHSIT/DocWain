from __future__ import annotations

from typing import Iterable, Optional

_RESUME_DOC_TYPES = {
    "resume",
    "cv",
    "curriculum_vitae",
    "curriculum vitae",
    "linkedin",
    "linkedin_profile",
    "linkedin-profile",
}

_RESUME_INTENT_TOKENS = {
    "extract",
    "list",
    "pull",
    "summarize",
    "summarise",
    "summary",
    "overview",
    "recap",
    "rank",
    "ranking",
    "top",
    "best",
}


def _normalize(value: Optional[str]) -> str:
    cleaned = (value or "").strip().lower().replace(" ", "_")
    if cleaned == "cv":
        return "resume"
    if cleaned in {"linkedin_profile", "linkedin-profile"}:
        return "linkedin"
    return cleaned


def _intent_matches_resume(query: str) -> bool:
    lowered = (query or "").lower()
    return any(token in lowered for token in _RESUME_INTENT_TOKENS)


def _is_resume_doc_type(values: Iterable[str]) -> bool:
    return any(_normalize(value) in _RESUME_DOC_TYPES for value in values or [])


class DomainRouter:
    @staticmethod
    def resolve(
        query: str,
        tool_hint: Optional[str],
        retrieved_metadata: Optional[dict],
    ) -> str:
        if tool_hint and str(tool_hint).strip().lower() == "resume":
            return "resume"
        _ = (query, retrieved_metadata)
        return "generic"


__all__ = ["DomainRouter"]
