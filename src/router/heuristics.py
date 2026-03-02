from __future__ import annotations

import re

from src.policy.response_policy import INFO_MODE, ResponseModeClassifier
from src.prompting.persona import is_meta_question

_META_HINTS = [
    re.compile(r"\bpersona\b"),
    re.compile(r"\bidentity\b"),
    re.compile(r"\bcapabilities\b"),
    re.compile(r"\bwhat\s+can\s+you\s+do\b"),
    re.compile(r"\bwhat\s+else\s+can\s+you\s+do\b"),
    re.compile(r"\bwhat\s+all\s+can\s+(?:you|docwain)\s+do\b"),
    re.compile(r"\bwhat\s+else\s+can\s+you\s+help\s+with\b"),
    re.compile(r"\bhow\s+can\s+(?:you|docwain)\s+help(?:\s+me)?\b"),
    re.compile(r"\bshow\s+(?:me\s+)?what\s+(?:you|docwain)\s+can\s+do\b"),
    re.compile(r"\bwho\s+are\s+you\b"),
    re.compile(r"\bwhat\s+are\s+you\b"),
    re.compile(r"\babout\s+you\b"),
]


def is_meta_query(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if ResponseModeClassifier.classify(normalized) == INFO_MODE:
        return True
    if is_meta_question(normalized):
        return True
    return any(pattern.search(normalized) for pattern in _META_HINTS)


__all__ = ["is_meta_query"]
