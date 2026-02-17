"""Auto-tool selector for agent mode.

Analyzes query intent, domain, and keywords to automatically select
relevant tools when agent_mode is enabled, without requiring explicit
tool specification from the caller.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.api.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent → tool mapping
# ---------------------------------------------------------------------------

_INTENT_TOOLS: Dict[str, List[str]] = {
    "generate": ["content_generate"],
    "compare": ["content_generate"],
    "extract": ["resumes"],
    "contact": ["resumes"],
    "rank": ["resumes"],
    "summarize": ["content_generate"],
}

# ---------------------------------------------------------------------------
# Domain → tool mapping
# ---------------------------------------------------------------------------

_DOMAIN_TOOLS: Dict[str, List[str]] = {
    "resume": ["resumes"],
    "hr": ["resumes"],
    "legal": ["lawhere"],
    "medical": ["medical"],
    "invoice": ["content_generate"],
}

# ---------------------------------------------------------------------------
# Keyword patterns → tool mapping
# ---------------------------------------------------------------------------

_KEYWORD_TOOL_PATTERNS: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:cover\s+letter|professional\s+summary|skills?\s+matrix|interview\s+prep)\b", re.IGNORECASE), "content_generate"),
    (re.compile(r"\b(?:summarize|summary|key\s+points|overview)\b", re.IGNORECASE), "content_generate"),
    (re.compile(r"\b(?:compare|comparison|versus|vs\.?)\b.*\b(?:candidate|resume|profile|document)s?\b", re.IGNORECASE), "content_generate"),
    (re.compile(r"\b(?:rank|ranking|top\s+\d+|best\s+candidate|order\s+by)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:contact\s+info|phone|email\s+address|linkedin|github)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:extract|list)\b.*\b(?:skills?|experience|education|certifications?)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:resume|cv|candidate)\b.*\b(?:detail|info|data|profile)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:legal|clause|contract|liability|indemnit|govern\w*\s*law|tort|statute)\b", re.IGNORECASE), "lawhere"),
    (re.compile(r"\b(?:medical|patient|diagnosis|medication|symptom|clinical|health\s*record)\b", re.IGNORECASE), "medical"),
    (re.compile(r"\b(?:invoice|billing|payment|receipt|purchase\s+order)\b", re.IGNORECASE), "content_generate"),
    (re.compile(r"\b(?:screen|pii|readability)\b", re.IGNORECASE), "screen_pii"),
    (re.compile(r"\b(?:translat(?:e|ion|or))\b", re.IGNORECASE), "translator"),
    (re.compile(r"\b(?:email|draft|compose)\b", re.IGNORECASE), "email_drafting"),
    (re.compile(r"\b(?:jira|confluence|ticket)\b", re.IGNORECASE), "jira_confluence"),
    (re.compile(r"\b(?:code\s+docs?|documentation|api\s+docs?)\b", re.IGNORECASE), "code_docs"),
    (re.compile(r"\b(?:tutor|teach|explain\s+(?:how|what|why)|learn)\b", re.IGNORECASE), "tutor"),
    (re.compile(r"\b(?:web\s*extract|scrape|fetch\s*url)\b", re.IGNORECASE), "web_extract"),
]

# ---------------------------------------------------------------------------
# Tools that must never be auto-selected (require external data / user action)
# ---------------------------------------------------------------------------

_NEVER_AUTO_SELECT = frozenset({"stt", "tts", "db_connector"})


class ToolSelector:
    """Selects tools automatically based on query analysis."""

    def __init__(self, *, registered_tools: Optional[frozenset[str]] = None):
        """Initialize selector.

        Args:
            registered_tools: Set of tool names known to be registered.
                If None, attempts to read from the global tool registry.
        """
        self._registered = registered_tools

    def _get_registered(self) -> Optional[frozenset[str]]:
        """Lazily resolve the set of registered tool names.

        Returns None when the registry could not be resolved (skip filtering).
        Returns a frozenset when explicitly provided or resolved from registry.
        """
        if self._registered is not None:
            return self._registered
        try:
            from src.tools.base import registry
            return frozenset(registry._registry.keys())
        except Exception:
            return None  # Can't resolve — skip registration check

    def select_tools(
        self,
        query: str,
        intent_parse: Optional[Any] = None,
        analysis: Optional[Any] = None,
    ) -> List[str]:
        """Select tools based on query signals.

        Args:
            query: The user's query text.
            intent_parse: IntentParse or QueryAnalysis with .intent/.domain attrs.
            analysis: QueryAnalysis object (alternative signal source).

        Returns:
            List of 0 to MAX_AUTO_TOOLS tool names to invoke.
        """
        if not getattr(Config.Execution, "AGENT_AUTO_TOOLS", True):
            return []

        max_tools = int(getattr(Config.Execution, "AGENT_MAX_AUTO_TOOLS", 3))
        if not query or not query.strip():
            return []

        candidates: List[str] = []

        # Source 1: Intent-based selection
        candidates.extend(self._from_intent(intent_parse, analysis))

        # Source 2: Domain-based selection
        candidates.extend(self._from_domain(intent_parse, analysis))

        # Source 3: Keyword-based selection
        candidates.extend(self._from_keywords(query))

        # Deduplicate preserving order
        seen: set[str] = set()
        deduped: List[str] = []
        for name in candidates:
            if name not in seen:
                seen.add(name)
                deduped.append(name)

        # Filter out never-auto-select and unregistered tools
        registered = self._get_registered()
        result = [
            t for t in deduped
            if t not in _NEVER_AUTO_SELECT
            and (registered is None or t in registered)
        ]

        # Cap to max
        result = result[:max_tools]

        if result:
            logger.info("Auto-selected tools: %s for query: %.80s", result, query)

        return result

    def _from_intent(
        self,
        intent_parse: Optional[Any],
        analysis: Optional[Any],
    ) -> List[str]:
        """Select tools based on intent classification."""
        intent = None
        if intent_parse:
            intent = getattr(intent_parse, "intent", None)
        if not intent and analysis:
            intent = getattr(analysis, "intent", None)
        if not intent:
            return []
        return list(_INTENT_TOOLS.get(str(intent).lower(), []))

    def _from_domain(
        self,
        intent_parse: Optional[Any],
        analysis: Optional[Any],
    ) -> List[str]:
        """Select tools based on domain classification."""
        domain = None
        if intent_parse:
            domain = getattr(intent_parse, "domain", None)
        if not domain and analysis:
            domain = getattr(analysis, "domain", None)
        if not domain:
            return []
        return list(_DOMAIN_TOOLS.get(str(domain).lower(), []))

    def _from_keywords(self, query: str) -> List[str]:
        """Select tools based on keyword pattern matching."""
        tools: List[str] = []
        for pattern, tool_name in _KEYWORD_TOOL_PATTERNS:
            if pattern.search(query):
                tools.append(tool_name)
        return tools
