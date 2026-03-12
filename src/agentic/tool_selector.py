"""Auto-agent selector for agent mode.

Analyzes query intent, domain, and NLU understanding to automatically select
relevant agents when agent_mode is enabled, without requiring explicit
agent specification from the caller.

Agent selection uses three signal sources (in priority order):
1. ML-based intent classification (from intent_parse / analysis objects)
2. ML-based domain classification (from intent_parse / analysis objects)
3. NLU-based semantic understanding — parses query structure (action verbs,
   target nouns) and compares against agent capability *descriptions* using
   embedding similarity + structural NLP overlap.  No hardcoded patterns,
   prototypes, or keyword lists — agents are matched purely by understanding
   what the user wants and what each agent can do.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

from src.api.config import Config

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Intent + domain → agent mapping  (ML-based, primary signal)
# ---------------------------------------------------------------------------

_INTENT_AGENTS: Dict[str, List[str]] = {
    "generate": ["content_generate"],
    "contact": ["resumes"],
}
_INTENT_TOOLS = _INTENT_AGENTS  # backward-compat alias

_INTENT_DOMAIN_AGENTS: Dict[str, List[str]] = {
    "extract:resume": ["resumes"],
    "extract:hr": ["resumes"],
    "rank:resume": ["resumes"],
    "rank:hr": ["resumes"],
    "contact:resume": ["resumes"],
    "contact:hr": ["resumes"],
}
_INTENT_DOMAIN_TOOLS = _INTENT_DOMAIN_AGENTS  # backward-compat alias

# ---------------------------------------------------------------------------
# Domain → agent mapping  (ML-based, secondary signal)
# ---------------------------------------------------------------------------

_DOMAIN_AGENTS: Dict[str, List[str]] = {
    "resume": ["resumes"],
    "hr": ["resumes"],
    "legal": ["lawhere"],
    "medical": ["medical"],
    "image": ["image_analysis"],
    "customer_service": ["customer_service"],
    "support": ["customer_service"],
}
_DOMAIN_TOOLS = _DOMAIN_AGENTS  # backward-compat alias

# ---------------------------------------------------------------------------
# Agents that must never be auto-selected (require external data / user action)
# ---------------------------------------------------------------------------

_NEVER_AUTO_SELECT = frozenset({"stt", "tts", "db_connector"})

def _get_embedder() -> Any:
    """Get the sentence-transformer embedder (singleton, via NLU engine)."""
    try:
        from src.nlp.nlu_engine import get_embedder
        return get_embedder()
    except Exception:
        return None

class AgentSelector:
    """Selects agents automatically based on query analysis.

    Uses three signal sources in priority order:
    1. ML intent+domain classification (from intent_parse/analysis objects)
    2. ML domain classification
    3. Semantic similarity (embedding-based, replaces old regex patterns)
    """

    def __init__(self, *, registered_tools: Optional[frozenset[str]] = None):
        self._registered = registered_tools

    def _get_registered(self) -> Optional[frozenset[str]]:
        """Lazily resolve the set of registered agent names."""
        if self._registered is not None:
            return self._registered
        try:
            from src.tools.base import registry
            return frozenset(registry._registry.keys())
        except Exception:
            return None

    def select_agents(
        self,
        query: str,
        intent_parse: Optional[Any] = None,
        analysis: Optional[Any] = None,
    ) -> List[str]:
        """Select agents based on query signals.

        Args:
            query: The user's query text.
            intent_parse: IntentParse or QueryAnalysis with .intent/.domain attrs.
            analysis: QueryAnalysis object (alternative signal source).

        Returns:
            List of 0 to MAX_AUTO_TOOLS agent names to invoke.
        """
        if not getattr(Config.Execution, "AGENT_AUTO_TOOLS", True):
            return []

        max_tools = int(getattr(Config.Execution, "AGENT_MAX_AUTO_TOOLS", 3))
        if not query or not query.strip():
            return []

        candidates: List[str] = []

        # Source 1: Intent-based selection (ML — primary signal)
        candidates.extend(self._from_intent(intent_parse, analysis))

        # Source 2: Domain-based selection (ML — secondary signal)
        candidates.extend(self._from_domain(intent_parse, analysis))

        # Source 3: Semantic similarity (embedding-based — tertiary signal)
        candidates.extend(self._from_semantic(query))

        # Deduplicate preserving order
        seen: set[str] = set()
        deduped: List[str] = []
        for name in candidates:
            if name not in seen:
                seen.add(name)
                deduped.append(name)

        # Filter out never-auto-select and unregistered agents
        registered = self._get_registered()
        result = [
            t for t in deduped
            if t not in _NEVER_AUTO_SELECT
            and (registered is None or t in registered)
        ]

        # Cap to max
        result = result[:max_tools]

        if result:
            logger.info("Auto-selected agents: %s for query: %.80s", result, query)

        return result

    # Backward-compat alias
    select_tools = select_agents

    def _from_intent(
        self,
        intent_parse: Optional[Any],
        analysis: Optional[Any],
    ) -> List[str]:
        """Select agents based on intent + domain classification."""
        intent = None
        domain = None
        if intent_parse:
            intent = getattr(intent_parse, "intent", None)
            domain = getattr(intent_parse, "domain", None)
        if not intent and analysis:
            intent = getattr(analysis, "intent", None)
        if not domain and analysis:
            domain = getattr(analysis, "domain", None)
        if not intent:
            return []

        result: List[str] = []
        intent_str = str(intent).lower()

        # Domain-agnostic intents
        agents = _INTENT_AGENTS.get(intent_str)
        if agents:
            result.extend(agents)

        # Domain-specific intents — only when domain also matches
        if domain:
            domain_str = str(domain).lower()
            combo_key = f"{intent_str}:{domain_str}"
            combo_agents = _INTENT_DOMAIN_AGENTS.get(combo_key)
            if combo_agents:
                result.extend(combo_agents)

        return result

    def _from_domain(
        self,
        intent_parse: Optional[Any],
        analysis: Optional[Any],
    ) -> List[str]:
        """Select agents based on domain classification."""
        domain = None
        if intent_parse:
            domain = getattr(intent_parse, "domain", None)
        if not domain and analysis:
            domain = getattr(analysis, "domain", None)
        if not domain:
            return []
        return list(_DOMAIN_AGENTS.get(str(domain).lower(), []))

    def _from_semantic(self, query: str) -> List[str]:
        """Select agents using NLU understanding of query structure and meaning.

        Uses the nlu_agent_matcher module which:
        1. Parses query into semantic components (action verbs, target nouns)
        2. Compares against agent capability *descriptions* (not prototypes)
        3. Combines embedding similarity (when available) with structural NLP

        No hardcoded patterns, prototypes, or keyword lists — agents are
        matched purely by understanding what the user wants to do and what
        each agent is capable of.
        """
        try:
            from src.agentic.nlu_agent_matcher import match_agents
            embedder = _get_embedder()
            return match_agents(query, embedder=embedder)
        except Exception as exc:
            logger.debug("NLU agent matching failed: %s", exc)
            return []

# Backward-compat aliases
ToolSelector = AgentSelector
_KEYWORD_AGENT_PATTERNS = []  # Deprecated — kept for backward compat imports
_KEYWORD_TOOL_PATTERNS = _KEYWORD_AGENT_PATTERNS
