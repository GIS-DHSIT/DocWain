"""Auto-agent selector for agent mode.

Analyzes query intent, domain, and keywords to automatically select
relevant agents when agent_mode is enabled, without requiring explicit
agent specification from the caller.

Agent selection uses three signal sources (in priority order):
1. ML-based intent classification (from intent_parse / analysis objects)
2. ML-based domain classification (from intent_parse / analysis objects)
3. Keyword patterns — ONLY for unambiguous, explicit agent invocations
   (e.g., "translate to French", "write a cover letter"). Single-word
   patterns are avoided to prevent false positives on generic queries.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.api.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent + domain → agent mapping  (ML-based, primary signal)
# ---------------------------------------------------------------------------
# Intent alone is only used for domain-agnostic intents like "generate".
# Domain-specific intents (extract, rank, contact) MUST combine with
# domain to select the right agent — otherwise "extract invoice data"
# would incorrectly select the resumes agent.

_INTENT_AGENTS: Dict[str, List[str]] = {
    "generate": ["content_generate"],
    "contact": ["resumes"],
}
_INTENT_TOOLS = _INTENT_AGENTS  # backward-compat alias

# Intent+domain combinations for domain-specific intents.
# Key format: "intent:domain" — only triggers when BOTH match.
_INTENT_DOMAIN_AGENTS: Dict[str, List[str]] = {
    "extract:resume": ["resumes"],
    "extract:hr": ["resumes"],
    "rank:resume": ["resumes"],
    "rank:hr": ["resumes"],
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
}
_DOMAIN_TOOLS = _DOMAIN_AGENTS  # backward-compat alias

# ---------------------------------------------------------------------------
# Keyword patterns → agent mapping
# ---------------------------------------------------------------------------
# DESIGN: Only match unambiguous, multi-word phrases that clearly indicate
# the user wants a specific AGENT, not just asking about document content.
# Single words like "medical", "legal", "email", "task" are NOT matched
# because they appear in normal conversational queries across all domains.
# Domain-specific routing is handled by _DOMAIN_AGENTS (ML-based).

_KEYWORD_AGENT_PATTERNS: List[tuple[re.Pattern, str]] = [
    # ── Domain agent tasks (highly specific multi-word phrases) ─────────
    (re.compile(r"\b(?:interview\s+question|interview\s+prep)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:skill\s+gap|gap\s+analysis|missing\s+skill)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:fit\s+for|suitable\s+for|good\s+fit|role\s+fit)\b.*\b(?:role|position|job)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:drug\s+interactions?|contraindication|interaction\s+check)\b", re.IGNORECASE), "medical"),
    (re.compile(r"\b(?:treatment\s+plan|care\s+plan)\b.*\b(?:review|assess|analyz)\b", re.IGNORECASE), "medical"),
    (re.compile(r"\b(?:lab\s+result|blood\s+work|test\s+result)\b.*\b(?:interpret|analyz|review)\b", re.IGNORECASE), "medical"),
    (re.compile(r"\b(?:risky\s+clause|risk\s+assess|red\s+flag)\b.*\b(?:contract|agreement|legal)\b", re.IGNORECASE), "lawhere"),
    (re.compile(r"\b(?:(?:compliance|compliant|regulation)\b.*\b(?:check|assess|review)|(?:check|assess|review)\b.*\b(?:compliance|compliant|regulation))\b", re.IGNORECASE), "lawhere"),
    (re.compile(r"\b(?:payment\s+anomal\w*|unusual\s+payment|suspicious\s+charge)\b", re.IGNORECASE), "insights"),
    (re.compile(r"\b(?:expense\s+categori|classify\s+expense)\b", re.IGNORECASE), "insights"),

    # ── Content generation (explicit creation verbs + output type) ──────
    (re.compile(r"\b(?:cover\s+letter|professional\s+summary|skills?\s+matrix)\b", re.IGNORECASE), "content_generate"),
    (re.compile(r"\b(?:create|draft|generate|write)\b.*\b(?:invoice|billing|receipt|purchase\s+order)\b", re.IGNORECASE), "content_generate"),

    # ── Resume-specific patterns (require resume/candidate context) ─────
    (re.compile(r"\b(?:compare|comparison|versus|vs\.?)\b.*\b(?:candidate|resume|profile)s?\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:best\s+candidate|rank\s+(?:the\s+)?candidates?|top\s+\d+\s+candidates?)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:contact\s+info|phone\s+number|email\s+address|linkedin|github)\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:extract|list)\b.*\b(?:skills?|experience|education|certifications?)\b.*\b(?:resume|cv|candidate)s?\b", re.IGNORECASE), "resumes"),
    (re.compile(r"\b(?:resume|cv|candidate)\s+(?:details?|information|profile|summary)\b", re.IGNORECASE), "resumes"),

    # ── Legal-specific (only multi-word, highly specific phrases) ───────
    (re.compile(r"\b(?:indemnit\w+|force\s+majeure|non[\s-]?compete|governing\s+law|tort|statute)\b", re.IGNORECASE), "lawhere"),

    # ── Screening (require PII context, not bare "screen") ─────────────
    (re.compile(r"\b(?:screen\s+(?:\w+\s+)?(?:for\s+)?pii|pii\s+(?:detection|scan|check)|check\s+readability|readability\s+(?:score|check|analysis))\b", re.IGNORECASE), "screen_pii"),

    # ── Translation (clear translation intent) ─────────────────────────
    (re.compile(r"\b(?:translat(?:e|ion|or))\b", re.IGNORECASE), "translator"),
    (re.compile(r"\b(?:convert)\b.*\b(?:documents?|text|content)\s+(?:(?:in)?to|in)\s+(?:english|french|spanish|german|dutch|chinese|japanese|korean|arabic|hindi|portuguese|russian|italian)\b", re.IGNORECASE), "translator"),

    # ── Email drafting (require creation verb + email/message) ──────────
    (re.compile(r"\b(?:draft|compose|write)\s+(?:an?\s+)?(?:email|message|letter|memo)\b", re.IGNORECASE), "email_drafting"),

    # ── Integrations (specific product names) ──────────────────────────
    (re.compile(r"\b(?:jira|confluence|ticket)\b", re.IGNORECASE), "jira_confluence"),

    # ── Code/API docs (require "code" or "api" prefix) ─────────────────
    (re.compile(r"\b(?:code\s+docs?|api\s+docs?|technical\s+documentation|software\s+documentation)\b", re.IGNORECASE), "code_docs"),

    # ── Tutoring (require explicit teaching context) ───────────────────
    (re.compile(r"\b(?:tutor\s+me|teach\s+me|explain\s+(?:the\s+concept|step\s+by\s+step)|learn\s+about)\b", re.IGNORECASE), "tutor"),

    # ── Web tools (specific action phrases) ────────────────────────────
    (re.compile(r"\b(?:web\s*extract|scrape|fetch\s*url)\b", re.IGNORECASE), "web_extract"),
    (re.compile(r"\b(?:ocr|image\s+analysis|analy[sz]e\s+(?:this\s+)?image|extract\s+text\s+from\s+(?:an?\s+)?image|screenshot)\b", re.IGNORECASE), "image_analysis"),
    (re.compile(r"\b(?:search\s+(?:the\s+)?(?:web|internet|online)|google|look\s+up|find\s+online|latest\s+version\s+of|current\s+version\s+of|what\s+is\s+\w+\s+(?:framework|library|tool|platform)\s+used\s+for)\b", re.IGNORECASE), "web_search"),

    # ── Insights (require analytical context, not bare "unusual") ──────
    (re.compile(r"\b(?:what(?:'s| is)\s+interesting|find\s+anomal\w*|any\s+risks?\s+in|(?:what|common|find|any)\s+patterns?\s+(?:in|across|among)|patterns?\s+across)\b", re.IGNORECASE), "insights"),

    # ── Action items (require "action items" phrase, not bare "task") ──
    (re.compile(r"\b(?:action\s+items?|pending\s+tasks?|what\s+(?:needs|must|should)\s+(?:(?:to\s+)?be\s+)?done)\b", re.IGNORECASE), "action_items"),

    # ── Cloud platform / SharePoint (specific service names) ───────────
    (re.compile(r"\b(?:cloud\s+platform|azure\s+blob|s3\s+bucket|gcs\s+bucket|cloud\s+storage)\b", re.IGNORECASE), "cloud_platform"),
    (re.compile(r"\b(?:sharepoint|share\s+point|onedrive)\b", re.IGNORECASE), "sharepoint"),
]
_KEYWORD_TOOL_PATTERNS = _KEYWORD_AGENT_PATTERNS  # backward-compat alias

# ---------------------------------------------------------------------------
# Agents that must never be auto-selected (require external data / user action)
# ---------------------------------------------------------------------------

_NEVER_AUTO_SELECT = frozenset({"stt", "tts", "db_connector"})


class AgentSelector:
    """Selects agents automatically based on query analysis."""

    def __init__(self, *, registered_tools: Optional[frozenset[str]] = None):
        """Initialize selector.

        Args:
            registered_tools: Set of agent/tool names known to be registered.
                If None, attempts to read from the global registry.
        """
        self._registered = registered_tools

    def _get_registered(self) -> Optional[frozenset[str]]:
        """Lazily resolve the set of registered agent names.

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

        # Source 3: Keyword-based selection (only unambiguous patterns)
        candidates.extend(self._from_keywords(query))

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
        """Select agents based on intent + domain classification.

        Domain-agnostic intents (generate, contact) map directly to agents.
        Domain-specific intents (extract, rank) require matching domain
        to avoid selecting wrong agents (e.g., extract+invoice → resumes).
        """
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

    def _from_keywords(self, query: str) -> List[str]:
        """Select agents based on keyword pattern matching."""
        agents: List[str] = []
        for pattern, agent_name in _KEYWORD_AGENT_PATTERNS:
            if pattern.search(query):
                agents.append(agent_name)
        return agents


# Backward-compat alias
ToolSelector = AgentSelector
