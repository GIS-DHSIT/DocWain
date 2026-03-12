"""Tiered LLM routing — selects the optimal LLM provider based on query complexity.

Implements a three-tier fallback chain:

| Tier | Provider        | When                                          |
|------|-----------------|-----------------------------------------------|
| T1   | DocWain-Agent (local) | Simple: factual, extraction, classification   |
| T2   | Gemini Flash    | Medium: summaries, comparisons, tool exec      |
| T3   | GPT-4o/Claude   | Complex: multi-doc reasoning, ranking, agentic |

``ComplexityScorer`` produces a 0-to-1 score from query length, entity count,
cross-document signals, intent type, and sentence count.  ``IntelligenceRouter``
maps that score to a tier and returns the appropriate LLM client, falling back
to lower tiers when a provider is unavailable.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import threading
from typing import Any, Dict, Optional

logger = get_logger(__name__)

__all__ = [
    "IntelligenceRouter",
    "ComplexityScorer",
    "get_intelligence_router",
    "set_intelligence_router",
]

# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

_CROSS_DOC_RE = re.compile(
    r"\b(?:compare|across|all|versus|vs\.?|rank)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

# Intents that indicate inherently complex reasoning
_COMPLEX_INTENTS = frozenset({"compare", "rank", "analytics", "cross_doc"})

# Intents that sit in the medium-complexity band
_MEDIUM_INTENTS = frozenset({"summary", "comparison"})

# Intents that are inherently simple regardless of score
_SIMPLE_INTENTS = frozenset({"factual", "extraction", "classification"})

class ComplexityScorer:
    """Scores query complexity on a 0-to-1 scale.

    The score is an additive combination of five independent signals, each
    contributing a partial weight.  The final value is clamped to ``[0, 1]``.
    """

    def score(
        self,
        query: str,
        intent: Optional[str] = None,
        entity_count: int = 0,
        task_spec_complexity: Optional[str] = None,
    ) -> float:
        """Return a complexity score in ``[0, 1]``.

        Parameters
        ----------
        query:
            The raw user query text.
        intent:
            Parsed intent label (e.g. ``"compare"``, ``"factual"``).
        entity_count:
            Number of named entities detected in the query.
        task_spec_complexity:
            When provided by the fine-tuned model (``"simple"``/``"medium"``
            /``"complex"``), biases the heuristic score toward the model's
            classification.
        """
        s = 0.0

        # -- Query length --------------------------------------------------
        qlen = len(query)
        if qlen > 500:
            s += 0.3
        elif qlen > 200:
            s += 0.2

        # -- Entity count ---------------------------------------------------
        if entity_count > 6:
            s += 0.25
        elif entity_count > 3:
            s += 0.15

        # -- Cross-document signals -----------------------------------------
        if _CROSS_DOC_RE.search(query):
            s += 0.2

        # -- Intent complexity ----------------------------------------------
        if intent:
            lower_intent = intent.lower()
            if lower_intent in _COMPLEX_INTENTS:
                s += 0.2
            elif lower_intent in _MEDIUM_INTENTS:
                s += 0.1

        # -- Multi-sentence -------------------------------------------------
        sentences = [
            seg.strip()
            for seg in _SENTENCE_SPLIT_RE.split(query)
            if seg.strip()
        ]
        if len(sentences) > 3:
            s += 0.1

        s = min(max(s, 0.0), 1.0)

        # -- TaskSpec complexity override -----------------------------------
        # The fine-tuned model's complexity classification biases the score
        # to ensure complex queries reach T2/T3 and simple ones stay on T1.
        if task_spec_complexity == "complex":
            s = max(s, 0.75)
        elif task_spec_complexity == "simple":
            s = min(s, 0.35)

        return s

# ---------------------------------------------------------------------------
# Intelligence router
# ---------------------------------------------------------------------------

class IntelligenceRouter:
    """Three-tier fallback chain with complexity-based routing.

    When cloud routing is disabled (the default), every call returns the local
    client regardless of complexity.  When enabled, the router inspects the
    query complexity score and intent to select the cheapest tier capable of
    handling the request, falling back to lower tiers when a provider is not
    configured.
    """

    def __init__(
        self,
        local_client: Any,
        gemini_client: Any = None,
        openai_client: Any = None,
        claude_client: Any = None,
    ) -> None:
        self.local = local_client
        self.gemini = gemini_client
        self.openai = openai_client
        self.claude = claude_client
        self.scorer = ComplexityScorer()
        self._t2_threshold: float = 0.4
        self._t3_threshold: float = 0.7
        self._enabled: bool = False  # Cloud routing disabled by default

    # -- Configuration ------------------------------------------------------

    def configure(
        self,
        enabled: bool,
        t2_threshold: float = 0.4,
        t3_threshold: float = 0.7,
    ) -> None:
        """Apply runtime configuration.

        Parameters
        ----------
        enabled:
            When ``False``, :meth:`route` always returns the local client.
        t2_threshold:
            Minimum complexity score to consider Tier 2 (Gemini).
        t3_threshold:
            Minimum complexity score to consider Tier 3 (GPT-4o / Claude).
        """
        self._enabled = enabled
        self._t2_threshold = t2_threshold
        self._t3_threshold = t3_threshold
        logger.info(
            "IntelligenceRouter configured: enabled=%s  T2>=%.2f  T3>=%.2f",
            enabled,
            t2_threshold,
            t3_threshold,
        )

    # -- Routing ------------------------------------------------------------

    def route(
        self,
        query: str,
        intent: Optional[str] = None,
        entity_count: int = 0,
        task_spec_complexity: Optional[str] = None,
    ) -> Any:
        """Return the appropriate LLM client for the query complexity.

        When cloud routing is disabled, always returns the local client.
        """
        if not self._enabled:
            return self.local

        score = self.scorer.score(query, intent, entity_count, task_spec_complexity)
        tier = self._select_tier(intent, score)
        client = self._get_client(tier)
        logger.debug(
            "IntelligenceRouter: score=%.3f  tier=%d  client=%s",
            score,
            tier,
            type(client).__name__,
        )
        return client

    def _select_tier(self, intent: Optional[str], score: float) -> int:
        """Map a complexity score + intent to a tier index (0/1/2)."""
        lower_intent = intent.lower() if intent else None

        # Simple intents always stay local regardless of score
        if score < self._t2_threshold or lower_intent in _SIMPLE_INTENTS:
            return 0

        # Medium band
        if score < self._t3_threshold or lower_intent in _MEDIUM_INTENTS:
            return 1

        return 2

    def _get_client(self, tier: int) -> Any:
        """Get client for *tier* with automatic fallback.

        Fallback prefers escalation to a higher-capability cloud provider
        before falling back to local.  E.g. if Gemini (T2) is unavailable,
        try Azure OpenAI / Claude (T3) before dropping to local (T1).
        """
        if tier >= 2:
            if self.openai:
                return self.openai
            if self.claude:
                return self.claude
            tier = 1  # fallback

        if tier >= 1:
            if self.gemini:
                return self.gemini
            # Escalate to T3 cloud if T2 is unavailable
            if self.openai:
                return self.openai
            if self.claude:
                return self.claude

        return self.local

    # -- Debugging ----------------------------------------------------------

    def explain(
        self,
        query: str,
        intent: Optional[str] = None,
        entity_count: int = 0,
        task_spec_complexity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return debug info about the routing decision.

        Useful for logging and the ``/api/admin`` diagnostic endpoints.
        """
        score = self.scorer.score(query, intent, entity_count, task_spec_complexity)
        tier = self._select_tier(intent, score) if self._enabled else 0
        client = self.route(query, intent, entity_count, task_spec_complexity)
        return {
            "complexity_score": round(score, 3),
            "tier": tier,
            "tier_label": ["T1:local", "T2:gemini", "T3:cloud"][min(tier, 2)],
            "cloud_enabled": self._enabled,
            "client_backend": getattr(client, "backend", "unknown"),
            "client_model": getattr(client, "model_name", "unknown"),
        }

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_ROUTER: Optional[IntelligenceRouter] = None
_ROUTER_LOCK = threading.Lock()

def get_intelligence_router() -> Optional[IntelligenceRouter]:
    """Return the global ``IntelligenceRouter`` instance, or ``None``."""
    return _ROUTER

def set_intelligence_router(router: IntelligenceRouter) -> None:
    """Install *router* as the global ``IntelligenceRouter`` singleton."""
    global _ROUTER
    with _ROUTER_LOCK:
        _ROUTER = router
    logger.info("IntelligenceRouter singleton installed")
