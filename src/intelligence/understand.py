"""
QueryUnderstanding — single-LLM-call query analysis replacing 7 separate classifiers.

Decomposes user queries into structured intent, entities, format, and complexity
using one carefully designed prompt enriched with domain context from uploaded documents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubIntent:
    """A decomposed sub-intent within a complex query."""
    intent: str
    target: str
    scope: str


@dataclass
class DomainHints:
    """Domain-specific hints derived from document profile data."""
    relevant_fields: list[str] = field(default_factory=list)
    terminology_context: str = ""


@dataclass
class UnderstandResult:
    """Complete understanding of a user query."""
    primary_intent: str
    sub_intents: list[SubIntent]
    entities: list[str]
    output_format: str
    complexity: str
    needs_clarification: bool
    clarification_question: Optional[str]
    resolved_query: str
    thinking_required: bool
    domain_hints: DomainHints


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPLEXITY_SIGNALS = frozenset({
    "and", "compare", "all", "each", "vs", "summarize", "rank", "analyze",
})

_DEFAULT_RESULT = UnderstandResult(
    primary_intent="extract",
    sub_intents=[],
    entities=[],
    output_format="prose",
    complexity="simple",
    needs_clarification=False,
    clarification_question=None,
    resolved_query="",
    thinking_required=False,
    domain_hints=DomainHints(),
)

_JSON_SCHEMA = """{
  "primary_intent": "extract | compare | summarize | list | explain | aggregate | lookup",
  "sub_intents": [{"intent": "...", "target": "...", "scope": "..."}],
  "entities": ["entity1", "entity2"],
  "output_format": "table | bullets | sections | numbered | prose",
  "complexity": "simple | moderate | complex",
  "needs_clarification": false,
  "clarification_question": null,
  "resolved_query": "fully resolved query with pronouns replaced",
  "thinking_required": true,
  "domain_hints": {
    "relevant_fields": ["field1", "field2"],
    "terminology_context": "brief note on relevant domain terms"
  }
}"""

_SYSTEM_PROMPT = (
    "You are a document intelligence query analyzer. "
    "Given a user query and conversation context, produce a JSON analysis."
)


# ---------------------------------------------------------------------------
# QueryUnderstanding
# ---------------------------------------------------------------------------

class QueryUnderstanding:
    """Replaces 7 separate classifiers with a single LLM call."""

    def __init__(self, llm_gateway) -> None:
        """
        Args:
            llm_gateway: Object with ``.generate(prompt, **kwargs) -> str``.
        """
        self._llm = llm_gateway

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def understand(
        self,
        query: str,
        conversation_history: list[dict],
        domain_context: dict,
    ) -> UnderstandResult:
        """Analyse *query* in one LLM round-trip and return structured understanding.

        Args:
            query: The raw user query.
            conversation_history: List of ``{"role": ..., "content": ...}`` dicts.
            domain_context: Merged document profile data containing keys such as
                ``domain_labels``, ``key_terminology``, ``field_types``, and
                ``structure_patterns``.

        Returns:
            An :class:`UnderstandResult` instance.
        """
        prompt = self._build_prompt(query, conversation_history, domain_context)

        try:
            if hasattr(self._llm, "generate_with_metadata"):
                raw, _ = self._llm.generate_with_metadata(
                    prompt, options={"temperature": 0.0, "max_tokens": 512}
                )
            else:
                raw = self._llm.generate(prompt, temperature=0.0)
            result = self._parse_response(raw, query)
        except Exception:
            logger.warning(
                "LLM call failed during UNDERSTAND phase; returning default result",
                exc_info=True,
            )
            result = _make_default(query)

        logger.info(
            "UNDERSTAND completed",
            extra={
                "primary_intent": result.primary_intent,
                "complexity": result.complexity,
                "entity_count": len(result.entities),
                "sub_intent_count": len(result.sub_intents),
            },
        )
        return result

    @staticmethod
    def is_trivially_simple(query: str, history: list) -> bool:
        """Return ``True`` when the query is simple enough to skip UNDERSTAND.

        Criteria — **all** must hold:
        * 6 tokens or fewer
        * No prior conversation history
        * No complexity signal words present
        """
        if history:
            return False
        tokens = query.split()
        if len(tokens) > 6:
            return False
        lower_tokens = {t.lower().strip("?.,!") for t in tokens}
        if lower_tokens & _COMPLEXITY_SIGNALS:
            return False
        return True

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        conversation_history: list[dict],
        domain_context: dict,
    ) -> str:
        domain_labels = domain_context.get("domain_labels", "unknown")
        key_terminology = domain_context.get("key_terminology", "none")
        field_types = domain_context.get("field_types", "none")
        structure_patterns = domain_context.get("structure_patterns", "none")

        # Keep only the last 3 turns to stay within context limits.
        recent_history = conversation_history[-3:] if conversation_history else []
        history_text = json.dumps(recent_history, default=str) if recent_history else "[]"

        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"DOMAIN CONTEXT (from uploaded documents):\n"
            f"- Documents contain: {domain_labels}\n"
            f"- Key terminology: {key_terminology}\n"
            f"- Available fields: {field_types}\n"
            f"- Document structures: {structure_patterns}\n\n"
            f"Rules:\n"
            f"- Decompose multi-part queries into sub-intents\n"
            f"- Resolve pronouns using conversation history\n"
            f"- Infer output format from query semantics "
            f"(table for comparisons, bullets for lists, sections for summaries, "
            f"numbered for procedures, prose for factual)\n"
            f"- Assess complexity by semantic content, not word count\n"
            f"- Extract ALL entities mentioned or implied\n\n"
            f"USER:\n"
            f'Query: "{query}"\n'
            f"Conversation: {history_text}\n\n"
            f"Respond ONLY with JSON:\n{_JSON_SCHEMA}"
        )

    # ------------------------------------------------------------------
    # Response parsing with resilience
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, original_query: str) -> UnderstandResult:
        """Parse the LLM response into an :class:`UnderstandResult`.

        Tries three strategies in order:
        1. Direct ``json.loads``
        2. Extract from ````json ... ```` code blocks
        3. Find the first ``{ ... }`` block
        Falls back to a sensible default on failure.
        """
        data = self._try_parse_json(raw)
        if data is None:
            logger.warning("Failed to parse UNDERSTAND response; using default")
            return _make_default(original_query)

        try:
            return self._dict_to_result(data, original_query)
        except Exception:
            logger.warning(
                "Failed to convert parsed JSON to UnderstandResult; using default",
                exc_info=True,
            )
            return _make_default(original_query)

    @staticmethod
    def _try_parse_json(raw: str) -> Optional[dict]:
        """Attempt to extract a JSON object from *raw* using multiple strategies."""

        # Strategy 1: direct parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Strategy 2: ```json ... ``` code block
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 3: first { ... } block (greedy innermost braces won't work;
        # find the outermost pair by matching first '{' and last '}')
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last > first:
            try:
                data = json.loads(raw[first : last + 1])
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    @staticmethod
    def _dict_to_result(data: dict, original_query: str) -> UnderstandResult:
        """Convert a parsed JSON dict into an :class:`UnderstandResult`."""

        sub_intents = [
            SubIntent(
                intent=si.get("intent", ""),
                target=si.get("target", ""),
                scope=si.get("scope", ""),
            )
            for si in data.get("sub_intents", [])
            if isinstance(si, dict)
        ]

        dh_raw = data.get("domain_hints", {})
        if not isinstance(dh_raw, dict):
            dh_raw = {}

        domain_hints = DomainHints(
            relevant_fields=dh_raw.get("relevant_fields", []),
            terminology_context=dh_raw.get("terminology_context", ""),
        )

        return UnderstandResult(
            primary_intent=data.get("primary_intent", "extract"),
            sub_intents=sub_intents,
            entities=data.get("entities", []),
            output_format=data.get("output_format", "prose"),
            complexity=data.get("complexity", "simple"),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question"),
            resolved_query=data.get("resolved_query", original_query),
            thinking_required=bool(data.get("thinking_required", False)),
            domain_hints=domain_hints,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_default(query: str) -> UnderstandResult:
    """Return a safe default :class:`UnderstandResult` for *query*."""
    return UnderstandResult(
        primary_intent=_DEFAULT_RESULT.primary_intent,
        sub_intents=list(_DEFAULT_RESULT.sub_intents),
        entities=list(_DEFAULT_RESULT.entities),
        output_format=_DEFAULT_RESULT.output_format,
        complexity=_DEFAULT_RESULT.complexity,
        needs_clarification=_DEFAULT_RESULT.needs_clarification,
        clarification_question=_DEFAULT_RESULT.clarification_question,
        resolved_query=query,
        thinking_required=_DEFAULT_RESULT.thinking_required,
        domain_hints=DomainHints(),
    )
