"""Intent Analyzer — the UNDERSTAND step of the Core Agent pipeline.

Replaces six competing intent classifiers with one LLM-native system.
Fast-paths conversational queries (greetings, farewells, meta questions)
without an LLM call; everything else gets analyzed via build_understand_prompt.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.generation.prompts import build_understand_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid enum values
# ---------------------------------------------------------------------------

_VALID_TASK_TYPES = frozenset(
    {"extract", "compare", "summarize", "investigate", "lookup", "aggregate", "list", "conversational"}
)
_VALID_OUTPUT_FORMATS = frozenset({"table", "bullets", "sections", "numbered", "prose"})
_VALID_COMPLEXITIES = frozenset({"simple", "complex"})

# ---------------------------------------------------------------------------
# Conversational detection patterns
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings|yo)\b",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^\s*(?:bye|goodbye|see\s+you|thanks|thank\s+you|cheers)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"^\s*(?:who\s+are\s+you|what\s+can\s+you\s+do|help)\s*[?!.]?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# QueryUnderstanding dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryUnderstanding:
    """Result of the UNDERSTAND step — structured intent analysis."""

    task_type: str
    complexity: str
    resolved_query: str
    output_format: str
    relevant_documents: List[Dict[str, Any]]
    cross_profile: bool
    sub_tasks: Optional[List[str]] = None
    entities: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

    @property
    def is_conversational(self) -> bool:
        return self.task_type == "conversational"

    @property
    def is_complex(self) -> bool:
        return self.complexity == "complex" and bool(self.sub_tasks)


# ---------------------------------------------------------------------------
# IntentAnalyzer
# ---------------------------------------------------------------------------


class IntentAnalyzer:
    """LLM-native intent analysis with conversational fast-path."""

    def __init__(self, llm_gateway: Any) -> None:
        self._llm = llm_gateway

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        doc_intelligence: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> QueryUnderstanding:
        """Analyze user intent and return a structured QueryUnderstanding.

        Fast-paths greetings/farewells/meta questions without an LLM call.
        For real queries, builds a prompt via ``build_understand_prompt`` and
        parses the LLM's JSON response.
        """
        # Fast-path: conversational queries need no LLM call
        if self._is_conversational(query):
            logger.debug("Fast-path conversational: %s", query[:60])
            return QueryUnderstanding(
                task_type="conversational",
                complexity="simple",
                resolved_query=query,
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
            )

        # Build prompt and call LLM
        prompt = build_understand_prompt(query, doc_intelligence, conversation_history)
        try:
            raw = self._llm.generate(
                prompt,
                system="You are a document intelligence query analyzer. Respond ONLY with JSON.",
                temperature=0.1,
                max_tokens=512,
            )
        except Exception:
            logger.exception("LLM call failed for intent analysis")
            return self._safe_defaults(query)

        return self._parse_response(raw, query)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _is_conversational(query: str) -> bool:
        """Return True if *query* is a greeting, farewell, or meta question."""
        return bool(
            _GREETING_RE.search(query)
            or _FAREWELL_RE.search(query)
            or _META_RE.search(query)
        )

    @staticmethod
    def _parse_response(raw: str, original_query: str) -> QueryUnderstanding:
        """Parse LLM JSON response into QueryUnderstanding.

        Strips markdown fences, validates enum fields, and falls back to
        safe defaults on any parse failure.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            # Remove closing fence
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse intent JSON: %.200s", raw)
            return IntentAnalyzer._safe_defaults(original_query)

        # Validate and coerce enum fields
        task_type = data.get("task_type", "lookup")
        if task_type not in _VALID_TASK_TYPES:
            logger.warning("Invalid task_type '%s', falling back to 'lookup'", task_type)
            task_type = "lookup"

        output_format = data.get("output_format", "prose")
        if output_format not in _VALID_OUTPUT_FORMATS:
            output_format = "prose"

        complexity = data.get("complexity", "simple")
        if complexity not in _VALID_COMPLEXITIES:
            complexity = "simple"

        return QueryUnderstanding(
            task_type=task_type,
            complexity=complexity,
            resolved_query=data.get("resolved_query", original_query),
            output_format=output_format,
            relevant_documents=data.get("relevant_documents", []),
            cross_profile=bool(data.get("cross_profile", False)),
            sub_tasks=data.get("sub_tasks"),
            entities=data.get("entities", []),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question"),
        )

    @staticmethod
    def _safe_defaults(query: str) -> QueryUnderstanding:
        """Return a safe-default QueryUnderstanding when parsing fails."""
        return QueryUnderstanding(
            task_type="lookup",
            complexity="simple",
            resolved_query=query,
            output_format="prose",
            relevant_documents=[],
            cross_profile=False,
        )
