"""Unified TaskSpec dataclass — single structured output of query understanding.

The fine-tuned DocWain-Agent-v2 model (or NLU fallback) produces a TaskSpec
for every user query.  Downstream pipeline stages use its routing methods
instead of ad-hoc classification scattered across modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Valid value enums ────────────────────────────────────────────────────────

VALID_INTENTS = frozenset({
    "factual", "compare", "rank", "summarize", "extract",
    "generate", "analyze", "timeline", "redirect", "clarify",
})

VALID_DOMAINS = frozenset({
    "hr", "medical", "legal", "invoice", "insurance",
    "policy", "general", "content", "translation", "education",
})

VALID_OUTPUT_FORMATS = frozenset({
    "paragraph", "table", "bullets", "numbered", "chart_data", "json",
})

VALID_SCOPES = frozenset({
    "all_documents", "specific_document", "cross_document",
})

VALID_COMPLEXITIES = frozenset({"simple", "medium", "complex"})

# ── Intent→tools routing matrix ─────────────────────────────────────────────

_INTENT_TOOL_MAP: Dict[str, List[str]] = {
    "compare":   ["resumes", "insights"],
    "rank":      ["resumes", "insights"],
    "summarize": ["resumes", "medical", "lawhere"],
    "extract":   ["resumes", "medical"],
    "generate":  ["email_drafting", "content_generate"],
    "analyze":   ["insights"],
    "timeline":  ["resumes", "medical"],
}

_DOMAIN_TOOL_MAP: Dict[str, List[str]] = {
    "hr":          ["resumes"],
    "medical":     ["medical"],
    "legal":       ["lawhere"],
    "invoice":     [],
    "insurance":   ["lawhere"],
    "policy":      ["lawhere"],
    "translation": ["translator"],
    "content":     ["content_generate"],
    "education":   [],
}


@dataclass
class TaskSpec:
    """Unified output of query understanding — drives the entire RAG pipeline."""

    intent: str = "factual"
    domain: str = "general"
    output_format: str = "paragraph"
    entities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    scope: str = "all_documents"
    complexity: str = "medium"
    confidence: float = 0.5

    # ── Routing helpers ──────────────────────────────────────────────────

    def get_chunk_limit(self) -> int:
        """Max chunks to retrieve based on complexity."""
        return {"simple": 4, "medium": 10, "complex": 16}.get(
            self.complexity, 10
        )

    def get_context_chars(self) -> int:
        """Max context characters for LLM extraction."""
        return {"simple": 3072, "medium": 8192, "complex": 12288}.get(
            self.complexity, 8192
        )

    def get_num_predict(self) -> int:
        """Max output tokens for generation.

        Values include ~1K token overhead for Qwen3 thinking (the model
        generates thinking tokens even with think=False, consuming from
        the num_predict budget).
        """
        return {"simple": 1536, "medium": 2048, "complex": 2048}.get(
            self.complexity, 2048
        )

    def get_intent_output_tokens(self) -> int:
        """Intent-calibrated output token budget.

        Contact/factual queries need fewer tokens than summaries/generation.
        Values include ~1K Qwen3 thinking overhead.
        """
        _INTENT_TOKENS = {
            "factual": 1536,
            "extract": 1536,
            "contact": 1280,
            "clarify": 1280,
            "redirect": 1024,
            "compare": 2048,
            "rank": 2048,
            "summarize": 2048,
            "generate": 2048,
            "analyze": 2048,
            "timeline": 2048,
        }
        base = _INTENT_TOKENS.get(self.intent, 1536)
        # Complexity can boost: complex always gets at least 2048
        if self.complexity == "complex":
            base = max(base, 2048)
        return base

    def get_num_ctx(self) -> int:
        """Context window size for Ollama. Scaled by complexity.

        gpt-oss supports 131K but T4 GPU handles up to 32K comfortably.
        Larger context = more evidence visible = better answers.
        """
        return {"simple": 8192, "medium": 16384, "complex": 32768}.get(
            self.complexity, 16384
        )

    def should_use_agent_mode(self) -> bool:
        return self.complexity == "complex"

    def get_auto_tools(self) -> List[str]:
        """Merge intent-based and domain-based tool suggestions."""
        tools: List[str] = []
        tools.extend(_INTENT_TOOL_MAP.get(self.intent, []))
        tools.extend(_DOMAIN_TOOL_MAP.get(self.domain, []))
        # deduplicate while preserving order
        seen: set = set()
        result: List[str] = []
        for t in tools:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def to_natural_text(self) -> str:
        """Human-readable task description for LLM prompt injection."""
        parts: List[str] = []
        parts.append(f"Intent: {self.intent}")
        if self.domain != "general":
            parts.append(f"Domain: {self.domain}")
        if self.entities:
            parts.append(f"Key entities: {', '.join(self.entities)}")
        if self.constraints:
            constraint_str = ", ".join(
                f"{k}={v}" for k, v in self.constraints.items()
            )
            parts.append(f"Constraints: {constraint_str}")
        parts.append(f"Scope: {self.scope}")
        if self.output_format != "paragraph":
            parts.append(f"Output format: {self.output_format}")
        return " | ".join(parts)

    def get_format_instruction(self) -> str:
        """Return an explicit formatting instruction for the LLM prompt."""
        _FORMAT_INSTRUCTIONS = {
            "table": (
                "FORMAT: Present the answer as a **markdown table** with clear column headers. "
                "Use | header1 | header2 | ... | format. Every row must be separated by |."
            ),
            "bullets": (
                "FORMAT: Present the answer as a **bulleted list** using - or * for each point."
            ),
            "numbered": (
                "FORMAT: Present the answer as a **numbered list** (1. 2. 3. ...) ranked by relevance."
            ),
            "chart_data": (
                "FORMAT: Present the answer as **structured data** suitable for charting, "
                "with clear labels and numeric values."
            ),
            "json": (
                "FORMAT: Present the answer as valid **JSON** with descriptive keys."
            ),
        }
        return _FORMAT_INSTRUCTIONS.get(self.output_format, "")

    def get_few_shot_example(self) -> str:
        """Return a concrete output example for the requested format.

        Few-shot examples dramatically improve format compliance — models
        learn better from examples than from instructions alone.
        """
        _FEW_SHOT_EXAMPLES = {
            "table": (
                "EXAMPLE OUTPUT:\n"
                "**3 candidates analyzed across 2 criteria:**\n\n"
                "| Name | Experience | Key Skills | Rating |\n"
                "|------|-----------|------------|--------|\n"
                "| **John Smith** | **8 years** | Python, AWS, Docker | **A** |\n"
                "| Jane Doe | 5 years | Java, Azure | B |\n"
                "| Bob Lee | 3 years | JavaScript, React | C |\n\n"
                "**Top candidate: John Smith** — strongest experience and broadest skill set."
            ),
            "bullets": (
                "EXAMPLE OUTPUT:\n"
                "**Key findings from 3 documents:**\n\n"
                "- **Patient diagnosed with Type 2 Diabetes** on 2024-01-15 (Source: medical_report.pdf)\n"
                "- Current medications: **Metformin 500mg** twice daily, **Lisinopril 10mg** daily\n"
                "- Lab results show **HbA1c at 7.2%**, above target range of 6.5%\n"
                "- Next follow-up scheduled for **2024-04-15**"
            ),
            "numbered": (
                "EXAMPLE OUTPUT:\n"
                "**Ranking of 3 candidates for Senior Python Developer:**\n\n"
                "1. **John Smith** — 8 years Python, AWS certified, led team of 12. Best fit.\n"
                "2. **Jane Doe** — 5 years Python, strong data science background.\n"
                "3. **Bob Lee** — 3 years Python, junior level, limited system design."
            ),
        }
        return _FEW_SHOT_EXAMPLES.get(self.output_format, "")

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSpec":
        """Parse a dict into a validated TaskSpec, coercing bad values."""
        intent = d.get("intent", "factual")
        if intent not in VALID_INTENTS:
            intent = "factual"
        domain = d.get("domain", "general")
        if domain not in VALID_DOMAINS:
            domain = "general"
        output_format = d.get("output_format", "paragraph")
        if output_format not in VALID_OUTPUT_FORMATS:
            output_format = "paragraph"
        scope = d.get("scope", "all_documents")
        if scope not in VALID_SCOPES:
            scope = "all_documents"
        complexity = d.get("complexity", "medium")
        if complexity not in VALID_COMPLEXITIES:
            complexity = "medium"
        confidence = float(d.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        entities = d.get("entities") or []
        if not isinstance(entities, list):
            entities = [str(entities)]
        entities = [str(e) for e in entities]

        constraints = d.get("constraints") or {}
        if not isinstance(constraints, dict):
            constraints = {}

        return cls(
            intent=intent,
            domain=domain,
            output_format=output_format,
            entities=entities,
            constraints=constraints,
            scope=scope,
            complexity=complexity,
            confidence=confidence,
        )

    @classmethod
    def from_json(cls, raw: str) -> "TaskSpec":
        """Parse a JSON string into a TaskSpec; raises on invalid JSON."""
        return cls.from_dict(json.loads(raw))
