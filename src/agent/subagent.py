"""Dynamic Sub-Agent — spawned by CoreAgent for complex multi-part queries.

Each sub-agent gets a focused role and a subset of evidence, calls the LLM,
and returns a structured SubAgentResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.generation.prompts import build_subagent_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SubAgentResult:
    """Result from a single sub-agent execution."""

    task: str
    text: str
    sources: List[Dict]
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# DynamicSubAgent
# ---------------------------------------------------------------------------


class DynamicSubAgent:
    """Executes a focused sub-task against a subset of evidence."""

    def __init__(
        self,
        llm_gateway: Any,
        role: str,
        evidence: List[Dict[str, Any]],
        doc_context: Optional[Dict[str, Any]],
    ) -> None:
        self._llm = llm_gateway
        self._role = role
        self._evidence = evidence
        self._doc_context = doc_context

    def execute(self) -> SubAgentResult:
        """Run the sub-agent and return a SubAgentResult."""
        try:
            prompt = build_subagent_prompt(self._role, self._evidence, self._doc_context)
            text = self._llm.generate(
                prompt,
                system="You are a document analysis sub-agent. Be precise and grounded.",
                temperature=0.2,
                max_tokens=1024,
            )
            sources = [
                {
                    "source_name": item.get("source_name", "unknown"),
                    "source_index": item.get("source_index"),
                }
                for item in self._evidence
            ]
            return SubAgentResult(
                task=self._role,
                text=text,
                sources=sources,
                success=True,
            )
        except Exception as exc:
            logger.exception("Sub-agent failed for role: %s", self._role)
            return SubAgentResult(
                task=self._role,
                text="",
                sources=[],
                success=False,
                error=str(exc),
            )
