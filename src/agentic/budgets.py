from __future__ import annotations

from typing import Any, Dict

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_agent_budgets(
    query: str,
    analysis: Dict[str, Any],
    *,
    base_steps: int,
    base_evidence: int,
) -> Dict[str, int]:
    logger.debug("compute_agent_budgets called with base_steps=%s, base_evidence=%s, intent=%s",
                 base_steps, base_evidence, analysis.get("intent"))
    token_count = len((query or "").split())
    sub_queries = analysis.get("sub_queries") or []
    intent = str(analysis.get("intent") or "")

    complexity = 1 + (token_count / 12)
    complexity += min(len(sub_queries), 4) * 0.4
    if intent in {"comparison", "summary"}:
        complexity += 0.6
    if intent in {"procedural", "reasoning"}:
        complexity += 0.4

    steps = max(6, int(base_steps * complexity))
    # Evidence budget: use a reasonable minimum that doesn't exceed what
    # retrieval typically returns. Previous base of 20 was too high —
    # typical retrieval yields 5-12 chunks, causing false "no evidence" results.
    evidence = max(3, int(min(base_evidence, 10) * complexity))

    # Cap to keep latency reasonable
    steps = min(steps, base_steps * 3)
    evidence = min(evidence, 30)  # Hard cap: no query needs 60 evidence items

    logger.debug("compute_agent_budgets returning max_steps=%s, max_evidence=%s, complexity=%.2f",
                 steps, evidence, complexity)
    return {"max_steps": steps, "max_evidence": evidence}
