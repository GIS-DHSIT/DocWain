from __future__ import annotations

from typing import Any, Dict


def compute_agent_budgets(
    query: str,
    analysis: Dict[str, Any],
    *,
    base_steps: int,
    base_evidence: int,
) -> Dict[str, int]:
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
    evidence = max(10, int(base_evidence * complexity))

    # Cap to keep latency reasonable
    steps = min(steps, base_steps * 3)
    evidence = min(evidence, base_evidence * 3)

    return {"max_steps": steps, "max_evidence": evidence}
