from __future__ import annotations

from typing import List

from ..legality import LegalityAgent, serialize_legality_result
from ..models import ScreeningContext
from .base import ScreeningTool


class LegalityAgentTool(ScreeningTool):
    name = "legality_agent"
    category = "legality"
    default_weight = 0.14
    requires_internet = False
    supported_doc_types = None
    tool_version = "0.1"

    def __init__(self, agent: LegalityAgent | None = None) -> None:
        self.agent = agent or LegalityAgent()

    def applies_to(self, doc_type: str | None) -> bool:
        # Legality agent self-classifies when doc_type is absent.
        if doc_type and doc_type.upper() in {"RESUME"}:
            return False
        return True

    def _reasons(self, findings) -> List[str]:
        order = {"high": 0, "medium": 1, "low": 2}
        reasons: List[str] = []
        for finding in sorted(findings, key=lambda f: (order.get(getattr(f, "severity", ""), 3), f.title))[:5]:
            reasons.append(f"{finding.severity.upper()}: {finding.title}")
        return reasons or ["Legality screening completed"]

    def run(self, ctx: ScreeningContext):
        result = self.agent.analyze(ctx, region=ctx.region, jurisdiction=ctx.jurisdiction, doc_type=ctx.doc_type)
        payload = serialize_legality_result(result)
        score = result.scores.risk_score_0_1 if result.scores else 0.0
        evidence = []
        for finding in result.findings:
            if finding.evidence_snippets:
                evidence.append({"finding_id": finding.id, "snippet": finding.evidence_snippets[0]})
        return self.result(
            ctx,
            score=score,
            reasons=self._reasons(result.findings),
            raw_features={"legality": payload, "warnings": result.warnings, "errors": result.errors},
            actions=["Review legality.narrative_report and consult counsel as needed."],
            evidence_spans=evidence,
        )
