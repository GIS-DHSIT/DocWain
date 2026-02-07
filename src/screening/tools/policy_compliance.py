from __future__ import annotations

from typing import List

from ..models import ScreeningContext
from .base import ScreeningTool, clamp


class PolicyComplianceTool(ScreeningTool):
    name = "policy_compliance"
    category = "Compliance & Template Conformance"
    default_weight = 0.10
    tool_version = "1.0"

    def _lower_contains(self, text: str, needle: str) -> bool:
        return needle.lower() in text.lower()

    def run(self, ctx: ScreeningContext):
        doc_type = (ctx.doc_type or ctx.metadata.get("doc_type") or "").upper()
        rules = (ctx.config.policy_rules.get(doc_type) if ctx.config else None) or {}
        forbidden = rules.get("forbidden_phrases", [])
        required = rules.get("required_keywords", [])
        required_disclaimers = rules.get("required_disclaimers", [])

        text_lower = ctx.text.lower()
        reasons: List[str] = []
        score = 0.0
        actions: List[str] = ["tag"]

        found_forbidden = [phrase for phrase in forbidden if self._lower_contains(ctx.text, phrase)]
        if found_forbidden:
            reasons.append(f"Forbidden phrases detected: {', '.join(found_forbidden)}.")
            score += min(1.0, 0.6 + 0.1 * len(found_forbidden))
            actions.append("warn")

        missing_required = [kw for kw in required if not self._lower_contains(ctx.text, kw)]
        if missing_required:
            reasons.append(f"Missing expected keywords: {', '.join(missing_required)}.")
            score += min(0.4, 0.1 * len(missing_required))

        missing_disclaimers = [d for d in required_disclaimers if not self._lower_contains(ctx.text, d)]
        if missing_disclaimers:
            reasons.append(f"Missing required disclaimers: {', '.join(missing_disclaimers)}.")
            score += 0.25
            actions.append("warn")

        if not (forbidden or required or required_disclaimers):
            reasons.append("No policy rules configured for this document type.")

        if not reasons:
            reasons.append("No policy compliance issues detected.")

        raw_features = {
            "forbidden_hits": found_forbidden,
            "missing_required": missing_required,
            "missing_disclaimers": missing_disclaimers,
        }

        return self.result(ctx, clamp(score), reasons, raw_features=raw_features, actions=actions)
