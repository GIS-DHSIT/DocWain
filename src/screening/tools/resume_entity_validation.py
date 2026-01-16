from __future__ import annotations

from typing import List

from ..models import ScreeningContext
from .base import ScreeningTool, clamp
from .resume_screening import _get_analysis


class ResumeEntityValidationTool(ScreeningTool):
    name = "resume_entity_validation"
    category = "Resume Screening"
    default_weight = 0.10
    requires_internet = True
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "1.1"

    def run(self, ctx: ScreeningContext):
        analysis = _get_analysis(ctx)
        entities = analysis.validations.companies + analysis.validations.institutions
        total_entities = max(len(entities), 1)
        not_found = [e for e in entities if not e.exists]
        uncertain = [e for e in entities if e.status == "uncertain"]

        score = clamp(((len(not_found) * 0.7) + (len(uncertain) * 0.3)) / total_entities)
        reasons: List[str] = []
        if not_found:
            reasons.append(
                f"Entities not found in open-web existence checks: {', '.join(e.name for e in not_found)}."
            )
        if uncertain and not (ctx.config and ctx.config.internet_enabled):
            reasons.append("Internet validation disabled; entity existence marked as UNCERTAIN.")
        if not reasons:
            reasons.append("Entities appear legitimate based on existence-only checks (no employment/education verification).")

        raw_features = {
            "entities": [e.model_dump() for e in entities],
            "resume_detected": True,
            "internet_enabled": bool(ctx.config and ctx.config.internet_enabled),
        }
        return self.result(ctx, score, reasons, raw_features=raw_features, actions=["tag"])
