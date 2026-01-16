from __future__ import annotations

from ..models import ScreeningContext
from .base import PASSIVE_PATTERN, ScreeningTool, clamp, split_sentences


class PassiveVoiceTool(ScreeningTool):
    name = "passive_voice"
    category = "Language & Readability"
    default_weight = 0.05
    tool_version = "1.0"

    def run(self, ctx: ScreeningContext):
        sentences = split_sentences(ctx.text)
        passive_matches = list(PASSIVE_PATTERN.finditer(ctx.text))
        ratio = len(passive_matches) / max(len(sentences), 1)

        reasons = []
        actions = ["tag"]

        if ratio > 0.6:
            reasons.append("Heavy passive voice usage detected.")
            actions.append("warn")
        elif ratio > 0.3:
            reasons.append("Moderate passive voice usage.")
        else:
            reasons.append("Low passive voice usage.")

        score = clamp(ratio)  # 1.0 means passive in every sentence
        evidence = [
            {"span": match.group(0), "start": match.start(), "end": match.end()} for match in passive_matches[:10]
        ]

        raw_features = {"passive_ratio": ratio, "passive_count": len(passive_matches), "sentence_count": len(sentences)}
        return self.result(ctx, score, reasons, raw_features=raw_features, actions=actions, evidence_spans=evidence)
