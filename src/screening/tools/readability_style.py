from __future__ import annotations

from statistics import stdev
from typing import List

from ..models import ScreeningContext
from .base import ScreeningTool, clamp, flesch_reading_ease, tokenize_words


class ReadabilityStyleTool(ScreeningTool):
    name = "readability_style"
    category = "Language & Readability"
    default_weight = 0.07
    tool_version = "1.0"

    def _paragraph_scores(self, text: str) -> List[float]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [flesch_reading_ease(text)]
        scores = []
        for para in paragraphs:
            if len(tokenize_words(para)) < 5:
                continue
            scores.append(flesch_reading_ease(para))
        return scores or [flesch_reading_ease(text)]

    def run(self, ctx: ScreeningContext):
        overall_readability = flesch_reading_ease(ctx.text)
        paragraph_scores = self._paragraph_scores(ctx.text)
        oscillation = stdev(paragraph_scores) if len(paragraph_scores) > 1 else 0.0

        reasons: List[str] = []
        actions: List[str] = ["tag"]

        if overall_readability < 40:
            reasons.append(f"Text is hard to read (Flesch {overall_readability:.1f}).")
            actions.append("warn")
        elif overall_readability < 60:
            reasons.append(f"Text readability is moderate (Flesch {overall_readability:.1f}).")
        else:
            reasons.append(f"Good readability (Flesch {overall_readability:.1f}).")

        if oscillation < 5 and len(paragraph_scores) > 1:
            reasons.append("Readability is uniformly flat; consider varied sentence structure.")
        elif oscillation > 15:
            reasons.append("Readability oscillates widely between sections.")

        score = 0.0
        if overall_readability < 60:
            score += min(1.0, (60 - overall_readability) / 60)
        if oscillation < 5 and len(paragraph_scores) > 1:
            score += 0.1
        if oscillation > 20:
            score += 0.2

        raw_features = {
            "overall_readability": overall_readability,
            "oscillation": oscillation,
            "paragraph_scores": [round(s, 2) for s in paragraph_scores],
        }

        return self.result(ctx, clamp(score), reasons, raw_features=raw_features, actions=actions)
