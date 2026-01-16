from __future__ import annotations

from typing import List

from ..models import ScreeningContext
from .base import ScreeningTool, clamp, tokenize_words

MODAL_WORDS = {
    "may",
    "might",
    "could",
    "approximately",
    "roughly",
    "perhaps",
    "potentially",
    "possibly",
    "somewhat",
    "largely",
    "generally",
    "about",
    "various",
    "often",
    "frequently",
}


class AmbiguityVaguenessTool(ScreeningTool):
    name = "ambiguity_vagueness"
    category = "Information Quality"
    default_weight = 0.06
    tool_version = "1.0"

    def run(self, ctx: ScreeningContext):
        words = tokenize_words(ctx.text)
        total_words = max(len(words), 1)
        modal_count = sum(1 for w in words if w in MODAL_WORDS)
        density = modal_count / total_words

        reasons: List[str] = []
        actions: List[str] = ["tag"]

        if density > 0.08:
            reasons.append(f"High vagueness/hedging density ({density:.2%}).")
            actions.append("warn")
        elif density > 0.04:
            reasons.append(f"Moderate vagueness detected ({density:.2%}).")
        else:
            reasons.append("Low vagueness density.")

        score = clamp(density * 5)  # 0.04 density -> 0.2 score, 0.1 density -> 0.5 score
        raw_features = {"modal_count": modal_count, "word_count": total_words, "density": density}
        return self.result(ctx, score, reasons, raw_features=raw_features, actions=actions)
