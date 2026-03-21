"""Tournament — ranks trained models by weighted composite score."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ModelResult:
    name: str
    scores: Dict[str, float]
    composite: float = 0.0


class Tournament:
    def __init__(self, weights: Dict[str, float]):
        self._weights = weights

    def compute_composite(self, scores: Dict[str, float]) -> float:
        raw = sum(scores.get(k, 0) * w for k, w in self._weights.items())
        return round(raw * 100, 2)

    def rank(self, results: List[ModelResult]) -> List[ModelResult]:
        for r in results:
            r.composite = self.compute_composite(r.scores)
        return sorted(results, key=lambda r: r.composite, reverse=True)

    def best_per_criterion(self, results: List[ModelResult]) -> Dict[str, str]:
        best = {}
        for criterion in self._weights:
            top = max(results, key=lambda r: r.scores.get(criterion, 0))
            best[criterion] = top.name
        return best

    def save_results(self, ranked: List[ModelResult], path: Path) -> None:
        data = {
            "rankings": [{"name": r.name, "composite": r.composite, "scores": r.scores} for r in ranked],
            "weights": self._weights,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
