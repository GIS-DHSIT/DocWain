"""Quality gate -- decides whether a trained model can be promoted to production."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    composite: float = 0.0


class QualityGate:
    def __init__(self, composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True):
        self._composite_minimum = composite_minimum
        self._criterion_floor = criterion_floor
        self._must_beat_previous = must_beat_previous

    def evaluate(self, composite, scores, previous_composite=None):
        failures = []
        if composite < self._composite_minimum:
            failures.append(f"Composite {composite:.1f} below minimum {self._composite_minimum:.1f}")
        for criterion, score in scores.items():
            if score < self._criterion_floor:
                failures.append(f"{criterion} score {score:.1f} below floor {self._criterion_floor:.1f}")
        if self._must_beat_previous and previous_composite is not None and composite <= previous_composite:
            failures.append(f"Composite {composite:.1f} does not beat previous {previous_composite:.1f}")
        if failures:
            return GateResult(passed=False, reason="; ".join(failures), composite=composite)
        return GateResult(passed=True, reason="", composite=composite)
