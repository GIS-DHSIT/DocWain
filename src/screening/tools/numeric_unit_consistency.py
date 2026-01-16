from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Tuple

from ..models import ScreeningContext
from .base import ScreeningTool, clamp


NUM_UNIT_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%|[A-Za-z]{1,6})?")
METRIC_UNITS = {"km", "kg", "m", "cm", "mm", "g", "l"}
IMPERIAL_UNITS = {"mile", "mi", "lb", "ft", "inch", "in"}


class NumericUnitConsistencyTool(ScreeningTool):
    name = "numeric_unit_consistency"
    category = "Information Quality"
    default_weight = 0.07
    tool_version = "1.0"

    def _extract(self, text: str) -> List[Tuple[float, str]]:
        pairs: List[Tuple[float, str]] = []
        for match in NUM_UNIT_RE.finditer(text):
            try:
                value = float(match.group("num"))
            except (TypeError, ValueError):
                continue
            unit = (match.group("unit") or "").lower()
            pairs.append((value, unit))
        return pairs

    def run(self, ctx: ScreeningContext):
        pairs = self._extract(ctx.text)
        if not pairs:
            return self.result(
                ctx, 0.0, ["No numeric quantities detected."], raw_features={"count": 0}, actions=["tag"]
            )

        value_units = defaultdict(set)
        for value, unit in pairs:
            if unit:
                value_units[value].add(unit)

        conflicts = [value for value, units in value_units.items() if len(units) > 1]
        mixed_systems = any(u in METRIC_UNITS for _, u in pairs) and any(u in IMPERIAL_UNITS for _, u in pairs)
        unit_switches = sum(1 for idx in range(1, len(pairs)) if pairs[idx][1] != pairs[idx - 1][1])

        score = 0.0
        reasons: List[str] = []
        actions: List[str] = ["tag"]

        if conflicts:
            reasons.append(f"Same value reported with multiple units: {conflicts[:3]}.")
            score += 0.5
            actions.append("warn")
        if mixed_systems:
            reasons.append("Mixed metric and imperial units detected; ensure conversions are consistent.")
            score += 0.25
        if unit_switches > max(2, len(pairs) // 3):
            reasons.append("Frequent unit switching may indicate inconsistent reporting.")
            score += 0.2

        if not reasons:
            reasons.append("Numeric quantities and units look consistent.")

        raw_features = {
            "count": len(pairs),
            "conflicts": conflicts,
            "mixed_systems": mixed_systems,
            "unit_switches": unit_switches,
        }
        return self.result(ctx, clamp(score), reasons, raw_features=raw_features, actions=actions)
