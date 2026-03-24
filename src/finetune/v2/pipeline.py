"""V2 Pipeline Orchestrator — manages the four-phase fine-tuning pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Each phase is considered complete when its marker artifact exists.
PHASE_MARKERS: Dict[str, str] = {
    "phase1": "projection.pt",
    "phase2": "phase2_config.json",
    "phase3": "phase3_config.json",
    "phase4": "phase4_plan.json",
}


class V2Pipeline:
    """Orchestrates the V2 multi-phase fine-tuning pipeline.

    Parameters
    ----------
    base_dir : Path, optional
        Root directory where phase artifacts are stored.
        Defaults to ``finetune_artifacts/v2``.
    """

    phases: List[str] = ["phase1", "phase2", "phase3", "phase4"]

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path("finetune_artifacts/v2")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _completed_phases(self) -> List[str]:
        """Return the list of phases whose marker artifact exists."""
        completed: List[str] = []
        for phase in self.phases:
            marker = PHASE_MARKERS.get(phase)
            if marker and (self.base_dir / phase / marker).exists():
                completed.append(phase)
        return completed

    def next_phase(self) -> Optional[str]:
        """Return the first phase that has not been completed, or ``None``."""
        completed = set(self._completed_phases())
        for phase in self.phases:
            if phase not in completed:
                return phase
        return None

    def status(self) -> Dict:
        """Return a status dictionary describing pipeline progress."""
        completed = self._completed_phases()
        nxt = self.next_phase()
        # current_phase is only set when a phase is actively running
        # (i.e. its directory exists but its marker artifact does not).
        current: Optional[str] = None
        if nxt and (self.base_dir / nxt).exists():
            current = nxt
        return {
            "current_phase": current,
            "completed_phases": completed,
            "next_phase": nxt,
            "base_dir": str(self.base_dir),
        }
