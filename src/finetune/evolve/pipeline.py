"""EvolvePipeline — orchestrates the full observe->harvest->teach->train->gate cycle."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvolveConfig
from .observer import Observer
from .harvester import Harvester
from .teacher import Teacher
from .trainer import MultiModelTrainer
from .tournament import Tournament
from .distiller import Distiller
from .gate import QualityGate, GateResult
from .registry import ModelRegistry
from .deployer import Deployer


class EvolvePipeline:
    """Main orchestrator for interactive Claude Code sessions."""

    def __init__(self, config: EvolveConfig, signals_dir: Path, artifact_dir: Path, registry_path: Path) -> None:
        self._config = config
        self._signals_dir = Path(signals_dir)
        self._artifact_dir = Path(artifact_dir)

        self.observer = Observer(config, output_dir=self._signals_dir)
        self.harvester = Harvester(signals_dir=self._signals_dir)
        self.teacher = Teacher(output_dir=self._signals_dir)
        self.trainer = MultiModelTrainer(config, artifact_dir=self._artifact_dir)
        self.tournament = Tournament(weights=config.gate.weights)
        self.distiller = Distiller(distill_every_n=config.pipeline.distillation_every_n)
        self.gate = QualityGate(
            composite_minimum=config.gate.composite_minimum,
            criterion_floor=config.gate.criterion_floor,
            must_beat_previous=config.gate.must_beat_previous,
        )
        self.registry = ModelRegistry(registry_path)
        self.deployer = Deployer(ollama_host=config.deployment.ollama_host)

    # ------------------------------------------------------------------
    # Iteration tracking
    # ------------------------------------------------------------------

    def current_iteration(self) -> int:
        """Return the highest iter_N found in signals_dir, or 0 if none."""
        if not self._signals_dir.exists():
            return 0
        iters: List[int] = []
        for d in self._signals_dir.iterdir():
            if d.is_dir():
                match = re.match(r"iter_(\d+)", d.name)
                if match:
                    iters.append(int(match.group(1)))
        return max(iters) if iters else 0

    def next_iteration(self) -> int:
        return self.current_iteration() + 1

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a summary dict of the pipeline's current state."""
        return {
            "current_iteration": self.current_iteration(),
            "next_iteration": self.next_iteration(),
            "enabled_students": [s.name for s in self.trainer.list_enabled_students()],
            "registry_models": [
                {"tag": m.tag, "score": m.composite_score, "status": m.status}
                for m in self.registry.list_models()
            ],
            "gate_config": {
                "composite_minimum": self._config.gate.composite_minimum,
                "criterion_floor": self._config.gate.criterion_floor,
            },
            "distillation_due": self.distiller.should_distill(self.next_iteration()),
        }

    # ------------------------------------------------------------------
    # Step runners
    # ------------------------------------------------------------------

    def run_observe(self, prompts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run the observer step: probe the model and save signals."""
        iteration = self.next_iteration()
        signals = self.observer.probe_model(prompts)
        self.observer._save_signals(signals, iteration)
        return {
            "iteration": iteration,
            "signals_found": len(signals),
            "weak_areas": self._summarize_weak_areas(signals),
        }

    def run_harvest(self, iteration: int) -> Dict[str, Any]:
        """Merge observation + interaction signals, balance, and save."""
        obs_signals = self.harvester.load_observation_signals(iteration)
        int_signals = self.harvester.load_interaction_signals()
        merged = self.harvester.merge_and_dedup(obs_signals, int_signals)
        balanced = self.harvester.balance_categories(merged)
        self.harvester.save_harvest(balanced, iteration)
        return self.harvester.summarize(balanced)

    def run_gate(self, composite: float, scores: Dict[str, float]) -> GateResult:
        """Evaluate a model against the quality gate."""
        previous = self.registry.get("DocWain:latest")
        prev_composite = previous.composite_score if previous else None
        return self.gate.evaluate(composite, scores, prev_composite)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarize_weak_areas(self, signals) -> Dict[str, int]:
        areas: Dict[str, int] = {}
        for s in signals:
            sub = s.subcategory
            areas[sub] = areas.get(sub, 0) + 1
        return areas
