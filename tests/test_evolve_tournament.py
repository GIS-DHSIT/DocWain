import json
import pytest
from pathlib import Path


class TestTournament:
    def test_compute_composite_score(self):
        from src.finetune.evolve.tournament import Tournament
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        scores = {"accuracy": 0.87, "groundedness": 0.83, "reasoning": 0.82, "formatting": 0.81, "tone": 0.79}
        composite = t.compute_composite(scores)
        expected = 0.87*0.30 + 0.83*0.25 + 0.82*0.20 + 0.81*0.15 + 0.79*0.10
        assert abs(composite - expected * 100) < 0.1

    def test_rank_models(self):
        from src.finetune.evolve.tournament import Tournament, ModelResult
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        results = [
            ModelResult(name="qwen3-8b", scores={"accuracy": 0.87, "groundedness": 0.83, "reasoning": 0.82, "formatting": 0.81, "tone": 0.79}),
            ModelResult(name="llama-8b", scores={"accuracy": 0.81, "groundedness": 0.78, "reasoning": 0.77, "formatting": 0.75, "tone": 0.73}),
        ]
        ranked = t.rank(results)
        assert ranked[0].name == "qwen3-8b"
        assert ranked[1].name == "llama-8b"
        assert ranked[0].composite > ranked[1].composite

    def test_empty_results(self):
        from src.finetune.evolve.tournament import Tournament
        t = Tournament(weights={"accuracy": 1.0})
        ranked = t.rank([])
        assert ranked == []

    def test_save_results(self, tmp_path):
        from src.finetune.evolve.tournament import Tournament, ModelResult
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        results = [ModelResult(name="qwen3-8b", scores={"accuracy": 0.87, "groundedness": 0.83, "reasoning": 0.82, "formatting": 0.81, "tone": 0.79})]
        ranked = t.rank(results)
        t.save_results(ranked, tmp_path / "tournament.json")
        assert (tmp_path / "tournament.json").exists()
        data = json.loads((tmp_path / "tournament.json").read_text())
        assert len(data["rankings"]) == 1
        assert data["rankings"][0]["name"] == "qwen3-8b"

    def test_per_category_best(self):
        from src.finetune.evolve.tournament import Tournament, ModelResult
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        results = [
            ModelResult(name="qwen3-8b", scores={"accuracy": 0.90, "groundedness": 0.70, "reasoning": 0.80, "formatting": 0.80, "tone": 0.80}),
            ModelResult(name="llama-8b", scores={"accuracy": 0.70, "groundedness": 0.95, "reasoning": 0.80, "formatting": 0.80, "tone": 0.80}),
        ]
        best = t.best_per_criterion(results)
        assert best["accuracy"] == "qwen3-8b"
        assert best["groundedness"] == "llama-8b"
