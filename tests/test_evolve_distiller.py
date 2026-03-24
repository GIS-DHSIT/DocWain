# tests/test_evolve_distiller.py
import json
import pytest
from pathlib import Path


class TestDistiller:
    def test_should_distill_every_nth_iteration(self):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        assert not d.should_distill(iteration=1)
        assert not d.should_distill(iteration=2)
        assert d.should_distill(iteration=3)
        assert not d.should_distill(iteration=4)
        assert d.should_distill(iteration=6)

    def test_cherry_pick_best_responses(self):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        eval_results = {
            "model_a": [
                {"query": "q1", "response": "a_resp1", "scores": {"accuracy": 0.95, "groundedness": 0.60}},
                {"query": "q2", "response": "a_resp2", "scores": {"accuracy": 0.90, "groundedness": 0.65}},
            ],
            "model_b": [
                {"query": "q1", "response": "b_resp1", "scores": {"accuracy": 0.70, "groundedness": 0.95}},
                {"query": "q2", "response": "b_resp2", "scores": {"accuracy": 0.65, "groundedness": 0.90}},
            ],
        }
        best = d.cherry_pick(eval_results, criterion="accuracy")
        assert len(best) == 2
        assert best[0]["response"] == "a_resp1"

    def test_build_distillation_dataset(self):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        d._system_prompt = "You are DocWain."
        best_responses = [
            {"query": "q1", "response": "best resp 1"},
            {"query": "q2", "response": "best resp 2"},
        ]
        dataset = d.build_dataset(best_responses)
        assert len(dataset) == 2
        assert dataset[0]["messages"][0]["role"] == "system"
        assert dataset[0]["messages"][1]["role"] == "user"
        assert dataset[0]["messages"][2]["role"] == "assistant"

    def test_save_distillation_dataset(self, tmp_path):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        dataset = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        path = d.save_dataset(dataset, tmp_path / "distill.jsonl")
        assert path.exists()
        with open(path) as f:
            data = json.loads(f.readline())
        assert "messages" in data
