import json
import pytest
from pathlib import Path


class TestObserverPrompts:
    def test_get_eval_prompts_returns_list(self):
        from src.finetune.evolve.prompts.observer_prompts import get_eval_prompts
        prompts = get_eval_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) >= 30

    def test_eval_prompts_have_required_fields(self):
        from src.finetune.evolve.prompts.observer_prompts import get_eval_prompts
        for p in get_eval_prompts():
            assert "query" in p
            assert "category" in p
            assert "subcategory" in p
            assert p["category"] in ("document_understanding", "interaction_quality")

    def test_prompts_cover_all_subcategories(self):
        from src.finetune.evolve.prompts.observer_prompts import get_eval_prompts
        subcats = {p["subcategory"] for p in get_eval_prompts()}
        assert "table_extraction" in subcats
        assert "layout_parsing" in subcats
        assert "cross_reference" in subcats
        assert "section_hierarchy" in subcats
        assert "uncertainty_handling" in subcats


class TestObserver:
    def test_score_response_returns_criteria_dict(self):
        from src.finetune.evolve.observer import Observer
        obs = Observer.__new__(Observer)
        obs._weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        score = obs._score_response(
            query="What are the Q3 figures from the table?",
            response="The Q3 revenue was $1.2M as shown in the financial table on page 3.",
            category="document_understanding",
            subcategory="table_extraction",
        )
        assert "accuracy" in score
        assert "groundedness" in score
        assert "reasoning" in score
        assert "formatting" in score
        assert "tone" in score
        for v in score.values():
            assert 0.0 <= v <= 1.0

    def test_build_signal_from_observation(self):
        from src.finetune.evolve.observer import Observer, ObservationSignal
        obs = Observer.__new__(Observer)
        obs._weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        signal = obs._build_signal(
            query="test query",
            response="test response",
            category="document_understanding",
            subcategory="table_extraction",
            scores={"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.5, "formatting": 0.6, "tone": 0.7},
        )
        assert isinstance(signal, ObservationSignal)
        assert signal.signal_type == "table_extraction_weakness"
        assert signal.category == "document_understanding"
        assert signal.confidence_score < 1.0

    def test_classify_weakness_threshold(self):
        from src.finetune.evolve.observer import Observer
        obs = Observer.__new__(Observer)
        obs._weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        assert obs._is_weak({"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.3, "formatting": 0.5, "tone": 0.5})
        assert not obs._is_weak({"accuracy": 0.9, "groundedness": 0.9, "reasoning": 0.9, "formatting": 0.9, "tone": 0.9})

    def test_save_signals_to_dir(self, tmp_path):
        from src.finetune.evolve.observer import Observer, ObservationSignal
        obs = Observer.__new__(Observer)
        obs._output_dir = tmp_path
        signals = [
            ObservationSignal(
                signal_type="table_extraction_weakness",
                query="test",
                model_response="test resp",
                category="document_understanding",
                subcategory="table_extraction",
                confidence_score=0.35,
                scores={"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.3, "formatting": 0.5, "tone": 0.5},
            )
        ]
        obs._save_signals(signals, iteration=1)
        signal_dir = tmp_path / "iter_1"
        assert signal_dir.exists()
        assert (signal_dir / "observation_signals.jsonl").exists()
        with open(signal_dir / "observation_signals.jsonl") as f:
            data = json.loads(f.readline())
        assert data["signal_type"] == "table_extraction_weakness"
