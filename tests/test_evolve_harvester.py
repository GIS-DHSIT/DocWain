# tests/test_evolve_harvester.py
import json
import pytest
from pathlib import Path


class TestHarvester:
    def test_load_observation_signals(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        iter_dir = tmp_path / "iter_1"
        iter_dir.mkdir()
        signal = {"signal_type": "table_weakness", "query": "test", "model_response": "resp",
                  "category": "document_understanding", "subcategory": "table_extraction",
                  "confidence_score": 0.4, "scores": {}, "timestamp": "2026-03-21T00:00:00"}
        (iter_dir / "observation_signals.jsonl").write_text(json.dumps(signal) + "\n")
        h = Harvester(signals_dir=tmp_path)
        signals = h.load_observation_signals(iteration=1)
        assert len(signals) == 1
        assert signals[0]["signal_type"] == "table_weakness"

    def test_load_interaction_signals(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        feedback_file = tmp_path / "feedback.jsonl"
        entry = {"messages": [{"role": "user", "content": "test"}], "metadata": {"feedback": "negative", "correction": "x" * 60}}
        feedback_file.write_text(json.dumps(entry) + "\n")
        h = Harvester(signals_dir=tmp_path)
        signals = h.load_interaction_signals(feedback_path=feedback_file)
        assert len(signals) == 1

    def test_merge_and_dedup(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        obs = [
            {"query": "What are Q3 figures?", "category": "doc_understanding", "subcategory": "table"},
            {"query": "What are Q3 figures?", "category": "doc_understanding", "subcategory": "table"},
        ]
        interaction = [
            {"query": "Different question", "category": "interaction", "subcategory": "feedback"},
        ]
        merged = h.merge_and_dedup(obs, interaction)
        assert len(merged) == 2

    def test_category_balance(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        signals = [{"query": f"q{i}", "category": "doc_understanding", "subcategory": "table"} for i in range(100)]
        signals += [{"query": "layout_q", "category": "doc_understanding", "subcategory": "layout"}]
        balanced = h.balance_categories(signals, max_per_subcategory=20)
        table_count = sum(1 for s in balanced if s["subcategory"] == "table")
        assert table_count <= 20
        assert any(s["subcategory"] == "layout" for s in balanced)

    def test_save_harvest(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        signals = [{"query": "test", "category": "doc", "subcategory": "table"}]
        h.save_harvest(signals, iteration=1)
        path = tmp_path / "iter_1" / "harvested_signals.jsonl"
        assert path.exists()

    def test_harvest_summary(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        signals = [
            {"query": "q1", "category": "document_understanding", "subcategory": "table_extraction"},
            {"query": "q2", "category": "document_understanding", "subcategory": "layout_parsing"},
            {"query": "q3", "category": "interaction_quality", "subcategory": "feedback"},
        ]
        summary = h.summarize(signals)
        assert summary["total"] == 3
        assert summary["by_category"]["document_understanding"] == 2
        assert summary["by_category"]["interaction_quality"] == 1
        assert "table_extraction" in summary["by_subcategory"]
