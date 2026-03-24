"""Tests for the recursive visualization fine-tuning loop.

All tests work OFFLINE -- no GPU, no Ollama, no model required.
Tests cover: config defaults, dataset generation, state tracking.
"""

import json
import os
import pytest
from pathlib import Path

from src.finetune.viz_finetune_loop import (
    VizFinetuneConfig,
    VizFinetuneState,
    IterationResult,
    generate_iteration_dataset,
)


# ---------------------------------------------------------------------------
# VizFinetuneConfig defaults
# ---------------------------------------------------------------------------


class TestVizFinetuneConfigDefaults:
    """Config dataclass has correct defaults."""

    def test_max_iterations_at_least_3(self):
        config = VizFinetuneConfig()
        assert config.max_iterations >= 3

    def test_pass_threshold_is_80(self):
        config = VizFinetuneConfig()
        assert config.pass_threshold == 80.0

    def test_marginal_threshold_is_60(self):
        config = VizFinetuneConfig()
        assert config.marginal_threshold == 60.0

    def test_data_refresh_each_iteration_is_true(self):
        config = VizFinetuneConfig()
        assert config.data_refresh_each_iteration is True

    def test_base_model_default(self):
        config = VizFinetuneConfig()
        assert "unsloth" in config.base_model

    def test_model_name_default(self):
        config = VizFinetuneConfig()
        assert config.model_name == "DHS/DocWain"

    def test_sft_epochs_positive(self):
        config = VizFinetuneConfig()
        assert config.sft_epochs > 0

    def test_dpo_epochs_positive(self):
        config = VizFinetuneConfig()
        assert config.dpo_epochs > 0

    def test_dpo_beta_positive(self):
        config = VizFinetuneConfig()
        assert config.dpo_beta > 0

    def test_learning_rate_positive(self):
        config = VizFinetuneConfig()
        assert config.learning_rate > 0

    def test_dpo_learning_rate_positive(self):
        config = VizFinetuneConfig()
        assert config.dpo_learning_rate > 0

    def test_lora_r_positive(self):
        config = VizFinetuneConfig()
        assert config.lora_r > 0

    def test_custom_override(self):
        config = VizFinetuneConfig(max_iterations=10, pass_threshold=90.0)
        assert config.max_iterations == 10
        assert config.pass_threshold == 90.0


# ---------------------------------------------------------------------------
# generate_iteration_dataset
# ---------------------------------------------------------------------------


class TestGenerateIterationDataset:
    """Dataset generation produces valid files with expected structure."""

    def test_returns_dict_with_required_keys(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path),
        )
        assert "sft_path" in result
        assert "dpo_path" in result
        assert "sft_count" in result
        assert "dpo_count" in result
        assert "iteration" in result
        assert "seed" in result

    def test_sft_count_positive(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path),
        )
        assert result["sft_count"] > 0

    def test_dpo_count_positive(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path),
        )
        assert result["dpo_count"] > 0

    def test_sft_file_exists(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path),
        )
        assert Path(result["sft_path"]).exists()

    def test_dpo_file_exists(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path),
        )
        assert Path(result["dpo_path"]).exists()

    def test_sft_file_has_valid_jsonl(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path),
        )
        with open(result["sft_path"]) as f:
            lines = f.readlines()
        assert len(lines) > 0
        # Each line should be valid JSON
        for line in lines[:3]:
            data = json.loads(line)
            assert "messages" in data

    def test_dpo_file_has_valid_jsonl(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path),
        )
        with open(result["dpo_path"]) as f:
            lines = f.readlines()
        assert len(lines) > 0
        for line in lines[:3]:
            data = json.loads(line)
            assert "prompt" in data
            assert "chosen" in data
            assert "rejected" in data

    def test_different_iterations_produce_datasets(self, tmp_path):
        r1 = generate_iteration_dataset(
            iteration=1, seed=42, output_dir=str(tmp_path / "iter1"),
        )
        r2 = generate_iteration_dataset(
            iteration=2, seed=42, output_dir=str(tmp_path / "iter2"),
        )
        assert r1["sft_count"] > 0
        assert r2["sft_count"] > 0
        assert r1["iteration"] == 1
        assert r2["iteration"] == 2
        # Seed should differ
        assert r1["seed"] != r2["seed"]

    def test_iteration_stored_in_result(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=3, seed=42, output_dir=str(tmp_path),
        )
        assert result["iteration"] == 3

    def test_seed_computed_from_iteration(self, tmp_path):
        result = generate_iteration_dataset(
            iteration=2, seed=42, output_dir=str(tmp_path),
        )
        assert result["seed"] == 42 + 2 * 1000


# ---------------------------------------------------------------------------
# VizFinetuneState tracking
# ---------------------------------------------------------------------------


class TestVizFinetuneState:
    """State tracking: record_iteration, best_score, has_passed."""

    def test_initial_state(self):
        state = VizFinetuneState()
        assert state.iterations == []
        assert state.best_score == 0.0
        assert state.best_iteration == 0

    def test_record_iteration_appends(self):
        state = VizFinetuneState()
        state.record_iteration(iteration=1, score=65.0, verdict="marginal")
        assert len(state.iterations) == 1
        assert state.iterations[0].iteration == 1
        assert state.iterations[0].composite_score == 65.0
        assert state.iterations[0].verdict == "marginal"

    def test_best_score_updates_on_improvement(self):
        state = VizFinetuneState()
        state.record_iteration(iteration=1, score=50.0, verdict="fail")
        state.record_iteration(iteration=2, score=75.0, verdict="marginal")
        state.record_iteration(iteration=3, score=60.0, verdict="marginal")
        assert state.best_score == 75.0
        assert state.best_iteration == 2

    def test_best_score_does_not_decrease(self):
        state = VizFinetuneState()
        state.record_iteration(iteration=1, score=80.0, verdict="pass")
        state.record_iteration(iteration=2, score=70.0, verdict="marginal")
        assert state.best_score == 80.0
        assert state.best_iteration == 1

    def test_has_passed_true(self):
        state = VizFinetuneState()
        state.record_iteration(iteration=1, score=85.0, verdict="pass")
        assert state.has_passed(80.0) is True

    def test_has_passed_false(self):
        state = VizFinetuneState()
        state.record_iteration(iteration=1, score=60.0, verdict="marginal")
        assert state.has_passed(80.0) is False

    def test_has_passed_exact_threshold(self):
        state = VizFinetuneState()
        state.record_iteration(iteration=1, score=80.0, verdict="pass")
        assert state.has_passed(80.0) is True

    def test_has_passed_no_iterations(self):
        state = VizFinetuneState()
        assert state.has_passed(80.0) is False

    def test_record_iteration_returns_result(self):
        state = VizFinetuneState()
        result = state.record_iteration(
            iteration=1, score=72.0, verdict="marginal",
            chart_avg=68.0, text_avg=80.0,
            sft_count=50, dpo_count=30,
            duration_seconds=120.5,
        )
        assert isinstance(result, IterationResult)
        assert result.iteration == 1
        assert result.composite_score == 72.0
        assert result.chart_avg == 68.0
        assert result.text_avg == 80.0
        assert result.sft_count == 50
        assert result.dpo_count == 30
        assert result.duration_seconds == 120.5

    def test_multiple_iterations_tracked(self):
        state = VizFinetuneState()
        for i in range(1, 6):
            state.record_iteration(iteration=i, score=float(i * 15), verdict="fail")
        assert len(state.iterations) == 5
        assert state.best_score == 75.0
        assert state.best_iteration == 5


# ---------------------------------------------------------------------------
# IterationResult dataclass
# ---------------------------------------------------------------------------


class TestIterationResult:
    """IterationResult holds expected fields."""

    def test_fields(self):
        r = IterationResult(
            iteration=1,
            composite_score=82.5,
            verdict="pass",
            chart_avg=80.0,
            text_avg=90.0,
            sft_count=45,
            dpo_count=30,
            duration_seconds=300.0,
        )
        assert r.iteration == 1
        assert r.composite_score == 82.5
        assert r.verdict == "pass"
        assert r.chart_avg == 80.0
        assert r.text_avg == 90.0
        assert r.sft_count == 45
        assert r.dpo_count == 30
        assert r.duration_seconds == 300.0
