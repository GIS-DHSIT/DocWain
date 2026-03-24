"""Tests for visualization DPO preference pairs."""

import json
import re

import pytest

from src.finetune.dpo_data_generator import (
    build_dpo_dataset,
    generate_viz_preference_pairs,
)


class TestGenerateVizPreferencePairs:
    """Core generation tests."""

    def test_returns_at_least_10_pairs(self):
        pairs = generate_viz_preference_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) >= 10

    def test_pair_structure(self):
        """Each pair must have prompt, chosen, rejected."""
        for pair in generate_viz_preference_pairs():
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair

    def test_prompts_contain_evidence(self):
        """Every prompt should include [EVIDENCE]."""
        for pair in generate_viz_preference_pairs():
            assert "[EVIDENCE]" in pair["prompt"]


class TestChosenResponses:
    """Tests for the chosen (preferred) responses."""

    def test_every_chosen_has_viz_directive(self):
        for pair in generate_viz_preference_pairs():
            assert "<!--DOCWAIN_VIZ" in pair["chosen"], (
                f"Chosen response missing VIZ directive for prompt: {pair['prompt'][:60]}"
            )

    def test_every_chosen_has_markdown_table(self):
        for pair in generate_viz_preference_pairs():
            assert "|" in pair["chosen"], (
                f"Chosen response missing markdown table for prompt: {pair['prompt'][:60]}"
            )

    def test_viz_json_is_valid(self):
        """VIZ directives contain valid JSON with required fields."""
        pattern = re.compile(r"<!--DOCWAIN_VIZ\n(.+?)\n-->", re.DOTALL)
        for pair in generate_viz_preference_pairs():
            match = pattern.search(pair["chosen"])
            assert match, f"Could not parse VIZ directive in: {pair['chosen'][:80]}"
            payload = json.loads(match.group(1))
            assert "chart_type" in payload
            assert "title" in payload
            assert "labels" in payload
            assert "values" in payload
            assert isinstance(payload["labels"], list)
            assert isinstance(payload["values"], list)
            assert len(payload["labels"]) > 0
            assert len(payload["values"]) > 0


class TestRejectedResponses:
    """Tests for the rejected responses."""

    def test_no_rejected_has_viz_directive(self):
        for pair in generate_viz_preference_pairs():
            assert "<!--DOCWAIN_VIZ" not in pair["rejected"], (
                f"Rejected response should NOT have VIZ directive for prompt: {pair['prompt'][:60]}"
            )

    def test_rejected_is_plain_prose(self):
        """Rejected responses should not have markdown tables."""
        for pair in generate_viz_preference_pairs():
            # Rejected should be prose — no table pipes
            assert "| " not in pair["rejected"] or pair["rejected"].count("|") < 3


class TestChartTypeCoverage:
    """Ensure diverse chart types are covered."""

    REQUIRED_TYPES = {
        "bar", "donut", "line", "grouped_bar", "radar",
        "horizontal_bar", "waterfall", "scatter", "area",
        "gauge", "stacked_bar",
    }

    def test_all_chart_types_present(self):
        pattern = re.compile(r"<!--DOCWAIN_VIZ\n(.+?)\n-->", re.DOTALL)
        chart_types = set()
        for pair in generate_viz_preference_pairs():
            match = pattern.search(pair["chosen"])
            if match:
                payload = json.loads(match.group(1))
                chart_types.add(payload["chart_type"])
        missing = self.REQUIRED_TYPES - chart_types
        assert not missing, f"Missing chart types: {missing}"


class TestBuildDpoDatasetIntegration:
    """Verify viz pairs are included in the full dataset."""

    def test_dataset_includes_viz_pairs(self, tmp_path):
        result = build_dpo_dataset(output_dir=tmp_path)
        # The total should include viz pairs (at least 11 more than without)
        assert result["total"] >= 36  # 25 existing + 11 viz

    def test_viz_directives_present_in_output(self, tmp_path):
        build_dpo_dataset(output_dir=tmp_path)
        train_path = tmp_path / "dpo_train.jsonl"
        eval_path = tmp_path / "dpo_eval.jsonl"

        viz_count = 0
        for path in [train_path, eval_path]:
            with open(path) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if "<!--DOCWAIN_VIZ" in obj.get("chosen", ""):
                        viz_count += 1
        assert viz_count >= 10, f"Expected >= 10 viz pairs in output, got {viz_count}"
