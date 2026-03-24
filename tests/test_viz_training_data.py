"""Tests for visualization SFT training data generator."""

import json
import re

import pytest

from src.finetune.viz_training_data import (
    SYSTEM_PROMPT,
    _msg,
    _viz,
    generate_viz_sft_examples,
    write_viz_sft_dataset,
)


class TestGenerateVizSftExamples:
    """Core generation tests."""

    def test_returns_list_with_minimum_count(self):
        examples = generate_viz_sft_examples()
        assert isinstance(examples, list)
        assert len(examples) >= 50

    def test_message_structure(self):
        """Each example has messages with system/user/assistant roles."""
        examples = generate_viz_sft_examples()
        for ex in examples:
            assert "messages" in ex
            msgs = ex["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"

    def test_system_prompt_matches(self):
        examples = generate_viz_sft_examples()
        for ex in examples:
            assert ex["messages"][0]["content"] == SYSTEM_PROMPT

    def test_deterministic_with_same_seed(self):
        a = generate_viz_sft_examples(seed=42)
        b = generate_viz_sft_examples(seed=42)
        assert len(a) == len(b)
        for ea, eb in zip(a, b):
            assert ea["messages"][1]["content"] == eb["messages"][1]["content"]


class TestVizDirectives:
    """Tests for chart/visualization examples."""

    def test_minimum_viz_count(self):
        """At least 30 examples contain a VIZ directive."""
        examples = generate_viz_sft_examples()
        viz_count = sum(
            1 for e in examples
            if "<!--DOCWAIN_VIZ" in e["messages"][2]["content"]
        )
        assert viz_count >= 30

    def test_viz_json_is_valid(self):
        """VIZ directives contain valid JSON with required fields."""
        examples = generate_viz_sft_examples()
        pattern = re.compile(r"<!--DOCWAIN_VIZ\n(.+?)\n-->", re.DOTALL)
        for ex in examples:
            content = ex["messages"][2]["content"]
            match = pattern.search(content)
            if match:
                payload = json.loads(match.group(1))
                assert "chart_type" in payload
                assert "title" in payload
                assert "labels" in payload
                assert "values" in payload
                assert isinstance(payload["labels"], list)
                assert isinstance(payload["values"], list)
                assert len(payload["labels"]) > 0
                assert len(payload["values"]) > 0

    def test_all_thirteen_chart_types_covered(self):
        """All 13 required chart types must be present."""
        required = {
            "bar", "horizontal_bar", "grouped_bar", "stacked_bar",
            "donut", "line", "multi_line", "area", "scatter",
            "radar", "waterfall", "gauge", "treemap",
        }
        examples = generate_viz_sft_examples()
        pattern = re.compile(r"<!--DOCWAIN_VIZ\n(.+?)\n-->", re.DOTALL)
        chart_types = set()
        for ex in examples:
            match = pattern.search(ex["messages"][2]["content"])
            if match:
                payload = json.loads(match.group(1))
                chart_types.add(payload["chart_type"])
        missing = required - chart_types
        assert not missing, f"Missing chart types: {missing}"

    def test_viz_examples_have_tables(self):
        """Chart examples should include markdown tables."""
        examples = generate_viz_sft_examples()
        viz_examples = [
            e for e in examples
            if "<!--DOCWAIN_VIZ" in e["messages"][2]["content"]
        ]
        with_tables = sum(1 for e in viz_examples if "|" in e["messages"][2]["content"])
        assert with_tables >= len(viz_examples) * 0.8, "Most viz examples should have tables"


class TestFlowExamples:
    """Tests for flow analysis examples."""

    def test_minimum_flow_count(self):
        """At least 5 flow analysis examples (contain arrow, no VIZ)."""
        examples = generate_viz_sft_examples()
        flow_count = sum(
            1 for e in examples
            if "\u2192" in e["messages"][2]["content"]
            and "<!--DOCWAIN_VIZ" not in e["messages"][2]["content"]
        )
        assert flow_count >= 5

    def test_flow_examples_have_numbered_steps(self):
        """Flow examples should contain numbered steps."""
        examples = generate_viz_sft_examples()
        for ex in examples:
            content = ex["messages"][2]["content"]
            if "\u2192" in content and "<!--DOCWAIN_VIZ" not in content:
                assert re.search(r"\d+\.", content), "Flow example missing numbered steps"


class TestTextOnlyExamples:
    """Tests for text-only examples."""

    def test_minimum_text_only_count(self):
        """At least 10 text-only examples (no VIZ, no arrow)."""
        examples = generate_viz_sft_examples()
        text_count = sum(
            1 for e in examples
            if "<!--DOCWAIN_VIZ" not in e["messages"][2]["content"]
            and "\u2192" not in e["messages"][2]["content"]
        )
        assert text_count >= 10

    def test_text_only_are_concise(self):
        """Text-only examples should be relatively short answers."""
        examples = generate_viz_sft_examples()
        for ex in examples:
            content = ex["messages"][2]["content"]
            if "<!--DOCWAIN_VIZ" not in content and "\u2192" not in content:
                # Should not be excessively long
                assert len(content) < 1000, "Text-only answer unexpectedly long"


class TestNoRealContent:
    """Ensure no real document content leaks into training data."""

    def test_no_subscription_id(self):
        examples = generate_viz_sft_examples()
        for ex in examples:
            for msg in ex["messages"]:
                assert "subscription_id" not in msg["content"]

    def test_no_profile_id(self):
        examples = generate_viz_sft_examples()
        for ex in examples:
            for msg in ex["messages"]:
                assert "profile_id" not in msg["content"]


class TestHelpers:
    """Test helper functions."""

    def test_viz_helper(self):
        result = _viz("bar", "Test", ["A", "B"], [1, 2])
        assert "<!--DOCWAIN_VIZ" in result
        assert "-->" in result
        payload = json.loads(result.split("\n", 1)[1].rsplit("\n", 1)[0])
        assert payload["chart_type"] == "bar"
        assert payload["labels"] == ["A", "B"]
        assert payload["values"] == [1, 2]

    def test_viz_helper_with_optional_fields(self):
        result = _viz("grouped_bar", "T", ["X"], [1], unit="$", secondary_values=[2], secondary_name="Y")
        payload = json.loads(result.split("\n", 1)[1].rsplit("\n", 1)[0])
        assert payload["unit"] == "$"
        assert payload["secondary_values"] == [2]
        assert payload["secondary_name"] == "Y"

    def test_msg_helper(self):
        result = _msg("Hello", "World")
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["content"] == "Hello"
        assert result["messages"][2]["content"] == "World"


class TestWriteDataset:
    """Test dataset writing."""

    def test_creates_files(self, tmp_path):
        result = write_viz_sft_dataset(output_dir=tmp_path, seed=42)
        assert result["total"] >= 50
        assert result["train_count"] > result["eval_count"]
        assert (tmp_path / "viz_sft_train.jsonl").exists()
        assert (tmp_path / "viz_sft_eval.jsonl").exists()

    def test_output_is_valid_jsonl(self, tmp_path):
        write_viz_sft_dataset(output_dir=tmp_path, seed=42)
        for fname in ["viz_sft_train.jsonl", "viz_sft_eval.jsonl"]:
            with open(tmp_path / fname) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    assert "messages" in obj

    def test_90_10_split(self, tmp_path):
        result = write_viz_sft_dataset(output_dir=tmp_path, seed=42)
        total = result["total"]
        expected_train = int(total * 0.9)
        assert result["train_count"] == expected_train
        assert result["eval_count"] == total - expected_train
