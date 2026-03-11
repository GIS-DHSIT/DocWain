"""Tests for the evaluation harness."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.finetune.evaluate_model import (
    evaluate_model,
    identify_weak_categories,
    generate_augmentation_data,
    _load_eval_set,
    _identify_weak_from_results,
    PASS_COMPOSITE,
    MARGINAL_COMPOSITE,
)


def _write_eval_file(tmp_path: Path, examples: list) -> Path:
    """Helper to write eval examples to a JSONL file."""
    path = tmp_path / "eval.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path


def _make_example(query: str, intent: str, domain: str, entities: list = None):
    """Build a chat-format eval example."""
    gt = {
        "intent": intent,
        "domain": domain,
        "output_format": "paragraph",
        "entities": entities or [],
        "constraints": {},
        "scope": "all_documents",
        "complexity": "medium",
        "confidence": 0.95,
    }
    return {
        "messages": [
            {"role": "system", "content": "Parse query into TaskSpec JSON."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(gt)},
        ]
    }


class TestLoadEvalSet:
    """Test evaluation set loading."""

    def test_loads_valid_jsonl(self, tmp_path):
        examples = [
            _make_example("List candidates", "extract", "hr"),
            _make_example("What medications?", "extract", "medical"),
        ]
        path = _write_eval_file(tmp_path, examples)
        loaded = _load_eval_set(path)
        assert len(loaded) == 2
        assert loaded[0]["query"] == "List candidates"
        assert loaded[0]["ground_truth"]["intent"] == "extract"

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "eval.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_make_example("Q1", "factual", "hr")) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(_make_example("Q2", "compare", "hr")) + "\n")
        loaded = _load_eval_set(path)
        assert len(loaded) == 2

    def test_empty_file(self, tmp_path):
        path = tmp_path / "eval.jsonl"
        path.write_text("")
        loaded = _load_eval_set(path)
        assert len(loaded) == 0


class TestIdentifyWeakCategories:
    """Test weak category identification from results."""

    def test_identifies_weak_intent(self):
        results = [
            {"ground_truth": {"intent": "compare", "domain": "hr"}, "intent_match": False, "domain_match": True},
            {"ground_truth": {"intent": "compare", "domain": "hr"}, "intent_match": False, "domain_match": True},
            {"ground_truth": {"intent": "compare", "domain": "hr"}, "intent_match": False, "domain_match": True},
            {"ground_truth": {"intent": "factual", "domain": "hr"}, "intent_match": True, "domain_match": True},
            {"ground_truth": {"intent": "factual", "domain": "hr"}, "intent_match": True, "domain_match": True},
        ]
        weak = _identify_weak_from_results(results)
        assert "compare" in weak

    def test_identifies_weak_domain(self):
        results = [
            {"ground_truth": {"intent": "factual", "domain": "medical"}, "intent_match": True, "domain_match": False},
            {"ground_truth": {"intent": "factual", "domain": "medical"}, "intent_match": True, "domain_match": False},
            {"ground_truth": {"intent": "factual", "domain": "medical"}, "intent_match": True, "domain_match": False},
        ]
        weak = _identify_weak_from_results(results)
        assert "medical" in weak

    def test_no_weak_when_all_correct(self):
        results = [
            {"ground_truth": {"intent": "factual", "domain": "hr"}, "intent_match": True, "domain_match": True},
            {"ground_truth": {"intent": "factual", "domain": "hr"}, "intent_match": True, "domain_match": True},
        ]
        weak = _identify_weak_from_results(results)
        assert len(weak) == 0


class TestEvaluateModel:
    """Test full model evaluation with mocked model calls."""

    def test_perfect_score(self, tmp_path):
        """Model returning exact ground truth should score 100%."""
        examples = [
            _make_example("List candidates", "extract", "hr", ["candidates"]),
            _make_example("What medications?", "extract", "medical", ["medications"]),
        ]
        path = _write_eval_file(tmp_path, examples)

        # Mock model to return exact ground truth
        def mock_call(model_name, query):
            for ex in examples:
                if ex["messages"][1]["content"] == query:
                    return json.loads(ex["messages"][2]["content"])
            return None

        with patch("src.finetune.evaluate_model._call_model", side_effect=lambda m, q: mock_call(m, q)):
            result = evaluate_model(model_name="test", eval_set_path=path)

        assert result["json_parse_rate"] == 100.0
        assert result["intent_accuracy"] == 100.0
        assert result["domain_accuracy"] == 100.0
        assert result["composite_score"] >= PASS_COMPOSITE
        assert result["verdict"] == "pass"

    def test_zero_parse_rate(self, tmp_path):
        """Model returning garbage should get 0% parse rate."""
        examples = [
            _make_example("List candidates", "extract", "hr"),
            _make_example("What medications?", "extract", "medical"),
        ]
        path = _write_eval_file(tmp_path, examples)

        with patch("src.finetune.evaluate_model._call_model", return_value=None):
            result = evaluate_model(model_name="test", eval_set_path=path)

        assert result["json_parse_rate"] == 0.0
        assert result["verdict"] == "fail"

    def test_marginal_score(self, tmp_path):
        """Model with partial accuracy should get marginal verdict."""
        examples = [
            _make_example("List candidates", "extract", "hr", ["candidates"]),
            _make_example("What medications?", "extract", "medical", ["medications"]),
            _make_example("Compare contracts", "compare", "legal"),
            _make_example("Summarize invoices", "summarize", "invoice"),
        ]
        path = _write_eval_file(tmp_path, examples)

        # Return partially correct: right domain, wrong intent
        def mock_call(model_name, query):
            return {
                "intent": "factual",  # always wrong for non-factual
                "domain": "hr",  # always hr (sometimes wrong)
                "entities": [],
                "constraints": {},
            }

        with patch("src.finetune.evaluate_model._call_model", side_effect=lambda m, q: mock_call(m, q)):
            result = evaluate_model(model_name="test", eval_set_path=path)

        assert 0 < result["composite_score"] < 100
        assert result["verdict"] in ("marginal", "fail")


class TestGenerateAugmentationData:
    """Test targeted augmentation data generation."""

    def test_generates_examples(self):
        augmented = generate_augmentation_data(["hr", "compare"], count=50)
        assert isinstance(augmented, list)
        assert len(augmented) <= 50

    def test_empty_weak_categories(self):
        augmented = generate_augmentation_data([], count=50)
        assert len(augmented) == 0

    def test_format_is_valid(self):
        augmented = generate_augmentation_data(["medical"], count=20)
        for ex in augmented:
            assert "messages" in ex
            assert len(ex["messages"]) == 3
            # Verify assistant content is valid JSON
            parsed = json.loads(ex["messages"][2]["content"])
            assert "intent" in parsed


class TestIdentifyWeakCategoriesFromEvalResult:
    """Test the public identify_weak_categories() function."""

    def test_extracts_from_result(self):
        eval_result = {"weak_categories": ["compare", "medical"]}
        weak = identify_weak_categories(eval_result)
        assert weak == ["compare", "medical"]

    def test_empty_when_missing(self):
        assert identify_weak_categories({}) == []
