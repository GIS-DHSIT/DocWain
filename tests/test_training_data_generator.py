"""Tests for training data generation."""

import json
import pytest
from pathlib import Path

from src.finetune.training_data_generator import (
    generate_curated_examples,
    generate_paraphrases,
    generate_edge_cases,
    build_training_dataset,
    _chat_example,
    _expand_paraphrases_offline,
    SYSTEM_PROMPT,
    DatasetBuildResult,
)


class TestCuratedExamples:
    """Test curated example generation from intensive test queries."""

    def test_returns_list(self):
        examples = generate_curated_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_count_matches_intensive_test(self):
        """Should have ~96 curated examples (one per intensive test query)."""
        examples = generate_curated_examples()
        assert len(examples) >= 80  # at least 80 curated

    def test_chat_format(self):
        """Each example has messages with system/user/assistant roles."""
        examples = generate_curated_examples()
        for ex in examples[:5]:
            assert "messages" in ex
            msgs = ex["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"

    def test_system_prompt_matches(self):
        """System prompt in examples matches the inference-time prompt."""
        examples = generate_curated_examples()
        for ex in examples[:3]:
            assert ex["messages"][0]["content"] == SYSTEM_PROMPT

    def test_assistant_is_valid_json(self):
        """Assistant content should be valid JSON TaskSpec."""
        examples = generate_curated_examples()
        for ex in examples:
            content = ex["messages"][2]["content"]
            parsed = json.loads(content)
            assert "intent" in parsed
            assert "domain" in parsed
            assert "confidence" in parsed


class TestParaphraseGeneration:
    """Test domain:task paraphrase expansion."""

    def test_returns_list(self):
        examples = generate_paraphrases()
        assert isinstance(examples, list)

    def test_generates_substantial_count(self):
        """Should generate 500+ paraphrased examples."""
        examples = generate_paraphrases()
        assert len(examples) >= 500

    def test_valid_chat_format(self):
        examples = generate_paraphrases()
        for ex in examples[:10]:
            assert "messages" in ex
            assert len(ex["messages"]) == 3

    def test_covers_multiple_domains(self):
        """Paraphrases should cover HR, medical, legal, invoice, etc."""
        examples = generate_paraphrases()
        domains = set()
        for ex in examples:
            content = json.loads(ex["messages"][2]["content"])
            domains.add(content.get("domain", ""))
        assert len(domains) >= 5


class TestEdgeCases:
    """Test edge case generation."""

    def test_returns_list(self):
        examples = generate_edge_cases()
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_includes_redirect(self):
        """Should include off-topic redirect examples."""
        examples = generate_edge_cases()
        intents = set()
        for ex in examples:
            content = json.loads(ex["messages"][2]["content"])
            intents.add(content.get("intent", ""))
        assert "redirect" in intents

    def test_includes_clarify(self):
        """Should include ambiguous/clarify examples."""
        examples = generate_edge_cases()
        intents = set()
        for ex in examples:
            content = json.loads(ex["messages"][2]["content"])
            intents.add(content.get("intent", ""))
        assert "clarify" in intents

    def test_includes_constraints(self):
        """Some edge cases should have non-empty constraints."""
        examples = generate_edge_cases()
        has_constraints = any(
            json.loads(ex["messages"][2]["content"]).get("constraints")
            for ex in examples
        )
        assert has_constraints


class TestOfflineParaphrases:
    """Test the template-based paraphrase expansion."""

    def test_returns_requested_count(self):
        result = _expand_paraphrases_offline("Test query", count=10)
        assert len(result) <= 10

    def test_includes_original(self):
        result = _expand_paraphrases_offline("Test query", count=5)
        assert "Test query" in result

    def test_no_duplicates(self):
        result = _expand_paraphrases_offline("Check for issues", count=20)
        lowered = [r.lower() for r in result]
        assert len(lowered) == len(set(lowered))


class TestChatExample:
    """Test the chat example builder."""

    def test_format(self):
        ex = _chat_example("Hello", {"intent": "redirect", "confidence": 0.9})
        assert ex["messages"][0]["role"] == "system"
        assert ex["messages"][1]["content"] == "Hello"
        parsed = json.loads(ex["messages"][2]["content"])
        assert parsed["intent"] == "redirect"


class TestBuildDataset:
    """Test full dataset build."""

    def test_build_creates_files(self, tmp_path):
        result = build_training_dataset(output_dir=tmp_path, eval_split=0.1)
        assert isinstance(result, DatasetBuildResult)
        assert result.train_path is not None
        assert result.eval_path is not None
        assert result.train_path.exists()
        assert result.eval_path.exists()
        assert result.total_examples > 0
        assert result.train_count > result.eval_count

    def test_build_source_counts(self, tmp_path):
        result = build_training_dataset(output_dir=tmp_path, eval_split=0.1)
        assert "curated" in result.source_counts
        assert "paraphrases" in result.source_counts
        assert "edge_cases" in result.source_counts
        assert result.source_counts["curated"] >= 80

    def test_output_is_valid_jsonl(self, tmp_path):
        result = build_training_dataset(output_dir=tmp_path, eval_split=0.1)
        with open(result.train_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                assert "messages" in obj
