"""Tests for behavioral fine-tuning data generator."""
import json
import tempfile
from pathlib import Path

import pytest


def test_generate_identity_examples():
    from src.finetune.behavioral_data_generator import generate_identity_examples
    examples = generate_identity_examples()
    assert len(examples) >= 25
    for ex in examples:
        assert "messages" in ex
        msgs = ex["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        # Identity responses mention DocWain, documents, or DHS
        content_lower = msgs[2]["content"].lower()
        assert "docwain" in content_lower or "document" in content_lower or "dhs" in content_lower


def test_generate_pipeline_examples():
    from src.finetune.behavioral_data_generator import generate_pipeline_examples
    examples = generate_pipeline_examples()
    assert len(examples) >= 15
    for ex in examples:
        msgs = ex["messages"]
        assert len(msgs) == 3


def test_generate_formatting_examples():
    from src.finetune.behavioral_data_generator import generate_formatting_examples
    examples = generate_formatting_examples()
    assert len(examples) >= 15
    has_bold = any("**" in ex["messages"][2]["content"] for ex in examples)
    has_table = any("|" in ex["messages"][2]["content"] for ex in examples)
    assert has_bold
    assert has_table


def test_generate_feature_examples():
    from src.finetune.behavioral_data_generator import generate_feature_examples
    examples = generate_feature_examples()
    assert len(examples) >= 10


def test_generate_domain_examples():
    from src.finetune.behavioral_data_generator import generate_domain_examples
    examples = generate_domain_examples()
    assert len(examples) >= 10
    domains_seen = set()
    for ex in examples:
        content = ex["messages"][2]["content"].lower()
        for d in ["resume", "invoice", "contract", "medical", "financial"]:
            if d in content:
                domains_seen.add(d)
    assert len(domains_seen) >= 3


def test_generate_gap_handling_examples():
    from src.finetune.behavioral_data_generator import generate_gap_handling_examples
    examples = generate_gap_handling_examples()
    assert len(examples) >= 10
    has_gap_indicator = any(
        "not found" in ex["messages"][2]["content"].lower()
        or "don't address" in ex["messages"][2]["content"].lower()
        or "insufficient" in ex["messages"][2]["content"].lower()
        or "don't cover" in ex["messages"][2]["content"].lower()
        or "don't contain" in ex["messages"][2]["content"].lower()
        or "couldn't find" in ex["messages"][2]["content"].lower()
        for ex in examples
    )
    assert has_gap_indicator


def test_generate_isolation_examples():
    from src.finetune.behavioral_data_generator import generate_isolation_examples
    examples = generate_isolation_examples()
    assert len(examples) >= 10
    for ex in examples:
        resp = ex["messages"][2]["content"]
        assert "subscription_id" not in resp or "don't expose" in resp.lower() or "don't reveal" in resp.lower()


def test_generate_conversation_examples():
    from src.finetune.behavioral_data_generator import generate_conversation_examples
    examples = generate_conversation_examples()
    assert len(examples) >= 15


def test_build_behavioral_dataset_writes_files():
    from src.finetune.behavioral_data_generator import build_behavioral_dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_behavioral_dataset(output_dir=Path(tmpdir))
        train_path = Path(tmpdir) / "behavioral_train.jsonl"
        eval_path = Path(tmpdir) / "behavioral_eval.jsonl"
        assert train_path.exists()
        assert eval_path.exists()
        assert result["train_count"] >= 100
        assert result["eval_count"] >= 10
        with open(train_path) as f:
            for line in f:
                obj = json.loads(line)
                assert "messages" in obj


def test_system_prompt_is_consistent():
    from src.finetune.behavioral_data_generator import BEHAVIORAL_SYSTEM_PROMPT
    assert "DocWain" in BEHAVIORAL_SYSTEM_PROMPT
    assert "document intelligence" in BEHAVIORAL_SYSTEM_PROMPT.lower()


def test_dhs_company_knowledge():
    """Verify DHS IT Solutions company knowledge is baked into training data."""
    from src.finetune.behavioral_data_generator import generate_identity_examples
    examples = generate_identity_examples()
    all_responses = " ".join(ex["messages"][2]["content"] for ex in examples)
    assert "DHS IT Solutions" in all_responses
    assert "Sreekanth Kamtam" in all_responses
    assert "Rajasekar" in all_responses
    assert "2016" in all_responses
    assert "Newcastle" in all_responses
