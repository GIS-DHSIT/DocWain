"""Tests for intelligence_v2 prompt templates."""

from src.intelligence_v2.prompts import build_analysis_prompt


def test_build_analysis_prompt_contains_text():
    """Input text appears in the generated prompt."""
    text = "The quarterly revenue was $4.2 million in Q3 2025."
    prompt = build_analysis_prompt(text, "report.pdf", "financial")
    assert "The quarterly revenue was $4.2 million in Q3 2025." in prompt


def test_build_analysis_prompt_requests_json_keys():
    """All 7 required JSON keys are mentioned in the prompt."""
    prompt = build_analysis_prompt("Some text", "doc.pdf", "general")
    required_keys = [
        "document_type",
        "language",
        "summary",
        "section_summaries",
        "entities",
        "facts",
        "relationships",
        "answerable_topics",
    ]
    for key in required_keys:
        assert key in prompt, f"Key '{key}' not found in prompt"


def test_build_analysis_prompt_truncates_long_text():
    """50K char input results in a prompt under 40K chars."""
    long_text = "A" * 50_000
    prompt = build_analysis_prompt(long_text, "big.pdf", "general")
    assert len(prompt) < 40_000


def test_build_analysis_prompt_entity_roles():
    """The word 'role' is mentioned in the prompt for entity extraction."""
    prompt = build_analysis_prompt("Some document text.", "doc.pdf", "general")
    assert "role" in prompt.lower()
