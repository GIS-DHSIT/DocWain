"""Verify document understanding processes full text, not just 8K."""
import pytest


def test_ollama_understand_uses_full_text():
    """The LLM prompt must include the full document text, not truncate at 8K."""
    from unittest.mock import MagicMock

    captured_prompts = []

    class MockLLM:
        def generate(self, prompt):
            captured_prompts.append(prompt)
            return '{"doc_summary": "test", "section_summaries": [], "key_entities": [], "key_facts": [], "intent_tags": []}'

    from src.doc_understanding.understand import _ollama_understand

    # Create a 20K char document
    long_text = "Important fact number one. " * 800  # ~20K chars
    long_text += "CRITICAL_MARKER_AT_END."

    _ollama_understand(long_text, "contract", model_name=None, llm_client=MockLLM())

    assert captured_prompts, "LLM should have been called"
    prompt = captured_prompts[0]
    assert "CRITICAL_MARKER_AT_END" in prompt, (
        f"Full text not passed to LLM — truncated at {len(prompt)} chars. "
        "The 8K truncation is dropping end-of-document content."
    )


def test_ollama_understand_context_window_large():
    """num_ctx must be large enough for full documents."""
    captured_options = []

    class MockLLM:
        def generate_with_metadata(self, prompt, options=None):
            captured_options.append(options or {})
            return '{"doc_summary": "ok", "section_summaries": [], "key_entities": [], "key_facts": [], "intent_tags": []}', {}

    from src.doc_understanding.understand import _ollama_understand

    text = "A" * 30000
    _ollama_understand(text, "invoice", model_name=None, llm_client=MockLLM())

    assert captured_options, "generate_with_metadata should have been called"
    opts = captured_options[0]
    assert opts.get("num_ctx", 0) >= 16384, f"num_ctx too small: {opts.get('num_ctx')}"
