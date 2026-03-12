"""Tests for dynamic formatting principles in ReasoningEngine (Task 6)."""
import pytest

from src.intelligence.reasoning_engine import ReasoningEngine, _FORMAT_PRINCIPLES


class TestFormatPrinciples:
    """Verify _FORMAT_PRINCIPLES constant exists and contains key rules."""

    def test_format_principles_exists(self):
        assert isinstance(_FORMAT_PRINCIPLES, str)
        assert len(_FORMAT_PRINCIPLES) > 50

    def test_contains_table_rule(self):
        assert "table" in _FORMAT_PRINCIPLES.lower()

    def test_contains_bullet_rule(self):
        assert "bullet" in _FORMAT_PRINCIPLES.lower()

    def test_contains_adapt_language(self):
        assert "adapt" in _FORMAT_PRINCIPLES.lower()

    def test_contains_citation_rule(self):
        assert "SOURCE" in _FORMAT_PRINCIPLES or "cite" in _FORMAT_PRINCIPLES.lower()

    def test_contains_prose_guidance(self):
        assert "prose" in _FORMAT_PRINCIPLES.lower()

    def test_contains_length_matching(self):
        assert "length" in _FORMAT_PRINCIPLES.lower() or "complexity" in _FORMAT_PRINCIPLES.lower()


class TestIntentGuidanceRemoved:
    """Verify _intent_guidance method no longer exists on ReasoningEngine."""

    def test_intent_guidance_removed(self):
        assert not hasattr(ReasoningEngine, "_intent_guidance"), (
            "_intent_guidance should be removed; formatting is now principle-based"
        )
