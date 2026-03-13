"""Tests for FormatEnforcer — deterministic post-processing of LLM output."""
import pytest

from src.intelligence.format_enforcer import FormatEnforcer


class TestFormatEnforcer:
    """Verify FormatEnforcer fixes common markdown issues."""

    def setup_method(self):
        self.enforcer = FormatEnforcer()

    def test_enforce_returns_string(self):
        result = self.enforcer.enforce("Hello world", "prose")
        assert isinstance(result, str)

    def test_enforce_preserves_clean_text(self):
        text = "This is a clean response with no issues."
        assert self.enforcer.enforce(text, "prose") == text

    def test_normalizes_bullet_styles(self):
        text = "* Item one\n* Item two\n* Item three"
        result = self.enforcer.enforce(text, "bullets")
        assert "Item one" in result
        assert "Item two" in result
        # Should normalize * to -
        assert "- Item one" in result

    def test_fixes_broken_table_separator(self):
        text = "| Name | Value |\n|---|---|\n| A | 1 |\n| B | 2 |"
        result = self.enforcer.enforce(text, "table")
        assert "|" in result
        assert "Name" in result

    def test_empty_input(self):
        assert self.enforcer.enforce("", "prose") == ""

    def test_renumbers_broken_list(self):
        text = "1. First\n3. Second\n7. Third"
        result = self.enforcer.enforce(text, "numbered")
        assert "1. First" in result
        assert "2. Second" in result
        assert "3. Third" in result

    def test_sections_format(self):
        text = "Introduction\n## Section One\nContent here"
        result = self.enforcer.enforce(text, "sections")
        assert "## Section One" in result

    def test_merges_adjacent_source_citations(self):
        text = "The revenue is $10M [SOURCE-1][SOURCE-2] total."
        result = self.enforcer.enforce(text, "prose")
        # Should merge adjacent citations
        assert "[SOURCE-1, SOURCE-2]" in result

    def test_closes_unclosed_bold(self):
        text = "This is **bold text without closing"
        result = self.enforcer.enforce(text, "prose")
        # Should close the unclosed **
        count = result.count("**")
        assert count % 2 == 0


class TestIntentGuidanceRemoved:
    """Verify old _intent_guidance method no longer exists on ReasoningEngine."""

    def test_intent_guidance_removed(self):
        from src.intelligence.reasoning_engine import ReasoningEngine
        assert not hasattr(ReasoningEngine, "_intent_guidance"), (
            "_intent_guidance should be removed; formatting is now handled by FormatEnforcer"
        )
