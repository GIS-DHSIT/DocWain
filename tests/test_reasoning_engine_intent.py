"""Tests for the new intelligence pipeline components — QueryUnderstanding and dataclasses."""
import pytest

from src.intelligence.understand import (
    UnderstandResult,
    SubIntent,
    DomainHints,
    QueryUnderstanding,
    _make_default,
)


class TestUnderstandResultFields:
    """Verify UnderstandResult dataclass has required fields."""

    def test_has_primary_intent_field(self):
        r = _make_default("test query")
        assert hasattr(r, "primary_intent")
        assert r.primary_intent == "extract"

    def test_has_complexity_field(self):
        r = _make_default("test")
        assert hasattr(r, "complexity")
        assert r.complexity == "simple"

    def test_has_entities_field(self):
        r = _make_default("test")
        assert isinstance(r.entities, list)

    def test_has_sub_intents_field(self):
        r = _make_default("test")
        assert isinstance(r.sub_intents, list)

    def test_has_output_format_field(self):
        r = _make_default("test")
        assert hasattr(r, "output_format")
        assert r.output_format == "prose"

    def test_has_thinking_required_field(self):
        r = _make_default("test")
        assert hasattr(r, "thinking_required")
        assert r.thinking_required is False

    def test_has_needs_clarification_field(self):
        r = _make_default("test")
        assert r.needs_clarification is False

    def test_has_domain_hints(self):
        r = _make_default("test")
        assert isinstance(r.domain_hints, DomainHints)

    def test_resolved_query_set_from_input(self):
        r = _make_default("what is revenue?")
        assert r.resolved_query == "what is revenue?"


class TestSubIntentFields:
    """Verify SubIntent dataclass."""

    def test_sub_intent_creation(self):
        si = SubIntent(intent="list", target="all skills", scope="resume")
        assert si.intent == "list"
        assert si.target == "all skills"
        assert si.scope == "resume"


class TestDomainHintsFields:
    """Verify DomainHints dataclass."""

    def test_defaults(self):
        dh = DomainHints()
        assert dh.relevant_fields == []
        assert dh.terminology_context == ""

    def test_with_values(self):
        dh = DomainHints(relevant_fields=["revenue", "profit"], terminology_context="financial terms")
        assert "revenue" in dh.relevant_fields
        assert dh.terminology_context == "financial terms"


class TestQueryUnderstandingTrivialDetection:
    """Verify static trivial query detection."""

    def test_short_greeting_is_trivial(self):
        assert QueryUnderstanding.is_trivially_simple("hi", [])

    def test_short_question_is_trivial(self):
        assert QueryUnderstanding.is_trivially_simple("what is revenue?", [])

    def test_long_query_not_trivial(self):
        assert not QueryUnderstanding.is_trivially_simple(
            "Compare the revenue figures from Q1 and Q2 and show trends", []
        )

    def test_query_with_history_not_trivial(self):
        assert not QueryUnderstanding.is_trivially_simple(
            "yes", [{"role": "user", "content": "what is revenue?"}]
        )

    def test_compare_signal_not_trivial(self):
        assert not QueryUnderstanding.is_trivially_simple("compare A and B", [])

    def test_summarize_signal_not_trivial(self):
        assert not QueryUnderstanding.is_trivially_simple("summarize the doc", [])


class TestQueryUnderstandingJsonParsing:
    """Verify JSON parsing resilience."""

    def test_direct_json(self):
        raw = '{"primary_intent":"extract","sub_intents":[],"entities":["revenue"],"output_format":"prose","complexity":"simple","needs_clarification":false,"clarification_question":null,"resolved_query":"what is revenue","thinking_required":false,"domain_hints":{}}'
        result = QueryUnderstanding._try_parse_json(raw)
        assert result is not None
        assert result["primary_intent"] == "extract"

    def test_json_in_code_block(self):
        raw = '```json\n{"primary_intent":"list"}\n```'
        result = QueryUnderstanding._try_parse_json(raw)
        assert result is not None
        assert result["primary_intent"] == "list"

    def test_json_with_surrounding_text(self):
        raw = 'Here is my analysis:\n{"primary_intent":"compare"}\nDone.'
        result = QueryUnderstanding._try_parse_json(raw)
        assert result is not None
        assert result["primary_intent"] == "compare"

    def test_invalid_json_returns_none(self):
        result = QueryUnderstanding._try_parse_json("not json at all")
        assert result is None
