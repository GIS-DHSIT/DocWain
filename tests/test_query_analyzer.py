"""Tests for the algorithmic query geometry analyzer."""
import pytest

from src.docwain_intel.query_router import route_query
from src.docwain_intel.query_analyzer import analyze_query, QueryGeometry


def _analyze(query: str) -> QueryGeometry:
    """Helper: route then analyze a query."""
    analysis = route_query(query)
    return analyze_query(query, analysis)


class TestIntentType:
    def test_entity_lookup_possessive(self):
        g = _analyze("What is John's email?")
        assert g.intent_type == "entity_lookup"
        assert g.granularity >= 0.5, "specific single-field query should have high granularity"
        assert g.focus_type == "attribute_centric"
        assert "email" in g.requested_attributes

    def test_comparative(self):
        g = _analyze("Compare John and Sarah")
        assert g.intent_type == "comparative"
        assert g.is_comparison is True
        assert g.expected_entity_count >= 2

    def test_temporal_inquiry(self):
        g = _analyze("When did he start working?")
        assert g.intent_type == "temporal_inquiry"
        assert g.temporal_ordering is True
        assert g.question_word == "when"

    def test_process_inquiry(self):
        g = _analyze("How does the payment process work?")
        assert g.intent_type == "process_inquiry"
        assert g.focus_type == "process_centric"
        assert g.question_word == "how"

    def test_enumerative(self):
        g = _analyze("List all employees")
        assert g.intent_type == "enumerative"
        assert g.granularity <= 0.5, "enumerative queries should be detailed"

    def test_narrative_tell_me_about(self):
        g = _analyze("Tell me about John Smith")
        assert g.intent_type in ("narrative", "entity_lookup")
        assert g.granularity <= 0.5, "'tell me about' should push granularity low (detailed)"

    def test_causal_inquiry(self):
        g = _analyze("Why was the claim denied?")
        assert g.intent_type == "causal_inquiry"
        assert g.question_word == "why"

    def test_quantitative_aggregation(self):
        g = _analyze("How many invoices?")
        assert g.intent_type == "quantitative"
        assert g.is_aggregation is True
        assert g.question_word == "how"


class TestGranularity:
    def test_short_specific_high_granularity(self):
        g = _analyze("John's phone")
        assert g.granularity >= 0.5, "short possessive query should have high granularity"

    def test_long_vague_low_granularity(self):
        g = _analyze(
            "Can you give me a comprehensive overview of all the documents?"
        )
        assert g.granularity <= 0.3, "long vague query should have low granularity"


class TestEdgeCases:
    def test_empty_query(self):
        analysis = route_query("")
        g = analyze_query("", analysis)
        assert g.intent_type == "narrative"
        assert g.expected_entity_count == 0
        assert g.granularity == 0.5
        assert g.temporal_ordering is False
        assert g.requested_attributes == []

    def test_multiple_attributes(self):
        g = _analyze("What are the name, skills, and experience?")
        # Should find at least some of these as requested attributes.
        attrs_lower = [a.lower() for a in g.requested_attributes]
        found = sum(1 for a in ("name", "skill", "experience") if a in attrs_lower)
        assert found >= 2, (
            f"Expected at least 2 of name/skill/experience in {attrs_lower}"
        )


class TestFocusType:
    def test_who_entity_centric(self):
        g = _analyze("Who is the project manager?")
        assert g.focus_type == "entity_centric"

    def test_what_copula_attribute(self):
        g = _analyze("What is the total revenue?")
        assert g.focus_type == "attribute_centric"

    def test_possessive_attribute(self):
        g = _analyze("John's salary")
        assert g.focus_type == "attribute_centric"


class TestTemporalOrdering:
    def test_date_entity_triggers_temporal(self):
        g = _analyze("What happened on March 15, 2024?")
        assert g.temporal_ordering is True

    def test_temporal_lemma(self):
        g = _analyze("Show me the recent changes")
        assert g.temporal_ordering is True


class TestQuestionWord:
    def test_where(self):
        g = _analyze("Where is the headquarters located?")
        assert g.question_word == "where"

    def test_no_question_word(self):
        g = _analyze("John's phone number")
        assert g.question_word is None
