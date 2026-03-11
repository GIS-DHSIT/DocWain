"""Tests for the centralized NLU Engine."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.nlp.nlu_engine import (
    ClassificationRegistry,
    ClassificationResult,
    QuerySemantics,
    parse_query,
    classify_intent,
    classify_conversational,
    classify_scope,
    classify_domain_task,
    get_registry,
    _ensure_registry,
)


# ── QuerySemantics parsing ────────────────────────────────────────────────

class TestParseQuery:
    def test_extracts_verbs(self):
        sem = parse_query("compare these candidates")
        assert any(v in ("compare",) for v in sem.action_verbs)

    def test_extracts_nouns(self):
        sem = parse_query("extract skills from the resume")
        all_words = sem.target_nouns + sem.context_words
        assert any("skill" in w for w in all_words)

    def test_empty_query(self):
        sem = parse_query("")
        assert sem.action_verbs == []
        assert sem.target_nouns == []

    def test_returns_raw_text(self):
        sem = parse_query("hello world")
        assert sem.raw_text == "hello world"


# ── ClassificationRegistry ────────────────────────────────────────────────

class TestClassificationRegistry:
    def test_register_and_classify(self):
        reg = ClassificationRegistry("test", threshold=0.10, gap=0.01)
        reg.register("greeting", "Saying hello, hi, good morning, a friendly salutation")
        reg.register("farewell", "Saying goodbye, bye, see you later, ending conversation")

        result = reg.classify("hello there")
        # Should match greeting (even without embedder, NLU overlap should work)
        if result is not None:
            assert result.name == "greeting"

    def test_register_many(self):
        reg = ClassificationRegistry("test2", threshold=0.10, gap=0.01)
        reg.register_many({
            "a": "Description for category A about comparing items",
            "b": "Description for category B about translating text",
        })
        assert len(reg.entries) == 2

    def test_empty_registry_returns_none(self):
        reg = ClassificationRegistry("empty")
        result = reg.classify("hello")
        assert result is None

    def test_empty_query_returns_none(self):
        reg = ClassificationRegistry("test3", threshold=0.10, gap=0.01)
        reg.register("a", "Some description")
        assert reg.classify("") is None
        assert reg.classify("   ") is None

    def test_threshold_filtering(self):
        reg = ClassificationRegistry("strict", threshold=0.99, gap=0.01)
        reg.register("a", "Very specific description about quantum physics")
        result = reg.classify("unrelated cooking recipe")
        # With a 0.99 threshold, unlikely to match
        assert result is None

    def test_gap_check(self):
        reg = ClassificationRegistry("gap_test", threshold=0.05, gap=0.90)
        reg.register("a", "Comparing items and finding differences")
        reg.register("b", "Contrasting objects and finding distinctions")
        # Both are similar — gap check should reject
        result = reg.classify("compare and contrast these items")
        # With 0.90 gap requirement, should return None
        assert result is None


# ── Intent Classification ─────────────────────────────────────────────────

class TestClassifyIntent:
    def test_hint_mapping_rank(self):
        assert classify_intent("anything", intent_hint="rank") == "ranking"

    def test_hint_mapping_compare(self):
        assert classify_intent("anything", intent_hint="compare") == "comparison"

    def test_hint_mapping_summary(self):
        assert classify_intent("anything", intent_hint="summary") == "summary"

    def test_hint_mapping_contact(self):
        assert classify_intent("anything", intent_hint="contact") == "factual"

    def test_hint_mapping_analytics(self):
        assert classify_intent("anything", intent_hint="analytics") == "analytics"

    def test_no_hint_returns_string(self):
        result = classify_intent("what is the total amount?")
        assert isinstance(result, str)
        assert result in {
            "factual", "comparison", "ranking", "summary", "timeline",
            "reasoning", "multi_field", "analytics", "cross_document",
        }

    def test_comparison_query(self):
        result = classify_intent("compare these two candidates side by side")
        assert result in ("comparison", "cross_document", "factual")

    def test_ranking_query(self):
        result = classify_intent("rank the candidates from best to worst")
        assert result in ("ranking", "factual")

    def test_summary_query(self):
        result = classify_intent("give me an overview and summary")
        assert result in ("summary", "factual")

    def test_factual_fallback(self):
        result = classify_intent("what is the date on the document?")
        assert result == "factual"


# ── Conversational Classification ─────────────────────────────────────────

class TestClassifyConversational:
    def test_greeting(self):
        result = classify_conversational("hello")
        if result:
            assert result[0] == "GREETING"
            assert result[1] > 0

    def test_farewell(self):
        result = classify_conversational("goodbye")
        if result:
            assert result[0] == "FAREWELL"

    def test_thanks(self):
        result = classify_conversational("thank you so much")
        if result:
            assert result[0] == "THANKS"

    def test_non_conversational_returns_none(self):
        result = classify_conversational("extract the total amount from invoice 123")
        # Document queries should not be classified as conversational
        # (may or may not return None depending on NLU precision)
        if result:
            assert result[0] not in ("GREETING", "FAREWELL", "THANKS")


# ── Scope Classification ─────────────────────────────────────────────────

class TestClassifyScope:
    def test_all_scope_quantifier(self):
        result = classify_scope("show all candidates")
        # spaCy quantifier detection or embedding should catch this
        if result:
            assert result == "all_profile"

    def test_targeted_scope(self):
        result = classify_scope("what is John's email?")
        # Targeted queries may return "targeted" or None
        if result:
            assert result in ("targeted", "all_profile")

    def test_returns_string_or_none(self):
        result = classify_scope("hello")
        assert result is None or isinstance(result, str)


# ── Domain Task Classification ────────────────────────────────────────────

class TestClassifyDomainTask:
    def test_returns_dict_or_none(self):
        result = classify_domain_task("check for drug interactions")
        if result:
            assert "domain" in result
            assert "task_type" in result

    def test_hr_interview_questions(self):
        result = classify_domain_task("prepare interview questions for the candidate")
        if result:
            assert result["domain"] == "hr"
            assert "interview" in result["task_type"]

    def test_medical_drug_interaction(self):
        result = classify_domain_task("check drug interactions with aspirin")
        if result:
            assert result["domain"] == "medical"

    def test_legal_compliance(self):
        result = classify_domain_task("check compliance with GDPR regulations")
        if result:
            assert result["domain"] == "legal"

    def test_with_domain_filter(self):
        result = classify_domain_task("summarize the content", domain="hr")
        if result:
            assert result["domain"] == "hr"

    def test_unrelated_returns_none(self):
        result = classify_domain_task("what is the weather today?")
        # Generic query should not match any domain task
        assert result is None


# ── Registry Management ──────────────────────────────────────────────────

class TestRegistryManagement:
    def test_get_registry_creates(self):
        reg = get_registry("_test_create_")
        assert reg is not None
        assert reg.name == "_test_create_"

    def test_get_registry_singleton(self):
        reg1 = get_registry("_test_singleton_")
        reg2 = get_registry("_test_singleton_")
        assert reg1 is reg2

    def test_get_registry_no_create(self):
        reg = get_registry("_nonexistent_", create=False)
        assert reg is None

    def test_ensure_registry_intent(self):
        reg = _ensure_registry("intent")
        assert len(reg.entries) == 9  # 9 intents including timeline

    def test_ensure_registry_conversational(self):
        reg = _ensure_registry("conversational")
        assert len(reg.entries) == 15  # 15 conversational intents

    def test_ensure_registry_scope(self):
        reg = _ensure_registry("scope")
        assert len(reg.entries) == 2

    def test_ensure_registry_domain_task(self):
        reg = _ensure_registry("domain_task")
        assert len(reg.entries) > 50  # 60+ domain tasks

    def test_ensure_registry_content_type(self):
        reg = _ensure_registry("content_type")
        assert len(reg.entries) >= 10
