"""Tests for the intelligent USAGE_HELP module.

Covers sub-intent classification, example bank structure, context-aware
example selection, response composition, integration with conversational_nlp,
and edge cases.
"""
from __future__ import annotations

import pytest

from src.intelligence.conversational_nlp import (
    CAPABILITY,
    USAGE_HELP,
    ConversationalContext,
    classify_conversational_intent,
    collect_context,
    compose_response,
    generate_conversational_response,
)
from src.intelligence.usage_help import (
    ADVANCED_FEATURES,
    CAPABILITY_OVERVIEW,
    CONTENT_GENERATION_HELP,
    DOMAIN_EXAMPLES,
    FILE_TYPES,
    QUERY_EXAMPLES,
    QUICK_START,
    SCREENING_HELP,
    TASK_HELP,
    UPLOAD_HELP,
    ExampleQuery,
    HelpSubIntent,
    _EXAMPLE_BANK,
    classify_help_sub_intent,
    compose_usage_help_response,
    select_examples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    doc_count: int = 0,
    domains: list = None,
    names: list = None,
) -> ConversationalContext:
    """Build a ConversationalContext for testing."""
    domains = domains or []
    names = names or []
    return ConversationalContext(
        document_count=doc_count,
        document_names=names,
        dominant_domains=domains,
        profile_is_empty=(doc_count == 0),
        is_first_message=True,
        is_returning_user=False,
        conversation_turn_count=0,
        time_of_day="morning",
        total_points=doc_count * 10,
    )


# ---------------------------------------------------------------------------
# A. TestHelpSubIntentClassifier
# ---------------------------------------------------------------------------

class TestHelpSubIntentClassifier:
    """Sub-intent classifier tests."""

    def test_quick_start_bare_help(self):
        result = classify_help_sub_intent("help")
        assert result.name == QUICK_START

    def test_quick_start_getting_started(self):
        result = classify_help_sub_intent("getting started")
        assert result.name == QUICK_START

    def test_upload_help(self):
        result = classify_help_sub_intent("how do I upload a document?")
        assert result.name == UPLOAD_HELP

    def test_file_types(self):
        result = classify_help_sub_intent("what file types are supported?")
        assert result.name == FILE_TYPES

    def test_file_types_can_upload(self):
        result = classify_help_sub_intent("can I upload images?")
        assert result.name == FILE_TYPES

    def test_query_examples(self):
        result = classify_help_sub_intent("what can I ask?")
        assert result.name == QUERY_EXAMPLES

    def test_show_examples(self):
        result = classify_help_sub_intent("show me examples")
        assert result.name == QUERY_EXAMPLES

    def test_domain_examples_resume(self):
        result = classify_help_sub_intent("resume examples")
        assert result.name == DOMAIN_EXAMPLES
        assert result.domain_hint == "resume"

    def test_domain_examples_invoice(self):
        result = classify_help_sub_intent("show me invoice examples")
        assert result.name == DOMAIN_EXAMPLES
        assert result.domain_hint == "invoice"

    def test_task_help_compare(self):
        result = classify_help_sub_intent("how do I compare documents?")
        assert result.name == TASK_HELP
        assert result.task_hint == "compare"

    def test_task_help_rank(self):
        result = classify_help_sub_intent("how can I rank candidates?")
        assert result.name == TASK_HELP
        assert result.task_hint == "rank"

    def test_screening_help(self):
        result = classify_help_sub_intent("screening help")
        assert result.name == SCREENING_HELP

    def test_screening_pii(self):
        result = classify_help_sub_intent("check for PII")
        assert result.name == SCREENING_HELP

    def test_content_generation_help(self):
        result = classify_help_sub_intent("how can I generate content?")
        assert result.name == CONTENT_GENERATION_HELP

    def test_capability_overview(self):
        result = classify_help_sub_intent("what else can you do?")
        assert result.name == CAPABILITY_OVERVIEW

    def test_advanced_features(self):
        result = classify_help_sub_intent("advanced features")
        assert result.name == ADVANCED_FEATURES

    def test_finetuning_help(self):
        result = classify_help_sub_intent("fine-tuning help")
        assert result.name == ADVANCED_FEATURES

    def test_domain_hint_extraction(self):
        result = classify_help_sub_intent("show me legal examples")
        assert result.domain_hint == "legal"

    def test_task_hint_extraction(self):
        result = classify_help_sub_intent("how to summarize?")
        assert result.task_hint == "summarize"


# ---------------------------------------------------------------------------
# B. TestExampleBank
# ---------------------------------------------------------------------------

class TestExampleBank:
    """Example bank structure tests."""

    def test_all_expected_domains_present(self):
        expected = {"resume", "invoice", "legal", "medical", "general"}
        assert expected.issubset(set(_EXAMPLE_BANK.keys()))

    def test_resume_has_minimum_examples(self):
        total = sum(len(v) for v in _EXAMPLE_BANK["resume"].values())
        assert total >= 15

    def test_invoice_has_minimum_examples(self):
        total = sum(len(v) for v in _EXAMPLE_BANK["invoice"].values())
        assert total >= 8

    def test_no_duplicate_queries_within_domain(self):
        for domain, tasks in _EXAMPLE_BANK.items():
            queries = []
            for examples in tasks.values():
                queries.extend(e.query for e in examples)
            assert len(queries) == len(set(queries)), f"Duplicate in {domain}"

    def test_all_examples_have_description(self):
        for domain, tasks in _EXAMPLE_BANK.items():
            for task_type, examples in tasks.items():
                for ex in examples:
                    assert ex.description, f"Missing description: {ex.query}"

    def test_example_query_is_dataclass(self):
        ex = _EXAMPLE_BANK["resume"]["qa"][0]
        assert isinstance(ex, ExampleQuery)
        assert ex.domain == "resume"
        assert ex.task_type == "qa"


# ---------------------------------------------------------------------------
# C. TestContextAwareSelection
# ---------------------------------------------------------------------------

class TestContextAwareSelection:
    """Context-aware example selection tests."""

    def test_empty_profile_returns_general(self):
        ctx = _ctx(doc_count=0)
        sub = HelpSubIntent(name=QUERY_EXAMPLES)
        examples = select_examples(ctx, sub, seed=42, max_examples=5)
        assert len(examples) > 0
        assert all(e.domain == "general" for e in examples)

    def test_domain_hint_filters_to_domain(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        sub = HelpSubIntent(name=DOMAIN_EXAMPLES, domain_hint="invoice")
        examples = select_examples(ctx, sub, seed=42, max_examples=5)
        assert all(e.domain == "invoice" for e in examples)

    def test_task_hint_filters_to_task(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        sub = HelpSubIntent(name=TASK_HELP, task_hint="compare")
        examples = select_examples(ctx, sub, seed=42, max_examples=5)
        assert all(e.task_type == "compare" for e in examples)

    def test_dominant_domain_preference(self):
        ctx = _ctx(doc_count=10, domains=["resume", "invoice"])
        sub = HelpSubIntent(name=QUERY_EXAMPLES)
        examples = select_examples(ctx, sub, seed=42, max_examples=5)
        # At least one should be from the primary domain.
        domains = {e.domain for e in examples}
        assert "resume" in domains

    def test_seed_determinism(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        sub = HelpSubIntent(name=QUERY_EXAMPLES)
        ex1 = select_examples(ctx, sub, seed=999, max_examples=5)
        ex2 = select_examples(ctx, sub, seed=999, max_examples=5)
        assert [e.query for e in ex1] == [e.query for e in ex2]

    def test_different_seeds_different_results(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        sub = HelpSubIntent(name=QUERY_EXAMPLES)
        ex1 = select_examples(ctx, sub, seed=1, max_examples=5)
        ex2 = select_examples(ctx, sub, seed=999999, max_examples=5)
        # Not guaranteed to be different, but with enough examples they should be.
        q1 = [e.query for e in ex1]
        q2 = [e.query for e in ex2]
        # At least a statistical check — they shouldn't ALL be identical.
        # (With 20+ resume examples, two different seeds should diverge.)
        assert q1 != q2 or len(q1) == 0

    def test_max_examples_respected(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        sub = HelpSubIntent(name=QUERY_EXAMPLES)
        examples = select_examples(ctx, sub, seed=42, max_examples=3)
        assert len(examples) <= 3

    def test_no_domain_no_docs_returns_all(self):
        ctx = _ctx(doc_count=5, domains=[])
        sub = HelpSubIntent(name=QUERY_EXAMPLES)
        examples = select_examples(ctx, sub, seed=42, max_examples=5)
        assert len(examples) > 0


# ---------------------------------------------------------------------------
# D. TestResponseComposition
# ---------------------------------------------------------------------------

class TestResponseComposition:
    """Response composition tests."""

    def test_quick_start_empty_profile(self):
        ctx = _ctx(doc_count=0)
        resp = compose_usage_help_response("help", ctx, "test_key")
        assert "Upload" in resp or "upload" in resp
        assert "document" in resp.lower()

    def test_quick_start_with_docs(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_usage_help_response("help", ctx, "test_key")
        assert "5 document(s)" in resp or "document" in resp.lower()

    def test_upload_help_response(self):
        ctx = _ctx(doc_count=0)
        resp = compose_usage_help_response("how do I upload a document?", ctx)
        assert "Upload" in resp or "upload" in resp
        assert "PDF" in resp

    def test_file_types_response(self):
        ctx = _ctx(doc_count=0)
        resp = compose_usage_help_response("what file types are supported?", ctx)
        assert "PDF" in resp
        assert "DOCX" in resp or "docx" in resp

    def test_screening_help_response(self):
        ctx = _ctx(doc_count=5)
        resp = compose_usage_help_response("screening help", ctx)
        assert "PII" in resp
        assert "Screening" in resp or "screening" in resp

    def test_content_gen_help_response(self):
        ctx = _ctx(doc_count=5)
        resp = compose_usage_help_response("how can I generate content?", ctx)
        assert "Content Generation" in resp or "generation" in resp.lower()

    def test_advanced_features_response(self):
        ctx = _ctx(doc_count=5)
        resp = compose_usage_help_response("advanced features", ctx)
        assert "Fine-Tuning" in resp or "fine-tuning" in resp.lower()

    def test_capability_overview_response(self):
        ctx = _ctx(doc_count=3, domains=["resume"])
        resp = compose_usage_help_response("what else can you do?", ctx)
        low = resp.lower()
        assert "docwain" in low
        assert "what docwain can do" in low or "capabilities" in low
        assert "try these queries" in low

    def test_task_help_compare(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_usage_help_response("how do I compare documents?", ctx)
        assert "Compare" in resp or "compare" in resp

    def test_domain_examples_resume(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_usage_help_response("resume examples", ctx)
        assert len(resp) > 50

    def test_query_examples_response(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_usage_help_response("show me examples", ctx)
        assert len(resp) > 50

    def test_no_placeholder_leaks(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_usage_help_response("help", ctx, "test_key")
        assert "{" not in resp
        assert "}" not in resp


# ---------------------------------------------------------------------------
# E. TestIntegrationWithConversationalNLP
# ---------------------------------------------------------------------------

class TestIntegrationWithConversationalNLP:
    """Integration tests with the conversational NLP classifier."""

    def test_how_to_compare_classified_as_usage_help(self):
        result = classify_conversational_intent("how do I compare candidates?")
        assert result is not None
        intent, conf = result
        assert intent == USAGE_HELP
        assert conf >= 0.40  # NLU-based scores range 0.4-0.7

    def test_how_to_rank_classified_as_usage_help(self):
        result = classify_conversational_intent("how can I rank resumes?")
        assert result is not None
        intent, conf = result
        assert intent == USAGE_HELP

    def test_help_classified_as_usage_help(self):
        result = classify_conversational_intent("help")
        assert result is not None
        intent, _conf = result
        assert intent == USAGE_HELP

    def test_what_else_can_you_do_classified_as_capability(self):
        result = classify_conversational_intent("what else can you do?")
        assert result is not None
        intent, _conf = result
        assert intent == CAPABILITY

    def test_compose_response_delegates_to_usage_help(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_response(USAGE_HELP, ctx, user_key="test", user_text="resume examples")
        # Should come from usage_help module — contains domain-specific content.
        assert len(resp) > 50

    def test_full_pipeline_generates_response(self):
        catalog = {
            "documents": [
                {"source_name": "resume1.pdf"},
                {"source_name": "resume2.pdf"},
            ],
            "dominant_domains": {"resume": 2},
        }
        result = generate_conversational_response(
            "how do I compare candidates?",
            catalog=catalog,
            collection_point_count=20,
        )
        assert result is not None
        assert result.intent == USAGE_HELP
        assert len(result.text) > 20

    def test_compose_response_fallback_without_user_text(self):
        """compose_response without user_text falls back to fragment pools."""
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_response(USAGE_HELP, ctx, user_key="test", user_text="")
        # Should still produce a valid response from fragment pools.
        assert len(resp) > 10


# ---------------------------------------------------------------------------
# F. TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_text_returns_quick_start(self):
        result = classify_help_sub_intent("")
        assert result.name == QUICK_START

    def test_unknown_domain_in_hint(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        sub = HelpSubIntent(name=DOMAIN_EXAMPLES, domain_hint="unknown_domain")
        examples = select_examples(ctx, sub, seed=42, max_examples=5)
        # Should fall back to general examples.
        assert len(examples) > 0
        assert all(e.domain == "general" for e in examples)

    def test_none_domains_in_context(self):
        ctx = ConversationalContext(
            document_count=5,
            document_names=[],
            dominant_domains=[],
            profile_is_empty=False,
        )
        resp = compose_usage_help_response("help", ctx, "test")
        assert len(resp) > 10

    def test_content_generation_help_response_valid(self):
        ctx = _ctx(doc_count=3, domains=["invoice"])
        resp = compose_usage_help_response("content generation help", ctx)
        assert "Content Generation" in resp

    def test_screening_pii_detection(self):
        result = classify_help_sub_intent("how to check for PII")
        assert result.name == SCREENING_HELP

    def test_task_help_with_examples(self):
        ctx = _ctx(doc_count=5, domains=["resume"])
        resp = compose_usage_help_response("how to extract information?", ctx)
        assert "Extract" in resp or "extract" in resp
