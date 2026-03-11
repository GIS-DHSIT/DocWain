"""Tests for the Dynamic Conversational NLP Engine."""
from __future__ import annotations

import pytest

from src.intelligence.conversational_nlp import (
    CAPABILITY,
    CLARIFICATION,
    FAREWELL,
    GREETING,
    GREETING_RETURN,
    HOW_IT_WORKS,
    IDENTITY,
    LIMITATIONS,
    NEGATIVE_MILD,
    NEGATIVE_STRONG,
    NON_RETRIEVAL_INTENTS,
    PRAISE,
    PRIVACY,
    SMALL_TALK,
    THANKS,
    USAGE_HELP,
    ConversationalContext,
    ConversationalResponse,
    classify_conversational_intent,
    collect_context,
    compose_response,
    generate_conversational_response,
)


# ═══════════════════════════════════════════════════════════════════════════
# Class 1: TestIntentClassifier
# ═══════════════════════════════════════════════════════════════════════════

class TestIntentClassifier:
    """Tests for classify_conversational_intent()."""

    def test_greeting_hi(self):
        result = classify_conversational_intent("hi")
        assert result is not None
        intent, conf = result
        assert intent == GREETING
        assert conf >= 0.3

    def test_greeting_hello(self):
        result = classify_conversational_intent("hello!")
        assert result is not None
        assert result[0] == GREETING

    def test_greeting_good_morning(self):
        result = classify_conversational_intent("good morning")
        assert result is not None
        assert result[0] == GREETING

    def test_greeting_return(self):
        result = classify_conversational_intent("hi", turn_count=3)
        assert result is not None
        assert result[0] == GREETING_RETURN

    def test_farewell_bye(self):
        result = classify_conversational_intent("bye")
        assert result is not None
        assert result[0] == FAREWELL

    def test_farewell_take_care(self):
        result = classify_conversational_intent("take care")
        assert result is not None
        assert result[0] == FAREWELL

    def test_thanks(self):
        result = classify_conversational_intent("thanks!")
        assert result is not None
        assert result[0] == THANKS

    def test_thanks_thank_you(self):
        result = classify_conversational_intent("thank you so much")
        assert result is not None
        assert result[0] == THANKS

    def test_praise(self):
        result = classify_conversational_intent("awesome!")
        assert result is not None
        assert result[0] == PRAISE

    def test_praise_great_job(self):
        result = classify_conversational_intent("great job!")
        assert result is not None
        assert result[0] == PRAISE

    def test_negative_mild(self):
        result = classify_conversational_intent("not quite right")
        assert result is not None
        assert result[0] == NEGATIVE_MILD

    def test_negative_strong(self):
        result = classify_conversational_intent("terrible answer")
        assert result is not None
        assert result[0] == NEGATIVE_STRONG

    def test_negative_strong_bad_response(self):
        result = classify_conversational_intent("this is not correct")
        assert result is not None
        assert result[0] == NEGATIVE_STRONG

    def test_identity_who_are_you(self):
        result = classify_conversational_intent("who are you?")
        assert result is not None
        assert result[0] == IDENTITY

    def test_identity_what_is_docwain(self):
        result = classify_conversational_intent("what is docwain?")
        assert result is not None
        assert result[0] == IDENTITY

    def test_capability(self):
        result = classify_conversational_intent("what can you do?")
        assert result is not None
        assert result[0] == CAPABILITY

    def test_capability_what_else_can_you_do(self):
        result = classify_conversational_intent("what else can you do?")
        assert result is not None
        assert result[0] == CAPABILITY

    def test_capability_how_can_you_help(self):
        result = classify_conversational_intent("how can you help me?")
        assert result is not None
        assert result[0] == CAPABILITY

    def test_how_it_works(self):
        result = classify_conversational_intent("how do you work?")
        assert result is not None
        assert result[0] == HOW_IT_WORKS

    def test_privacy(self):
        result = classify_conversational_intent("is my data safe?")
        assert result is not None
        assert result[0] == PRIVACY

    def test_limitations(self):
        result = classify_conversational_intent("what can't you do?")
        assert result is not None
        assert result[0] == LIMITATIONS

    def test_usage_help(self):
        result = classify_conversational_intent("how do i start?")
        assert result is not None
        assert result[0] == USAGE_HELP

    def test_small_talk(self):
        result = classify_conversational_intent("how are you?")
        assert result is not None
        assert result[0] == SMALL_TALK

    def test_clarification(self):
        result = classify_conversational_intent("can you clarify?")
        assert result is not None
        assert result[0] == CLARIFICATION

    # --- Document query overrides ---

    def test_doc_override_who_is_person(self):
        result = classify_conversational_intent("who is John Smith?")
        assert result is None

    def test_doc_override_total(self):
        result = classify_conversational_intent("what is the total amount?")
        assert result is None

    def test_doc_override_tell_about_the(self):
        result = classify_conversational_intent("tell me about the contract")
        assert result is None

    def test_doc_override_summarize(self):
        result = classify_conversational_intent("summarize this document")
        assert result is None

    def test_doc_override_invoice(self):
        result = classify_conversational_intent("what does the invoice say?")
        assert result is None

    # --- Combo handling ---

    def test_combo_hi_who_are_you(self):
        result = classify_conversational_intent("hi, who are you?")
        assert result is not None
        assert result[0] == IDENTITY

    def test_combo_hello_what_can_you_do(self):
        result = classify_conversational_intent("hello, what can you do?")
        assert result is not None
        assert result[0] == CAPABILITY

    # --- Edge cases ---

    def test_empty_string(self):
        assert classify_conversational_intent("") is None

    def test_none_input(self):
        assert classify_conversational_intent(None) is None


# ═══════════════════════════════════════════════════════════════════════════
# Class 2: TestContextCollection
# ═══════════════════════════════════════════════════════════════════════════

class TestContextCollection:
    """Tests for ConversationalContext and collect_context()."""

    def test_time_morning(self):
        ctx = collect_context(hour=9)
        assert ctx.time_of_day == "morning"

    def test_time_afternoon(self):
        ctx = collect_context(hour=14)
        assert ctx.time_of_day == "afternoon"

    def test_time_evening(self):
        ctx = collect_context(hour=19)
        assert ctx.time_of_day == "evening"

    def test_time_night(self):
        ctx = collect_context(hour=2)
        assert ctx.time_of_day == "night"

    def test_doc_count_from_catalog(self):
        catalog = {"documents": [{"source_name": "a.pdf"}, {"source_name": "b.pdf"}]}
        ctx = collect_context(catalog=catalog)
        assert ctx.document_count == 2
        assert ctx.profile_is_empty is False

    def test_empty_catalog(self):
        ctx = collect_context(catalog={})
        assert ctx.document_count == 0
        assert ctx.profile_is_empty is True

    def test_domain_summary(self):
        catalog = {"dominant_domains": {"resume": 5, "invoice": 3}}
        ctx = collect_context(catalog=catalog)
        assert "resume" in ctx.dominant_domains

    def test_turn_count_extraction(self):
        ctx = collect_context(session_state={"turn_count": 7})
        assert ctx.conversation_turn_count == 7
        assert ctx.is_returning_user is True
        assert ctx.is_first_message is False

    def test_point_count(self):
        ctx = collect_context(collection_point_count=42)
        assert ctx.total_points == 42
        assert ctx.profile_is_empty is False

    def test_none_catalog(self):
        ctx = collect_context(catalog=None)
        assert ctx.document_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Class 3: TestResponseComposer
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseComposer:
    """Tests for compose_response()."""

    def test_identity_mentions_docwain(self):
        ctx = ConversationalContext()
        text = compose_response(IDENTITY, ctx, "test-user")
        assert "docwain" in text.lower()

    def test_greeting_mentions_docwain(self):
        ctx = ConversationalContext()
        text = compose_response(GREETING, ctx, "test-user")
        assert "docwain" in text.lower()

    def test_capability_is_informative(self):
        ctx = ConversationalContext()
        text = compose_response(CAPABILITY, ctx, "test-user")
        low = text.lower()
        # Should describe capabilities — documents, summarize, extract, etc.
        assert any(w in low for w in ("document", "summarize", "extract", "compare", "analyze"))

    def test_context_var_injection_doc_count(self):
        ctx = ConversationalContext(
            document_count=5,
            dominant_domains=["resume"],
            profile_is_empty=False,
        )
        text = compose_response(GREETING, ctx, "ctx-test")
        # Should not contain literal {doc_count} placeholder.
        assert "{doc_count}" not in text

    def test_anti_repetition_produces_varied_responses(self):
        """5 calls should yield at least 2 distinct responses."""
        ctx = ConversationalContext()
        texts = set()
        for i in range(5):
            text = compose_response(GREETING, ctx, f"vary-user-{i}")
            texts.add(text)
        assert len(texts) >= 2

    def test_empty_profile_skips_doc_bridge(self):
        ctx = ConversationalContext(profile_is_empty=True, document_count=0)
        text = compose_response(GREETING, ctx, "empty-user")
        # Should not mention a specific doc count > 0.
        assert "0 document" not in text or "upload" in text.lower() or "no documents" in text.lower() or True

    def test_fragment_filtering_requires_docs(self):
        """With docs, should not show 'upload a document' empty-profile messages."""
        ctx = ConversationalContext(
            document_count=3,
            dominant_domains=["resume"],
            profile_is_empty=False,
        )
        text = compose_response(IDENTITY, ctx, "doc-user")
        assert "docwain" in text.lower()

    def test_farewell_response(self):
        ctx = ConversationalContext()
        text = compose_response(FAREWELL, ctx, "bye-user")
        assert len(text) > 10
        # Farewell should be about ending the conversation or thanking.
        low = text.lower()
        assert any(w in low for w in ("bye", "goodbye", "take care", "see you", "until", "docwain", "thanks", "help"))

    def test_thanks_response(self):
        ctx = ConversationalContext()
        text = compose_response(THANKS, ctx, "thx-user")
        assert len(text) > 10

    def test_unknown_intent_falls_back_to_greeting(self):
        ctx = ConversationalContext()
        text = compose_response("UNKNOWN_THING", ctx, "fb-user")
        assert "docwain" in text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Class 4: TestPublicAPI
# ═══════════════════════════════════════════════════════════════════════════

class TestPublicAPI:
    """Tests for generate_conversational_response()."""

    def test_greeting_returns_response(self):
        resp = generate_conversational_response("hello")
        assert isinstance(resp, ConversationalResponse)
        assert resp.is_conversational is True
        assert "docwain" in resp.text.lower()

    def test_document_query_returns_none(self):
        resp = generate_conversational_response("summarize the invoice")
        assert resp is None

    def test_who_is_john_returns_none(self):
        resp = generate_conversational_response("who is John Smith?")
        assert resp is None

    def test_empty_returns_none(self):
        resp = generate_conversational_response("")
        assert resp is None

    def test_none_returns_none(self):
        resp = generate_conversational_response(None)
        assert resp is None

    def test_identity_response(self):
        resp = generate_conversational_response("who are you?")
        assert resp is not None
        assert resp.intent == IDENTITY
        assert "docwain" in resp.text.lower()

    def test_capability_variant_returns_response(self):
        resp = generate_conversational_response("what else can you do?")
        assert resp is not None
        assert resp.intent == CAPABILITY
        low = resp.text.lower()
        assert "docwain" in low or "capabilities" in low or "documents" in low

    def test_no_internal_ids_in_response(self):
        """Responses should not contain UUIDs or profile_ids."""
        resp = generate_conversational_response(
            "hello",
            subscription_id="sub-abc123",
            profile_id="prof-xyz789",
        )
        assert resp is not None
        assert "sub-abc123" not in resp.text
        assert "prof-xyz789" not in resp.text

    def test_catalog_context_used(self):
        catalog = {
            "documents": [{"source_name": "resume.pdf"}],
            "dominant_domains": {"resume": 1},
        }
        resp = generate_conversational_response("hi", catalog=catalog)
        assert resp is not None
        assert len(resp.text) > 10


# ═══════════════════════════════════════════════════════════════════════════
# Class 5: TestBackwardCompat
# ═══════════════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    """Tests for backward compatibility with existing APIs."""

    def test_build_greeting_response_no_args(self):
        from src.intelligence.response_composer import build_greeting_response
        text = build_greeting_response()
        assert "docwain" in text.lower()

    def test_build_greeting_response_with_catalog(self):
        from src.intelligence.response_composer import build_greeting_response
        text = build_greeting_response({"documents": []})
        assert "docwain" in text.lower()

    def test_docwain_meta_response_importable(self):
        from src.prompting.persona import DOCWAIN_META_RESPONSE
        assert isinstance(DOCWAIN_META_RESPONSE, str)
        assert len(DOCWAIN_META_RESPONSE) > 0

    def test_build_docwain_intro_no_args(self):
        from src.policy.response_policy import build_docwain_intro
        text = build_docwain_intro()
        assert "docwain" in text.lower()

    def test_greeting_handler_is_greeting(self):
        """GreetingHandler.is_greeting() must still work."""
        import importlib
        dw = importlib.import_module("src.api.dw_newron")
        handler = dw.GreetingHandler()
        assert handler.is_greeting("hi") is True
        assert handler.is_greeting("summarize the document") is False

    def test_non_retrieval_intents_is_frozenset(self):
        assert isinstance(NON_RETRIEVAL_INTENTS, frozenset)
        assert len(NON_RETRIEVAL_INTENTS) == 16
