"""Tests for the 0-source pipeline bug fix and domain detection expansion.

Phase 1 of the multi-agent plan:
- 1A: Entity-hint soft fallback in profile scan path
- 1B: Domain detection for summary/reasoning queries
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Domain detection tests (extract.py)
# ---------------------------------------------------------------------------

from src.rag_v3.extract import _ml_query_domain


@dataclass(frozen=True)
class _FakeIntentParse:
    """Lightweight mock for IntentParse — simulates ML classifier output."""
    intent: str = "qa"
    output_format: str = "text"
    requested_fields: list = dc_field(default_factory=list)
    domain: str = "generic"
    constraints: dict = dc_field(default_factory=dict)
    entity_hints: list = dc_field(default_factory=list)
    source: str = "test"


def _hr_intent(**kwargs):
    """Create an intent_parse with domain='resume' (maps to 'hr')."""
    return _FakeIntentParse(domain="resume", **kwargs)


def _invoice_intent(**kwargs):
    return _FakeIntentParse(domain="invoice", **kwargs)


def _legal_intent(**kwargs):
    return _FakeIntentParse(domain="legal", **kwargs)


class TestDomainDetectionExpansion:
    """Test _ml_query_domain with intent_parse providing domain signal."""

    # HR domain via intent_parse
    def test_resume_strong(self):
        assert _ml_query_domain("show me the resume", _hr_intent()) == "hr"

    def test_candidate_strong(self):
        assert _ml_query_domain("list all candidates", _hr_intent()) == "hr"

    # Expanded weak signals — ML classifier provides domain
    def test_profile_weak(self):
        assert _ml_query_domain("what is their profile", _hr_intent()) == "hr"

    def test_qualified_weak(self):
        assert _ml_query_domain("is the person qualified", _hr_intent()) == "hr"

    def test_suitable_weak(self):
        assert _ml_query_domain("are they suitable for the role", _hr_intent()) == "hr"

    def test_background_weak(self):
        assert _ml_query_domain("check their background", _hr_intent()) == "hr"

    def test_career_weak(self):
        assert _ml_query_domain("describe their career history", _hr_intent()) == "hr"

    def test_designation_weak(self):
        assert _ml_query_domain("what is the designation", _hr_intent()) == "hr"

    def test_work_history_weak(self):
        assert _ml_query_domain("work history details", _hr_intent()) == "hr"

    # Person-name possessive patterns — ML classifier detects resume domain
    def test_possessive_profile(self):
        assert _ml_query_domain("Gaurav's profile", _hr_intent()) == "hr"

    def test_possessive_summary(self):
        assert _ml_query_domain("Dhayal's summary", _hr_intent()) == "hr"

    def test_possessive_background(self):
        assert _ml_query_domain("Dev's background", _hr_intent()) == "hr"

    def test_possessive_career(self):
        assert _ml_query_domain("Gokul's career", _hr_intent()) == "hr"

    def test_possessive_strengths(self):
        assert _ml_query_domain("Bharath's strengths", _hr_intent()) == "hr"

    # Reasoning about a person
    def test_is_qualified(self):
        assert _ml_query_domain("Is Gokul qualified for a senior developer role?", _hr_intent()) == "hr"

    def test_is_suitable(self):
        assert _ml_query_domain("Is Dev suitable for the position?", _hr_intent()) == "hr"

    def test_can_fit(self):
        assert _ml_query_domain("Can Bharath fit the lead role?", _hr_intent()) == "hr"

    def test_would_eligible(self):
        assert _ml_query_domain("Would Dhayal be eligible for this position?", _hr_intent()) == "hr"

    # Summarize/describe a person
    def test_summarize_person(self):
        assert _ml_query_domain("Summarize Gaurav", _hr_intent()) == "hr"

    def test_describe_person(self):
        assert _ml_query_domain("Describe Gokul's experience", _hr_intent()) == "hr"

    def test_tell_me_about(self):
        assert _ml_query_domain("Tell me about Dev", _hr_intent()) == "hr"

    # Non-HR queries should still return None or correct domain
    def test_generic_query_returns_none(self):
        assert _ml_query_domain("what is the weather today") is None

    def test_invoice_query_returns_invoice(self):
        assert _ml_query_domain("show me the invoice totals", _invoice_intent()) == "invoice"

    def test_legal_query_returns_legal(self):
        assert _ml_query_domain("review the contract terms", _legal_intent()) == "legal"

    def test_empty_query_returns_none(self):
        assert _ml_query_domain("") is None

    def test_none_query_returns_none(self):
        assert _ml_query_domain(None) is None


# ---------------------------------------------------------------------------
# Entity filter soft fallback tests (pipeline.py)
# ---------------------------------------------------------------------------

class TestEntityFilterSoftFallback:
    """Test that entity filter uses soft fallback instead of hard early-return."""

    def _make_chunk(self, text="chunk text", source_name="doc.pdf", score=0.5):
        meta = {"source_name": source_name, "filename": source_name}
        return SimpleNamespace(text=text, score=score, meta=meta, id="c1")

    def test_filter_returns_matching_when_found(self):
        """When entity matches, filtered list is used."""
        from src.rag_v3.pipeline import _filter_chunks_by_entity_hint

        chunks = [
            self._make_chunk("Gaurav has 5 years experience", "Gaurav_Resume.pdf"),
            self._make_chunk("Invoice totals for Q3", "invoice.pdf"),
        ]
        result = _filter_chunks_by_entity_hint(chunks, "Gaurav", "test-cid")
        # Should return only Gaurav's chunk
        assert len(result) >= 1
        assert any("Gaurav" in (getattr(c, "text", "") or "") for c in result)

    def test_filter_returns_empty_when_no_match(self):
        """When entity doesn't match, returns empty list."""
        from src.rag_v3.pipeline import _filter_chunks_by_entity_hint

        chunks = [
            self._make_chunk("Invoice totals for Q3", "invoice.pdf"),
            self._make_chunk("General document content", "doc.pdf"),
        ]
        result = _filter_chunks_by_entity_hint(chunks, "Gaurav", "test-cid")
        assert result == []

    def test_profile_scan_path_soft_fallback(self):
        """Verify the profile scan path uses soft fallback (not hard early-return).

        The bug was: when entity filter returned empty in profile scan path,
        pipeline returned immediately with sources=[], domain='unknown'.
        After fix: pipeline keeps all chunks and continues extraction.
        """
        # This tests the code path at pipeline.py line ~738
        # We verify by checking the code pattern (integration test would need live data)
        import inspect
        from src.rag_v3 import pipeline as mod

        source = inspect.getsource(mod)

        # The old buggy pattern: hard early-return with empty sources
        old_pattern = (
            'if not reranked:\n'
            '                    return _build_answer(\n'
            '                        response_text=f"No documents found matching'
        )
        assert old_pattern not in source, (
            "Found old hard early-return pattern in profile scan path — fix not applied"
        )

        # The new soft fallback pattern should exist
        assert "Entity filter for" in source
        assert "returned empty in profile scan" in source
        assert "keeping all" in source


# ---------------------------------------------------------------------------
# Integration: metadata shape tests
# ---------------------------------------------------------------------------

class TestMetadataShape:
    """Ensure metadata always has correct types for scope and domain."""

    def test_scope_should_be_dict_in_profile_scan(self):
        """Profile scan path sets scope as dict, not string."""
        import inspect
        from src.rag_v3 import pipeline as mod

        source = inspect.getsource(mod)
        # In profile scan success path, scope should be dict
        assert '"scope": {"profile_id": profile_id}' in source

    def test_no_bare_targeted_string_scope(self):
        """The old bug set scope='targeted' (string). Verify it's gone."""
        import inspect
        from src.rag_v3 import pipeline as mod

        source = inspect.getsource(mod)
        # The old pattern was: "scope": "targeted"
        # We should not find this exact pattern in the profile scan early-return
        # (it was removed by the soft fallback fix)
        # Note: "targeted" may appear in other contexts (comments, logging) — that's fine
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if '"scope": "targeted"' in line and "_build_answer" in lines[max(0, i - 5):i + 1].__repr__():
                # Check it's not in a return _build_answer context (the old bug)
                context = "\n".join(lines[max(0, i - 5):i + 5])
                assert "entity_hint" not in context or "profile scan" not in context.lower(), (
                    f"Found bare 'targeted' scope near _build_answer at line {i}"
                )
