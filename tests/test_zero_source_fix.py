"""Tests for the 0-source pipeline bug fix and domain detection expansion.

Phase 1 of the multi-agent plan:
- 1A: Entity-hint soft fallback in profile scan path
- 1B: Domain detection for summary/reasoning queries
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Domain detection tests (extract.py)
# ---------------------------------------------------------------------------

from src.rag_v3.extract import _query_domain_override


class TestDomainDetectionExpansion:
    """Test _QUERY_HR_WEAK expansion and person-name pattern matching."""

    # Existing strong signals still work
    def test_resume_strong(self):
        assert _query_domain_override("show me the resume") == "hr"

    def test_candidate_strong(self):
        assert _query_domain_override("list all candidates") == "hr"

    # Expanded weak signals
    def test_profile_weak(self):
        assert _query_domain_override("what is their profile") == "hr"

    def test_qualified_weak(self):
        assert _query_domain_override("is the person qualified") == "hr"

    def test_suitable_weak(self):
        assert _query_domain_override("are they suitable for the role") == "hr"

    def test_background_weak(self):
        assert _query_domain_override("check their background") == "hr"

    def test_career_weak(self):
        assert _query_domain_override("describe their career history") == "hr"

    def test_designation_weak(self):
        assert _query_domain_override("what is the designation") == "hr"

    def test_work_history_weak(self):
        assert _query_domain_override("work history details") == "hr"

    # Person-name possessive patterns
    def test_possessive_profile(self):
        assert _query_domain_override("Gaurav's profile") == "hr"

    def test_possessive_summary(self):
        assert _query_domain_override("Dhayal's summary") == "hr"

    def test_possessive_background(self):
        assert _query_domain_override("Dev's background") == "hr"

    def test_possessive_career(self):
        assert _query_domain_override("Gokul's career") == "hr"

    def test_possessive_strengths(self):
        assert _query_domain_override("Bharath's strengths") == "hr"

    # Reasoning about a person
    def test_is_qualified(self):
        assert _query_domain_override("Is Gokul qualified for a senior developer role?") == "hr"

    def test_is_suitable(self):
        assert _query_domain_override("Is Dev suitable for the position?") == "hr"

    def test_can_fit(self):
        assert _query_domain_override("Can Bharath fit the lead role?") == "hr"

    def test_would_eligible(self):
        assert _query_domain_override("Would Dhayal be eligible for this position?") == "hr"

    # Summarize/describe a person
    def test_summarize_person(self):
        assert _query_domain_override("Summarize Gaurav") == "hr"

    def test_describe_person(self):
        assert _query_domain_override("Describe Gokul's experience") == "hr"

    def test_tell_me_about(self):
        assert _query_domain_override("Tell me about Dev") == "hr"

    # Non-HR queries should still return None or correct domain
    def test_generic_query_returns_none(self):
        assert _query_domain_override("what is the weather today") is None

    def test_invoice_query_returns_invoice(self):
        assert _query_domain_override("show me the invoice totals") == "invoice"

    def test_legal_query_returns_legal(self):
        assert _query_domain_override("review the contract terms") == "legal"

    def test_empty_query_returns_none(self):
        assert _query_domain_override("") is None

    def test_none_query_returns_none(self):
        assert _query_domain_override(None) is None


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
