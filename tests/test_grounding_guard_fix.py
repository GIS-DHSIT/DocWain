"""Tests for grounding guard label replacement fix and evidence fallback improvements."""

from __future__ import annotations

import pytest

from src.orchestrator.grounding_guard import apply_grounding_guard, _sanitize_labelled_entities


# ---------------------------------------------------------------------------
# Grounding guard: labelled entity sanitisation
# ---------------------------------------------------------------------------


def test_verified_label_preserved():
    """Label with value found in evidence should be kept unchanged."""
    # The regex captures everything after 'Brand:' up to the first period,
    # so the captured value is 'Acme Corp' and evidence must contain it.
    answer = "The Brand: Acme Corp. They are well known."
    evidence = "Acme Corp manufactures high quality widgets."
    result = _sanitize_labelled_entities(answer, evidence)
    assert "Brand: Acme Corp" in result


def test_unverified_label_removed():
    """Label with value NOT in evidence should be removed, not replaced
    with 'not explicitly mentioned'."""
    answer = "The Brand: FakeCorp produces these items."
    evidence = "The document discusses production processes."
    result = _sanitize_labelled_entities(answer, evidence)
    assert "not explicitly mentioned" not in result
    assert "FakeCorp" not in result


# ---------------------------------------------------------------------------
# Evidence fallback: _format_evidence_fallback
# ---------------------------------------------------------------------------

from src.api.dw_newron import _format_evidence_fallback


def test_evidence_fallback_missing_field_omitted():
    """When a requested field has no matching value in ledger, it should
    be omitted entirely — no 'Not available' message."""
    ledger = [{"doc_name": "doc1.pdf", "snippet": "Name: Alice"}]
    query = "extract name and salary from the document"
    result = _format_evidence_fallback(query, ledger)
    assert "Not available" not in result
    assert "not available" not in result.lower() or "no matching" in result.lower()
    # 'name' should still appear (it has a value)
    assert "name" in result.lower()


def test_evidence_fallback_present_field_included():
    """When a requested field has a matching value, it should appear."""
    ledger = [{"doc_name": "doc1.pdf", "snippet": "Name: Alice"}]
    query = "extract name from the document"
    result = _format_evidence_fallback(query, ledger)
    assert "Alice" in result


def test_evidence_fallback_no_ledger():
    """When the ledger is empty, the message should say 'No matching
    information was found in the documents.' — not 'Not available'."""
    result = _format_evidence_fallback("extract name", [])
    assert "No matching information was found in the documents." in result
    assert "Not available in retrieved context" not in result
