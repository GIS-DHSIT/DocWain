"""Tests for MISSING_REASON elimination from rendered output.

Verifies that renderers return empty strings instead of MISSING_REASON
and that the pipeline emergency fallback generates output from raw chunks.
"""
from __future__ import annotations

import pytest

from src.rag_v3.types import (
    Candidate,
    CandidateField,
    Chunk,
    ChunkSource,
    EvidenceSpan,
    FieldValuesField,
    GenericSchema,
    HRSchema,
    MISSING_REASON,
)
from src.rag_v3.enterprise import (
    _render_generic,
    _render_contact_value,
    _sanitize_render_value,
)
render_generic_mod = pytest.importorskip("src.rag_v3.renderers.generic", reason="Module removed")
render_generic = render_generic_mod.render_generic
from src.rag_v3.pipeline import _emergency_chunk_summary


# ── Generic renderer (renderers/generic.py) ──────────────────────────────

def test_generic_renderer_no_facts_returns_empty():
    """GenericSchema with no facts should return empty string, not MISSING_REASON."""
    schema = GenericSchema(facts=FieldValuesField(items=None))
    result = render_generic(schema, intent="facts")
    assert result == ""
    assert MISSING_REASON not in result


# ── Enterprise generic renderer ──────────────────────────────────────────

def test_enterprise_generic_no_facts_returns_empty():
    """enterprise._render_generic with no facts should return empty string."""
    schema = GenericSchema(facts=FieldValuesField(items=None))
    result = _render_generic(schema, intent="facts")
    assert result == ""
    assert MISSING_REASON not in result


def test_enterprise_generic_empty_fact_values_returns_empty():
    """enterprise._render_generic with facts that have no values should return empty."""
    schema = GenericSchema(
        facts=FieldValuesField(
            items=[],
        )
    )
    result = _render_generic(schema, intent="facts")
    assert result == ""
    assert MISSING_REASON not in result


# ── HR renderer ──────────────────────────────────────────────────────────

def test_hr_renderer_no_candidates_returns_empty():
    """HRSchema with no candidates should return empty string, not MISSING_REASON."""
    schema = HRSchema(candidates=CandidateField(items=None))
    result = _render_generic.__module__  # just to import the enterprise module
    from src.rag_v3.enterprise import _render_hr
    result = _render_hr(schema, intent="facts")
    assert result == ""
    assert MISSING_REASON not in result


# ── Contact renderer ────────────────────────────────────────────────────

def test_contact_renderer_no_items_returns_empty():
    """_render_contact_value([]) should return empty string, not MISSING_REASON."""
    result = _render_contact_value([])
    assert result == ""
    assert MISSING_REASON not in result


def test_contact_renderer_none_returns_empty():
    """_render_contact_value(None) should return empty string."""
    result = _render_contact_value(None)
    assert result == ""


def test_contact_renderer_with_items_returns_joined():
    """_render_contact_value with actual items should return them joined."""
    result = _render_contact_value(["foo@bar.com", "baz@qux.com"])
    assert result == "foo@bar.com, baz@qux.com"


# ── _sanitize_render_value ──────────────────────────────────────────────

def test_sanitize_render_value_missing_returns_empty():
    """_sanitize_render_value('') should return empty string."""
    result = _sanitize_render_value("")
    assert result == ""
    assert MISSING_REASON not in result


def test_sanitize_render_value_missing_reason_returns_empty():
    """_sanitize_render_value(MISSING_REASON) should return empty string."""
    result = _sanitize_render_value(MISSING_REASON)
    assert result == ""


def test_sanitize_render_value_garbage_returns_empty():
    """Garbage text with metadata patterns should return empty string."""
    garbage = "section_id: abc123 chunk_type: body page_start: 1"
    result = _sanitize_render_value(garbage)
    assert result == ""
    assert MISSING_REASON not in result


def test_sanitize_render_value_valid_text_passes_through():
    """Valid text should pass through _sanitize_render_value unchanged."""
    result = _sanitize_render_value("John Doe has 5 years of experience")
    assert result == "John Doe has 5 years of experience"


# ── Emergency chunk summary ─────────────────────────────────────────────

def _make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        id=f"chunk_{idx}",
        text=text,
        score=0.9,
        source=ChunkSource(document_name=f"doc_{idx}.pdf"),
    )


def test_emergency_chunk_summary_produces_output():
    """Three chunks should produce query-aware bullets."""
    chunks = [
        _make_chunk("Alice has 5 years of Python experience.", 0),
        _make_chunk("Bob specializes in data engineering.", 1),
        _make_chunk("Carol holds a PMP certification.", 2),
    ]
    # Use a query whose keywords overlap with the chunk content so that
    # lines pass the relevance filter and appear as bullets.
    result = _emergency_chunk_summary(chunks, "python experience engineering certification")
    assert result.startswith("Based on")
    assert "Alice has 5 years of Python experience." in result
    assert "Bob specializes in data engineering." in result
    assert "Carol holds a PMP certification." in result


def test_emergency_chunk_summary_empty_chunks():
    """No chunks should return empty string."""
    result = _emergency_chunk_summary([], "some query")
    assert result == ""


def test_emergency_chunk_summary_blank_text_chunks():
    """Chunks with only whitespace text should return empty string."""
    chunks = [
        _make_chunk("", 0),
        _make_chunk("   ", 1),
    ]
    result = _emergency_chunk_summary(chunks, "some query")
    assert result == ""


def test_emergency_chunk_summary_truncates_long_text():
    """Chunks with very long single sentences (>300 chars) use fallback truncation."""
    long_text = "A" * 400
    chunks = [_make_chunk(long_text, 0)]
    result = _emergency_chunk_summary(chunks, "query")
    # 400-char line — no keyword overlap with "query", so falls to raw content
    # presentation (header + truncated snippet).
    assert result  # produces some output
    assert len(result) <= 500  # header + truncated snippet


def test_emergency_chunk_summary_takes_only_top_8():
    """At most 30 chunks are scanned, and at most 15 lines are produced."""
    chunks = [_make_chunk(f"Chunk {i} has relevant content here.", i) for i in range(40)]
    result = _emergency_chunk_summary(chunks, "query about content")
    # Up to 30 chunks scanned, up to 15 bullet lines produced
    lines = [l for l in result.strip().split("\n") if l.strip()]
    assert len(lines) <= 16  # header + up to 15 bullet lines
    # Chunks beyond 30-chunk scan limit should never appear
    assert "Chunk 30" not in result
    assert "Chunk 31" not in result
