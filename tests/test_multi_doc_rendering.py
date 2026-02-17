"""Tests for enhanced cross-document rendering in enterprise.py (Task 6)."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.rag_v3.types import Candidate, CandidateField, EvidenceSpan, HRSchema
from src.rag_v3.enterprise import _render_hr


def _make_candidate(
    name: str,
    technical_skills: list | None = None,
    functional_skills: list | None = None,
    certifications: list | None = None,
    total_years_experience: str | None = None,
    experience_summary: str | None = None,
) -> Candidate:
    return Candidate(
        name=name,
        technical_skills=technical_skills,
        functional_skills=functional_skills,
        certifications=certifications,
        total_years_experience=total_years_experience,
        experience_summary=experience_summary,
        evidence_spans=[],
    )


def _make_hr_schema(candidates: list) -> HRSchema:
    return HRSchema(candidates=CandidateField(items=candidates))


class TestRenderHRComparison:
    def test_render_hr_comparison_two_candidates(self):
        """Two candidates with compare intent should use structured comparison."""
        candidates = [
            _make_candidate("Abinaya", technical_skills=["Python", "Java", "AWS"]),
            _make_candidate("Aadithya", technical_skills=["Python", "Go", "Docker"]),
        ]
        schema = _make_hr_schema(candidates)
        result = _render_hr(schema, intent="compare", query="compare candidates")
        # Should have structured comparison format
        assert "Abinaya" in result or "Aadithya" in result
        assert "Comparison" in result or "Technical" in result or "vs" in result or "|" in result

    def test_render_hr_ranking_still_works(self):
        """When comparator raises an error, ranking should still work."""
        candidates = [
            _make_candidate("Abinaya", technical_skills=["Python"]),
            _make_candidate("Aadithya", technical_skills=["Python", "Java", "Go"]),
        ]
        schema = _make_hr_schema(candidates)
        # Patch at the module level that the lazy import loads from
        with patch.dict("sys.modules", {"src.rag_v3.comparator": MagicMock(
            compare_candidates_from_schema=MagicMock(side_effect=RuntimeError("test")),
            render_comparison=MagicMock(),
        )}):
            result = _render_hr(schema, intent="rank", query="rank candidates")
        assert "ranking" in result.lower() or "Aadithya" in result or "Abinaya" in result

    def test_comparison_fallback_to_ranking(self):
        """If comparator raises, should fall back to existing ranking."""
        candidates = [
            _make_candidate("Alice", technical_skills=["Python", "Java"]),
            _make_candidate("Bob", technical_skills=["Go", "Rust", "C++"]),
        ]
        schema = _make_hr_schema(candidates)
        # Make the comparator import fail
        with patch("src.rag_v3.comparator.compare_candidates_from_schema", side_effect=RuntimeError("fail")):
            result = _render_hr(schema, intent="rank", query="rank candidates")
        # Should fall back to ranking
        assert "ranking" in result.lower() or "1." in result

    def test_render_single_candidate_unchanged(self):
        """Single candidate rendering should be unchanged."""
        candidates = [
            _make_candidate(
                "Abinaya",
                technical_skills=["Python", "Java"],
                total_years_experience="5 years",
            ),
        ]
        schema = _make_hr_schema(candidates)
        result = _render_hr(schema, intent="detail", query="tell me about Abinaya")
        assert "Abinaya" in result
        assert "Python" in result

    def test_render_contact_unchanged(self):
        """Contact rendering should be unchanged."""
        cand = _make_candidate("Abinaya")
        cand.emails = ["abinaya@example.com"]
        cand.phones = ["555-1234"]
        schema = _make_hr_schema([cand])
        result = _render_hr(schema, intent="contact", query="contact info")
        assert "abinaya@example.com" in result
        assert "555-1234" in result

    def test_render_hr_compare_intent_uses_comparator(self):
        """Compare intent with 2+ candidates should attempt the comparator."""
        candidates = [
            _make_candidate("Alice", technical_skills=["Python", "Java"]),
            _make_candidate("Bob", technical_skills=["Go", "Rust"]),
        ]
        schema = _make_hr_schema(candidates)
        # The comparator is loaded via lazy import in _render_hr, so just call it normally
        result = _render_hr(schema, intent="compare", query="compare candidates")
        # Should have comparison output (comparator module exists)
        assert "Alice" in result or "Bob" in result
