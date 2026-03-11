"""Tests for cross-document comparison engine (Task 2)."""
from __future__ import annotations

import pytest
from src.rag_v3.types import Candidate, EvidenceSpan
from src.rag_v3.comparator import (
    ComparisonResult,
    FieldComparison,
    compare_candidates_from_schema,
    compare_documents,
    render_comparison,
)
from src.rag_v3.document_context import DocumentContext
from src.rag_v3.types import Chunk, ChunkSource


def _make_context(
    doc_id: str,
    doc_name: str,
    fields: dict | None = None,
    doc_domain: str = "resume",
) -> DocumentContext:
    return DocumentContext(
        document_id=doc_id,
        document_name=doc_name,
        doc_domain=doc_domain,
        chunks=[],
        fields=fields or {},
        section_kinds=[],
    )


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


class TestCompareDocuments:
    def test_compare_two_resumes_skills(self):
        ctx_a = _make_context("d1", "Abinaya.pdf", fields={
            "Technical Skills": "Python, Java, AWS",
        })
        ctx_b = _make_context("d2", "Aadithya.pdf", fields={
            "Technical Skills": "Python, Docker, Kubernetes",
        })
        result = compare_documents([ctx_a, ctx_b], "compare skills")
        assert len(result.documents) == 2
        assert "Abinaya.pdf" in result.documents
        assert "Aadithya.pdf" in result.documents

    def test_compare_numeric_experience(self):
        ctx_a = _make_context("d1", "Abinaya.pdf", fields={"Total Years Experience": "5 years"})
        ctx_b = _make_context("d2", "Aadithya.pdf", fields={"Total Years Experience": "8 years"})
        result = compare_documents([ctx_a, ctx_b], "compare experience")
        # Should have a numeric comparison
        numeric = [c for c in result.field_comparisons if c.comparison_type == "numeric"]
        assert len(numeric) >= 1

    def test_compare_text_fields_side_by_side(self):
        ctx_a = _make_context("d1", "A.pdf", fields={"Summary": "Senior developer with cloud expertise"})
        ctx_b = _make_context("d2", "B.pdf", fields={"Summary": "Junior developer learning web technologies"})
        result = compare_documents([ctx_a, ctx_b], "compare summaries")
        text_comps = [c for c in result.field_comparisons if c.comparison_type == "text"]
        assert len(text_comps) >= 1
        assert "A.pdf" in text_comps[0].values
        assert "B.pdf" in text_comps[0].values

    def test_focus_fields_filters_comparison(self):
        ctx_a = _make_context("d1", "A.pdf", fields={
            "Technical Skills": "Python",
            "Summary": "Long summary",
        })
        ctx_b = _make_context("d2", "B.pdf", fields={
            "Technical Skills": "Java",
            "Summary": "Another summary",
        })
        result = compare_documents([ctx_a, ctx_b], "compare", focus_fields=["technical_skills"])
        field_names = [c.field_name for c in result.field_comparisons]
        assert "technical_skills" in field_names
        assert "summary" not in field_names

    def test_strengths_computed_correctly(self):
        ctx_a = _make_context("d1", "Abinaya.pdf", fields={
            "Total Years Experience": "8 years",
        })
        ctx_b = _make_context("d2", "Aadithya.pdf", fields={
            "Total Years Experience": "3 years",
        })
        result = compare_documents([ctx_a, ctx_b], "compare experience")
        # Abinaya should have the experience strength
        assert any("Higher" in s for s in result.strengths.get("Abinaya.pdf", []))

    def test_compare_single_document_graceful(self):
        ctx = _make_context("d1", "Solo.pdf")
        result = compare_documents([ctx], "compare")
        assert len(result.documents) == 1
        assert "Only one document" in result.summary

    def test_compare_empty_contexts(self):
        result = compare_documents([], "compare")
        assert result.documents == []
        assert result.field_comparisons == []


class TestRenderComparison:
    def test_render_comparison_two_docs(self):
        result = ComparisonResult(
            documents=["A.pdf", "B.pdf"],
            field_comparisons=[
                FieldComparison(
                    field_name="technical_skills",
                    values={"A.pdf": ["Python", "Java"], "B.pdf": ["Python", "Go"]},
                    comparison_type="overlap",
                    overlap=["Python"],
                    differences={"A.pdf": ["Java"], "B.pdf": ["Go"]},
                ),
            ],
            summary="Comparison of 2 docs",
            strengths={"A.pdf": [], "B.pdf": []},
        )
        rendered = render_comparison(result, "compare")
        assert "**Comparison: A.pdf vs B.pdf**" in rendered
        # Table format: shared skills row and per-doc values
        assert "| Criterion |" in rendered
        assert "| Technical Skills |" in rendered
        assert "Python" in rendered
        assert "Java" in rendered
        assert "Go" in rendered

    def test_render_comparison_three_docs(self):
        result = ComparisonResult(
            documents=["A.pdf", "B.pdf", "C.pdf"],
            field_comparisons=[
                FieldComparison(
                    field_name="role",
                    values={"A.pdf": "Developer", "B.pdf": "Manager", "C.pdf": "Analyst"},
                    comparison_type="text",
                ),
            ],
            summary="Comparison of 3 docs",
            strengths={"A.pdf": [], "B.pdf": [], "C.pdf": []},
        )
        rendered = render_comparison(result, "compare")
        assert "**Comparison of 3 documents**" in rendered
        # 3+ docs use markdown table format
        assert "Developer" in rendered
        assert "Manager" in rendered
        assert "Analyst" in rendered

    def test_compare_candidates_from_schema(self):
        cand_a = _make_candidate(
            "Abinaya", technical_skills=["Python", "Java", "AWS"],
            certifications=["CAPM"], total_years_experience="5 years",
        )
        cand_b = _make_candidate(
            "Aadithya", technical_skills=["Python", "Go", "Docker"],
            certifications=["PMP", "AWS SAA"], total_years_experience="8 years",
        )
        result = compare_candidates_from_schema([cand_a, cand_b], "compare all candidates")
        assert len(result.documents) == 2
        assert "Abinaya" in result.documents
        assert "Aadithya" in result.documents
        # Should have list comparisons for skills
        list_comps = [c for c in result.field_comparisons if c.comparison_type == "overlap"]
        assert len(list_comps) >= 1
