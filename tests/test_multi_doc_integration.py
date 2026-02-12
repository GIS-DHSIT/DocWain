"""End-to-end integration tests for multi-document intelligence (Task 7).

Uses FakeQdrant/FakeEmbedder/FakeRedis for pipeline-level testing.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, FakeRedis, make_point
from src.rag_v3.types import LLMBudget


def _build_resume_points():
    """Create synthetic Qdrant points for 2 resume documents."""
    return [
        make_point(
            pid="p1", profile_id="prof-1", document_id="doc1",
            file_name="Abinaya Resume.pdf", text="Name: Abinaya\nTechnical Skills: Python, Java, SAP S/4 HANA",
            page=1, section_kind="skills_technical", doc_domain="resume",
        ),
        make_point(
            pid="p2", profile_id="prof-1", document_id="doc1",
            file_name="Abinaya Resume.pdf", text="Experience: 5 years at Deloitte as Business Analyst",
            page=1, section_kind="experience", doc_domain="resume",
        ),
        make_point(
            pid="p3", profile_id="prof-1", document_id="doc1",
            file_name="Abinaya Resume.pdf", text="Education: MBA from XLRI\nCertifications: CAPM, SAP Certified",
            page=2, section_kind="education", doc_domain="resume",
        ),
        make_point(
            pid="p4", profile_id="prof-1", document_id="doc2",
            file_name="Aadithya Profile.pdf", text="Name: Aadithya\nTechnical Skills: Python, Go, Docker, Kubernetes",
            page=1, section_kind="skills_technical", doc_domain="resume",
        ),
        make_point(
            pid="p5", profile_id="prof-1", document_id="doc2",
            file_name="Aadithya Profile.pdf", text="Experience: 8 years as Senior Software Engineer at Google",
            page=1, section_kind="experience", doc_domain="resume",
        ),
        make_point(
            pid="p6", profile_id="prof-1", document_id="doc2",
            file_name="Aadithya Profile.pdf", text="Education: B.Tech from IIT Madras\nCertifications: AWS Solutions Architect",
            page=2, section_kind="education", doc_domain="resume",
        ),
    ]


def _build_mixed_points():
    """Create points with mixed domains (resumes + invoices)."""
    resume_points = _build_resume_points()
    invoice_points = [
        make_point(
            pid="p7", profile_id="prof-1", document_id="doc3",
            file_name="Invoice_001.pdf", text="Invoice Number: INV-001\nTotal Amount: $15,000.00",
            page=1, section_kind="invoice_header", doc_domain="invoice",
        ),
        make_point(
            pid="p8", profile_id="prof-1", document_id="doc3",
            file_name="Invoice_001.pdf", text="Line Items:\nConsulting Services: $10,000\nTravel: $5,000",
            page=1, section_kind="invoice_items", doc_domain="invoice",
        ),
    ]
    return resume_points + invoice_points


def _run_pipeline_direct(query: str, points: list):
    """Run _run_all_profile_analysis directly with mocked infrastructure."""
    from src.rag_v3.pipeline import _run_all_profile_analysis

    fake_qdrant = FakeQdrant(points)
    budget = LLMBudget(llm_client=None, max_calls=0)

    with patch("src.rag_v3.pipeline.build_collection_name", return_value="test-col"), \
         patch("src.rag_v3.pipeline.expand_full_scan_by_profile", return_value=[
             # Convert FakePoints to Chunks
             _fake_point_to_chunk(pt) for pt in points
         ]), \
         patch("src.rag_v3.pipeline.rerank", side_effect=lambda query, chunks, **kw: chunks), \
         patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x):

        return _run_all_profile_analysis(
            query=query,
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=fake_qdrant,
            embedder=FakeEmbedder(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )


def _fake_point_to_chunk(pt):
    """Convert a FakePoint to a Chunk for pipeline consumption."""
    from src.rag_v3.types import Chunk, ChunkSource
    payload = pt.payload or {}
    return Chunk(
        id=str(pt.id),
        text=payload.get("canonical_text", ""),
        score=pt.score,
        source=ChunkSource(
            document_name=payload.get("source_name", ""),
            page=payload.get("page"),
        ),
        meta={
            "document_id": payload.get("document_id", ""),
            "source_name": payload.get("source_name", ""),
            "section_kind": payload.get("section_kind", ""),
            "doc_domain": payload.get("doc_domain", ""),
        },
    )


class TestE2EMultiDocIntelligence:
    def test_e2e_analytics_how_many_resumes(self):
        """'How many resumes?' should return count from analytics engine."""
        points = _build_resume_points()
        result = _run_pipeline_direct("How many resumes are there?", points)
        answer = result["response"]
        # Should mention 2 resumes
        assert "2" in answer

    def test_e2e_compare_all_candidates(self):
        """'Compare all candidates' should produce comparison output."""
        points = _build_resume_points()
        result = _run_pipeline_direct("Compare all candidates side by side", points)
        answer = result["response"]
        # Should mention both candidates or have comparison structure
        assert any(name in answer for name in ["Abinaya", "Aadithya", "Comparison", "compare"])

    def test_e2e_single_candidate_query(self):
        """Single-candidate factual query should still work through existing path."""
        points = _build_resume_points()
        # This is a factual query — will fall through to extract_schema.
        # With llm_client=None it goes deterministic, which should produce something.
        with patch("src.rag_v3.pipeline.extract_schema") as mock_extract, \
             patch("src.rag_v3.pipeline._extract_render_judge") as mock_erj:
            from src.rag_v3.judge import JudgeResult
            mock_extraction = MagicMock()
            mock_extraction.domain = "hr"
            mock_extraction.intent = "detail"
            mock_extract.return_value = mock_extraction
            mock_erj.return_value = ("Skills: Python, Java, SAP", JudgeResult(status="pass", reason="ok"))

            result = _run_pipeline_direct("What are Abinaya's technical skills?", points)
            answer = result["response"]
            assert len(answer) > 5
            mock_extract.assert_called_once()

    def test_e2e_all_profile_with_mixed_domains(self):
        """Mixed domains should be handled with domain-aware grouping."""
        points = _build_mixed_points()
        result = _run_pipeline_direct("How many documents in total?", points)
        answer = result["response"]
        # Should count all documents (3 total: 2 resumes + 1 invoice)
        assert "3" in answer

    def test_e2e_ranking_query(self):
        """Ranking query should fall through to extract_schema path for structured HR extraction."""
        points = _build_resume_points()
        # Ranking goes through extract_schema, not comparator, so mock the extraction
        with patch("src.rag_v3.pipeline.extract_schema") as mock_extract, \
             patch("src.rag_v3.pipeline._extract_render_judge") as mock_erj:
            from src.rag_v3.judge import JudgeResult
            mock_extraction = MagicMock()
            mock_extraction.domain = "hr"
            mock_extraction.intent = "rank"
            mock_extract.return_value = mock_extraction
            mock_erj.return_value = ("1. Aadithya - 8 years\n2. Abinaya - 5 years", JudgeResult(status="pass", reason="ok"))

            result = _run_pipeline_direct("Rank candidates by experience", points)
            answer = result["response"]
            assert len(answer) > 10
            mock_extract.assert_called_once()

    def test_e2e_factual_multi_doc(self):
        """Factual multi-doc query ('all candidates') routes to cross_document intent."""
        points = _build_resume_points()
        result = _run_pipeline_direct("What are the skills of all candidates?", points)
        answer = result["response"]
        # Cross-document intent — falls through to extract_schema with multi-doc data.
        # Should have some content.
        assert len(answer) > 5
