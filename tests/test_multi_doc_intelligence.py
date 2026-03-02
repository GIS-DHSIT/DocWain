"""Tests for analytics intent classification (Task 4) and pipeline integration (Task 5)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.rag_v3.llm_extract import classify_query_intent, _GENERATION_TEMPLATES
from src.rag_v3.types import Chunk, ChunkSource, LLMBudget


# ── Task 4: Intent classification tests ──────────────────────────────

class TestAnalyticsIntentClassification:
    def test_analytics_intent_how_many(self):
        assert classify_query_intent("How many invoices?") == "analytics"

    def test_analytics_intent_total(self):
        assert classify_query_intent("Total amount across invoices?") == "analytics"

    def test_comparison_intent_unchanged(self):
        assert classify_query_intent("Compare candidates") == "comparison"

    def test_factual_intent_unchanged(self):
        assert classify_query_intent("What are Abinaya's skills?") == "factual"

    def test_analytics_in_generation_templates(self):
        assert "analytics" in _GENERATION_TEMPLATES
        assert "aggregate" in _GENERATION_TEMPLATES["analytics"].lower()


# ── Task 5: Pipeline integration tests ───────────────────────────────

def _make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc1",
    doc_name: str = "Resume.pdf",
    section_kind: str = "skills_technical",
    doc_domain: str = "resume",
    score: float = 0.9,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        score=score,
        source=ChunkSource(document_name=doc_name, page=1),
        meta={
            "document_id": doc_id,
            "source_name": doc_name,
            "section_kind": section_kind,
            "doc_domain": doc_domain,
        },
    )


class TestPipelineRouting:
    """Test that the pipeline routes analytics/comparison/factual queries correctly."""

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.rerank")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    @patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x)
    def test_analytics_query_routes_to_corpus(
        self, mock_dedup, mock_build, mock_rerank, mock_scan,
    ):
        """Analytics query should use corpus_analytics, not extract_schema."""
        from src.rag_v3.pipeline import _run_all_profile_analysis

        chunks = [
            _make_chunk("c1", "Python skills", doc_id="d1", doc_name="Resume1.pdf"),
            _make_chunk("c2", "Java skills", doc_id="d2", doc_name="Resume2.pdf"),
        ]
        mock_scan.return_value = chunks
        mock_rerank.return_value = chunks

        budget = LLMBudget(llm_client=None, max_calls=0)
        result = _run_all_profile_analysis(
            query="How many resumes are there?",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )
        answer_text = result["response"]
        # Analytics should detect 2 documents
        assert "2" in answer_text

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.rerank")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    @patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x)
    def test_comparison_query_routes_to_comparator(
        self, mock_dedup, mock_build, mock_rerank, mock_scan,
    ):
        """Comparison query with no LLM should use deterministic comparator."""
        from src.rag_v3.pipeline import _run_all_profile_analysis

        chunks = [
            _make_chunk("c1", "Technical Skills: Python, Java", doc_id="d1", doc_name="Abinaya.pdf"),
            _make_chunk("c2", "Technical Skills: Python, Go", doc_id="d2", doc_name="Aadithya.pdf"),
        ]
        mock_scan.return_value = chunks
        mock_rerank.return_value = chunks

        budget = LLMBudget(llm_client=None, max_calls=0)
        result = _run_all_profile_analysis(
            query="Compare candidates side by side",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )
        answer_text = result["response"]
        assert "Comparison" in answer_text or "Abinaya" in answer_text or "Aadithya" in answer_text

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.rerank")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    @patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x)
    @patch("src.rag_v3.pipeline.extract_schema")
    @patch("src.rag_v3.pipeline._extract_render_judge")
    def test_factual_query_uses_existing_path(
        self, mock_erj, mock_extract, mock_dedup, mock_build, mock_rerank, mock_scan,
    ):
        """Factual query should fall through to existing extract_schema path."""
        from src.rag_v3.pipeline import _run_all_profile_analysis
        from src.rag_v3.judge import JudgeResult

        chunks = [
            _make_chunk("c1", "Python, Java, AWS", doc_id="d1", doc_name="Abinaya.pdf"),
        ]
        mock_scan.return_value = chunks
        mock_rerank.return_value = chunks

        # Mock extraction + judge
        mock_extraction = MagicMock()
        mock_extraction.domain = "hr"
        mock_extraction.intent = "detail"
        mock_extract.return_value = mock_extraction
        mock_erj.return_value = ("Skills: Python, Java", JudgeResult(status="pass", reason="ok"))

        budget = LLMBudget(llm_client=None, max_calls=0)
        result = _run_all_profile_analysis(
            query="What are Abinaya's skills?",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )
        # extract_schema should have been called
        mock_extract.assert_called_once()

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    def test_all_profile_with_no_chunks_returns_no_info(
        self, mock_build, mock_scan,
    ):
        """Empty profile should return error message."""
        from src.rag_v3.pipeline import _run_all_profile_analysis, NO_CHUNKS_MESSAGE

        mock_scan.return_value = []
        budget = LLMBudget(llm_client=None, max_calls=0)
        result = _run_all_profile_analysis(
            query="How many resumes?",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )
        assert "couldn't find" in result["response"] or "Not enough information" in result["response"]

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.rerank")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    @patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x)
    def test_metadata_includes_document_count(
        self, mock_dedup, mock_build, mock_rerank, mock_scan,
    ):
        """Analytics result metadata should include document_count."""
        from src.rag_v3.pipeline import _run_all_profile_analysis

        chunks = [
            _make_chunk("c1", "text", doc_id="d1", doc_name="R1.pdf"),
            _make_chunk("c2", "text", doc_id="d2", doc_name="R2.pdf"),
            _make_chunk("c3", "text", doc_id="d3", doc_name="R3.pdf"),
        ]
        mock_scan.return_value = chunks
        mock_rerank.return_value = chunks

        budget = LLMBudget(llm_client=None, max_calls=0)
        result = _run_all_profile_analysis(
            query="How many documents in total?",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )
        assert result.get("metadata", {}).get("document_count") == 3

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.rerank")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    @patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x)
    def test_metadata_includes_intent_type(
        self, mock_dedup, mock_build, mock_rerank, mock_scan,
    ):
        """Analytics result metadata should have intent='analytics'."""
        from src.rag_v3.pipeline import _run_all_profile_analysis

        chunks = [
            _make_chunk("c1", "text", doc_id="d1", doc_name="R1.pdf"),
        ]
        mock_scan.return_value = chunks
        mock_rerank.return_value = chunks

        budget = LLMBudget(llm_client=None, max_calls=0)
        result = _run_all_profile_analysis(
            query="How many resumes are there?",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )
        assert result.get("metadata", {}).get("intent") == "analytics"

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.rerank")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    @patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x)
    def test_comparison_with_llm_uses_llm_response(
        self, mock_dedup, mock_build, mock_rerank, mock_scan,
    ):
        """When LLM is available, comparison should use LLM response."""
        from src.rag_v3.pipeline import _run_all_profile_analysis
        from src.rag_v3.types import LLMResponseSchema

        chunks = [
            _make_chunk("c1", "Python", doc_id="d1", doc_name="A.pdf"),
            _make_chunk("c2", "Java", doc_id="d2", doc_name="B.pdf"),
        ]
        mock_scan.return_value = chunks
        mock_rerank.return_value = chunks

        mock_llm = MagicMock()
        budget = LLMBudget(llm_client=mock_llm, max_calls=4)

        with patch("src.rag_v3.llm_extract.llm_extract_and_respond") as mock_llm_extract:
            mock_llm_extract.return_value = LLMResponseSchema(
                text="LLM comparison: A is better at Python, B at Java.",
            )
            result = _run_all_profile_analysis(
                query="Compare candidates",
                subscription_id="sub-1",
                profile_id="prof-1",
                qdrant_client=MagicMock(),
                embedder=MagicMock(),
                cross_encoder=None,
                llm_client=mock_llm,
                budget=budget,
                intent_parse=None,
                correlation_id="test",
                request_id="test",
            )
            assert "LLM comparison" in result["response"]

    @patch("src.rag_v3.pipeline.expand_full_scan_by_profile")
    @patch("src.rag_v3.pipeline.rerank")
    @patch("src.rag_v3.pipeline.build_collection_name", return_value="test-collection")
    @patch("src.rag_v3.pipeline.deduplicate_by_content", side_effect=lambda x: x)
    def test_comparison_without_llm_uses_deterministic(
        self, mock_dedup, mock_build, mock_rerank, mock_scan,
    ):
        """Without LLM, comparison should use deterministic comparator."""
        from src.rag_v3.pipeline import _run_all_profile_analysis

        chunks = [
            _make_chunk("c1", "Technical Skills: Python, Java", doc_id="d1", doc_name="Abinaya.pdf"),
            _make_chunk("c2", "Technical Skills: Go, Docker", doc_id="d2", doc_name="Aadithya.pdf"),
        ]
        mock_scan.return_value = chunks
        mock_rerank.return_value = chunks

        budget = LLMBudget(llm_client=None, max_calls=0)
        result = _run_all_profile_analysis(
            query="Compare all candidates",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=None,
            llm_client=None,
            budget=budget,
            intent_parse=None,
            correlation_id="test",
            request_id="test",
        )
        answer = result["response"]
        # Should contain comparison content from the deterministic comparator
        assert "Abinaya" in answer or "Aadithya" in answer or "Comparison" in answer
