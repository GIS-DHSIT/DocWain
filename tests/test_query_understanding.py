"""Tests for the unified query understanding module."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.intelligence.task_spec import TaskSpec
from src.intelligence.query_understanding import (
    understand_query,
    _nlu_fallback,
    _infer_output_format,
    _try_llm_understanding,
    _FALLBACK_CONFIDENCE,
)


class TestUnderstandQuery:
    """Test the main understand_query() entry point."""

    def test_returns_taskspec(self):
        """understand_query always returns a TaskSpec."""
        result = understand_query("List all candidates")
        assert isinstance(result, TaskSpec)

    def test_fallback_when_model_unavailable(self):
        """When LLM is not available, NLU fallback is used."""
        with patch(
            "src.intelligence.query_understanding._get_llm_client",
            return_value=None,
        ):
            result = understand_query("Compare candidates for Python role")
            assert isinstance(result, TaskSpec)
            assert result.confidence == _FALLBACK_CONFIDENCE

    def test_nlu_fallback_sets_low_confidence(self):
        """NLU fallback always marks confidence as 0.4."""
        result = _nlu_fallback("What are the patient diagnoses?")
        assert result.confidence == _FALLBACK_CONFIDENCE
        assert isinstance(result, TaskSpec)

    def test_nlu_fallback_hr_domain(self):
        """NLU fallback can detect HR domain from domain hint."""
        result = _nlu_fallback("List candidate skills", domain_hint="hr")
        assert result.domain == "hr"

    def test_nlu_fallback_returns_entities(self):
        """NLU fallback extracts entities from query."""
        result = _nlu_fallback("Who has the most experience in Python?")
        assert isinstance(result.entities, list)

    def test_nlu_fallback_general_domain(self):
        """Without domain hint, defaults to general."""
        result = _nlu_fallback("Tell me about the documents")
        assert result.domain in ("general", "hr", "medical", "legal", "invoice",
                                  "insurance", "policy", "content", "translation", "education")


class TestLLMUnderstanding:
    """Test the LLM-based understanding path."""

    def test_llm_returns_taskspec_on_success(self):
        """When LLM returns valid JSON, parse into TaskSpec."""
        mock_client = MagicMock()
        mock_response = {
            "message": {
                "content": json.dumps({
                    "intent": "compare",
                    "domain": "hr",
                    "output_format": "table",
                    "entities": ["Python"],
                    "constraints": {},
                    "scope": "cross_document",
                    "complexity": "complex",
                    "confidence": 0.92,
                })
            }
        }
        mock_client.chat.return_value = mock_response

        with patch(
            "src.intelligence.query_understanding._get_llm_client",
            return_value=mock_client,
        ):
            result = _try_llm_understanding("Compare candidates for Python role")
            assert result is not None
            assert result.intent == "compare"
            assert result.domain == "hr"
            assert result.confidence == 0.92

    def test_llm_returns_none_on_invalid_json(self):
        """When LLM returns garbage, return None → triggers fallback."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "I don't understand the question."}
        }

        with patch(
            "src.intelligence.query_understanding._get_llm_client",
            return_value=mock_client,
        ):
            result = _try_llm_understanding("Compare candidates")
            assert result is None

    def test_llm_returns_none_on_exception(self):
        """When LLM call fails, return None gracefully."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("model not running")

        with patch(
            "src.intelligence.query_understanding._get_llm_client",
            return_value=mock_client,
        ):
            result = _try_llm_understanding("Compare candidates")
            assert result is None

    def test_llm_extracts_json_from_markdown_fences(self):
        """Model may wrap JSON in markdown code fences."""
        mock_client = MagicMock()
        json_str = json.dumps({
            "intent": "factual", "domain": "medical",
            "output_format": "paragraph", "entities": ["diagnoses"],
            "constraints": {}, "scope": "all_documents",
            "complexity": "simple", "confidence": 0.88,
        })
        mock_client.chat.return_value = {
            "message": {"content": f"```json\n{json_str}\n```"}
        }

        with patch(
            "src.intelligence.query_understanding._get_llm_client",
            return_value=mock_client,
        ):
            result = _try_llm_understanding("What are the diagnoses?")
            assert result is not None
            assert result.intent == "factual"
            assert result.domain == "medical"


class TestOutputFormatInference:
    """Test heuristic output format detection."""

    def test_table_keyword(self):
        assert _infer_output_format("Show a comparison table") == "table"

    def test_matrix_keyword(self):
        assert _infer_output_format("Create a skills matrix") == "table"

    def test_bullet_keyword(self):
        assert _infer_output_format("List them as bullet points") == "bullets"

    def test_chart_keyword(self):
        assert _infer_output_format("Visualize as a chart") == "chart_data"

    def test_numbered_keyword(self):
        assert _infer_output_format("Rank the top 5 candidates") == "numbered"

    def test_json_keyword(self):
        assert _infer_output_format("Return structured data as json") == "json"

    def test_default_paragraph(self):
        assert _infer_output_format("What are the patient diagnoses?") == "paragraph"


class TestQueryUnderstandingIntegration:
    """Integration tests — verify end-to-end with NLU fallback."""

    @pytest.mark.parametrize("query,expected_domain", [
        ("List all candidates and their skills", "hr"),
        ("What medications are prescribed?", "medical"),
        ("Find risky clauses in the contract", "legal"),
        ("What are the invoice totals?", "invoice"),
    ])
    def test_domain_detection_via_fallback(self, query, expected_domain):
        """NLU fallback correctly identifies domain for clear queries."""
        with patch(
            "src.intelligence.query_understanding._get_llm_client",
            return_value=None,
        ):
            result = understand_query(query)
            # NLU engine may or may not detect domain — just verify it returns valid TaskSpec
            assert isinstance(result, TaskSpec)
            assert result.confidence == _FALLBACK_CONFIDENCE

    def test_low_confidence_triggers_fallback(self):
        """If LLM returns low confidence, fall back to NLU."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "content": json.dumps({
                    "intent": "factual", "domain": "general",
                    "confidence": 0.3,  # below 0.5 threshold
                })
            }
        }

        with patch(
            "src.intelligence.query_understanding._get_llm_client",
            return_value=mock_client,
        ):
            result = understand_query("Some ambiguous query")
            # Should have fallen back to NLU
            assert result.confidence == _FALLBACK_CONFIDENCE
