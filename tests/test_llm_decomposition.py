"""Tests for LLM-backed query decomposition."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from src.rag_v3.query_decomposer import decompose_query


class TestLLMDecomposition:

    def test_llm_decomposition_called_for_complex_query(self):
        """Long complex queries should attempt LLM decomposition."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"sub_queries": [{"text": "candidate cloud experience", "entity_scope": null}, {"text": "senior architect requirements", "entity_scope": null}]}'

        result = decompose_query(
            "Which candidate has the most relevant experience for a senior cloud architect role requiring AWS and Kubernetes?",
            llm_client=mock_llm,
        )
        assert mock_llm.generate.called
        assert len(result.sub_queries) >= 1

    def test_llm_failure_falls_back_to_heuristic(self):
        """If LLM decomposition fails, heuristic result should be used."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = TimeoutError("LLM timeout")

        result = decompose_query(
            "Which candidate has the most relevant experience for a senior cloud architect role?",
            llm_client=mock_llm,
        )
        assert len(result.sub_queries) >= 1
        assert result.sub_queries[0].text

    def test_short_query_skips_llm(self):
        """Short queries (< 10 words) should NOT trigger LLM decomposition."""
        mock_llm = MagicMock()
        result = decompose_query("What are Prudhvi's skills?", llm_client=mock_llm)
        assert not mock_llm.generate.called
        assert len(result.sub_queries) >= 1

    def test_llm_returns_empty_falls_back(self):
        """LLM returning empty response falls back to heuristic."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""

        result = decompose_query(
            "Which candidate has the most relevant experience for a senior cloud architect role requiring AWS?",
            llm_client=mock_llm,
        )
        assert len(result.sub_queries) >= 1

    def test_llm_returns_invalid_json_falls_back(self):
        """LLM returning invalid JSON falls back to heuristic."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not valid json at all"

        result = decompose_query(
            "Which candidate has the most relevant experience for a senior cloud architect role requiring AWS?",
            llm_client=mock_llm,
        )
        assert len(result.sub_queries) >= 1
