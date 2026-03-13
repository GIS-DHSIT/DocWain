"""Tests for Task 8 — CoreAgent KG probe at query time."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

from src.agent.core_agent import CoreAgent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mock_llm():
    """LLM that returns valid JSON for intent analysis and reasoning."""
    llm = MagicMock()
    intent_json = json.dumps({
        "task_type": "lookup",
        "complexity": "simple",
        "resolved_query": "test query",
        "output_format": "prose",
        "relevant_documents": [],
        "cross_profile": False,
        "entities": [],
    })
    # generate is used by IntentAnalyzer and Reasoner
    llm.generate.return_value = intent_json
    return llm


@dataclass
class _FakeRetrievalResult:
    chunks: list = field(default_factory=list)
    profiles_searched: list = field(default_factory=list)


@dataclass
class _FakeKGResult:
    doc_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)


def _mock_qdrant():
    return MagicMock()


def _mock_embedder():
    m = MagicMock()
    m.encode.return_value = [[0.0] * 1024]
    return m


def _mock_mongodb():
    m = MagicMock()
    m.find.return_value = []
    return m


def _mock_kg_service(doc_ids=None, chunk_ids=None):
    kg = MagicMock()
    kg.extract_entities.return_value = [{"entity_id": "e1", "value": "Alpha"}]
    kg.query.return_value = _FakeKGResult(
        doc_ids=doc_ids or ["doc_kg_1"],
        chunk_ids=chunk_ids or ["chunk_kg_1"],
    )
    return kg


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@patch("src.agent.core_agent.compose_response", return_value={"answer": "ok"})
@patch("src.agent.core_agent.build_context", return_value=([], {}))
@patch("src.agent.core_agent.rerank_chunks", return_value=[])
def test_kg_service_called_when_provided(mock_rerank, mock_ctx, mock_compose):
    """CoreAgent should call kg_query_service.query() when it's provided."""
    kg = _mock_kg_service()

    agent = CoreAgent(
        llm_gateway=_mock_llm(),
        qdrant_client=_mock_qdrant(),
        embedder=_mock_embedder(),
        mongodb=_mock_mongodb(),
        kg_query_service=kg,
    )

    # Patch retriever to return a fake result
    agent._retriever = MagicMock()
    agent._retriever.retrieve.return_value = _FakeRetrievalResult()

    # Patch reasoner
    agent._reasoner = MagicMock()
    agent._reasoner.reason.return_value = MagicMock(
        text="answer", grounded=True, usage={},
    )

    agent.handle(
        query="Tell me about Alpha",
        subscription_id="sub_1",
        profile_id="prof_1",
        user_id="user_1",
        session_id="sess_1",
        conversation_history=None,
    )

    kg.extract_entities.assert_called_once()
    kg.query.assert_called_once()

    # Verify the KG doc IDs were passed to retrieve
    call_kwargs = agent._retriever.retrieve.call_args
    doc_ids_arg = call_kwargs.kwargs.get("document_ids") or call_kwargs[1].get("document_ids")
    assert doc_ids_arg is not None
    assert "doc_kg_1" in doc_ids_arg


@patch("src.agent.core_agent.compose_response", return_value={"answer": "ok"})
@patch("src.agent.core_agent.build_context", return_value=([], {}))
@patch("src.agent.core_agent.rerank_chunks", return_value=[])
def test_core_agent_works_without_kg_service(mock_rerank, mock_ctx, mock_compose):
    """CoreAgent should work fine when kg_query_service is None (backward compat)."""
    agent = CoreAgent(
        llm_gateway=_mock_llm(),
        qdrant_client=_mock_qdrant(),
        embedder=_mock_embedder(),
        mongodb=_mock_mongodb(),
    )

    agent._retriever = MagicMock()
    agent._retriever.retrieve.return_value = _FakeRetrievalResult()

    agent._reasoner = MagicMock()
    agent._reasoner.reason.return_value = MagicMock(
        text="answer", grounded=True, usage={},
    )

    result = agent.handle(
        query="Tell me about Alpha",
        subscription_id="sub_1",
        profile_id="prof_1",
        user_id="user_1",
        session_id="sess_1",
        conversation_history=None,
    )

    assert result is not None


@patch("src.agent.core_agent.compose_response", return_value={"answer": "ok"})
@patch("src.agent.core_agent.build_context", return_value=([], {}))
@patch("src.agent.core_agent.rerank_chunks", return_value=[])
def test_kg_probe_failure_is_non_fatal(mock_rerank, mock_ctx, mock_compose):
    """If kg_query_service.query() raises, the pipeline should still work."""
    kg = MagicMock()
    kg.extract_entities.side_effect = RuntimeError("KG unavailable")

    agent = CoreAgent(
        llm_gateway=_mock_llm(),
        qdrant_client=_mock_qdrant(),
        embedder=_mock_embedder(),
        mongodb=_mock_mongodb(),
        kg_query_service=kg,
    )

    agent._retriever = MagicMock()
    agent._retriever.retrieve.return_value = _FakeRetrievalResult()

    agent._reasoner = MagicMock()
    agent._reasoner.reason.return_value = MagicMock(
        text="answer", grounded=True, usage={},
    )

    # Should not raise
    result = agent.handle(
        query="Tell me about Alpha",
        subscription_id="sub_1",
        profile_id="prof_1",
        user_id="user_1",
        session_id="sess_1",
        conversation_history=None,
    )
    assert result is not None
