"""Tests for CoreAgent orchestrator and DynamicSubAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.subagent import DynamicSubAgent, SubAgentResult
from src.agent.core_agent import CoreAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = "Sub-agent output text"
    llm.generate_with_metadata.return_value = (
        "Answer text",
        {"thinking": None, "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    )
    return llm


@pytest.fixture
def mock_qdrant():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1] * 1024]
    return embedder


@pytest.fixture
def mock_mongodb():
    return MagicMock()


@pytest.fixture
def core_agent(mock_llm, mock_qdrant, mock_embedder, mock_mongodb):
    return CoreAgent(
        llm_gateway=mock_llm,
        qdrant_client=mock_qdrant,
        embedder=mock_embedder,
        mongodb=mock_mongodb,
    )


# ---------------------------------------------------------------------------
# CoreAgent init
# ---------------------------------------------------------------------------


def test_core_agent_init_creates_subcomponents(mock_llm, mock_qdrant, mock_embedder, mock_mongodb):
    """CoreAgent init creates IntentAnalyzer, UnifiedRetriever, and Reasoner."""
    with patch("src.agent.core_agent.IntentAnalyzer") as MockIntent, \
         patch("src.agent.core_agent.UnifiedRetriever") as MockRetriever, \
         patch("src.agent.core_agent.Reasoner") as MockReasoner:
        agent = CoreAgent(mock_llm, mock_qdrant, mock_embedder, mock_mongodb)
        MockIntent.assert_called_once_with(llm_gateway=mock_llm)
        MockRetriever.assert_called_once_with(qdrant_client=mock_qdrant, embedder=mock_embedder)
        MockReasoner.assert_called_once_with(llm_gateway=mock_llm)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_handle_validates_subscription_id(core_agent):
    """handle() raises ValueError when subscription_id is empty."""
    with pytest.raises(ValueError, match="subscription_id"):
        core_agent.handle(
            query="test",
            subscription_id="",
            profile_id="p1",
            user_id="u1",
            session_id="s1",
            conversation_history=None,
        )


def test_handle_validates_profile_id(core_agent):
    """handle() raises ValueError when profile_id is empty."""
    with pytest.raises(ValueError, match="profile_id"):
        core_agent.handle(
            query="test",
            subscription_id="sub1",
            profile_id="",
            user_id="u1",
            session_id="s1",
            conversation_history=None,
        )


# ---------------------------------------------------------------------------
# Conversational short-circuit
# ---------------------------------------------------------------------------


def test_handle_conversational_shortcircuits(mock_llm, mock_qdrant, mock_embedder, mock_mongodb):
    """Conversational query short-circuits without retrieval."""
    from src.agent.intent import QueryUnderstanding

    understanding = QueryUnderstanding(
        task_type="conversational",
        complexity="simple",
        resolved_query="hello",
        output_format="prose",
        relevant_documents=[],
        cross_profile=False,
    )

    with patch("src.agent.core_agent.IntentAnalyzer") as MockIntent, \
         patch("src.agent.core_agent.UnifiedRetriever") as MockRetriever, \
         patch("src.agent.core_agent.Reasoner"):
        MockIntent.return_value.analyze.return_value = understanding
        agent = CoreAgent(mock_llm, mock_qdrant, mock_embedder, mock_mongodb)
        result = agent.handle(
            query="hello",
            subscription_id="sub1",
            profile_id="p1",
            user_id="u1",
            session_id="s1",
            conversation_history=None,
        )
        # Retriever should NOT have been called
        MockRetriever.return_value.retrieve.assert_not_called()
        assert "response" in result
        assert result["context_found"] is False


# ---------------------------------------------------------------------------
# Simple query — full pipeline
# ---------------------------------------------------------------------------


def test_handle_simple_query_full_pipeline(mock_llm, mock_qdrant, mock_embedder, mock_mongodb):
    """Simple query runs the full UNDERSTAND → RETRIEVE → REASON → COMPOSE pipeline."""
    from src.agent.intent import QueryUnderstanding
    from src.retrieval.retriever import EvidenceChunk, RetrievalResult
    from src.generation.reasoner import ReasonerResult

    understanding = QueryUnderstanding(
        task_type="lookup",
        complexity="simple",
        resolved_query="What is the contract value?",
        output_format="prose",
        relevant_documents=[],
        cross_profile=False,
    )

    chunk = EvidenceChunk(
        text="The contract value is $100,000.",
        source_name="contract.pdf",
        document_id="doc1",
        profile_id="p1",
        section="Terms",
        page_start=1,
        page_end=1,
        score=0.9,
        chunk_id="c1",
    )
    retrieval_result = RetrievalResult(chunks=[chunk], profiles_searched=["p1"], total_found=1)
    reason_result = ReasonerResult(
        text="The contract value is **$100,000** [SOURCE-1].",
        sources=[{"source_name": "contract.pdf"}],
        grounded=True,
        usage={"prompt_tokens": 100},
    )

    with patch("src.agent.core_agent.IntentAnalyzer") as MockIntent, \
         patch("src.agent.core_agent.UnifiedRetriever") as MockRetriever, \
         patch("src.agent.core_agent.Reasoner") as MockReasoner, \
         patch("src.agent.core_agent.rerank_chunks", return_value=[chunk]), \
         patch("src.agent.core_agent.build_context", return_value=([{"text": "evidence"}], {})):
        MockIntent.return_value.analyze.return_value = understanding
        MockRetriever.return_value.retrieve.return_value = retrieval_result
        MockReasoner.return_value.reason.return_value = reason_result
        agent = CoreAgent(mock_llm, mock_qdrant, mock_embedder, mock_mongodb)
        result = agent.handle(
            query="What is the contract value?",
            subscription_id="sub1",
            profile_id="p1",
            user_id="u1",
            session_id="s1",
            conversation_history=None,
        )

    assert "response" in result
    assert "sources" in result
    assert "grounded" in result
    assert "context_found" in result
    assert "metadata" in result


# ---------------------------------------------------------------------------
# AnswerPayload fields
# ---------------------------------------------------------------------------


def test_handle_returns_all_answer_payload_fields(mock_llm, mock_qdrant, mock_embedder, mock_mongodb):
    """handle() returns dict with response, sources, grounded, context_found, metadata."""
    from src.agent.intent import QueryUnderstanding
    from src.retrieval.retriever import EvidenceChunk, RetrievalResult
    from src.generation.reasoner import ReasonerResult

    understanding = QueryUnderstanding(
        task_type="lookup",
        complexity="simple",
        resolved_query="test",
        output_format="prose",
        relevant_documents=[],
        cross_profile=False,
    )
    chunk = EvidenceChunk(
        text="data", source_name="f.pdf", document_id="d1", profile_id="p1",
        section="S", page_start=1, page_end=1, score=0.8, chunk_id="c1",
    )
    retrieval_result = RetrievalResult(chunks=[chunk], profiles_searched=["p1"], total_found=1)
    reason_result = ReasonerResult(text="answer", sources=[], grounded=True, usage={})

    with patch("src.agent.core_agent.IntentAnalyzer") as MockIntent, \
         patch("src.agent.core_agent.UnifiedRetriever") as MockRetriever, \
         patch("src.agent.core_agent.Reasoner") as MockReasoner, \
         patch("src.agent.core_agent.rerank_chunks", return_value=[chunk]), \
         patch("src.agent.core_agent.build_context", return_value=([{"text": "ev"}], {})):
        MockIntent.return_value.analyze.return_value = understanding
        MockRetriever.return_value.retrieve.return_value = retrieval_result
        MockReasoner.return_value.reason.return_value = reason_result
        agent = CoreAgent(mock_llm, mock_qdrant, mock_embedder, mock_mongodb)
        result = agent.handle(
            query="test", subscription_id="sub1", profile_id="p1",
            user_id="u1", session_id="s1", conversation_history=None,
        )

    required_keys = {"response", "sources", "grounded", "context_found", "metadata"}
    assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# _handle_conversational
# ---------------------------------------------------------------------------


def test_handle_conversational_returns_greeting(core_agent):
    """_handle_conversational returns a friendly greeting for 'hello'."""
    result = core_agent._handle_conversational("hello")
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0


def test_handle_conversational_returns_farewell(core_agent):
    """_handle_conversational returns farewell text for 'goodbye'."""
    result = core_agent._handle_conversational("goodbye")
    assert "response" in result


def test_handle_conversational_returns_thanks(core_agent):
    """_handle_conversational returns thanks text for 'thank you'."""
    result = core_agent._handle_conversational("thank you")
    assert "response" in result


# ---------------------------------------------------------------------------
# DynamicSubAgent
# ---------------------------------------------------------------------------


def test_subagent_execute_success(mock_llm):
    """DynamicSubAgent.execute() returns SubAgentResult with success=True."""
    agent = DynamicSubAgent(
        llm_gateway=mock_llm,
        role="Extract contract value",
        evidence=[{"text": "The value is $100K", "source_index": 1, "source_name": "doc.pdf", "score": 0.9}],
        doc_context={"summary": "A contract document"},
    )
    result = agent.execute()
    assert isinstance(result, SubAgentResult)
    assert result.success is True
    assert result.text == "Sub-agent output text"
    assert result.error is None


def test_subagent_execute_failure(mock_llm):
    """DynamicSubAgent.execute() returns SubAgentResult with success=False on error."""
    mock_llm.generate.side_effect = RuntimeError("LLM timeout")
    agent = DynamicSubAgent(
        llm_gateway=mock_llm,
        role="Extract contract value",
        evidence=[],
        doc_context=None,
    )
    result = agent.execute()
    assert isinstance(result, SubAgentResult)
    assert result.success is False
    assert "LLM timeout" in result.error
