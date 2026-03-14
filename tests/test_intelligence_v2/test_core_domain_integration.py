"""Tests for CoreAgent domain dispatch integration."""
from unittest.mock import MagicMock, patch
import pytest


def _make_core_agent(**overrides):
    """Create a CoreAgent with mocked dependencies."""
    from src.agent.core_agent import CoreAgent
    defaults = {
        "llm_gateway": MagicMock(),
        "qdrant_client": MagicMock(),
        "embedder": MagicMock(),
        "mongodb": MagicMock(),
    }
    defaults.update(overrides)
    defaults["mongodb"].find.return_value = []
    return CoreAgent(**defaults)


class TestCoreDomainIntegration:
    """CoreAgent routes domain tasks through DomainDispatcher."""

    def test_domain_task_skips_retrieve_reason(self):
        """When DomainDispatcher handles query, RETRIEVE and REASON are skipped."""
        agent = _make_core_agent()

        domain_response = {
            "response": "Interview questions generated",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {"agent": "hr", "agent_task": "interview_questions", "agent_handled": True},
        }

        with patch.object(agent._intent_analyzer, "analyze") as mock_analyze, \
             patch.object(agent._domain_dispatcher, "try_handle", return_value=domain_response), \
             patch.object(agent._retriever, "retrieve") as mock_retrieve:

            mock_analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="extract",
                complexity="simple",
                resolved_query="Generate interview questions for this candidate",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )

            result = agent.handle(
                query="Generate interview questions for this candidate",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="hr",
            )

        assert result["response"] == "Interview questions generated"
        assert result["metadata"]["agent_handled"] is True
        # RETRIEVE should NOT have been called
        mock_retrieve.assert_not_called()

    def test_non_domain_task_continues_rag(self):
        """When DomainDispatcher returns None, standard RAG pipeline runs."""
        agent = _make_core_agent()

        with patch.object(agent._intent_analyzer, "analyze") as mock_analyze, \
             patch.object(agent._domain_dispatcher, "try_handle", return_value=None), \
             patch.object(agent._retriever, "retrieve") as mock_retrieve, \
             patch.object(agent._reasoner, "reason") as mock_reason, \
             patch("src.agent.core_agent.rerank_chunks", return_value=[]), \
             patch("src.agent.core_agent.build_context", return_value=([], {})):

            mock_analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="lookup",
                complexity="simple",
                resolved_query="What is the revenue?",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )

            mock_retrieve.return_value = MagicMock(
                chunks=[], profiles_searched=["prof_1"],
            )
            mock_reason.return_value = MagicMock(
                text="Revenue is $1M", grounded=True, usage={},
            )

            result = agent.handle(
                query="What is the revenue?",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
            )

        mock_retrieve.assert_called_once()
        mock_reason.assert_called_once()

    def test_agent_name_passed_to_dispatcher(self):
        """agent_name from request should be forwarded to DomainDispatcher."""
        agent = _make_core_agent()

        with patch.object(agent._intent_analyzer, "analyze") as mock_analyze, \
             patch.object(agent._domain_dispatcher, "try_handle", return_value={
                 "response": "Compliant", "sources": [], "grounded": True,
                 "context_found": True, "metadata": {"agent": "legal", "agent_handled": True},
             }) as mock_dispatch:

            mock_analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="extract",
                complexity="simple",
                resolved_query="Check compliance",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )

            agent.handle(
                query="Check compliance",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="legal",
            )

        # Verify agent_name was passed
        call_kwargs = mock_dispatch.call_args
        assert call_kwargs[1]["agent_name"] == "legal"
