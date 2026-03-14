"""End-to-end test: domain agent query flows through CoreAgent pipeline."""
from unittest.mock import MagicMock, patch
import pytest

from src.agent.core_agent import CoreAgent
from src.agentic.domain_agents import AgentTaskResult


def _build_core_agent(llm_generate_returns=None):
    """Build CoreAgent with mocked dependencies."""
    mock_llm = MagicMock()
    if llm_generate_returns:
        mock_llm.generate.side_effect = llm_generate_returns
    else:
        mock_llm.generate.return_value = '{"task_type": "extract", "complexity": "simple", "resolved_query": "test", "output_format": "prose", "relevant_documents": [], "cross_profile": false}'

    mock_mongodb = MagicMock()
    mock_mongodb.find.return_value = []

    return CoreAgent(
        llm_gateway=mock_llm,
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        mongodb=mock_mongodb,
    )


class TestE2EDomainRouting:
    """Domain agent queries through unified CoreAgent pipeline."""

    def test_explicit_agent_name_routes_to_domain_agent(self):
        """agent_name='hr' should route through DomainDispatcher to HR agent."""
        mock_agent_result = AgentTaskResult(
            task_type="interview_questions",
            success=True,
            output="Here are 5 interview questions:\n1. Tell me about your Python experience...",
            sources=[],
            structured_data={},
        )

        mock_agent = MagicMock()
        mock_agent.domain = "hr"
        mock_agent.get_capabilities.return_value = ["interview_questions"]
        mock_agent.execute.return_value = mock_agent_result

        core = _build_core_agent()

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "hr", "task_type": "interview_questions"}):

            result = core.handle(
                query="Generate interview questions for this candidate",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="hr",
            )

        assert "interview" in result["response"].lower()
        assert result["metadata"]["agent_handled"] is True
        assert result["metadata"]["agent"] == "hr"

    def test_auto_detect_domain_no_agent_name(self):
        """Without agent_name, auto-detection routes to domain agent."""
        mock_agent_result = AgentTaskResult(
            task_type="compliance_check",
            success=True,
            output="GDPR compliance analysis complete.",
            sources=[],
            structured_data={},
        )

        mock_agent = MagicMock()
        mock_agent.domain = "legal"
        mock_agent.get_capabilities.return_value = ["compliance_check"]
        mock_agent.execute.return_value = mock_agent_result

        core = _build_core_agent()

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "legal", "task_type": "compliance_check"}):

            result = core.handle(
                query="Check GDPR compliance of this contract",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
            )

        assert result["metadata"]["agent_handled"] is True
        assert result["metadata"]["agent"] == "legal"

    def test_non_domain_falls_through_to_rag(self):
        """A non-domain query bypasses dispatch and uses RAG."""
        core = _build_core_agent()

        with patch("src.agent.domain_dispatch.detect_agent_task", return_value=None), \
             patch.object(core._retriever, "retrieve") as mock_retrieve, \
             patch.object(core._reasoner, "reason") as mock_reason, \
             patch("src.agent.core_agent.rerank_chunks", return_value=[]), \
             patch("src.agent.core_agent.build_context", return_value=([], {})):

            mock_retrieve.return_value = MagicMock(
                chunks=[], profiles_searched=["prof_1"],
            )
            mock_reason.return_value = MagicMock(
                text="The Q3 revenue was $2.5M.", grounded=True, usage={},
            )

            result = core.handle(
                query="What is Q3 revenue?",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
            )

        assert result is not None
        assert result.get("metadata", {}).get("agent_handled") is not True
        mock_retrieve.assert_called_once()
        mock_reason.assert_called_once()

    def test_domain_agent_failure_falls_through(self):
        """If domain agent fails, query falls through to RAG pipeline."""
        mock_agent = MagicMock()
        mock_agent.domain = "hr"
        mock_agent.get_capabilities.return_value = ["interview_questions"]
        mock_agent.execute.return_value = AgentTaskResult(
            task_type="interview_questions",
            success=False,
            output="",
            error="LLM timeout",
        )

        core = _build_core_agent()

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "hr", "task_type": "interview_questions"}), \
             patch.object(core._retriever, "retrieve") as mock_retrieve, \
             patch.object(core._reasoner, "reason") as mock_reason, \
             patch("src.agent.core_agent.rerank_chunks", return_value=[]), \
             patch("src.agent.core_agent.build_context", return_value=([], {})):

            mock_retrieve.return_value = MagicMock(
                chunks=[], profiles_searched=["prof_1"],
            )
            mock_reason.return_value = MagicMock(
                text="Based on the documents...", grounded=True, usage={},
            )

            result = core.handle(
                query="Generate interview questions",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="hr",
            )

        # Should have fallen through to RAG
        mock_retrieve.assert_called_once()
