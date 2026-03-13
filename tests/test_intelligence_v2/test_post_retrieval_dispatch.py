"""Tests for post-retrieval domain dispatch — agent gets evidence context."""
from unittest.mock import MagicMock, patch, call
import pytest


def _make_core_agent():
    from src.agent.core_agent import CoreAgent
    mongodb = MagicMock()
    mongodb.find.return_value = []
    return CoreAgent(
        llm_gateway=MagicMock(),
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        mongodb=mongodb,
    )


class TestPostRetrievalDispatch:
    """Domain dispatch after RETRIEVE provides evidence to agents."""

    def test_post_retrieval_dispatch_with_evidence(self):
        """If pre-retrieval dispatch returns None but agent_name is set,
        try again after RETRIEVE with evidence."""
        agent = _make_core_agent()

        evidence = [
            {"text": "Resume: 5 years Python", "source_name": "resume.pdf",
             "document_id": "d1", "score": 0.9, "page": 1, "section": "Experience",
             "source_index": 1, "profile_id": "prof_1", "chunk_id": "c1"},
        ]

        domain_response = {
            "response": "Skill gap analysis complete",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {"agent": "hr", "agent_task": "skill_gap_analysis", "agent_handled": True},
        }

        with patch.object(agent._intent_analyzer, "analyze") as mock_analyze, \
             patch.object(agent._domain_dispatcher, "try_handle") as mock_dispatch, \
             patch.object(agent._retriever, "retrieve") as mock_retrieve, \
             patch("src.agent.core_agent.rerank_chunks", return_value=evidence), \
             patch("src.agent.core_agent.build_context", return_value=(evidence, {})):

            mock_analyze.return_value = MagicMock(
                is_conversational=False,
                task_type="extract",
                complexity="simple",
                resolved_query="Analyze skill gaps",
                output_format="prose",
                relevant_documents=[],
                cross_profile=False,
                sub_tasks=None,
                is_complex=False,
            )

            # Pre-retrieval: None, Post-retrieval: response
            mock_dispatch.side_effect = [None, domain_response]

            mock_retrieve.return_value = MagicMock(
                chunks=evidence, profiles_searched=["prof_1"],
            )

            result = agent.handle(
                query="Analyze skill gaps",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                agent_name="hr",
            )

        assert result["response"] == "Skill gap analysis complete"
        assert mock_dispatch.call_count == 2
        # Second call should have evidence
        second_call_kwargs = mock_dispatch.call_args_list[1]
        second_evidence = second_call_kwargs[1].get("evidence", second_call_kwargs[0][3] if len(second_call_kwargs[0]) > 3 else [])
        assert len(second_evidence) > 0

    def test_no_post_retrieval_without_agent_name(self):
        """Without agent_name, post-retrieval dispatch should NOT happen."""
        agent = _make_core_agent()

        with patch.object(agent._intent_analyzer, "analyze") as mock_analyze, \
             patch.object(agent._domain_dispatcher, "try_handle", return_value=None) as mock_dispatch, \
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

            agent.handle(
                query="What is the revenue?",
                subscription_id="sub_1",
                profile_id="prof_1",
                user_id="user_1",
                session_id="sess_1",
                conversation_history=None,
                # No agent_name
            )

        # Only pre-retrieval dispatch, no post-retrieval
        assert mock_dispatch.call_count == 1
