"""Tests for DomainDispatcher — detects domain tasks and delegates to agents."""
from unittest.mock import MagicMock, patch
import pytest

from src.agent.domain_dispatch import DomainDispatcher


class TestDomainDispatcher:
    """DomainDispatcher detection and delegation tests."""

    def test_no_detection_returns_none(self):
        """Non-domain queries should return None (fall through to RAG)."""
        dispatcher = DomainDispatcher(llm_gateway=MagicMock())
        with patch("src.agent.domain_dispatch.detect_agent_task", return_value=None):
            result = dispatcher.try_handle(
                query="What is the revenue for Q3?",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[],
                doc_context={},
            )
        assert result is None

    def test_explicit_agent_name_dispatches(self):
        """Explicit agent_name should dispatch to that domain agent."""
        dispatcher = DomainDispatcher(llm_gateway=MagicMock())

        mock_agent = MagicMock()
        mock_agent.domain = "hr"
        mock_agent.get_capabilities.return_value = ["interview_questions"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Here are interview questions...",
            sources=[],
            structured_data={},
            task_type="interview_questions",
        )

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "hr", "task_type": "interview_questions"}):
            result = dispatcher.try_handle(
                query="Generate interview questions for this candidate",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[{"text": "Resume content..."}],
                doc_context={},
                agent_name="hr",
            )
        assert result is not None
        assert result["response"] == "Here are interview questions..."
        assert result["metadata"]["agent_handled"] is True

    def test_auto_detect_dispatches(self):
        """Auto-detected domain task should dispatch without explicit agent_name."""
        dispatcher = DomainDispatcher(llm_gateway=MagicMock())

        mock_agent = MagicMock()
        mock_agent.domain = "legal"
        mock_agent.get_capabilities.return_value = ["compliance_check"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Compliance analysis...",
            sources=[],
            structured_data={},
            task_type="compliance_check",
        )

        with patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "legal", "task_type": "compliance_check"}), \
             patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent):
            result = dispatcher.try_handle(
                query="Check GDPR compliance of this contract",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[{"text": "Contract text..."}],
                doc_context={},
            )
        assert result is not None
        assert result["metadata"]["agent"] == "legal"

    def test_agent_failure_returns_none(self):
        """If agent execution fails, return None so RAG pipeline handles it."""
        dispatcher = DomainDispatcher(llm_gateway=MagicMock())

        mock_agent = MagicMock()
        mock_agent.domain = "hr"
        mock_agent.get_capabilities.return_value = ["interview_questions"]
        mock_agent.execute.return_value = MagicMock(
            success=False,
            output="",
            sources=[],
            structured_data={},
            task_type="interview_questions",
            error="LLM timeout",
        )

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "hr", "task_type": "interview_questions"}):
            result = dispatcher.try_handle(
                query="Generate interview questions",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[],
                doc_context={},
                agent_name="hr",
            )
        assert result is None

    def test_evidence_passed_as_context(self):
        """Retrieved evidence should be passed to the domain agent as text context."""
        dispatcher = DomainDispatcher(llm_gateway=MagicMock())

        mock_agent = MagicMock()
        mock_agent.domain = "invoice"
        mock_agent.get_capabilities.return_value = ["payment_anomaly_detection"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Anomaly found...",
            sources=[],
            structured_data={},
            task_type="payment_anomaly_detection",
        )

        evidence = [
            {"text": "Invoice line 1: $500", "source_name": "invoice.pdf"},
            {"text": "Invoice line 2: $1000", "source_name": "invoice.pdf"},
        ]

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "invoice", "task_type": "payment_anomaly_detection"}):
            dispatcher.try_handle(
                query="Check for payment anomalies",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=evidence,
                doc_context={},
                agent_name="invoice",
            )

        call_args = mock_agent.execute.call_args
        ctx = call_args[0][1]  # second positional arg is context
        assert "Invoice line 1: $500" in ctx["text"]
        assert "Invoice line 2: $1000" in ctx["text"]

    def test_media_propagated(self):
        """Media from agent structured_data should be propagated to response."""
        dispatcher = DomainDispatcher(llm_gateway=MagicMock())

        mock_agent = MagicMock()
        mock_agent.domain = "analytics_viz"
        mock_agent.get_capabilities.return_value = ["generate_chart"]
        mock_agent.execute.return_value = MagicMock(
            success=True,
            output="Chart generated",
            sources=[],
            structured_data={"media": [{"type": "image", "url": "/tmp/chart.png"}]},
            task_type="generate_chart",
        )

        with patch("src.agent.domain_dispatch.get_domain_agent", return_value=mock_agent), \
             patch("src.agent.domain_dispatch.detect_agent_task", return_value={"domain": "analytics_viz", "task_type": "generate_chart"}):
            result = dispatcher.try_handle(
                query="Generate a bar chart of revenue",
                subscription_id="sub_1",
                profile_id="prof_1",
                evidence=[],
                doc_context={},
                agent_name="analytics_viz",
            )
        assert result is not None
        assert result["media"] == [{"type": "image", "url": "/tmp/chart.png"}]
