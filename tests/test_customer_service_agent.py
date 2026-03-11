"""Tests for the Customer Service Agent.

Tests task detection, agent execution, and NLU registration for all 5 task types:
resolve_issue, troubleshoot, escalation_assessment, generate_response, faq_search.
"""
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Agent registration and capabilities
# ---------------------------------------------------------------------------

class TestCustomerServiceRegistration:
    def test_agent_exists_in_registry(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("customer_service")
        assert agent is not None
        assert agent.domain == "customer_service"

    def test_support_alias(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("support")
        assert agent is not None
        assert agent.domain == "customer_service"

    def test_helpdesk_alias(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("helpdesk")
        assert agent is not None
        assert agent.domain == "customer_service"

    def test_has_five_capabilities(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("customer_service")
        caps = agent.get_capabilities()
        assert len(caps) == 5
        assert "resolve_issue" in caps
        assert "troubleshoot" in caps
        assert "escalation_assessment" in caps
        assert "generate_response" in caps
        assert "faq_search" in caps

    def test_can_handle_valid_tasks(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("customer_service")
        for task in ["resolve_issue", "troubleshoot", "escalation_assessment",
                     "generate_response", "faq_search"]:
            assert agent.can_handle(task), f"Should handle {task}"

    def test_cannot_handle_invalid_task(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("customer_service")
        assert not agent.can_handle("nonexistent_task")

    def test_uses_thinking_model(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("customer_service")
        assert agent.use_thinking_model is True


# ---------------------------------------------------------------------------
# NLU task detection
# ---------------------------------------------------------------------------

class TestCustomerServiceDetection:
    def test_resolve_issue_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("resolve my customer support request about billing")
        assert r is not None
        assert r["domain"] == "customer_service"
        assert r["task_type"] == "resolve_issue"

    def test_troubleshoot_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("troubleshoot this problem step by step and diagnose the issue")
        assert r is not None
        assert r["domain"] == "customer_service"
        assert r["task_type"] == "troubleshoot"

    def test_escalation_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("assess whether this customer issue needs escalation to a manager")
        assert r is not None
        assert r["domain"] == "customer_service"
        assert r["task_type"] == "escalation_assessment"

    def test_generate_response_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("generate a professional customer support reply to this complaint")
        assert r is not None
        assert r["domain"] == "customer_service"
        assert r["task_type"] == "generate_response"

    def test_faq_search_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("search frequently asked questions for password reset help articles")
        assert r is not None
        assert r["domain"] == "customer_service"
        assert r["task_type"] == "faq_search"


# ---------------------------------------------------------------------------
# Agent execution (with mocked LLM)
# ---------------------------------------------------------------------------

class TestCustomerServiceExecution:
    def _mock_agent(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("customer_service")
        agent._llm = MagicMock()
        agent._llm.generate_with_metadata = MagicMock(
            return_value=("Thank you for reaching out. Based on our policy, your claim is covered under Section 3.", {})
        )
        return agent

    def test_resolve_issue(self):
        agent = self._mock_agent()
        result = agent.execute("resolve_issue", {
            "text": "Coverage includes dental and vision. Exclusions: cosmetic procedures.",
            "query": "Is my dental cleaning covered?",
        })
        assert result.success
        assert result.output
        assert result.task_type == "resolve_issue"

    def test_troubleshoot(self):
        agent = self._mock_agent()
        result = agent.execute("troubleshoot", {
            "text": "Upload guide: 1. Click Upload. 2. Select file. 3. Wait for processing.",
            "query": "My document upload keeps failing",
        })
        assert result.success
        assert result.output
        assert result.task_type == "troubleshoot"

    def test_escalation_assessment(self):
        agent = self._mock_agent()
        result = agent.execute("escalation_assessment", {
            "text": "SLA: Critical issues resolved within 4 hours.",
            "query": "Customer reports data breach affecting 1000 users",
        })
        assert result.success
        assert result.output
        assert result.task_type == "escalation_assessment"
        assert isinstance(result.structured_data, dict)

    def test_generate_response(self):
        agent = self._mock_agent()
        result = agent.execute("generate_response", {
            "text": "Return policy: 30-day refund window for unused items.",
            "query": "Customer wants to return a used product after 45 days",
        })
        assert result.success
        assert result.output
        assert result.task_type == "generate_response"

    def test_faq_search(self):
        agent = self._mock_agent()
        result = agent.execute("faq_search", {
            "text": "Q: How to reset password? A: Go to Settings > Security > Reset.",
            "query": "How do I change my password?",
        })
        assert result.success
        assert result.output
        assert result.task_type == "faq_search"

    def test_unknown_task_returns_error(self):
        agent = self._mock_agent()
        result = agent.execute("nonexistent_task", {"text": "test"})
        assert not result.success
        assert result.error


# ---------------------------------------------------------------------------
# NLU agent matcher integration
# ---------------------------------------------------------------------------

class TestCustomerServiceNLUMatcher:
    def test_customer_service_registered_in_nlu(self):
        from src.agentic.nlu_agent_matcher import _ensure_registry
        from src.nlp.nlu_engine import get_registry
        _ensure_registry()
        reg = get_registry("agent")
        assert "customer_service" in reg.entries
