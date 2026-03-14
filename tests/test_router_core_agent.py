"""Integration test: verify /api/ask routes through Core Agent."""
import pytest
from unittest.mock import MagicMock, patch


class TestPipelineRedirect:
    @patch("src.execution.router._get_core_agent")
    def test_execute_request_uses_core_agent(self, mock_get_agent):
        from src.execution.router import execute_request
        from src.execution.common import ExecutionResult

        mock_agent = MagicMock()
        mock_agent.handle.return_value = {
            "response": "Test answer",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {"task_type": "lookup"},
        }
        mock_get_agent.return_value = mock_agent

        request = MagicMock()
        request.query = "What is the salary?"
        request.subscription_id = "sub_1"
        request.profile_id = "hr"
        request.user_id = "user@test.com"
        request.session_id = "sess_1"
        request.document_id = None
        request.agent_name = None
        request.debug = False

        ctx = MagicMock()
        ctx.query = "What is the salary?"
        ctx.subscription_id = "sub_1"
        ctx.profile_id = "hr"
        ctx.user_id = "user@test.com"
        ctx.session_id = "sess_1"

        result = execute_request(request, MagicMock(), ctx, stream=False, debug=False)

        assert isinstance(result, ExecutionResult)
        assert result.answer["response"] == "Test answer"
        assert result.answer["grounded"] is True
        mock_agent.handle.assert_called_once()

    @patch("src.execution.router._get_core_agent")
    def test_returns_execution_result_structure(self, mock_get_agent):
        from src.execution.router import execute_request
        from src.mode.execution_mode import ExecutionMode

        mock_agent = MagicMock()
        mock_agent.handle.return_value = {
            "response": "Answer",
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {},
        }
        mock_get_agent.return_value = mock_agent

        request = MagicMock()
        request.query = "test"
        request.subscription_id = "sub_1"
        request.profile_id = "hr"
        request.user_id = "user@test.com"
        request.document_id = None
        request.agent_name = None

        ctx = MagicMock()
        ctx.session_id = None

        result = execute_request(request, MagicMock(), ctx, stream=False, debug=False)

        assert hasattr(result, "answer")
        assert hasattr(result, "mode")
        assert hasattr(result, "debug")
        assert result.mode == ExecutionMode.AGENT
        assert result.stream is None

    @patch("src.execution.router._get_core_agent")
    def test_debug_mode_includes_metadata(self, mock_get_agent):
        from src.execution.router import execute_request

        mock_agent = MagicMock()
        mock_agent.handle.return_value = {
            "response": "Answer",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {"task_type": "extract", "timing": {"total_ms": 500}},
        }
        mock_get_agent.return_value = mock_agent

        request = MagicMock()
        request.query = "test"
        request.subscription_id = "sub_1"
        request.profile_id = "hr"
        request.user_id = "user@test.com"
        request.document_id = None
        request.agent_name = None

        ctx = MagicMock()
        ctx.session_id = None

        result = execute_request(request, MagicMock(), ctx, stream=False, debug=True)

        assert result.debug.get("task_type") == "extract"
