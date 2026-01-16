from __future__ import annotations

from typing import Any

from src.agentic.orchestrator import run_agent_mode
from src.execution.normal_runner import run_normal_mode
from src.mode.execution_mode import ExecutionMode, resolve_execution_mode


def execute_request(request: Any, session_state: Any, ctx: Any, *, stream: bool = False, debug: bool = False):
    """
    Single entry point for /ask and /askStream. Dispatches to the appropriate runner.
    """
    mode = resolve_execution_mode(request, session_state)
    if mode == ExecutionMode.NORMAL:
        return run_normal_mode(request, ctx=ctx, session_state=session_state, stream=stream, debug=debug)
    return run_agent_mode(request, ctx=ctx, session_state=session_state, stream=stream, debug=debug)
