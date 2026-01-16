from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from src.api.config import Config


class ExecutionMode(str, Enum):
    NORMAL = "normal"
    AGENT = "agent"


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def resolve_execution_mode(request: Any, session_state: Any) -> ExecutionMode:
    """
    Resolve execution mode based on explicit request flags, query params, session defaults,
    and configuration. Priority:
      1) request.agent_mode (body)
      2) request.agent_mode_query (query param)
      3) session_state.preferred_execution_mode
      4) Config.Execution.DEFAULT_AGENT_MODE (defaults to NORMAL)
    """
    if not getattr(Config.Execution, "ALLOW_AGENT_MODE", True):
        return ExecutionMode.NORMAL

    explicit_agent_mode = _coerce_bool(getattr(request, "agent_mode", None))
    if explicit_agent_mode is not None:
        return ExecutionMode.AGENT if explicit_agent_mode else ExecutionMode.NORMAL

    query_agent_mode = _coerce_bool(getattr(request, "agent_mode_query", None))
    if query_agent_mode is not None:
        return ExecutionMode.AGENT if query_agent_mode else ExecutionMode.NORMAL

    preferred_mode = getattr(session_state, "preferred_execution_mode", None)
    if preferred_mode:
        return preferred_mode

    default_is_agent = bool(getattr(Config.Execution, "DEFAULT_AGENT_MODE", False))
    return ExecutionMode.AGENT if default_is_agent else ExecutionMode.NORMAL
