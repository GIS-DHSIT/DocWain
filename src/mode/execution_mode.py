from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from src.api.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Intents that benefit from multi-step agent reasoning
_AGENT_INTENTS = {"comparison", "summary", "reasoning", "procedural", "deep_analysis"}


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


def _infer_mode_from_query(request: Any) -> Optional[ExecutionMode]:
    """Use QueryIntelligence to auto-select agent mode for complex queries."""
    query = getattr(request, "query", None) or getattr(request, "question", None)
    if not query:
        return None
    try:
        from src.api.query_intelligence import QueryIntelligence
        analysis = QueryIntelligence().analyze(str(query))
        intent = getattr(analysis, "intent", "") or ""
        sub_queries = getattr(analysis, "sub_queries", None) or []

        if intent.lower() in _AGENT_INTENTS:
            logger.debug("Auto-selecting AGENT mode for intent=%s", intent)
            return ExecutionMode.AGENT

        if len(sub_queries) > 1:
            logger.debug("Auto-selecting AGENT mode: %d sub-queries detected", len(sub_queries))
            return ExecutionMode.AGENT
    except Exception:
        pass
    return None


def resolve_execution_mode(request: Any, session_state: Any) -> ExecutionMode:
    """
    Resolve execution mode based on explicit request flags, query analysis,
    session defaults, and configuration. Priority:
      1) request.agent_mode (body)
      2) request.agent_mode_query (query param)
      3) Query intent analysis (auto-select agent for complex queries)
      4) session_state.preferred_execution_mode
      5) Config.Execution.DEFAULT_AGENT_MODE (defaults to NORMAL)
    """
    if not getattr(Config.Execution, "ALLOW_AGENT_MODE", True):
        return ExecutionMode.NORMAL

    explicit_agent_mode = _coerce_bool(getattr(request, "agent_mode", None))
    if explicit_agent_mode is not None:
        return ExecutionMode.AGENT if explicit_agent_mode else ExecutionMode.NORMAL

    query_agent_mode = _coerce_bool(getattr(request, "agent_mode_query", None))
    if query_agent_mode is not None:
        return ExecutionMode.AGENT if query_agent_mode else ExecutionMode.NORMAL

    # Auto-select agent mode for complex queries (comparison, summary, reasoning)
    inferred = _infer_mode_from_query(request)
    if inferred is not None:
        return inferred

    preferred_mode = getattr(session_state, "preferred_execution_mode", None)
    if preferred_mode:
        return preferred_mode

    default_is_agent = bool(getattr(Config.Execution, "DEFAULT_AGENT_MODE", False))
    return ExecutionMode.AGENT if default_is_agent else ExecutionMode.NORMAL
