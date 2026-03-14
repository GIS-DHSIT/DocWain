from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any

from src.execution.common import ExecutionResult, chunk_text_stream, chunk_text_stream_with_metadata
from src.metrics.ai_metrics import get_metrics_store
from src.mode.execution_mode import ExecutionMode
from src.runtime.chain_factory import build_chain
from src.runtime.freshness_guard import FreshnessGuard

logger = get_logger(__name__)

def _ensure_debug_metadata(answer: dict, debug_info: dict) -> dict:
    metadata = answer.get("metadata") or {}
    metadata["debug"] = {**debug_info, **metadata.get("debug", {})}
    answer["metadata"] = metadata
    return answer

def run_normal_mode(
    request: Any,
    ctx: Any,
    session_state: Any,
    *,
    stream: bool = False,
    debug: bool = False,
) -> ExecutionResult:
    """
    Fast, deterministic execution path. Avoids planners and tools to keep latency low.
    """
    chain = build_chain(ctx, profile_filters=getattr(request, "profile_filters", None), mode=ExecutionMode.NORMAL)
    answer, evidence_ids = chain.run(stream=False, debug=debug)

    guard = FreshnessGuard(session_state)
    answer, evidence_ids = guard.enforce(
        ctx=ctx,
        answer=answer,
        evidence_ids=evidence_ids,
        regenerate=lambda: chain.run(stream=False, debug=debug, force_refresh=True),
    )

    debug_info = {
        "execution_mode": ExecutionMode.NORMAL.value,
        "planner_used": False,
        "tools_used": [],
        "cache": "disabled",
        "request_id": ctx.request_id,
        "evidence_ids": evidence_ids,
    }
    _ensure_debug_metadata(answer, debug_info)
    metrics_store = get_metrics_store()
    if metrics_store.available:
        metrics_store.record(
            distributions={"agent_execution": {ExecutionMode.NORMAL.value: 1}},
            agent=ExecutionMode.NORMAL.value,
            model_id=ctx.model_name,
        )

    stream_iterable = None
    if stream:
        stream_iterable = chunk_text_stream_with_metadata(
            answer.get("response") or "",
            metadata=answer,
        )

    if debug:
        logger.debug("Normal mode response metadata: %s", answer.get("metadata"))

    return ExecutionResult(
        answer=answer,
        mode=ExecutionMode.NORMAL,
        debug=debug_info,
        stream=stream_iterable,
    )
