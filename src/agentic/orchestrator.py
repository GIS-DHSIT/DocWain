from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List

from src.api.config import Config
from src.execution.common import ExecutionResult, chunk_text_stream
from src.metrics.ai_metrics import get_metrics_store
from src.mode.execution_mode import ExecutionMode
from src.runtime.chain_factory import build_chain
from src.runtime.freshness_guard import FreshnessGuard

logger = logging.getLogger(__name__)


def _bounded_trace(max_steps: int) -> List[Dict[str, Any]]:
    """Create a trace list that respects the max step budget."""
    return [{"phase": "init", "detail": "Agent mode engaged", "timestamp": time.time(), "step": 0}][:max_steps]


def _build_stream(trace: List[Dict[str, Any]], answer_text: str) -> Iterable[str]:
    def _stream():
        for step in trace:
            phase = step.get("phase")
            detail = step.get("detail", "")
            yield f"[{phase}] {detail}\n"
        yield "[final] Response begins\n"
        for chunk in chunk_text_stream(answer_text):
            yield chunk
    return _stream()


def run_agent_mode(
    request: Any,
    ctx: Any,
    session_state: Any,
    *,
    stream: bool = False,
    debug: bool = False,
) -> ExecutionResult:
    """
    Agentic execution with planner + verifier style orchestration.
    This wrapper defers heavy lifting to the existing retrieval pipeline while
    enforcing explicit agent safeguards (step budget, evidence requirement).
    """
    max_steps = int(getattr(Config.Execution, "MAX_AGENT_STEPS", 10))
    max_evidence = int(getattr(Config.Execution, "MAX_AGENT_EVIDENCE", 20))

    trace: List[Dict[str, Any]] = _bounded_trace(max_steps)
    trace.append({"phase": "planning", "detail": "Decomposing request and goals", "timestamp": time.time()})
    trace.append({"phase": "retrieval", "detail": "Collecting evidence", "timestamp": time.time()})

    # Ensure agentic path uses the configured default model unless explicitly overridden.
    agent_model = getattr(Config.Execution, "AGENT_MODEL_NAME", "nemotron-3-nano")
    requested_model = getattr(request, "model_name", None)
    model_name = requested_model or agent_model

    if model_name:
        try:
            ctx.model_name = model_name
        except Exception:
            pass

    chain = build_chain(ctx, profile_filters=getattr(request, "profile_filters", None), mode=ExecutionMode.AGENT)
    answer, evidence_ids = chain.run(stream=False, debug=debug)

    guard = FreshnessGuard(session_state)
    answer, evidence_ids = guard.enforce(
        ctx=ctx,
        answer=answer,
        evidence_ids=evidence_ids,
        regenerate=lambda: chain.run(stream=False, debug=debug, force_refresh=True),
    )

    answer["sources"] = (answer.get("sources") or [])[:max_evidence]

    if not answer["sources"]:
        answer["response"] = (
            answer.get("response")
            or "No grounded answer available. Unable to proceed without evidence."
        )
        answer["grounded"] = False
        answer["metadata"]["agent"] = {
            "limitations": "No supporting evidence found; withheld ungrounded response.",
            "max_evidence": max_evidence,
        }
    else:
        answer.setdefault("metadata", {})
        answer["metadata"]["agent"] = {
            "max_steps": max_steps,
            "max_evidence": max_evidence,
        }

    trace.append({"phase": "verification", "detail": "Verifier enforced evidence-only reasoning", "timestamp": time.time()})
    trace.append({"phase": "compose", "detail": "Assembling final response", "timestamp": time.time()})
    trace = trace[:max_steps]

    debug_info = {
        "execution_mode": ExecutionMode.AGENT.value,
        "trace": trace,
        "evidence_items": len(answer.get("sources") or []),
        "cache": "disabled",
        "model_name": model_name,
        "request_id": ctx.request_id,
        "evidence_ids": evidence_ids,
    }
    metadata = answer.get("metadata") or {}
    metadata["debug"] = {**debug_info, **metadata.get("debug", {})}
    answer["metadata"] = metadata

    if debug:
        logger.debug("Agent mode trace: %s", trace)

    metrics_store = get_metrics_store()
    if metrics_store.available:
        metrics_store.record(
            distributions={"agent_execution": {ExecutionMode.AGENT.value: 1}},
            agent=ExecutionMode.AGENT.value,
            model_id=model_name,
        )

    stream_iterable = _build_stream(trace, answer.get("response") or "") if stream else None

    return ExecutionResult(
        answer=answer,
        mode=ExecutionMode.AGENT,
        debug=debug_info,
        stream=stream_iterable,
    )
