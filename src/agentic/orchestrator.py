from __future__ import annotations

import concurrent.futures
from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, Iterable, List, Optional

from src.api.config import Config
from src.api.query_intelligence import QueryIntelligence
from src.agentic.budgets import compute_agent_budgets
from src.execution.common import ExecutionResult, chunk_text_stream
from src.metrics.ai_metrics import get_metrics_store
from src.mode.execution_mode import ExecutionMode
from src.runtime.chain_factory import build_chain
from src.runtime.freshness_guard import FreshnessGuard
from src.chat.companion_classifier import CompanionClassifier
from src.api.dw_newron import get_redis_client

logger = get_logger(__name__)

def _get_thinking_client() -> Optional[Any]:
    """Create an OllamaClient pointed at lfm2.5-thinking for MoE reasoning steps."""
    try:
        cfg = getattr(Config, "ThinkingModel", None)
        if cfg is None or not getattr(cfg, "ENABLED", True):
            return None
        from src.llm.clients import OllamaClient
        model = getattr(cfg, "MODEL", "lfm2.5-thinking:latest")
        return OllamaClient(model_name=model)
    except Exception as exc:
        logger.debug("Thinking client unavailable: %s", exc)
        return None

def _execute_agents_parallel(
    agents_and_contexts: List[tuple],
    timeout: float = 30.0,
    max_workers: int = 3,
) -> List[Any]:
    """Execute multiple domain agents in parallel using ThreadPoolExecutor.

    Args:
        agents_and_contexts: list of (agent, task_type, context) tuples
        timeout: per-agent timeout in seconds
        max_workers: max concurrent agents

    Returns:
        List of AgentTaskResult (or None for failures).
    """
    results = [None] * len(agents_and_contexts)

    def _run(idx: int, agent, task_type: str, context: Dict[str, Any]):
        try:
            return idx, agent.execute(task_type, context)
        except Exception as exc:
            logger.warning("Parallel agent %s/%s failed: %s", agent.domain, task_type, exc)
            return idx, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(agents_and_contexts))) as pool:
        futures = []
        for i, (agent, task_type, context) in enumerate(agents_and_contexts):
            futures.append(pool.submit(_run, i, agent, task_type, context))

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                idx, result = future.result(timeout=timeout)
                results[idx] = result
            except Exception as exc:
                logger.debug("Parallel agent future failed: %s", exc)

    return results

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
    base_steps = int(getattr(Config.Execution, "MAX_AGENT_STEPS", 10))
    base_evidence = int(getattr(Config.Execution, "MAX_AGENT_EVIDENCE", 20))
    max_steps = base_steps
    max_evidence = base_evidence

    query_text = getattr(request, "query", None) or getattr(request, "question", None) or ""
    try:
        analysis = QueryIntelligence().analyze(str(query_text))
        budgets = compute_agent_budgets(str(query_text), analysis.__dict__, base_steps=base_steps, base_evidence=base_evidence)
        max_steps = budgets.get("max_steps", base_steps)
        max_evidence = budgets.get("max_evidence", base_evidence)
    except Exception:
        pass

    companion_payload = None
    try:
        companion_classifier = CompanionClassifier(get_redis_client())
        companion_payload = companion_classifier.classify(
            user_query=str(query_text),
            last_turns_summary="",
            intent_from_query_intelligence=getattr(analysis, "intent", "") if "analysis" in locals() else "",
            session_id=getattr(request, "session_id", None),
        ).as_dict()
    except Exception:
        companion_payload = None

    # --- Domain agent detection ----------------------------------------
    # Check if the query requires a specialized domain agent (e.g. "generate
    # interview questions") before the standard retrieval pipeline.
    # MoE: reasoning agents use lfm2.5-thinking, generation agents use DocWain-Agent.
    try:
        from src.agentic.domain_agents import detect_agent_task, get_domain_agent
        _agent_det = detect_agent_task(str(query_text))
        if _agent_det:
            logger.info("Agent mode: detected domain agent task: %s", _agent_det)
            _thinking_client = _get_thinking_client()
            _domain_agent = get_domain_agent(_agent_det["domain"], thinking_client=_thinking_client)
            if _domain_agent:
                # Run standard retrieval to get document context for the agent
                chain = build_chain(ctx, profile_filters=getattr(request, "profile_filters", None), mode=ExecutionMode.AGENT)
                _rag_answer, _rag_evidence_ids = chain.run(stream=False, debug=debug)
                # Build context from retrieved sources
                _agent_context: Dict[str, Any] = {"query": str(query_text)}
                _rag_sources = _rag_answer.get("sources", [])
                if _rag_sources:
                    _source_texts = [s.get("text", "") for s in _rag_sources if s.get("text")]
                    _agent_context["text"] = "\n\n".join(_source_texts[:10])
                elif _rag_answer.get("response"):
                    _agent_context["text"] = _rag_answer["response"]

                _agent_result = _domain_agent.execute(_agent_det["task_type"], _agent_context)
                if _agent_result.success and _agent_result.output:
                    answer = _rag_answer
                    answer["response"] = _agent_result.output
                    answer.setdefault("metadata", {})
                    answer["metadata"]["agent"] = {
                        "domain_agent": _agent_det["domain"],
                        "task_type": _agent_det["task_type"],
                        "agent_handled": True,
                        "max_steps": max_steps,
                        "max_evidence": max_evidence,
                    }
                    trace.append({"phase": "domain_agent", "detail": f"{_agent_det['domain']}/{_agent_det['task_type']}", "timestamp": time.time()})
                    trace.append({"phase": "compose", "detail": "Agent response assembled", "timestamp": time.time()})
                    debug_info = {
                        "execution_mode": ExecutionMode.AGENT.value,
                        "trace": trace[:max_steps],
                        "evidence_items": len(answer.get("sources") or []),
                        "model_name": model_name if "model_name" in locals() else "",
                        "request_id": ctx.request_id,
                        "companion": companion_payload,
                    }
                    answer["metadata"]["debug"] = debug_info
                    stream_iterable = _build_stream(trace, answer.get("response", "")) if stream else None
                    return ExecutionResult(answer=answer, mode=ExecutionMode.AGENT, debug=debug_info, stream=stream_iterable)
    except Exception:
        logger.debug("Domain agent detection skipped", exc_info=True)

    # --- Auto-tool selection -------------------------------------------
    auto_tools: List[str] = []
    try:
        if getattr(Config.Execution, "AGENT_AUTO_TOOLS", True):
            from src.agentic.tool_selector import ToolSelector
            selector = ToolSelector()
            auto_tools = selector.select_tools(
                query=str(query_text),
                intent_parse=analysis if "analysis" in locals() else None,
            )
            if auto_tools:
                ctx.with_tools(auto_tools)
                logger.info("Agent auto-selected tools: %s", ctx.tools)
    except Exception:
        logger.debug("Auto-tool selection skipped", exc_info=True)

    trace: List[Dict[str, Any]] = _bounded_trace(max_steps)
    trace.append({"phase": "planning", "detail": "Decomposing request and goals", "timestamp": time.time()})
    if auto_tools:
        trace.append({"phase": "tool_selection", "detail": f"Auto-selected tools: {auto_tools}", "timestamp": time.time()})
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
        # Only refuse if the response itself is empty — don't override a
        # valid LLM response just because source metadata is missing.
        if not (answer.get("response") or "").strip():
            answer["response"] = (
                "No grounded answer available. Unable to proceed without evidence."
            )
        answer["grounded"] = False
        answer["metadata"]["agent"] = {
            "limitations": "No supporting evidence found in source metadata.",
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
        "companion": companion_payload,
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
