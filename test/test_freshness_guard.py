import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.mode.execution_mode import ExecutionMode
try:
    from src.mode.session_state import SessionState
except ImportError:
    from src.mode import SessionState
from src.runtime.chain_factory import build_chain
from src.runtime.freshness_guard import FreshnessGuard
from src.runtime.request_context import RequestContext


def _ctx(query: str, session: str = "s1", user: str = "user@example.com", mode: str = "normal") -> RequestContext:
    return RequestContext.build(
        query=query,
        session_id=session,
        user_id=user,
        mode=mode,
        debug=False,
        profile_id="p1",
        subscription_id="sub",
        model_name="test-model",
    )


def test_freshness_guard_prevents_stale_reuse():
    session_state = SessionState()
    guard = FreshnessGuard(session_state)
    ctx_a = _ctx("What is X?")
    first_answer = {"response": "Answer about X", "sources": [{"document_id": "doc-1"}], "metadata": {}}
    guard.enforce(ctx_a, first_answer, ["doc-1"], regenerate=lambda: (first_answer, ["doc-1"]))
    first_fp = session_state.last_request_fingerprint

    ctx_b = _ctx("What is Y?")
    stale_answer = {"response": "Answer about X", "sources": [{"document_id": "doc-2"}], "metadata": {}}
    regenerated = {"response": "Fresh answer about Y", "sources": [{"document_id": "doc-2"}], "metadata": {}}
    called = False

    def regen():
        nonlocal called
        called = True
        return regenerated, ["doc-2"]

    final_answer, evidence = guard.enforce(ctx_b, stale_answer, ["doc-2"], regenerate=regen)

    assert called, "Freshness guard must trigger regeneration on stale reuse"
    assert final_answer["response"] != stale_answer["response"]
    assert final_answer["metadata"]["freshness_guard"]["regenerated"] is True
    assert session_state.last_request_fingerprint != first_fp
    assert evidence == ["doc-2"]


def test_build_chain_is_request_scoped():
    ctx_a = _ctx("How to configure A?", session="s1")
    ctx_b = _ctx("How to configure B?", session="s2")
    chain_a = build_chain(ctx_a, mode=ExecutionMode.NORMAL)
    chain_b = build_chain(ctx_b, mode=ExecutionMode.NORMAL)

    assert chain_a is not chain_b
    assert chain_a.ctx.request_id != chain_b.ctx.request_id
    assert chain_a.ctx.query != chain_b.ctx.query


def test_parallel_requests_keep_unique_request_ids():
    session_state = SessionState()
    guard = FreshnessGuard(session_state)
    ctxs = [_ctx(f"Question {i}", session="shared") for i in range(10)]
    lock = threading.Lock()

    def worker(ctx: RequestContext):
        answer = {"response": f"Response {ctx.query}", "sources": [{"document_id": ctx.query}], "metadata": {}}
        final_answer, _ = guard.enforce(ctx, answer, [ctx.query], regenerate=lambda: (answer, [ctx.query]))
        with lock:
            return final_answer["metadata"]["request_id"]

    with ThreadPoolExecutor(max_workers=5) as pool:
        request_ids = list(pool.map(worker, ctxs))

    assert len(set(request_ids)) == len(ctxs), "Each parallel request must carry its own request_id"


def test_freshness_guard_returns_insufficient_when_regen_repeats():
    session_state = SessionState()
    guard = FreshnessGuard(session_state)
    ctx_a = _ctx("Find alpha")
    first = {"response": "Alpha", "sources": [{"document_id": "doc-alpha"}], "metadata": {}}
    guard.enforce(ctx_a, first, ["doc-alpha"], regenerate=lambda: (first, ["doc-alpha"]))

    ctx_b = _ctx("Find beta")
    stale = {"response": "Alpha", "sources": [{"document_id": "doc-beta"}], "metadata": {}}

    final_answer, _ = guard.enforce(
        ctx_b,
        stale,
        ["doc-beta"],
        regenerate=lambda: ({"response": "Alpha", "sources": [{"document_id": "doc-beta"}], "metadata": {}}, ["doc-beta"]),
    )

    assert "Not found in the selected documents" in final_answer["response"]
    assert final_answer["metadata"]["freshness_guard"]["regenerated"] is True
