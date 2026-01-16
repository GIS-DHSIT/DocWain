from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import docwain_api
from src.execution.common import ExecutionResult
from src.mode.execution_mode import ExecutionMode


@pytest.fixture
def client():
    return TestClient(docwain_api.app)


def _build_answer(mode: ExecutionMode):
    metadata = {"debug": {"execution_mode": mode.value}}
    sources = [{"id": "s1"}] if mode == ExecutionMode.AGENT else []
    return {
        "response": f"{mode.value} response",
        "sources": sources,
        "grounded": True,
        "context_found": True,
        "metadata": metadata,
    }


def test_default_ask_routes_to_normal(monkeypatch, client):
    from src.execution import router

    calls = {"normal": 0}

    def fake_normal(request, stream=False, debug=False):
        calls["normal"] += 1
        return ExecutionResult(
            answer=_build_answer(ExecutionMode.NORMAL),
            mode=ExecutionMode.NORMAL,
            debug={"execution_mode": ExecutionMode.NORMAL.value},
        )

    def fake_agent(*args, **kwargs):  # pragma: no cover - should never fire here
        raise AssertionError("Agent runner should not be called in default mode")

    monkeypatch.setattr(router, "run_normal_mode", fake_normal)
    monkeypatch.setattr(router, "run_agent_mode", fake_agent)

    response = client.post("/api/ask", json={"query": "hello world"})
    assert response.status_code == 200
    payload = response.json()
    assert calls["normal"] == 1
    assert payload["debug"]["execution_mode"] == "normal"
    assert payload["answer"]["metadata"]["debug"]["execution_mode"] == "normal"


def test_agent_mode_body_toggle(monkeypatch, client):
    from src.execution import router

    calls = {"agent": 0}

    def fake_normal(*args, **kwargs):  # pragma: no cover - not expected
        raise AssertionError("Normal runner should not be called when agent_mode=true")

    def fake_agent(request, stream=False, debug=False):
        calls["agent"] += 1
        return ExecutionResult(
            answer=_build_answer(ExecutionMode.AGENT),
            mode=ExecutionMode.AGENT,
            debug={"execution_mode": ExecutionMode.AGENT.value},
        )

    monkeypatch.setattr(router, "run_normal_mode", fake_normal)
    monkeypatch.setattr(router, "run_agent_mode", fake_agent)

    response = client.post("/api/ask", json={"query": "go deep", "agent_mode": True})
    assert response.status_code == 200
    payload = response.json()
    assert calls["agent"] == 1
    assert payload["debug"]["execution_mode"] == "agent"
    assert payload["answer"]["metadata"]["debug"]["execution_mode"] == "agent"


def test_mode_isolation_per_request(monkeypatch, client):
    from src.execution import router

    calls = {"agent": 0, "normal": 0}

    def fake_normal(request, stream=False, debug=False):
        calls["normal"] += 1
        return ExecutionResult(
            answer=_build_answer(ExecutionMode.NORMAL),
            mode=ExecutionMode.NORMAL,
            debug={"execution_mode": ExecutionMode.NORMAL.value},
        )

    def fake_agent(request, stream=False, debug=False):
        calls["agent"] += 1
        return ExecutionResult(
            answer=_build_answer(ExecutionMode.AGENT),
            mode=ExecutionMode.AGENT,
            debug={"execution_mode": ExecutionMode.AGENT.value},
        )

    monkeypatch.setattr(router, "run_normal_mode", fake_normal)
    monkeypatch.setattr(router, "run_agent_mode", fake_agent)

    session_id = "isolation-session"
    first = client.post("/api/ask", json={"query": "first", "agent_mode": True, "session_id": session_id})
    second = client.post("/api/ask", json={"query": "second", "agent_mode": False, "session_id": session_id})

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["agent"] == 1
    assert calls["normal"] == 1


def test_agent_mode_requires_evidence(monkeypatch):
    from src.agentic import orchestrator
    from src.api import dw_newron

    monkeypatch.setattr(
        dw_newron,
        "answer_question",
        lambda **kwargs: {"response": "speculative", "sources": []},
    )

    fake_request = SimpleNamespace(
        query="q",
        user_id="user",
        profile_id="profile",
        subscription_id="sub",
        model_name="m",
        persona="p",
        session_id="sess",
        new_session=False,
        agent_mode=True,
        debug=True,
    )

    result = orchestrator.run_agent_mode(fake_request, stream=False, debug=True)
    assert result.mode == ExecutionMode.AGENT
    assert result.answer["grounded"] is False
    assert "limitations" in result.answer["metadata"].get("agent", {})


def test_agent_mode_uses_default_agent_model(monkeypatch):
    from src.agentic import orchestrator
    from src.api import dw_newron
    from src.api.config import Config

    captured = {}

    def fake_answer_question(**kwargs):
        captured["model_name"] = kwargs.get("model_name")
        return {
            "response": "agent response",
            "sources": [{"id": "s1"}],
            "grounded": True,
            "context_found": True,
        }

    monkeypatch.setattr(dw_newron, "answer_question", fake_answer_question)

    fake_request = SimpleNamespace(
        query="agent",
        user_id="u",
        profile_id="p",
        subscription_id="s",
        model_name="llama3.2",  # default should be overridden
        persona="doc",
        session_id=None,
        new_session=False,
        agent_mode=True,
        debug=False,
    )

    orchestrator.run_agent_mode(fake_request, stream=False, debug=False)
    assert captured["model_name"] == Config.Execution.AGENT_MODEL_NAME


def test_ask_stream_defaults_to_normal(monkeypatch, client):
    from src.execution import router

    calls = {"normal": 0}

    def fake_normal(request, stream=False, debug=False):
        calls["normal"] += 1
        return ExecutionResult(
            answer=_build_answer(ExecutionMode.NORMAL),
            mode=ExecutionMode.NORMAL,
            debug={"execution_mode": ExecutionMode.NORMAL.value},
            stream=iter(["chunk-one", "chunk-two"]),
        )

    def fake_agent(*args, **kwargs):  # pragma: no cover
        raise AssertionError("Agent runner should not be called by default")

    monkeypatch.setattr(router, "run_normal_mode", fake_normal)
    monkeypatch.setattr(router, "run_agent_mode", fake_agent)

    with client.stream("POST", "/api/askStream", json={"query": "stream default"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert calls["normal"] == 1
    assert "chunk-one" in body


def test_ask_stream_uses_mode_router(monkeypatch, client):
    from src.execution import router

    def fake_normal(*args, **kwargs):  # pragma: no cover
        raise AssertionError("Normal runner should not stream when agent_mode=true")

    def fake_agent(request, stream=False, debug=False):
        return ExecutionResult(
            answer=_build_answer(ExecutionMode.AGENT),
            mode=ExecutionMode.AGENT,
            debug={"execution_mode": ExecutionMode.AGENT.value},
            stream=iter(["[planning]", "[retrieval]", "[final]"]),
        )

    monkeypatch.setattr(router, "run_normal_mode", fake_normal)
    monkeypatch.setattr(router, "run_agent_mode", fake_agent)

    with client.stream("POST", "/api/askStream?agent_mode=true", json={"query": "stream it"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "[planning]" in body
    assert "[final]" in body
