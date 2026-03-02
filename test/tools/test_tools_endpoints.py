from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from src import main as docwain_api


@pytest.fixture
def client():
    return TestClient(docwain_api.app)


def test_translator_endpoint(client: TestClient):
    response = client.post(
        "/api/tools/translator/translate",
        json={"text": "hello", "target_lang": "fr"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tool_name"] == "translator"
    assert data["status"] == "success"
    assert data["result"]["translated_text"]


def test_tool_run_endpoint(client: TestClient):
    response = client.post(
        "/api/tools/run",
        json={"tool_name": "translator", "input": {"text": "hola", "target_lang": "en"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["tool_name"] == "translator"


def test_tts_streaming_endpoint(client: TestClient):
    response = client.post("/api/tools/tts/speak", json={"text": "test audio"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")


def test_web_analyze_text_only(client: TestClient):
    response = client.post("/api/tools/web/analyze", json={"text": "Sample content for analysis."})
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["summary"]


def test_jira_endpoints_flat_paths(client: TestClient):
    response = client.post("/api/tools/jira/issues", json={"issues": [{"id": "JIRA-1", "title": "Issue"}]})
    assert response.status_code == 200
    body = response.json()
    assert body["tool_name"] == "jira_confluence"
    assert body["result"]["issues"]["count"] == 1


def test_smoke_ask_endpoint(monkeypatch, client: TestClient):
    class DummyMode:
        value = "normal"

    class DummyResult:
        def __init__(self):
            self.answer = {
                "response": "ok",
                "sources": [],
                "grounded": True,
                "context_found": True,
                "metadata": {},
            }
            self.mode = DummyMode()
            self.stream = None
            self.debug = {}

    def fake_execute(request: Any, session_state: Any, ctx: Any, stream: bool = False, debug: bool = False):
        return DummyResult()

    monkeypatch.setattr(docwain_api, "execute_request", fake_execute)
    response = client.post(
        "/api/ask",
        json={"query": "hello", "profile_id": "p1", "user_id": "u1", "subscription_id": "s1"},
    )
    assert response.status_code == 200
    assert response.json()["answer"]["response"] == "ok"


def test_smoke_ask_persists_chat_turn(monkeypatch, client: TestClient):
    class DummyMode:
        value = "normal"

    class DummyResult:
        def __init__(self):
            self.answer = {
                "response": "persist me",
                "sources": [],
                "grounded": True,
                "context_found": True,
                "metadata": {},
            }
            self.mode = DummyMode()
            self.stream = None
            self.debug = {}

    def fake_execute(request: Any, session_state: Any, ctx: Any, stream: bool = False, debug: bool = False):
        return DummyResult()

    captured = {}

    def fake_add_message_to_history(user_id: str, query: str, response: Any = None, session_id: str | None = None, new_session: bool = False, **kwargs: Any):
        captured["user_id"] = user_id
        captured["query"] = query
        captured["response"] = response
        captured["session_id"] = session_id
        captured["new_session"] = new_session
        return {"sessions": []}, "persisted-session-1"

    monkeypatch.setattr(docwain_api, "execute_request", fake_execute)
    monkeypatch.setattr(docwain_api, "add_message_to_history", fake_add_message_to_history)

    response = client.post(
        "/api/ask",
        json={"query": "hello", "profile_id": "p1", "user_id": "u1", "subscription_id": "s1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["response"] == "persist me"
    assert body["current_session_id"] == "persisted-session-1"
    assert captured["user_id"] == "u1"
    assert captured["query"] == "hello"


def test_smoke_teams_endpoint(monkeypatch, client: TestClient):
    docwain_api.activity_payload = {"type": "message", "text": "hi", "conversation": {"id": "c1"}}
    docwain_api.raw_body = b"{}"

    async def fake_parse(request):
        return docwain_api.activity_payload, docwain_api.raw_body

    async def fake_handle(activity_payload: Dict[str, Any], headers=None, raw_body=None):
        return {"type": "message", "text": "ack"}

    monkeypatch.setattr(docwain_api, "_parse_teams_activity", fake_parse)
    monkeypatch.setattr(docwain_api.teams_adapter, "handle_teams_activity", fake_handle)

    response = client.post("/api/teams/messages", json={"text": "hello"})
    assert response.status_code == 200
    assert response.json().get("text") == "ack"
