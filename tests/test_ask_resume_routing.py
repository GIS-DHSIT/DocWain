from __future__ import annotations

from src.api.rag_state import AppState, set_app_state
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, FakeRedis, make_point

import src.main as main


def _set_state(points):
    set_app_state(
        AppState(
            embedding_model=FakeEmbedder(),
            reranker=None,
            qdrant_client=FakeQdrant(points),
            redis_client=FakeRedis(),
            ollama_client=None,
            rag_system=None,
        )
    )


def _build_request(**kwargs):
    payload = dict(
        query="Summarize the document.",
        user_id="user-1",
        profile_id="profile-1",
        subscription_id="sub-1",
        document_id=None,
        tool_hint=None,
        model_name="DocWain-Agent",
        persona="DocWain",
        session_id=None,
        new_session=False,
        agent_mode=False,
        stream=False,
        debug=False,
        tools=None,
        tool_inputs=None,
        use_tools=False,
    )
    payload.update(kwargs)
    return main.QuestionRequest(**payload)


def test_ask_returns_acknowledgement_line():
    points = [
        make_point(
            pid="p1",
            profile_id="profile-1",
            document_id="doc-1",
            file_name="resume.pdf",
            text="Skills: Python, SQL",
            page=1,
            doc_domain="generic",
        )
    ]
    _set_state(points)

    request = _build_request(query="list skills")
    response = main.ask_question_api(request, stream=False)

    assert response.answer.startswith("I understand you want")


def test_ask_aggregates_multiple_documents():
    points = [
        make_point(
            pid="p1",
            profile_id="profile-1",
            document_id="doc-1",
            file_name="candidate_a.pdf",
            text="Skills: Python",
            page=1,
            doc_domain="generic",
        ),
        make_point(
            pid="p2",
            profile_id="profile-1",
            document_id="doc-2",
            file_name="candidate_b.pdf",
            text="Skills: SQL",
            page=1,
            doc_domain="generic",
        ),
    ]
    _set_state(points)

    request = _build_request(query="list skills")
    response = main.ask_question_api(request, stream=False)

    lowered = response.answer.lower()
    assert "candidate_a.pdf" in lowered
    assert "candidate_b.pdf" in lowered
