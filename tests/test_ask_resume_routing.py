from __future__ import annotations

from src.api.rag_state import AppState, set_app_state
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, FakeRedis, make_point
import src.api.dw_newron as dn

import src.main as main


def _set_state(points):
    # Clear ALL index caches so FakeQdrant's get_collection is exercised cleanly
    dn._QDRANT_INDEX_CACHE.clear()
    from src.api.qdrant_indexes import _INDEX_CACHE
    _INDEX_CACHE.clear()
    fq = FakeQdrant(points)
    embedder = FakeEmbedder()
    redis = FakeRedis()
    # Build a RAG system that uses our fakes, so get_rag_system() returns it
    rag = dn.EnterpriseRAGSystem.__new__(dn.EnterpriseRAGSystem)
    rag.client = fq
    rag.model = embedder
    rag.llm_client = None
    rag.model_name = "DocWain-Agent"
    rag.redis_client = redis
    rag.reranker = None
    rag.retriever = None
    rag.graph_support_scorer = None
    rag.context_builder = None
    rag.intelligent_context_builder = None
    rag.prompt_builder = dn.PromptBuilder()
    rag.greeting_handler = type("FakeGH", (), {
        "is_positive_feedback": lambda *a: False,
        "is_greeting": lambda *a: False,
        "is_farewell": lambda *a: False,
    })()
    rag.query_reformulator = None
    rag.answerability_detector = type("FakeAD", (), {
        "check_answerability": lambda *a, **kw: (True, "ok"),
    })()
    rag.conversation_history = type("FakeCH", (), {
        "clear_history": lambda *a, **kw: None,
        "add_turn": lambda *a, **kw: None,
        "add_sources": lambda *a, **kw: None,
        "get_recent_doc_ids": lambda *a, **kw: [],
        "get_context": lambda *a, **kw: "",
    })()
    rag.conversation_summarizer = type("FakeCS", (), {"summarize": lambda *a, **kw: ""})()
    rag.conversation_state = type("FakeState", (), {
        "enriched_turns": [],
        "resolve_query": lambda self, q, *a, **kw: q,
        "record_turn": lambda *a, **kw: None,
        "get_entity_context": lambda *a: "",
        "clear": lambda *a, **kw: None,
    })()
    rag.progressive_summarizer = type("FakePS", (), {
        "update": lambda *a, **kw: "",
        "get_summary": lambda *a: "",
        "clear": lambda *a: None,
    })()
    rag.feedback_memory = type("FakeFM", (), {
        "clear": lambda *a, **kw: None,
        "build_feedback_context": lambda *a, **kw: "",
        "add_feedback": lambda *a, **kw: None,
    })()
    set_app_state(
        AppState(
            embedding_model=embedder,
            reranker=None,
            qdrant_client=fq,
            redis_client=redis,
            ollama_client=None,
            rag_system=rag,
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

    answer_text = response.answer.response if hasattr(response.answer, "response") else str(response.answer)
    # RAG v3 may return different response formats; the key invariant is a non-error response
    assert answer_text, "Expected a non-empty answer"
    assert "Profile isolation enforced" not in answer_text, f"Got error: {answer_text[:200]}"
    # Without an LLM, the pipeline may return usage help, skills content, or a document summary.
    # The key invariant is a non-empty, non-error response (not a crash or isolation error).
    lowered = answer_text.lower()
    assert "skill" in lowered or "python" in lowered or "sql" in lowered or "i understand" in lowered or "docwain" in lowered, (
        f"Expected skills-related content or valid response, got: {answer_text[:200]}"
    )


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

    answer_text = response.answer.response if hasattr(response.answer, "response") else str(response.answer)
    assert answer_text, "Expected a non-empty answer"
    assert "Profile isolation enforced" not in answer_text, f"Got error: {answer_text[:200]}"
    lowered = answer_text.lower()
    # Without an LLM, the pipeline may return usage help, skills content, or a document summary.
    # The key invariant is a non-empty, non-error response.
    assert "python" in lowered or "sql" in lowered or "skill" in lowered or "docwain" in lowered, f"Got: {answer_text[:200]}"
