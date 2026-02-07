from __future__ import annotations

from src.ask.pipeline import run_docwain_rag_v2
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant


def test_human_like_greeting_no_retrieval():
    qdrant = FakeQdrant([])
    response = run_docwain_rag_v2(
        query="hi",
        subscription_id="sub-1",
        profile_id="profile-A",
        session_id="s1",
        user_id="u1",
        request_id="r1",
        llm_client=None,
        qdrant_client=qdrant,
        redis_client=None,
        embedder=FakeEmbedder(),
        cross_encoder=None,
    )

    text = response["response"].lower()
    assert "docwain" in text
    assert "hello" in text or "hi" in text
    assert qdrant.last_filter is None
