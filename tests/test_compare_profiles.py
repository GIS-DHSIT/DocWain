from __future__ import annotations

from src.ask.pipeline import run_docwain_rag_v2
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, make_point


def test_compare_profiles_output():
    points = [
        make_point(
            pid="1",
            profile_id="profile-A",
            document_id="doc-A",
            file_name="Alice Resume.pdf",
            text="Alice Johnson has skills in Python and SQL.",
            page=1,
            doc_domain="resume",
        ),
        make_point(
            pid="2",
            profile_id="profile-A",
            document_id="doc-B",
            file_name="Bob Resume.pdf",
            text="Bob Smith has skills in Java and Kotlin.",
            page=1,
            doc_domain="resume",
        ),
    ]
    qdrant = FakeQdrant(points)
    response = run_docwain_rag_v2(
        query="differences between both profiles",
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

    text = response["response"]
    assert "Alice Resume.pdf" in text
    assert "Bob Resume.pdf" in text
    assert "Python" in text
    assert "Java" in text
    assert "|" not in text
