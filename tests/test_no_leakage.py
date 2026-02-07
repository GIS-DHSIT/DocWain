from __future__ import annotations

from qdrant_client.models import FieldCondition

from src.ask.pipeline import run_docwain_rag_v2
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, make_point


def test_no_leakage_across_profiles():
    points = [
        make_point(
            pid="1",
            profile_id="profile-A",
            document_id="doc-A",
            file_name="Alice Resume.pdf",
            text="Alice Johnson has skills in Python and SQL.",
            page=1,
        ),
        make_point(
            pid="2",
            profile_id="profile-B",
            document_id="doc-B",
            file_name="Bob Resume.pdf",
            text="Bob Smith has skills in Java.",
            page=1,
        ),
    ]
    qdrant = FakeQdrant(points)
    response = run_docwain_rag_v2(
        query="Who is Bob Smith?",
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

    assert "Bob Smith" not in response["response"]
    assert "Note: Complete information is not available" in response["response"]

    filt = qdrant.last_filter
    assert filt is not None
    keys = [cond.key for cond in filt.must if isinstance(cond, FieldCondition)]
    assert "profile_id" in keys
