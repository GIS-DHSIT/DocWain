from __future__ import annotations

import re

from src.ask.pipeline import run_docwain_rag_v2
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, make_point


def test_no_internal_ids_in_response():
    points = [
        make_point(
            pid="1",
            profile_id="profile-A",
            document_id="doc-A",
            file_name="Alice Resume.pdf",
            text="Alice Johnson has skills in Python.",
            page=1,
        )
    ]
    qdrant = FakeQdrant(points)
    response = run_docwain_rag_v2(
        query="Who mentions Python?",
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
    assert "document_id" not in text
    assert "section_id" not in text
    assert not re.search(r"\bdoc-[A-Za-z0-9]+\b", text)
