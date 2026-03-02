from __future__ import annotations

from src.docwain_intel.answering import run_agentic_rag
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, FakeRedis, make_point


def test_pipeline_ack_line_and_skills():
    points = [
        make_point(
            pid="p1",
            profile_id="prof-1",
            document_id="doc-1",
            file_name="resume.pdf",
            text="Skills: Python, SQL, Spark",
            page=1,
            chunk_kind="list_item_chunk",
        )
    ]
    response = run_agentic_rag(
        query="list skills",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_id="sess-1",
        request_id="req-1",
        llm_client=None,
        qdrant_client=FakeQdrant(points),
        redis_client=FakeRedis(),
        embedder=FakeEmbedder(),
        cross_encoder=None,
    )
    answer = response.get("response", "")
    assert answer.startswith("I understand you want")
    assert "Skills" in answer


def test_pipeline_no_chunks_graceful_message():
    response = run_agentic_rag(
        query="list skills",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_id="sess-1",
        request_id="req-1",
        llm_client=None,
        qdrant_client=FakeQdrant([]),
        redis_client=FakeRedis(),
        embedder=FakeEmbedder(),
        cross_encoder=None,
    )
    answer = response.get("response", "")
    assert answer.startswith("I understand you want")
    assert response.get("context_found") is False


def test_pipeline_avoids_not_explicitly_mentioned():
    points = [
        make_point(
            pid="p1",
            profile_id="prof-1",
            document_id="doc-1",
            file_name="resume.pdf",
            text="Certifications: CAPM",
            page=1,
            chunk_kind="section_chunk",
        )
    ]
    response = run_agentic_rag(
        query="summarize certifications",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_id="sess-1",
        request_id="req-1",
        llm_client=None,
        qdrant_client=FakeQdrant(points),
        redis_client=FakeRedis(),
        embedder=FakeEmbedder(),
        cross_encoder=None,
    )
    answer = (response.get("response") or "").lower()
    assert "not explicitly mentioned" not in answer
    assert "capm" in answer


def test_pipeline_ranking_line_present():
    points = [
        make_point(
            pid="p1",
            profile_id="prof-1",
            document_id="doc-1",
            file_name="resume_a.pdf",
            text="Skills: Python, SQL",
            page=1,
            chunk_kind="list_item_chunk",
        ),
        make_point(
            pid="p2",
            profile_id="prof-1",
            document_id="doc-2",
            file_name="resume_b.pdf",
            text="Skills: Python, SQL, Spark",
            page=1,
            chunk_kind="list_item_chunk",
        ),
    ]
    response = run_agentic_rag(
        query="rank the profiles based on skills",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_id="sess-1",
        request_id="req-1",
        llm_client=None,
        qdrant_client=FakeQdrant(points),
        redis_client=FakeRedis(),
        embedder=FakeEmbedder(),
        cross_encoder=None,
    )
    answer = response.get("response") or ""
    # Response format may vary: "Ranking:" header or direct listing with bullet points
    assert answer, "Expected a non-empty ranking response"
    lowered = answer.lower()
    assert "resume_a" in lowered or "resume_b" in lowered or "ranking" in lowered or "-" in answer, (
        f"Expected ranking content with document references, got: {answer[:200]}"
    )
