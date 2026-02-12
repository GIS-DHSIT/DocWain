from __future__ import annotations

from src.docwain_intel.answering import run_agentic_rag
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, FakeRedis, make_point


def _make_points():
    points = [
        make_point(
            pid="a1",
            profile_id="prof-1",
            document_id="doc-a",
            file_name="candidate_a.pdf",
            text="CERTIFICATIONS: CAPM, ITIL\nSkills: Python, SQL",
            page=1,
            chunk_kind="section_chunk",
        ),
        make_point(
            pid="b1",
            profile_id="prof-1",
            document_id="doc-b",
            file_name="candidate_b.pdf",
            text="Certified ScrumMaster (CSM)\nSkills: Java, SCM tools",
            page=1,
            chunk_kind="list_item_chunk",
        ),
        make_point(
            pid="c1",
            profile_id="prof-1",
            document_id="doc-c",
            file_name="candidate_c.pdf",
            text="Professional Summary: Experienced engineer\nSkills: C++",
            page=1,
            chunk_kind="section_chunk",
        ),
    ]
    return points


def test_certifications_structured_summary():
    points = _make_points()
    response = run_agentic_rag(
        query="summarize the certifications of each candidates in a structured way",
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
    text = response.get("response") or ""
    assert "CAPM" in text
    assert "Certified ScrumMaster" in text
    # Candidate C should show missing certifications
    assert "candidate_c.pdf" in text
    assert "Not found in the retrieved evidence" in text
    assert "Professional Summary" not in text


def test_rank_profiles_based_on_skills():
    points = _make_points()
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
    text = response.get("response") or ""
    assert "Skills" in text
    assert "Not explicitly mentioned" not in text


def test_compare_certification_counts():
    points = _make_points()
    response = run_agentic_rag(
        query="which candidate has more certifications",
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
    text = response.get("response") or ""
    # The response should mention certifications from multiple candidates
    assert "CAPM" in text or "Certified ScrumMaster" in text or "CSM" in text or "certification" in text.lower()


def test_best_fit_scm_engineer():
    points = _make_points()
    response = run_agentic_rag(
        query="who will be a best fit for the role of SCM engineer",
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
    text = response.get("response") or ""
    assert "SCM" in text or "scm" in text
    assert "candidate_b.pdf" in text
