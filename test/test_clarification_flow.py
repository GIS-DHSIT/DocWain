from src.api.dw_newron import EnterpriseRAGSystem, RetrievedChunk


def _chunk(*, chunk_id: str, source: str, score: float, section: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=f"Content from {source} / {section}",
        score=score,
        metadata={
            "source_file": source,
            "section_title": section,
        },
        source=source,
    )


def test_clarification_requested_for_vague_query_with_unrelated_docs():
    chunks = [
        _chunk(chunk_id="1", source="Alpha_Report.pdf", score=0.52, section="Overview"),
        _chunk(chunk_id="2", source="Beta_Manual.pdf", score=0.50, section="Summary"),
    ]
    doc_options, section_options = EnterpriseRAGSystem._collect_focus_options(chunks, {})
    needs_clarification = EnterpriseRAGSystem._should_request_clarification(
        query="summarize",
        doc_options=doc_options,
        section_options=section_options,
        metadata_filters={},
        preprocessing_metadata={},
    )
    assert needs_clarification is True


def test_no_clarification_when_query_mentions_document():
    chunks = [
        _chunk(chunk_id="1", source="Alpha_Report.pdf", score=0.52, section="Overview"),
        _chunk(chunk_id="2", source="Beta_Manual.pdf", score=0.50, section="Summary"),
    ]
    doc_options, section_options = EnterpriseRAGSystem._collect_focus_options(chunks, {})
    needs_clarification = EnterpriseRAGSystem._should_request_clarification(
        query="summarize the alpha report",
        doc_options=doc_options,
        section_options=section_options,
        metadata_filters={},
        preprocessing_metadata={},
    )
    assert needs_clarification is False


def test_no_clarification_for_comparison_intent():
    chunks = [
        _chunk(chunk_id="1", source="Alpha_Report.pdf", score=0.52, section="Overview"),
        _chunk(chunk_id="2", source="Beta_Manual.pdf", score=0.50, section="Summary"),
    ]
    doc_options, section_options = EnterpriseRAGSystem._collect_focus_options(chunks, {})
    needs_clarification = EnterpriseRAGSystem._should_request_clarification(
        query="compare alpha and beta",
        doc_options=doc_options,
        section_options=section_options,
        metadata_filters={},
        preprocessing_metadata={"query_analysis": {"intent": "comparison"}},
    )
    assert needs_clarification is False


def test_evidence_response_includes_citations():
    sources = [
        {
            "source_id": 1,
            "section": "Overview",
            "page": 2,
            "excerpt": "This document describes the project scope and objectives.",
        }
    ]
    response = EnterpriseRAGSystem._build_evidence_response(
        query="What is the scope?",
        sources=sources,
        doc_options=[{"label": "Alpha Report"}],
        section_options=["Overview"],
    )
    assert "[SOURCE-1]" in response
    assert "project scope" in response
