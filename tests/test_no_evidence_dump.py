from src.rag.citation_policy import apply as apply_citation_policy


def test_evidence_block_removed_and_citations_once():
    answer = (
        "Here is the summary.\n\n"
        "Evidence:\n"
        "- Line one\n"
        "- Line two\n\n"
        "Takeaways:\n"
        "- Next step\n\n"
        "Citations: old source"
    )
    sources = [
        {"source_name": "doc.pdf", "section": "Summary", "page": "1"},
        {"source_name": "doc.pdf", "section": "Details", "page": "2"},
    ]
    result = apply_citation_policy(answer, sources, scope_type="single_doc", user_requested_evidence=False)

    assert "Evidence:" not in result
    assert result.lower().count("citations:") == 1
    assert result.strip().endswith("Citations: doc.pdf | Summary | 1; doc.pdf | Details | 2")
