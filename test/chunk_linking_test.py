from src.api.enhanced_retrieval import normalize_chunk_links


def test_normalize_chunk_links_sets_prev_next_consistently():
    chunk_metadata = [
        {"chunk_id": "old-0", "prev_chunk_id": None, "next_chunk_id": "old-1", "section_title": "One"},
        {"chunk_id": "old-1", "prev_chunk_id": "old-0", "next_chunk_id": "old-2", "section_title": "Two"},
        {"chunk_id": "old-2", "prev_chunk_id": "old-1", "next_chunk_id": None, "section_title": "Three"},
    ]
    chunks = ["alpha", "bravo", "charlie"]

    normalized = normalize_chunk_links(
        chunk_metadata,
        subscription_id="sub-123",
        profile_id="prof-456",
        document_id="doc-789",
        doc_name="sample.docx",
        chunks=chunks,
    )

    ids = [m["chunk_id"] for m in normalized]
    assert len(set(ids)) == 3  # unique deterministic ids

    assert normalized[0]["prev_chunk_id"] is None
    assert normalized[0]["next_chunk_id"] == ids[1]

    assert normalized[1]["prev_chunk_id"] == ids[0]
    assert normalized[1]["next_chunk_id"] == ids[2]

    assert normalized[2]["prev_chunk_id"] == ids[1]
    assert normalized[2]["next_chunk_id"] is None

    # document_id should be enforced for every chunk
    assert all(m["document_id"] == "doc-789" for m in normalized)
