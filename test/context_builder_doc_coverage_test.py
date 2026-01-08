from src.api.enhanced_context_builder import IntelligentContextBuilder


def test_context_builder_preserves_multi_doc_coverage():
    builder = IntelligentContextBuilder(max_context_chunks=3)
    chunks = [
        {"text": "Info A1 with enough detail to exceed filtering threshold for coverage checks.", "score": 0.92, "metadata": {"document_id": "docA", "source_file": "Doc A", "chunk_id": "a1", "chunk_index": 0}},
        {"text": "Info A2 with more detail to keep another chunk from the same document.", "score": 0.85, "metadata": {"document_id": "docA", "source_file": "Doc A", "chunk_id": "a2", "chunk_index": 1}},
        {"text": "Info B1 with coverage from a different document to validate round robin selection.", "score": 0.88, "metadata": {"document_id": "docB", "source_file": "Doc B", "chunk_id": "b1", "chunk_index": 0}},
        {"text": "Info C1 with content to ensure a third document is included in the context.", "score": 0.80, "metadata": {"document_id": "docC", "source_file": "Doc C", "chunk_id": "c1", "chunk_index": 0}},
    ]

    context, sources = builder.build_context(chunks, query="test query")

    assert len(sources) == 3  # limited by max_context_chunks
    source_names = {s["source_name"] for s in sources}
    # Should include top evidence from three different documents via round-robin selection
    assert {"Doc A", "Doc B", "Doc C"}.issubset(source_names)
    assert context.count("[SOURCE-") == 3
