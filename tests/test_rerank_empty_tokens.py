from src.api import dw_newron as dn


def _chunk(text, score, chunk_id):
    return dn.RetrievedChunk(
        id=chunk_id,
        text=text,
        score=score,
        metadata={},
        source=None,
        method="dense",
    )


def test_hybrid_rerank_handles_empty_tokens():
    reranker = dn.HybridReranker(cross_encoder=None)
    chunks = [
        _chunk("   ", 0.2, "c1"),
        _chunk("", 0.9, "c2"),
    ]

    reranked = reranker.rerank(chunks=chunks, query="", top_k=2, use_cross_encoder=False)

    assert [c.id for c in reranked] == ["c2", "c1"]
