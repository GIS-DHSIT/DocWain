from src.api.dw_newron import QdrantRetriever, RetrievedChunk


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload
        self.id = payload.get("chunk_id", "p-1")
        self.score = 0.5


class _FakeClient:
    def __init__(self, points):
        self.points = points
        self.scroll_calls = []

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        # Mimic qdrant_client.scroll returning an object with .points
        return type("ScrollResult", (), {"points": self.points})


class _FakeModel:
    """Placeholder for retriever constructor; not used in this test."""


def test_expand_with_neighbors_handles_missing_source_fields():
    # Neighbor payload lacks source/source_file; ensure expansion does not crash and falls back to "unknown".
    neighbor_point = _FakePoint(
        {
            "text": "neighbor chunk text",
            "document_id": "doc-123",
            "chunk_index": 1,
        }
    )
    client = _FakeClient(points=[neighbor_point])
    retriever = QdrantRetriever(client=client, model=_FakeModel())

    seed_chunk = RetrievedChunk(
        id="seed-0",
        text="seed text",
        score=1.0,
        metadata={"document_id": "doc-123", "chunk_index": 0},
    )

    expanded = retriever.expand_with_neighbors(
        collection_name="test-collection",
        seed_chunks=[seed_chunk],
        profile_id="prof-abc",
        window=1,
        max_new=2,
    )

    assert len(expanded) == 2
    neighbor = expanded[1]
    assert neighbor.text == "neighbor chunk text"
    assert neighbor.source == "unknown"
