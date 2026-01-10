import json

from src.finetune import dataset_builder


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload


class _FakeQdrantClient:
    def __init__(self, points):
        self.points = points
        self.scroll_calls = []

    def scroll(self, collection_name, scroll_filter, limit, with_vectors, with_payload, offset=None):
        self.scroll_calls.append((collection_name, offset))
        data = self.points
        start = offset or 0
        end = start + limit
        batch = [_FakePoint(payload) for payload in data[start:end]]
        next_offset = end if end < len(data) else None
        return batch, next_offset


def test_sample_chunks_deduplicates_by_metadata():
    points = [
        {"text": "Long chunk text A" * 10, "source_file": "a.pdf", "document_id": "doc1", "chunk_index": 1, "profile_id": "p1"},
        {"text": "Long chunk text A" * 10, "source_file": "a.pdf", "document_id": "doc1", "chunk_index": 1, "profile_id": "p1"},
        {"text": "Long chunk text B" * 10, "source_file": "b.pdf", "document_id": "doc2", "chunk_index": 2, "profile_id": "p1"},
    ]
    client = _FakeQdrantClient(points)
    chunks = dataset_builder._sample_chunks("p1", "sub", limit=10, client=client)
    assert len(chunks) == 2


def test_generate_pairs_validates_grounding(monkeypatch):
    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, max_retries=2):
            self.calls += 1
            if self.calls == 1:
                return json.dumps([{"instruction": "Q", "output": "not in chunk"}])
            return json.dumps([{"instruction": "Q", "output": "grounded answer"}])

    monkeypatch.setattr(dataset_builder, "_get_llm_client", lambda model_name=None: _FakeLLM())
    chunk_text = "This is a grounded answer inside the chunk."
    pairs = dataset_builder._generate_pairs_for_chunk(chunk_text, "profile", None, 1, retries=1)
    assert pairs[0]["output"] == "grounded answer"
