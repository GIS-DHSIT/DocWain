import json
from pathlib import Path

import numpy as np

from src.training.embedding_aware_pairgen import Chunk, LineFrequencyCleaner, build_pairs_for_profile
from src.training.qdrant_schema_probe import SchemaProbe, SchemaProbeResult, FieldStat


class _FakePoint:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload

    def dict(self):
        return {"id": self.id, "payload": self.payload}


class _FakeClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    def scroll(self, collection_name=None, limit=None, offset=None, **kwargs):
        limit = limit or len(self.payloads)
        offset = offset or 0
        batch = self.payloads[offset : offset + limit]
        next_offset = None if offset + limit >= len(self.payloads) else offset + limit
        return batch, next_offset


def test_schema_probe_detects_text_field(tmp_path: Path):
    payloads = [
        _FakePoint("p1", {"content": "hello world", "profile_id": "abc"}),
        _FakePoint("p2", {"content": "second text", "profile_id": "def"}),
    ]
    client = _FakeClient(payloads)
    probe = SchemaProbe(client=client, collection="c", run_dir=tmp_path, sample_size=2, scroll_page=2)
    result = probe.probe()
    assert result.text_field.path == ["content"]
    assert result.profile_field.path == ["profile_id"]
    assert result.profile_type == "str"


def test_profile_filter_matches_type(tmp_path: Path):
    payloads = [
        _FakePoint("p1", {"text": "hello world", "metadata": {"profileId": 123}}),
        _FakePoint("p2", {"text": "another", "metadata": {"profileId": 456}}),
    ]
    client = _FakeClient(payloads)
    probe = SchemaProbe(client=client, collection="c", run_dir=tmp_path, sample_size=2, scroll_page=2)
    result = probe.probe()
    assert result.profile_field.path == ["metadata", "profileId"]
    assert result.profile_type in {"int", "float", "str"}


def test_pairs_created_nonzero_when_text_present(tmp_path: Path):
    chunks = [
        Chunk(
            profile_id="p",
            text="Line one.\nLine two important detail.",
            embedding=list(np.ones(4)),
            metadata={"doc_id": "d1", "chunk_index": 0},
        ),
        Chunk(
            profile_id="p",
            text="Continuation line three.\nMore content.",
            embedding=list(np.ones(4)),
            metadata={"doc_id": "d1", "chunk_index": 1},
        ),
        Chunk(
            profile_id="p",
            text="Another section with data.",
            embedding=list(np.ones(4)),
            metadata={"doc_id": "d1", "chunk_index": 2},
        ),
    ]
    cleaner = LineFrequencyCleaner()
    pairs, drops = build_pairs_for_profile(chunks, cleaner=cleaner, use_ollama=False)
    assert len(pairs) > 0
