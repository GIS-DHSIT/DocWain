import json
from pathlib import Path

import pytest

from src.finetune.dataset_builder import build_dataset_from_qdrant
from src.finetune.pair_generator import ChunkRecord, MultiStrategyPairGenerator, merge_adjacent
from src.api.config import Config


class _FakePoint:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload
        self.vector = [0.1, 0.2, 0.3]


class _FakeQdrantClient:
    def __init__(self, points):
        self.points = points

    def scroll(self, collection_name, scroll_filter=None, limit=1, with_vectors=False, with_payload=True, offset=None):
        start = offset or 0
        end = start + limit
        batch = self.points[start:end]
        next_offset = end if end < len(self.points) else None
        return batch, next_offset


def test_merge_adjacent_respects_min_tokens():
    chunks = [
        ChunkRecord(text="alpha beta", metadata={"document_id": "d1", "chunk_index": 0}),
        ChunkRecord(text="gamma delta epsilon", metadata={"document_id": "d1", "chunk_index": 1}),
        ChunkRecord(text="zeta eta theta iota", metadata={"document_id": "d1", "chunk_index": 2}),
    ]
    merged = merge_adjacent(chunks, min_tokens=6, merge_window=3)
    assert len(merged) == 1
    assert "alpha beta" in merged[0].text


def test_multi_strategy_fallback_generates_pairs():
    block = ChunkRecord(text="Step one: do X. Step two: do Y.", metadata={"document_id": "d1"})
    gen = MultiStrategyPairGenerator(llm_client=None, min_pairs=1, max_pairs=5)
    pairs, strategies = gen.generate([block])
    assert len(pairs) >= 1
    assert strategies["span_instruction"] >= 1 or strategies["extractive_qa"] >= 1


def test_build_dataset_with_schema_probe_and_sparse_text(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Config.Finetune, "MIN_CHUNK_CHARS", 1, raising=False)
    monkeypatch.setattr(Config.Finetune, "MIN_MERGED_TOKENS", 1, raising=False)
    monkeypatch.setattr(Config.Finetune, "MIN_PAIRS_PER_PROFILE", 1, raising=False)
    monkeypatch.setattr(Config.Finetune, "MAX_PAIRS_PER_PROFILE", 5, raising=False)

    points = [
        _FakePoint("p1", {"content": "Short text one.", "profile_id": "p1", "document_id": "d1"}),
        _FakePoint("p2", {"content": "Short text two.", "profile_id": "p1", "document_id": "d1"}),
    ]
    client = _FakeQdrantClient(points)
    result = build_dataset_from_qdrant(
        profile_id="p1",
        subscription_id="sub",
        max_points=10,
        output_dir=tmp_path,
        client=client,
        collection_name="col",
        run_id="run-test",
    )
    assert result.status == "success"
    assert result.dataset_path and result.dataset_path.exists()
    lines = result.dataset_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
