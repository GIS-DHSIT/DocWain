from __future__ import annotations

import logging

import pytest

from src.ask.pipeline import run_docwain_rag_v2
from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, FakeRedis, make_point


class FakeCrossEncoder:
    def predict(self, pairs):  # noqa: ANN001
        torch = pytest.importorskip("torch")
        _ = pairs
        return torch.tensor([[0.1, 0.3, 0.2]])


def test_rag_v2_reranker_tensor_no_crash(caplog):
    caplog.set_level(logging.WARNING)
    points = [
        make_point(pid="p1", profile_id="prof-1", document_id="doc-1", file_name="a.pdf", text="Alpha text", page=1),
        make_point(pid="p2", profile_id="prof-1", document_id="doc-2", file_name="b.pdf", text="Beta text", page=2),
        make_point(pid="p3", profile_id="prof-1", document_id="doc-3", file_name="c.pdf", text="Gamma text", page=3),
    ]
    response = run_docwain_rag_v2(
        query="What is in the documents?",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_id="sess-1",
        user_id="user-1",
        request_id="req-1",
        llm_client=None,
        qdrant_client=FakeQdrant(points),
        redis_client=FakeRedis(),
        embedder=FakeEmbedder(),
        cross_encoder=FakeCrossEncoder(),
    )
    assert response
    assert response.get("response")
    assert response.get("sources")
    assert "only one element tensors can be converted to Python scalars" not in caplog.text
