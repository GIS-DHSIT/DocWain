from __future__ import annotations

import pytest

from src.ask.retriever import DocWainRetriever
from tests.rag_v2_helpers import FakeQdrant, make_point


class TorchEmbedder:
    def encode(self, text, convert_to_numpy=False, normalize_embeddings=True):  # noqa: ANN001
        _ = (text, convert_to_numpy, normalize_embeddings)
        torch = pytest.importorskip("torch")
        return torch.tensor([[0.1, 0.2, 0.3, 0.4]])


def test_docwain_retriever_accepts_tensor_embedding():
    points = [
        make_point(pid="p1", profile_id="prof-1", document_id="doc-1", file_name="a.pdf", text="Alpha", page=1),
    ]
    retriever = DocWainRetriever(FakeQdrant(points), TorchEmbedder())
    results = retriever.retrieve(
        query="test",
        subscription_id="sub-1",
        profile_id="prof-1",
        top_k=1,
        collection_name="sub-1",
    )
    assert len(results) == 1
