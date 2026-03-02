from __future__ import annotations

import pytest

from src.ask.retriever import _vector_to_list


def test_vector_to_list_handles_list_tensor():
    torch = pytest.importorskip("torch")
    vec = [torch.tensor([0.1, 0.2, 0.3])]
    out = _vector_to_list(vec)
    assert out == pytest.approx([0.1, 0.2, 0.3])
