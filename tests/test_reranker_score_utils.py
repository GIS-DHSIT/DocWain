from __future__ import annotations

import pytest

from src.services.retrieval.score_utils import RerankShapeError, normalize_scores


def test_normalize_scores_tensor_flat():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([0.1, 0.2, 0.3])
    normalized = normalize_scores(scores, expected_k=3)
    assert normalized == pytest.approx([0.1, 0.2, 0.3])


def test_normalize_scores_tensor_row():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[0.1, 0.2, 0.3]])
    normalized = normalize_scores(scores, expected_k=3)
    assert normalized == pytest.approx([0.1, 0.2, 0.3])


def test_normalize_scores_tensor_scalar():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[0.9]])
    normalized = normalize_scores(scores, expected_k=1)
    assert normalized == pytest.approx([0.9])


def test_normalize_scores_mismatch_raises():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[0.1, 0.2]])
    with pytest.raises(RerankShapeError):
        normalize_scores(scores, expected_k=3)
