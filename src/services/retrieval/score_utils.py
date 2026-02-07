from __future__ import annotations

from typing import Any, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore


class RerankShapeError(RuntimeError):
    """Raised when reranker scores do not align with candidate count."""

    def __init__(
        self,
        message: str,
        *,
        expected_k: int,
        actual_len: int,
        score_type: str,
        score_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__(message)
        self.expected_k = expected_k
        self.actual_len = actual_len
        self.score_type = score_type
        self.score_shape = score_shape


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and getattr(torch, "is_tensor", None) and torch.is_tensor(value)


def _is_numpy_array(value: Any) -> bool:
    return np is not None and isinstance(value, np.ndarray)


def _shape_of(value: Any) -> Optional[Tuple[int, ...]]:
    if _is_torch_tensor(value):
        try:
            return tuple(int(v) for v in value.shape)
        except Exception:
            return None
    if _is_numpy_array(value):
        try:
            return tuple(int(v) for v in value.shape)
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        try:
            if not value:
                return (0,)
            child_shape = _shape_of(value[0])
            if child_shape:
                return (len(value),) + child_shape
            return (len(value),)
        except Exception:
            return None
    return None


def _flatten_scores(value: Any) -> List[Any]:
    if _is_torch_tensor(value):
        return list(value.detach().cpu().flatten().tolist())
    if _is_numpy_array(value):
        return list(value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        flattened: List[Any] = []
        for item in value:
            flattened.extend(_flatten_scores(item))
        return flattened
    return [value]


def to_py_scalar(value: Any, *, reduce: str = "first") -> float:
    """
    Convert model outputs (tensor/np/list/scalar) into a python float safely.
    reduce:
      - "first": take first element after flatten
      - "max": take max
      - "mean": take mean
    """
    flattened = _flatten_scores(value)
    if not flattened:
        raise ValueError("Cannot convert empty value to scalar")
    floats = [float(v) for v in flattened]
    if len(floats) == 1:
        return floats[0]
    if reduce == "first":
        return floats[0]
    if reduce == "max":
        return max(floats)
    if reduce == "mean":
        return sum(floats) / len(floats)
    raise ValueError(f"Unknown reduce strategy: {reduce}")


def normalize_scores(scores: Any, *, expected_k: int) -> List[float]:
    """
    Normalize reranker scores into a list[float] with length expected_k.
    Accepts tensors, numpy arrays, list[float], list[tensor], etc.
    """
    flattened = _flatten_scores(scores)
    actual_len = len(flattened)
    if actual_len != expected_k:
        score_type = type(scores).__name__
        score_shape = _shape_of(scores)
        raise RerankShapeError(
            f"Reranker score length mismatch: expected {expected_k}, got {actual_len}",
            expected_k=expected_k,
            actual_len=actual_len,
            score_type=score_type,
            score_shape=score_shape,
        )
    return [float(v) for v in flattened]


def describe_scores(scores: Any) -> dict:
    return {
        "score_type": type(scores).__name__,
        "score_shape": _shape_of(scores),
    }


__all__ = ["RerankShapeError", "normalize_scores", "to_py_scalar", "describe_scores"]
