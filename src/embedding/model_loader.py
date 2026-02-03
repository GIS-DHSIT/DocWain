from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from src.api.config import Config

logger = logging.getLogger(__name__)

_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME: Optional[str] = None
_MODEL_DEVICE: Optional[str] = None
_MODEL_DIM: Optional[int] = None
_FALLBACK_USED = False


def _preferred_device() -> str:
    env_device = (os.getenv("EMBEDDING_DEVICE") or "").strip().lower()
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_meta_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "meta tensor" in msg or "cannot copy out of meta tensor" in msg


def _candidates() -> list[str]:
    candidates: list[str] = []
    for name in getattr(Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", []) or []:
        if name:
            candidates.append(str(name))
    if not candidates:
        fallback = getattr(Config.Model, "EMBEDDING_MODEL", None) or getattr(
            Config.Model, "SENTENCE_TRANSFORMERS", "BAAI/bge-large-en-v1.5"
        )
        candidates.append(str(fallback))
    return candidates


def _load_on_cpu(name: str) -> SentenceTransformer:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_default_dtype(torch.float32)
    return SentenceTransformer(name, device="cpu", model_kwargs={"device_map": None, "low_cpu_mem_usage": False})


def _load_model(name: str, device: str) -> SentenceTransformer:
    model = _load_on_cpu(name)
    if device != "cpu":
        model.to(device)
    return model


def _health_check(model: SentenceTransformer) -> None:
    model.encode(["health check"], convert_to_numpy=True, normalize_embeddings=False, batch_size=1)


def get_embedding_model(
    *,
    required_dim: Optional[int] = None,
    reload: bool = False,
    device: Optional[str] = None,
) -> SentenceTransformer:
    global _MODEL, _MODEL_NAME, _MODEL_DEVICE, _MODEL_DIM, _FALLBACK_USED
    target_device = (device or _preferred_device()).strip().lower()
    if _MODEL is not None and not reload and _MODEL_DEVICE == target_device:
        return _MODEL

    last_error: Optional[Exception] = None
    for candidate in _candidates():
        try:
            model = _load_model(candidate, target_device)
            _health_check(model)
            dim = model.get_sentence_embedding_dimension()
            if required_dim and dim != required_dim:
                logger.warning("Embedding dim mismatch: required=%s got=%s", required_dim, dim)
            _MODEL = model
            _MODEL_NAME = candidate
            _MODEL_DEVICE = target_device
            _MODEL_DIM = dim
            logger.info("Embedding model ready: %s (dim=%s, device=%s)", candidate, dim, target_device)
            return model
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if _is_meta_tensor_error(exc):
                logger.warning("Meta tensor error loading %s: %s", candidate, exc)
            else:
                logger.warning("Failed to load embedding model %s: %s", candidate, exc)

    if not _FALLBACK_USED:
        _FALLBACK_USED = True
        fallback_name = "sentence-transformers/all-mpnet-base-v2"
        try:
            model = _load_model(fallback_name, "cpu")
            _health_check(model)
            dim = model.get_sentence_embedding_dimension()
            _MODEL = model
            _MODEL_NAME = fallback_name
            _MODEL_DEVICE = "cpu"
            _MODEL_DIM = dim
            logger.info("Fallback embedding model ready: %s (dim=%s)", fallback_name, dim)
            return model
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error:
        raise last_error
    raise RuntimeError("No embedding models available")


def encode_with_fallback(
    texts: list[str],
    *,
    normalize_embeddings: bool = False,
    convert_to_numpy: bool = True,
    batch_size: Optional[int] = None,
) -> Any:
    model = get_embedding_model()
    try:
        return model.encode(
            texts,
            batch_size=batch_size or 32,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_meta_tensor_error(exc):
            logger.warning("Meta tensor error during encode; falling back to cpu model: %s", exc)
            model = get_embedding_model(reload=True, device="cpu")
            return model.encode(
                texts,
                batch_size=max(1, min(batch_size or 32, 16)),
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
        raise


def get_model_info() -> Tuple[Optional[str], Optional[int], Optional[str]]:
    return _MODEL_NAME, _MODEL_DIM, _MODEL_DEVICE


__all__ = ["get_embedding_model", "encode_with_fallback", "get_model_info"]
