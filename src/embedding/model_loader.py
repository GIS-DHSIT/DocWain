from __future__ import annotations

import contextlib
import warnings
from src.utils.logging_utils import get_logger
import os
import threading
from typing import Any, Dict, Optional, Tuple

# Suppress SentenceTransformer._target_device deprecation warning.
# This is a library-internal deprecation (replaced by .device) that
# fires on model instantiation — no action needed on our side.
warnings.filterwarnings(
    "ignore",
    message=r".*_target_device.*has been deprecated",
    category=FutureWarning,
)

# Suppress tqdm progress bars globally — they spam logs with
# "Batches: 1/1" for every encode call in sentence-transformers and
# cross-encoder.  Must be set before any tqdm import.
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from sentence_transformers import SentenceTransformer

from src.api.config import Config

logger = get_logger(__name__)

_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME: Optional[str] = None
_MODEL_DEVICE: Optional[str] = None
_MODEL_DIM: Optional[int] = None
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[SentenceTransformer, int]] = {}
_MODEL_LOCK = threading.Lock()
_FALLBACK_NAME = "sentence-transformers/all-mpnet-base-v2"
_FALLBACK_USED = False
_FORCED_CPU_CANDIDATES: set[str] = set()
_REQUEST_CONTEXT = threading.local()

@contextlib.contextmanager
def embed_request_context(request_id: Optional[str]):
    prev = getattr(_REQUEST_CONTEXT, "embed_request_id", None)
    if request_id:
        _REQUEST_CONTEXT.embed_request_id = request_id
    elif hasattr(_REQUEST_CONTEXT, "embed_request_id"):
        delattr(_REQUEST_CONTEXT, "embed_request_id")
    try:
        yield
    finally:
        if prev:
            _REQUEST_CONTEXT.embed_request_id = prev
        elif hasattr(_REQUEST_CONTEXT, "embed_request_id"):
            delattr(_REQUEST_CONTEXT, "embed_request_id")

def _preferred_device() -> str:
    # Always default to CPU for embeddings — GPU is reserved for the LLM (DocWain-Agent).
    # bge-large-en-v1.5 (335M params) runs efficiently on CPU.
    env_device = (os.getenv("EMBEDDING_DEVICE") or "cpu").strip().lower()
    return env_device

def _is_meta_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "meta tensor" in msg or "cannot copy out of meta tensor" in msg

def _is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or ("cuda error" in msg and "alloc" in msg)
    )

def _request_prefix() -> str:
    request_id = getattr(_REQUEST_CONTEXT, "embed_request_id", None)
    return f"[embed_request_id={request_id}] " if request_id else ""

_GPU_MIN_FREE_MB = int(os.getenv("EMBEDDING_GPU_MIN_FREE_MB", "1500"))

def _has_sufficient_gpu_memory(min_mb: int = 0) -> bool:
    """Check if GPU has enough free memory for embedding."""
    threshold = min_mb or _GPU_MIN_FREE_MB
    try:
        if not torch.cuda.is_available():
            return False
        free, _total = torch.cuda.mem_get_info()
        free_mb = free / (1024 * 1024)
        return free_mb > threshold
    except Exception:  # noqa: BLE001
        return False

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
    if getattr(Config.Model, "OFFLINE_ONLY", True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_default_dtype(torch.float32)
    return SentenceTransformer(
        name,
        device="cpu",
        model_kwargs={"device_map": None, "low_cpu_mem_usage": False},
        local_files_only=True,
    )

def _has_meta_tensors(model: SentenceTransformer) -> bool:
    try:
        params = model.parameters()
    except Exception:  # noqa: BLE001
        return False
    for param in params:
        if getattr(param, "is_meta", False):
            return True
    return False

def _ensure_not_meta(model: SentenceTransformer, stage: str) -> None:
    if _has_meta_tensors(model):
        raise RuntimeError(f"Model contains meta tensors after {stage}")

def _health_check(model: SentenceTransformer) -> None:
    model.encode(["health check"], convert_to_numpy=True, normalize_embeddings=False, batch_size=1, show_progress_bar=False)

def get_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    *,
    required_dim: Optional[int] = None,
    reload: bool = False,
) -> Tuple[SentenceTransformer, int]:
    global _MODEL, _MODEL_NAME, _MODEL_DEVICE, _MODEL_DIM, _FALLBACK_USED
    target_device = (device or _preferred_device()).strip().lower()
    candidates = [model_name] if model_name else _candidates()
    last_error: Optional[Exception] = None

    with _MODEL_LOCK:
        for candidate in candidates:
            effective_device = target_device
            if effective_device != "cpu" and candidate in _FORCED_CPU_CANDIDATES:
                effective_device = "cpu"
            cache_key = (candidate, effective_device)
            if not reload:
                cached = _MODEL_CACHE.get(cache_key)
                if cached:
                    cached_model, cached_dim = cached
                    if _has_meta_tensors(cached_model):
                        _MODEL_CACHE.pop(cache_key, None)
                    else:
                        _MODEL = cached_model
                        _MODEL_NAME = candidate
                        _MODEL_DEVICE = effective_device
                        _MODEL_DIM = cached_dim
                        return cached_model, cached_dim
            else:
                _MODEL_CACHE.pop(cache_key, None)

            try:
                logger.info(
                    "%sLoading embedding model: %s (device=%s)",
                    _request_prefix(),
                    candidate,
                    effective_device,
                )
                model = _load_on_cpu(candidate)
                _ensure_not_meta(model, "cpu_load")
                if effective_device != "cpu" and not _has_sufficient_gpu_memory():
                    logger.info(
                        "%sGPU memory low before device move; loading %s on cpu",
                        _request_prefix(),
                        candidate,
                    )
                    _FORCED_CPU_CANDIDATES.add(candidate)
                    effective_device = "cpu"
                if effective_device != "cpu":
                    try:
                        model.to(effective_device)
                        _ensure_not_meta(model, "device_move")
                    except Exception as exc:  # noqa: BLE001
                        if _is_cuda_oom(exc):
                            _FORCED_CPU_CANDIDATES.add(candidate)
                            logger.warning(
                                "%sCUDA OOM loading %s on %s; falling back to cpu",
                                _request_prefix(),
                                candidate,
                                effective_device,
                            )
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:  # noqa: BLE001
                                pass
                            model = _load_on_cpu(candidate)
                            _ensure_not_meta(model, "cpu_reload")
                            effective_device = "cpu"
                        else:
                            raise
                _health_check(model)
                # Suppress per-batch progress bars — they spam logs with
                # "Batches: 1/1" for every single-item encode call.
                model.show_progress_bar = False
                dim = model.get_sentence_embedding_dimension()
                if required_dim and dim != required_dim:
                    logger.warning("Embedding dim mismatch: required=%s got=%s", required_dim, dim)
                _MODEL_CACHE[(candidate, effective_device)] = (model, dim)
                _MODEL = model
                _MODEL_NAME = candidate
                _MODEL_DEVICE = effective_device
                _MODEL_DIM = dim
                logger.info(
                    "%sEmbedding model ready: %s (dim=%s, device=%s)",
                    _request_prefix(),
                    candidate,
                    dim,
                    effective_device,
                )
                return model, dim
            except Exception as exc:  # noqa: BLE001
                _MODEL_CACHE.pop((candidate, effective_device), None)
                last_error = exc
                if _is_meta_tensor_error(exc):
                    logger.warning("%sMeta tensor error loading %s: %s", _request_prefix(), candidate, exc)
                else:
                    logger.warning("%sFailed to load embedding model %s: %s", _request_prefix(), candidate, exc)

        if _FALLBACK_USED:
            fallback_key = (_FALLBACK_NAME, "cpu")
            cached = _MODEL_CACHE.get(fallback_key)
            if cached:
                cached_model, cached_dim = cached
                if not _has_meta_tensors(cached_model):
                    _MODEL = cached_model
                    _MODEL_NAME = _FALLBACK_NAME
                    _MODEL_DEVICE = "cpu"
                    _MODEL_DIM = cached_dim
                    return cached_model, cached_dim
                _MODEL_CACHE.pop(fallback_key, None)

        if not _FALLBACK_USED:
            _FALLBACK_USED = True
            fallback_key = (_FALLBACK_NAME, "cpu")
            _MODEL_CACHE.pop(fallback_key, None)
            try:
                logger.info(
                    "%sLoading fallback embedding model: %s (device=cpu)",
                    _request_prefix(),
                    _FALLBACK_NAME,
                )
                model = _load_on_cpu(_FALLBACK_NAME)
                _ensure_not_meta(model, "cpu_load")
                _health_check(model)
                # Suppress per-batch progress bars — they spam logs with
                # "Batches: 1/1" for every single-item encode call.
                model.show_progress_bar = False
                dim = model.get_sentence_embedding_dimension()
                _MODEL_CACHE[fallback_key] = (model, dim)
                _MODEL = model
                _MODEL_NAME = _FALLBACK_NAME
                _MODEL_DEVICE = "cpu"
                _MODEL_DIM = dim
                logger.info(
                    "%sFallback embedding model ready: %s (dim=%s)",
                    _request_prefix(),
                    _FALLBACK_NAME,
                    dim,
                )
                return model, dim
            except Exception as exc:  # noqa: BLE001
                _MODEL_CACHE.pop(fallback_key, None)
                last_error = exc

    if last_error:
        raise last_error
    raise RuntimeError("No embedding models available")

def _optimal_batch_size(device: str, requested: Optional[int], num_texts: int) -> int:
    """Select batch size based on device and workload."""
    if requested:
        return requested
    default = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    if device == "cpu":
        # Smaller batches improve CPU cache locality — 2-3x faster
        return min(default, 16)
    return default

def encode_with_fallback(
    texts: list[str],
    *,
    normalize_embeddings: bool = False,
    convert_to_numpy: bool = True,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Any:
    # Proactive GPU check — skip straight to CPU if GPU memory is low
    effective_device = device
    if effective_device != "cpu" and not _has_sufficient_gpu_memory():
        logger.info(
            "%sGPU memory low (<%.0fMB free); using CPU for encoding %d texts",
            _request_prefix(), float(_GPU_MIN_FREE_MB), len(texts),
        )
        effective_device = "cpu"

    model, _ = get_embedding_model(device=effective_device)
    actual_device = str(getattr(model, "device", effective_device) or "cpu")
    effective_batch = _optimal_batch_size(actual_device, batch_size, len(texts))

    try:
        return model.encode(
            texts,
            batch_size=effective_batch,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_meta_tensor_error(exc) or _is_cuda_oom(exc):
            if _is_cuda_oom(exc):
                logger.debug("%sCUDA OOM during encode; falling back to cpu model: %s", _request_prefix(), exc)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            else:
                logger.debug("%sMeta tensor error during encode; falling back to cpu model: %s", _request_prefix(), exc)
            model, _ = get_embedding_model(reload=True, device="cpu")
            cpu_batch = _optimal_batch_size("cpu", batch_size, len(texts))
            return model.encode(
                texts,
                batch_size=cpu_batch,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
        raise

def get_model_info() -> Tuple[Optional[str], Optional[int], Optional[str]]:
    return _MODEL_NAME, _MODEL_DIM, _MODEL_DEVICE

__all__ = ["get_embedding_model", "encode_with_fallback", "get_model_info", "embed_request_context"]
