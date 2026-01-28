import re
import json
import time
import threading
import logging
import hashlib
import concurrent.futures
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque, Counter
from datetime import datetime, timedelta
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from fastapi import HTTPException, status
from qdrant_client.models import Distance, PointStruct, SparseVector, Range
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import os
import redis
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from src.api.config import Config
from src.api.enhanced_context_builder import IntelligentContextBuilder
from src.api.query_intelligence import QueryIntelligence
from src.api.enhanced_retrieval import GraphGuidedRetriever
from src.api.reasoning_layer import EvidencePlanner, AnswerVerifier, ConfidenceScorer
from src.api.learning_signals import LearningSignalStore
from src.agentic.memory import AgentMemory
from src.agentic.clarification_agent import ClarificationAgent
from src.agentic.model_arbitration import ModelArbitrationLayer, ModelCandidate
from src.agentic.post_processor import PostProcessor
from src.agentic.response_templates import ResponseTemplateSelector
from src.chat.companion_classifier import CompanionClassifier
from src.chat.opener_generator import contains_banned_opener, generate_opener
from src.kg.neo4j_store import Neo4jStore
from src.utils.redis_cache import RedisJsonCache, hash_query, stamp_cache_payload
from src.agentic.retriever_manager import RetrieverManager
from src.agentic.verification_agent import VerificationAgent
from src.services.retrieval import (
    QueryUnderstanding,
    HybridRetriever,
    HybridRetrieverConfig,
    Reranker,
    RerankerConfig,
    RetrievalConfidenceScorer,
)
from src.retrieval import (
    QueryAnalyzer,
    EvidenceConstraints,
    EvidenceRequirements,
    HybridRanker,
    HybridRankerConfig,
    RetrievalQualityScorer,
    ContextAssembler,
    FallbackRepair,
    extract_required_attributes,
    filter_chunks_by_intent,
    extract_answer_requirements,
    validate_answer_requirements,
    build_intent_miss_response,
)
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from sklearn.feature_extraction.text import HashingVectorizer
from src.api.vector_store import build_collection_name, PAYLOAD_INDEX_FIELDS, PAYLOAD_INDEX_SCHEMAS
from src.api.genai_client import generate_text, get_genai_client
from src.finetune import resolve_model_for_profile
from src.metrics.ai_metrics import get_metrics_store
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store
from src.security.response_sanitizer import sanitize_user_payload
from src.quality.bad_answer_evaluator import BadAnswerEvaluator, EvalConfig
from src.quality.auto_repair import AutoRepairEngine, RepairConfig
from src.quality.telemetry import emit_quality_telemetry

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantCollectionNotFoundError(ValueError):
    def __init__(self, collection_name: str, available_collections: Optional[List[str]] = None):
        available = available_collections or []
        detail = f"qdrant_collection_not_found: collection '{collection_name}' not found"
        if available:
            detail = f"{detail}; available_collections={available}"
        super().__init__(detail)
        self.collection_name = collection_name
        self.available_collections = available


def _resolve_collection_name(
    *,
    collection_name: Optional[str] = None,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> str:
    if collection_name and str(collection_name).strip():
        return str(collection_name).strip()
    if subscription_id and str(subscription_id).strip():
        return build_collection_name(str(subscription_id).strip())
    if profile_id and str(profile_id).strip():
        return str(profile_id).strip()
    raise ValueError("collection_name could not be resolved; provide collection_name, subscription_id, or profile_id")

# Initialize models and clients (lazy loading to avoid startup errors)
_MODEL = None
_CROSS_ENCODER = None
_QDRANT_CLIENT = None
_REDIS_CLIENT = None
_MODEL_CACHE: Dict[int, SentenceTransformer] = {}
_MODEL_BY_NAME: Dict[str, SentenceTransformer] = {}
_METRICS_TRACKER = None
REDIS_MEMORY_TTL = int(os.getenv("REDIS_MEMORY_TTL", "86400"))
ANSWER_CACHE_TTL = int(os.getenv("RAG_ANSWER_CACHE_TTL", "1800"))
ANSWER_CACHE_VERSION = "v3"
NO_ANSWER_CACHE = os.getenv("NO_ANSWER_CACHE", "true").strip().lower() in {"true", "1", "yes", "on"}
ENABLE_ANSWER_CACHE = False if NO_ANSWER_CACHE else os.getenv("ENABLE_ANSWER_CACHE", "false").strip().lower() == "true"


def _torch_cuda_available() -> bool:
    try:
        return bool(torch) and bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _embedding_device() -> str:
    env_device = (os.getenv("EMBEDDING_DEVICE") or "").strip().lower()
    if env_device:
        return env_device
    return "cuda" if _torch_cuda_available() else "cpu"


def _is_meta_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "meta tensor" in msg or "cannot copy out of meta tensor" in msg


def _resolve_torch_dtype():
    if not torch:
        return None
    raw = (os.getenv("EMBEDDING_TORCH_DTYPE") or "").strip().lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32", "full"}:
        return torch.float32
    return torch.float32


def _model_kwargs_for_device(device: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"device_map": None, "low_cpu_mem_usage": False}
    dtype = _resolve_torch_dtype()
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return kwargs


def _parse_redis_connection_string(conn_str: str):
    """
    Lightweight parser for Azure Redis Cache connection strings.

    Returns a dict with host, port, password, and ssl settings. Falls
    back to conservative defaults when parsing fails so the app can still start.
    """
    settings = {
        "host": (getattr(Config.Redis, "HOST", "localhost") or "").strip(),
        "port": getattr(Config.Redis, "PORT", 6380),
        "password": getattr(Config.Redis, "PASSWORD", None) or None,
        "ssl": getattr(Config.Redis, "SSL", True),
    }

    if not conn_str:
        return settings

    conn_str = conn_str.strip()
    try:
        if conn_str.startswith(("redis://", "rediss://")):
            parsed_url = urlparse(conn_str)
            if parsed_url.hostname:
                settings["host"] = parsed_url.hostname.strip()
            if parsed_url.port:
                settings["port"] = int(parsed_url.port)
            if parsed_url.password:
                settings["password"] = parsed_url.password.strip()
            settings["ssl"] = conn_str.startswith("rediss://") or getattr(Config.Redis, "SSL", True)
            return settings

        parts = conn_str.split(",")
        host_port = parts[0]
        if ":" in host_port:
            host, port = host_port.split(":", 1)
            settings["host"] = host.strip() or settings["host"]
            settings["port"] = int(port or settings["port"])
        elif host_port:
            settings["host"] = host_port.strip() or settings["host"]

        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "password":
                settings["password"] = value
            elif key == "ssl":
                settings["ssl"] = value.lower() in {"true", "1", "yes", "on"}
    except Exception:
        return settings

    if isinstance(settings.get("password"), str):
        settings["password"] = settings["password"].strip() or None
    settings["port"] = int(settings.get("port") or getattr(Config.Redis, "PORT", 6380))
    settings["ssl"] = bool(settings.get("ssl"))
    return settings


def _slug(value: str) -> str:
    """Safe, lowercase slug for cache keys."""
    if not value:
        return "default"
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "default"


def _clean_password(password: Optional[str]) -> Optional[str]:
    """Normalize password strings, trimming accidental whitespace and empty values."""
    if password is None:
        return None
    cleaned = password.strip()
    return cleaned or None


def _configure_hf_env() -> None:
    """Set HuggingFace hub timeouts/retries to avoid default short timeouts."""
    if getattr(Config.Model, "DISABLE_HF", False):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        return
    if os.getenv("HF_HUB_READ_TIMEOUT") is None:
        os.environ["HF_HUB_READ_TIMEOUT"] = str(getattr(Config.Model, "HF_HUB_READ_TIMEOUT", 30))
    if os.getenv("HF_HUB_CONNECT_TIMEOUT") is None:
        os.environ["HF_HUB_CONNECT_TIMEOUT"] = str(getattr(Config.Model, "HF_HUB_CONNECT_TIMEOUT", 10))
    if os.getenv("HF_HUB_MAX_RETRIES") is None:
        os.environ["HF_HUB_MAX_RETRIES"] = str(getattr(Config.Model, "HF_HUB_MAX_RETRIES", 3))
    if getattr(Config.Model, "HF_DISABLE_TELEMETRY", True):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if getattr(Config.Model, "TRANSFORMERS_OFFLINE", False):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _coerce_bool(value: Optional[Any], default: Optional[bool] = None) -> bool:
    """Coerce truthy string/env values into a boolean with a sensible default."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


_METRICS_EMBED_SAMPLE_RATE = float(os.getenv("METRICS_EMBED_SAMPLE_RATE", "0.4"))
_METRICS_EMBED_MAX_CHARS = int(os.getenv("METRICS_EMBED_MAX_CHARS", "1200"))
_FIRST_METRICS_REQUEST = True


def _approx_token_count(text: Optional[str]) -> int:
    return len((text or "").split())


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", (left or "").lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", (right or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(overlap) / max(len(union), 1)


def _embedding_similarity_score(left: str, right: str) -> Optional[float]:
    if not left or not right:
        return None
    sample_rate = max(0.0, min(1.0, _METRICS_EMBED_SAMPLE_RATE))
    if sample_rate < 1.0 and random.random() > sample_rate:
        return None
    left_trimmed = (left or "")[:_METRICS_EMBED_MAX_CHARS]
    right_trimmed = (right or "")[:_METRICS_EMBED_MAX_CHARS]
    try:
        model = get_model()
        vectors = model.encode([left_trimmed, right_trimmed], normalize_embeddings=True)
        sim = float(np.dot(vectors[0], vectors[1]))
        return max(0.0, min(1.0, sim))
    except Exception:
        return None


def _semantic_similarity(left: str, right: str) -> float:
    sim = _embedding_similarity_score(left, right)
    if sim is not None:
        return sim
    return _token_overlap_score(left, right)


def _build_namespace(
        subscription_id: str,
        profile_id: str,
        model_name: str = "",
        session_id: Optional[str] = None
) -> str:
    """
    Compose a stable namespace for Redis keys so history/feedback/cache
    never bleed across tenants, profiles, models, or chat sessions.
    """
    parts = [
        _slug(subscription_id),
        _slug(profile_id),
        _slug(model_name or "default"),
    ]
    if session_id:
        parts.append(_slug(session_id))
    return ":".join(parts)


def _load_model_candidates(required_dim: Optional[int] = None) -> SentenceTransformer:
    candidates = []
    _configure_hf_env()
    local_only = bool(getattr(Config.Model, "DISABLE_HF", False))
    for name in getattr(Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", []):
        if name and name not in candidates:
            candidates.append(name)
    if not candidates:
        candidates.append(getattr(Config.Model, "SENTENCE_TRANSFORMERS", "sentence-transformers/all-mpnet-base-v2"))

    last_error = None
    for name in candidates:
        try:
            device = _embedding_device()
            logger.info("Loading sentence transformer model: %s (device=%s)", name, device)
            model_kwargs = _model_kwargs_for_device(device)
            try:
                model = SentenceTransformer(
                    name,
                    device=device,
                    model_kwargs=model_kwargs,
                    local_files_only=local_only,
                )
            except Exception as exc:  # noqa: BLE001
                if device != "cpu" and _is_meta_tensor_error(exc):
                    logger.warning(
                        "Model load on %s failed (%s); retrying on cpu for stability",
                        device,
                        exc,
                    )
                    cpu_kwargs = _model_kwargs_for_device("cpu")
                    model = SentenceTransformer(
                        name,
                        device="cpu",
                        model_kwargs=cpu_kwargs,
                        local_files_only=local_only,
                    )
                else:
                    raise
            dim = model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model '{name}' with dim={dim}")
            if required_dim is None or dim == required_dim:
                return model
            _MODEL_CACHE[dim] = model  # cache for possible future use
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load model '{name}': {e}")
    raise RuntimeError(f"Could not load any sentence transformer model from {candidates}: {last_error}")


def resolve_embedding_model_for_generation(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    mapping = getattr(Config.ModelArbitration, "EMBEDDING_MAP", {}) or {}
    if not mapping:
        return None
    key = str(model_name).lower()
    return mapping.get(key) or mapping.get(model_name) or mapping.get(key.split(":")[0])


def get_model_by_name(model_name: str, required_dim: Optional[int] = None) -> SentenceTransformer:
    """Load a specific embedding model by name with optional dimension enforcement."""
    _configure_hf_env()
    if model_name in _MODEL_BY_NAME:
        model = _MODEL_BY_NAME[model_name]
    else:
        device = _embedding_device()
        model_kwargs = _model_kwargs_for_device(device)
        try:
            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs,
                local_files_only=bool(getattr(Config.Model, "DISABLE_HF", False)),
            )
            _MODEL_BY_NAME[model_name] = model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding model %s failed to load: %s", model_name, exc)
            return get_model(required_dim=required_dim)
    if required_dim is None:
        return model
    dim = model.get_sentence_embedding_dimension()
    if dim == required_dim:
        return model
    logger.warning("Embedding model %s dim=%s does not match required=%s; using default embedder", model_name, dim, required_dim)
    return get_model(required_dim=required_dim)


def get_model(required_dim: Optional[int] = None):
    """Lazy load sentence transformer model."""
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = _load_model_candidates()
        except Exception as exc:
            if getattr(Config.Model, "DISABLE_HF", False):
                logger.warning("Embedding models disabled or unavailable; using fallback embedder: %s", exc)
                _MODEL = _FallbackEmbedder()
            else:
                raise

    if required_dim is None:
        return _MODEL

    dim = _MODEL.get_sentence_embedding_dimension()
    if dim == required_dim:
        return _MODEL

    # Try cached models by dimension
    if required_dim in _MODEL_CACHE:
        logger.info(f"Using cached model with dim {required_dim}")
        return _MODEL_CACHE[required_dim]

    # Try loading a candidate that matches required_dim
    try:
        candidate = _load_model_candidates(required_dim=required_dim)
        _MODEL_CACHE[required_dim] = candidate
        logger.info(f"Loaded fallback model with dim {required_dim}")
        return candidate
    except Exception as exc:
        logger.warning(f"Failed to load fallback model for dim {required_dim}: {exc}; continuing with primary model")
        return _MODEL


class _FallbackEmbedder:
    """Local stub to avoid HuggingFace network calls when disabled."""

    def __init__(self):
        self._dim = int(getattr(Config.Model, "EMBEDDING_DIM", 0) or 0)

    def encode(self, *args, **kwargs):
        raise RuntimeError("Embedding models are disabled (HF connections disabled).")

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def get_cross_encoder():
    """Lazy load cross encoder model."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        if getattr(Config.Model, "DISABLE_HF", False):
            logger.info("Cross-encoder disabled because HuggingFace usage is disabled")
            return None
        if not getattr(Config.Retrieval, "RERANKER_ENABLED", True):
            logger.info("Cross-encoder reranker disabled via configuration")
            return None
        model_name = getattr(Config.Model, "RERANKER_MODEL", "")
        if not model_name:
            logger.info("Cross-encoder reranker disabled via configuration")
            return None
        try:
            _CROSS_ENCODER = CrossEncoder(model_name)
            logger.info("Loaded cross-encoder model: %s", model_name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load cross-encoder '{model_name}', continuing without reranker: {e}")
            _CROSS_ENCODER = None
    return _CROSS_ENCODER


def get_qdrant_client():
    """Lazy load Qdrant client."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        try:
            _QDRANT_CLIENT = QdrantClient(
                url=Config.Qdrant.URL,
                api_key=Config.Qdrant.API,
                timeout=120
            )
            logger.info("Initialized Qdrant client")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    return _QDRANT_CLIENT


def get_redis_client():
    """Lazy init for Redis client with comprehensive error handling."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        try:
            raw_conn = (
                os.getenv("REDIS_URL")
                or os.getenv("REDIS_CONNECTION_STRING")
                or getattr(Config.Redis, "CONNECTION_STRING", "")
            )

            parsed = _parse_redis_connection_string(raw_conn)
            host = (
                os.getenv("REDIS_HOST")
                or parsed.get("host")
                or getattr(Config.Redis, "HOST", "rediscache.redis.cache.windows.net")
                or ""
            ).strip()
            if not host:
                host = getattr(Config.Redis, "HOST", "localhost")

            password = _clean_password(
                os.getenv("REDIS_PASSWORD")
                or parsed.get("password")
                or getattr(Config.Redis, "PASSWORD", "")
            )

            db_idx = int(os.getenv("REDIS_DB", getattr(Config.Redis, "DB", 0)))
            port = int(os.getenv("REDIS_PORT", parsed.get("port", getattr(Config.Redis, "PORT", 6380))))
            ssl_enabled = _coerce_bool(
                os.getenv("REDIS_SSL"),
                parsed.get("ssl", getattr(Config.Redis, "SSL", True)),
            )
            socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "10"))
            socket_connect_timeout = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", str(socket_timeout)))

            client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db_idx,
                ssl=ssl_enabled,
                decode_responses=True,
                socket_connect_timeout=socket_connect_timeout,
                socket_timeout=socket_timeout,
            )
            client.ping()
            _REDIS_CLIENT = client
            logger.info("Redis connected via SSL=%s to %s:%s/%s", ssl_enabled, host, port, db_idx)

        except Exception as e:
            logger.error(f" Redis connection failed: {e}")
            logger.warning("RAG system will work WITHOUT caching/history persistence. "
                           "Check REDIS_CONNECTION_STRING/REDIS_PASSWORD and network reachability.")
            _REDIS_CLIENT = None  # Graceful degradation

    return _REDIS_CLIENT


def configure_gemini():
    """Configure Gemini API with proper error handling."""
    try:
        api_key = getattr(Config.Model, "GEMINI_API_KEY", None) or getattr(Config.Gemini, "GEMINI_API_KEY", None)
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in configuration")
        get_genai_client(api_key)
        logger.info("Gemini API configured successfully")
        return api_key
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        raise


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict
    source: str | None = None
    method: str = "dense"


@dataclass
class ConversationTurn:
    """Data class for conversation history."""
    user_message: str
    assistant_response: str
    timestamp: float


@dataclass
class ProfileContextSnapshot:
    """Lightweight snapshot of profile-specific vocabulary and hints."""
    top_keywords: List[str]
    document_hints: List[str]
    total_chunks: int
    last_updated: float


class ChatFeedbackMemory:
    """Stores compact Q/A feedback to steer future responses."""

    def __init__(
            self,
            max_items: int = 12,
            redis_client: Optional[redis.Redis] = None,
            ttl_seconds: int = REDIS_MEMORY_TTL
    ):
        self.max_items = max_items
        self.memories: Dict[Tuple[str, str], deque] = {}
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _cache_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:feedback:{namespace}:{user_id}"

    def _load_from_cache(self, namespace: str, user_id: str):
        key = (namespace, user_id)
        if key in self.memories or not self.redis:
            return
        try:
            cached = self.redis.get(self._cache_key(namespace, user_id))
        except Exception as exc:
            logger.warning(f"Failed to read feedback memory from Redis: {exc}")
            return
        if not cached:
            return
        try:
            items = json.loads(cached)
            if isinstance(items, list):
                self.memories[key] = deque(items, maxlen=self.max_items)
        except Exception as exc:
            logger.warning(f"Failed to parse feedback memory from Redis: {exc}")

    def _persist(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            payload = list(self.memories.get((namespace, user_id), []))
            self.redis.setex(self._cache_key(namespace, user_id), self.ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Failed to persist feedback memory to Redis: {exc}")

    def clear(self, namespace: str, user_id: str):
        """Remove cached feedback so new sessions start fresh."""
        key = (namespace, user_id)
        if key in self.memories:
            self.memories[key].clear()
        if self.redis:
            try:
                self.redis.delete(self._cache_key(namespace, user_id))
            except Exception as exc:
                logger.warning(f"Failed to clear feedback memory from Redis: {exc}")

    def add_feedback(self, namespace: str, user_id: str, query: str, answer: str, sources: List[Dict[str, Any]]):
        self._load_from_cache(namespace, user_id)
        key = (namespace, user_id)
        if key not in self.memories:
            self.memories[key] = deque(maxlen=self.max_items)
        src_names = [s.get("source_name") for s in (sources or []) if s.get("source_name")]
        self.memories[key].append({
            "q": query.strip(),
            "a": answer.strip(),
            "sources": src_names,
            "ts": time.time()
        })
        self._persist(namespace, user_id)

    def build_feedback_context(self, namespace: str, user_id: str, limit: int = 5) -> str:
        self._load_from_cache(namespace, user_id)
        key = (namespace, user_id)
        if key not in self.memories or not self.memories[key]:
            return ""
        recent = list(self.memories[key])[-limit:]
        lines = []
        for idx, item in enumerate(recent, 1):
            source_hint = f" | sources: {', '.join(item['sources'][:3])}" if item.get("sources") else ""
            lines.append(f"{idx}) Q: {item['q']} | A: {item['a'][:180]}{source_hint}")
        return "RECENT CHAT FEEDBACK (reuse tone/precision):\n" + "\n".join(lines) + "\n"


class ConversationHistory:
    """Manages conversation history with a sliding window."""

    def __init__(
            self,
            max_turns: int = 3,
            redis_client: Optional[redis.Redis] = None,
            ttl_seconds: int = REDIS_MEMORY_TTL
    ):
        self.max_turns = max_turns
        self.histories: Dict[Tuple[str, str], deque] = {}
        self.recent_docs: Dict[Tuple[str, str], deque] = {}
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _history_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:history:{namespace}:{user_id}"

    def _docs_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:recent_docs:{namespace}:{user_id}"

    def _load_history(self, namespace: str, user_id: str):
        key = (namespace, user_id)
        if key in self.histories or not self.redis:
            return
        try:
            cached = self.redis.get(self._history_key(namespace, user_id))
        except Exception as exc:
            logger.warning(f"Failed to read conversation history from Redis: {exc}")
            return
        if not cached:
            return
        try:
            items = json.loads(cached)
            if isinstance(items, list):
                turns = deque(maxlen=self.max_turns)
                for item in items[-self.max_turns:]:
                    if not isinstance(item, dict):
                        continue
                    turns.append(ConversationTurn(
                        user_message=item.get("user_message", ""),
                        assistant_response=item.get("assistant_response", ""),
                        timestamp=float(item.get("timestamp", 0.0))
                    ))
                self.histories[key] = turns
        except Exception as exc:
            logger.warning(f"Failed to parse conversation history from Redis: {exc}")

    def _load_recent_docs(self, namespace: str, user_id: str):
        key = (namespace, user_id)
        if key in self.recent_docs or not self.redis:
            return
        try:
            cached = self.redis.get(self._docs_key(namespace, user_id))
        except Exception as exc:
            logger.warning(f"Failed to read recent docs from Redis: {exc}")
            return
        if not cached:
            return
        try:
            items = json.loads(cached)
            if isinstance(items, list):
                self.recent_docs[key] = deque(items, maxlen=10)
        except Exception as exc:
            logger.warning(f"Failed to parse recent docs from Redis: {exc}")

    def _persist_history(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            payload = []
            for turn in self.histories.get((namespace, user_id), []):
                payload.append({
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "timestamp": turn.timestamp
                })
            self.redis.setex(self._history_key(namespace, user_id), self.ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Failed to persist conversation history to Redis: {exc}")

    def _persist_recent_docs(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            payload = list(self.recent_docs.get((namespace, user_id), []))
            self.redis.setex(self._docs_key(namespace, user_id), self.ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Failed to persist recent docs to Redis: {exc}")

    def add_turn(self, namespace: str, user_id: str, user_message: str, assistant_response: str):
        """Add a conversation turn to history."""
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories:
            self.histories[key] = deque(maxlen=self.max_turns)

        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=time.time()
        )
        self.histories[key].append(turn)
        self._persist_history(namespace, user_id)

    def add_sources(self, namespace: str, user_id: str, doc_ids: List[str]):
        """Track recently used document IDs for recency-based boosting."""
        key = (namespace, user_id)
        self._load_recent_docs(namespace, user_id)
        if key not in self.recent_docs:
            self.recent_docs[key] = deque(maxlen=10)
        for doc_id in doc_ids:
            if doc_id:
                self.recent_docs[key].append(doc_id)
        self._persist_recent_docs(namespace, user_id)

    def get_context(
            self,
            namespace: str,
            user_id: str,
            max_turns: int = 2,
            include_assistant: bool = True,
            max_chars: int = 1200,
    ) -> str:
        """Get recent conversation context as formatted string."""
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories or not self.histories[key]:
            return ""

        context_parts = []
        recent_turns = list(self.histories[key])[-max_turns:]

        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            if include_assistant and turn.assistant_response:
                trimmed = " ".join(turn.assistant_response.split())
                if len(trimmed) > 280:
                    trimmed = trimmed[:277].rsplit(" ", 1)[0] + "..."
                context_parts.append(f"Assistant: {trimmed}")

        context = "\n".join(context_parts)
        if len(context) > max_chars:
            return context[-max_chars:]
        return context

    def get_recent_doc_ids(self, namespace: str, user_id: str) -> List[str]:
        """Return a list of recently cited document IDs for this user."""
        key = (namespace, user_id)
        self._load_recent_docs(namespace, user_id)
        if key not in self.recent_docs:
            return []
        return list(self.recent_docs[key])

    def get_last_user_message(self, namespace: str, user_id: str) -> str:
        """Return the most recent user message in history."""
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories or not self.histories[key]:
            return ""
        return (self.histories[key][-1].user_message or "").strip()

    def clear_history(self, namespace: str, user_id: str):
        """Clear conversation history for a user."""
        key = (namespace, user_id)
        if key in self.histories:
            self.histories[key].clear()
        if key in self.recent_docs:
            self.recent_docs[key].clear()
        if self.redis:
            try:
                self.redis.delete(self._history_key(namespace, user_id))
                self.redis.delete(self._docs_key(namespace, user_id))
            except Exception as exc:
                logger.warning(f"Failed to clear Redis memory for user {user_id}: {exc}")


class TextPreprocessor:
    """Handles text preprocessing for consistent tokenization."""

    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'about', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'that', 'this', 'these', 'those', 'it', 'its', 'their', 'them', 'they', 'you',
            'your', 'we', 'our', 'us', 'i', 'me', 'my', 'he', 'she', 'his', 'her', 'not',
            'do', 'does', 'did', 'done', 'have', 'has', 'had', 'will', 'would', 'can', 'could',
            'should', 'may', 'might', 'must', 'if', 'then', 'so', 'than', 'such', 'also'
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by lowercasing and removing extra whitespace."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with lemmatization and stopword removal for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        lemmatized = []
        for token in tokens:
            if token.endswith('ing'):
                lemmatized.append(token[:-3])
            elif token.endswith('ed'):
                lemmatized.append(token[:-2])
            elif token.endswith('s') and len(token) > 3:
                lemmatized.append(token[:-1])
            else:
                lemmatized.append(token)

        return lemmatized


class GreetingHandler:
    """Handles greeting and farewell detection."""

    GREETINGS = {
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'greetings', 'howdy', 'what\'s up', 'how are you', 'how do you do',
        'nice to meet you', 'good day', 'hiya', 'sup', 'yo', 'salutations',
        'welcome', 'hola', 'bonjour', 'namaste', 'aloha', 'hii'
    }

    FAREWELLS = {
        'bye', 'goodbye', 'good bye', 'see you', 'see ya', 'farewell',
        'adieu', 'cheerio', 'tata', 'ta ta', 'catch you later', 'later',
        'take care', 'until next time', 'signing off', 'gotta go', 'gtg',
        'peace out', 'ciao', 'au revoir', 'sayonara', 'hasta la vista',
        'talk to you later', 'ttyl', 'see you soon', 'see you around',
        'good night', 'goodnight', 'have a good day', 'have a great day',
        'thanks for your help', 'that\'s all', 'i\'m done', 'end chat',
        'quit', 'exit', 'close', 'finish', 'terminate', 'stop'
    }

    POSITIVE_FEEDBACK = {
        'thanks', 'thank you', 'thanks a lot', 'thank you so much',
        'appreciate it', 'much appreciated', 'thx', 'ty', 'tysm',
        'great answer', 'good answer', 'very good', 'awesome', 'perfect',
        'well done', 'nice', 'excellent'
    }

    NEGATIVE_FEEDBACK = {
        'bad answer', 'wrong answer', 'not right', 'not correct', 'incorrect',
        'not accurate', 'this is bad', 'this is wrong', 'not good',
        'doesn\'t make sense', 'does not make sense', 'poor answer',
        'not helpful', 'this is not right response', 'this is not right',
        'this is not correct', 'this is not accurate', 'bad response'
    }

    QUESTION_CUES = {
        'can you', 'could you', 'would you', 'should you', 'help me', 'help with', 'help',
        'how', 'what', 'why', 'when', 'where', 'which', 'who', 'can u', 'could u',
        'tell me', 'show me', 'explain', 'walk me through', 'guide me', 'need help'
    }

    FILLER_WORDS = {'there', 'team', 'docwain', 'assistant', 'bot', 'buddy', 'friend', 'please'}

    @staticmethod
    def _clean_message(message: str) -> str:
        cleaned = re.sub(r"[^\w\s?.!]", " ", message.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _strip_phrases(cls, message: str, phrases: Set[str]) -> str:
        cleaned = cls._clean_message(message)
        for phrase in sorted(phrases, key=len, reverse=True):
            cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _has_question_cue(cls, message: str) -> bool:
        if "?" in message:
            return True
        normalized = cls._clean_message(message)
        return any(cue in normalized for cue in cls.QUESTION_CUES)

    @classmethod
    def _has_follow_on_content(cls, message: str, phrases: Set[str]) -> bool:
        """
        Check if there is meaningful content beyond the matched phrase.
        Prevents treating greetings/thanks with real questions as small talk.
        """
        remainder = cls._strip_phrases(message, phrases)
        if not remainder:
            return False

        if cls._has_question_cue(remainder):
            return True

        tokens = [t for t in remainder.split() if t not in cls.FILLER_WORDS]
        return len(tokens) >= 2

    @classmethod
    def is_greeting(cls, message: str) -> bool:
        """Check if message is a greeting."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(g) for g in cls.GREETINGS) + r')\b'
        if not re.search(pattern, normalized):
            return False

        if cls._has_follow_on_content(message, cls.GREETINGS):
            return False

        return True

    @classmethod
    def is_farewell(cls, message: str) -> bool:
        """Check if message is a farewell."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.FAREWELLS) + r')\b'
        if not re.search(pattern, normalized):
            return False

        if cls._has_follow_on_content(message, cls.FAREWELLS):
            return False

        return True

    @classmethod
    def is_positive_feedback(cls, message: str) -> bool:
        """Check if message is positive feedback."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.POSITIVE_FEEDBACK) + r')\b'
        if not re.search(pattern, normalized):
            return False

        if cls._has_follow_on_content(message, cls.POSITIVE_FEEDBACK):
            return False

        return True

    @classmethod
    def is_negative_feedback(cls, message: str) -> bool:
        """Check if message is negative feedback."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.NEGATIVE_FEEDBACK) + r')\b'
        if not re.search(pattern, normalized):
            return False

        return True


class OllamaClient:
    """Handles local Ollama model calls with controlled generation to reduce hallucinations."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        if not self.model_name:
            raise ValueError("OLLAMA_MODEL environment variable is not set")
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
        backoff: float = 1.0
    ) -> str:
        metrics_store = get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        temperature = getattr(Config.LLM, "TEMPERATURE", 0.2)
        top_p = getattr(Config.LLM, "TOP_P", 0.85)
        max_tokens = getattr(Config.LLM, "MAX_TOKENS", 2048)
        for attempt in range(1, max_retries + 1):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": temperature,   #  LOWER = less hallucination
                        "top_p": top_p,
                        "top_k": 40,
                        "repeat_penalty": 1.1,
                        "num_ctx": 4096,
                        "num_predict": max_tokens,
                    }
                )

                text = (response.get("response") or "").strip()

                if not text:
                    logger.warning(f"Ollama returned empty response: {response}")
                    return "I don’t have enough information in the documents to answer that."
                if metrics_store.available:
                    latency_ms = (time.time() - request_started) * 1000
                    metrics_store.record(
                        values={"llm_latency_ms": latency_ms},
                        histograms={"llm_latency_ms": latency_ms},
                        model_id=self.model_name,
                    )
                    if attempt > 1:
                        metrics_store.record(
                            counters={"llm_retry_count": attempt - 1},
                            model_id=self.model_name,
                        )
                return text

            except Exception as e:
                logger.warning(f"Ollama attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error(f"All Ollama retries failed")
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "I’m unable to answer that based on the available information."


class GeminiClient:
    def __init__(self, model_name: Optional[str] = None):
        self.api_key = configure_gemini()
        self.model_name = model_name or Config.Model.GEMINI_MODEL_NAME
        if not self.model_name:
            raise ValueError("Gemini model name is not configured")
        self.generation_config = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.3),
            "top_p": getattr(Config.LLM, "TOP_P", 0.95),
            "top_k": 40,
            "max_output_tokens": getattr(Config.LLM, "MAX_TOKENS", 2048),
        }
        logger.info(f"Initialized GeminiClient with model: {self.model_name}")

    def generate(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: float = 1.0
    ) -> str:
        """Generate response with retry logic and robust parsing."""
        metrics_store = get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        for attempt in range(1, max_retries + 1):
            try:
                text, response = generate_text(
                    api_key=self.api_key,
                    model=self.model_name,
                    prompt=prompt,
                    generation_config=self.generation_config,
                )
                if text:
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                        if attempt > 1:
                            metrics_store.record(
                                counters={"llm_retry_count": attempt - 1},
                                model_id=self.model_name,
                            )
                    return text
                logger.warning(f"No text in response: {response}")
                return "I apologize, but I couldn't generate a proper response."
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "I apologize, but I encountered an error generating a response."


class OpenAICompatibleClient:
    """Handles OpenAI-compatible local LLM endpoints (chat completions)."""

    def __init__(
            self,
            model_name: Optional[str] = None,
            endpoint: Optional[str] = None,
            api_key: Optional[str] = None
    ):
        self.endpoint = endpoint or os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        if self.endpoint.rstrip("/").endswith("/v1"):
            self.endpoint = self.endpoint.rstrip("/") + "/chat/completions"
        self.model_name = model_name or os.getenv("LOCAL_LLM_MODEL", "local-model")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "")
        self.temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", str(getattr(Config.LLM, "TEMPERATURE", 0.0))))
        self.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", str(getattr(Config.LLM, "MAX_TOKENS", 2048))))
        self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "30"))
        logger.info("Initialized OpenAI-compatible client at %s with model %s", self.endpoint, self.model_name)

    def warm_up(self):
        try:
            self.generate("ping", max_retries=1, backoff=0.0)
        except Exception as exc:
            logger.warning("Local LLM warm-up failed (continuing): %s", exc)

    def generate(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: float = 1.0
    ) -> str:
        metrics_store = get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, max_retries + 1):
            try:
                req = request.Request(self.endpoint, data=data, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                response = json.loads(body)
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or choice.get("text") or ""
                text = text.strip()
                if text:
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                        if attempt > 1:
                            metrics_store.record(
                                counters={"llm_retry_count": attempt - 1},
                                model_id=self.model_name,
                            )
                    return text
                logger.warning("No text in local LLM response: %s", response)
                return "I apologize, but I couldn't generate a proper response."
            except (HTTPError, URLError, ValueError) as e:
                logger.warning("Local LLM attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error("All local LLM retry attempts failed: %s", e)
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "I apologize, but I encountered an error generating a response."


class QueryReformulator:
    """Reformulates conversational queries into clear, concise search queries."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def reformulate(self, query: str, conversation_context: str = "") -> str:
        """Reformulate query using LLM to make it more search-friendly."""
        if len(query.split()) <= 5 and not conversation_context:
            return query

        prompt = f"""You are a query reformulation assistant. Convert the user's conversational question into a clear, concise search query optimized for semantic search.

RULES:
1. Extract the core information need
2. Remove filler words and conversational elements
3. Keep domain-specific terms and technical vocabulary
4. Make it 3-10 words maximum
5. If conversation context is provided, resolve pronouns and references
6. Output ONLY the reformulated query, nothing else

{f"CONVERSATION CONTEXT:\\n{conversation_context}\\n" if conversation_context else ""}
USER QUERY: {query}

REFORMULATED QUERY:"""

        try:
            reformulated = self.llm_client.generate(
                prompt,
                max_retries=2,
                backoff=0.5
            )

            reformulated = reformulated.strip().strip('"\'')

            if 2 <= len(reformulated.split()) <= 15 and reformulated.lower() != query.lower():
                logger.info(f"Reformulated: '{query}' � '{reformulated}'")
                return reformulated
            else:
                return query

        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return query


class QdrantRetriever:
    """Handles retrieval from Qdrant using native search functionality."""

    def __init__(self, client: QdrantClient, model: SentenceTransformer):
        self.client = client
        self.model = model
        self.preprocessor = TextPreprocessor()
        self.profile_context_cache: Dict[Tuple[str, str], ProfileContextSnapshot] = {}
        self.collection_dims: Dict[str, int] = {}
        self._payload_index_cache: Dict[str, Set[str]] = {}
        self._hash_vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    @staticmethod
    def _ensure_profile(profile_id: Optional[str]):
        if not profile_id:
            logger.error("Security: retrieval attempted without profile_id filter")
            raise ValueError("profile_id is required for retrieval to enforce isolation")

    def _build_filter(
        self,
        profile_id: str,
        document_ids: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
        section_titles: Optional[List[str]] = None,
        section_paths: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        min_confidence: Optional[float] = None,
    ) -> Filter:
        self._ensure_profile(profile_id)
        conditions = [
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        ]
        if document_ids:
            conditions.append(FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids])))
        if source_files:
            conditions.append(FieldCondition(key="source_file", match=MatchAny(any=[str(s) for s in source_files])))
        if doc_types:
            conditions.append(FieldCondition(key="doc_type", match=MatchAny(any=[str(d) for d in doc_types])))
        if section_titles:
            conditions.append(FieldCondition(key="section_title", match=MatchAny(any=[str(s) for s in section_titles])))
        if section_paths:
            conditions.append(FieldCondition(key="section_path", match=MatchAny(any=[str(s) for s in section_paths])))
        if page_numbers:
            values = []
            for page in page_numbers:
                try:
                    values.append(int(page))
                except Exception:
                    continue
                values.append(str(page))
            if values:
                conditions.append(FieldCondition(key="page", match=MatchAny(any=values)))
        if min_confidence is not None:
            try:
                min_conf = float(min_confidence)
                conditions.append(FieldCondition(key="ocr_confidence", range=Range(gte=min_conf)))
            except Exception:
                pass
        return Filter(must=conditions)

    def _build_sparse_query(self, query: str) -> SparseVector:
        matrix = self._hash_vectorizer.transform([query])
        coo = matrix.tocoo()
        return SparseVector(indices=coo.col.tolist(), values=coo.data.astype(np.float32).tolist())

    def _list_collection_names(self) -> List[str]:
        try:
            collections = self.client.get_collections().collections
            return [col.name for col in collections]
        except Exception as exc:
            logger.warning("Could not list Qdrant collections: %s", exc)
            return []

    @staticmethod
    def _iter_filter_conditions(query_filter: Any):
        if query_filter is None:
            return
        if isinstance(query_filter, Filter):
            for attr in ("must", "should", "must_not"):
                for cond in getattr(query_filter, attr, []) or []:
                    yield from QdrantRetriever._iter_filter_conditions(cond)
            return
        if isinstance(query_filter, FieldCondition):
            yield query_filter
            return
        if isinstance(query_filter, dict):
            key = query_filter.get("key")
            if key:
                yield query_filter
            for attr in ("must", "should", "must_not"):
                for cond in query_filter.get(attr, []) or []:
                    yield from QdrantRetriever._iter_filter_conditions(cond)
            nested_filter = query_filter.get("filter")
            if nested_filter:
                yield from QdrantRetriever._iter_filter_conditions(nested_filter)

    def _extract_filter_keys(self, query_filter: Any) -> Set[str]:
        keys: Set[str] = set()
        for cond in self._iter_filter_conditions(query_filter):
            key = getattr(cond, "key", None)
            if key is None and isinstance(cond, dict):
                key = cond.get("key")
            if key:
                keys.add(str(key))
        return keys

    def _ensure_filter_indexes(self, collection_name: str, query_filter: Any) -> None:
        keys = self._extract_filter_keys(query_filter)
        if not keys:
            return
        fields_to_index = {
            key for key in keys if key in PAYLOAD_INDEX_FIELDS or key in PAYLOAD_INDEX_SCHEMAS
        }
        if not fields_to_index:
            return

        cached = self._payload_index_cache.setdefault(collection_name, set())
        for field in sorted(fields_to_index):
            if field in cached:
                continue
            schema = PAYLOAD_INDEX_SCHEMAS.get(field, "keyword")
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "already exists" in msg or "index exists" in msg:
                    cached.add(field)
                    continue
                logger.warning(
                    "Payload index ensure failed for %s on %s: %s", field, collection_name, exc
                )
                continue
            cached.add(field)

    def _ensure_collection_exists(self, collection_name: str) -> None:
        try:
            self.client.get_collection(collection_name)
            return
        except Exception as exc:
            available = self._list_collection_names()
            if collection_name not in available:
                logger.error("Qdrant collection not found: %s (available=%s)", collection_name, available)
                raise QdrantCollectionNotFoundError(collection_name, available) from exc
            logger.error("Qdrant collection lookup failed for '%s': %s", collection_name, exc)
            raise ValueError(f"qdrant_collection_unavailable: collection '{collection_name}'") from exc

    def ensure_collection_exists(self, collection_name: str) -> None:
        self._ensure_collection_exists(collection_name)

    @staticmethod
    def _points_to_chunks(results, method: str) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        points = []
        if hasattr(results, "points"):
            points = results.points or []
        elif isinstance(results, tuple) and results:
            points = results[0] or []

        for pt in points:
            payload = pt.payload or {}
            source = payload.get("source_file") or payload.get("source") or ""
            chunks.append(
                RetrievedChunk(
                    id=str(pt.id),
                    text=payload.get("text", ""),
                    score=float(pt.score),
                    metadata=payload,
                    source=source or None,
                    method=method,
                )
            )
        return chunks

    @staticmethod
    def _rrf_merge(dense: List[RetrievedChunk], sparse: List[RetrievedChunk], k: int = 60) -> List[RetrievedChunk]:
        scored: Dict[str, RetrievedChunk] = {}
        weights = getattr(Config.Retrieval, "HYBRID_WEIGHTS", {"dense": 0.6, "sparse": 0.4})
        dense_w = float(weights.get("dense", 0.6))
        sparse_w = float(weights.get("sparse", 0.4))

        for rank, chunk in enumerate(dense, start=1):
            score = dense_w * (1.0 / (k + rank))
            scored.setdefault(chunk.id, chunk)
            scored[chunk.id].score = scored[chunk.id].score + score if scored[chunk.id].score else score

        for rank, chunk in enumerate(sparse, start=1):
            score = sparse_w * (1.0 / (k + rank))
            if chunk.id in scored:
                scored[chunk.id].score += score
                scored[chunk.id].metadata.setdefault("methods", []).append("sparse")
            else:
                chunk.score = score
                scored[chunk.id] = chunk

        return sorted(scored.values(), key=lambda c: float(c.score), reverse=True)

    def run_search(
            self,
            collection_name: str,
            query_vector: List[float],
            query_filter: Optional[dict] = None,
            limit: int = 50,
            vector_name: str = "content_vector",
            score_threshold: Optional[float] = None
    ):
        """Execute a vector search against Qdrant using query_points."""
        self._ensure_collection_exists(collection_name)
        try:
            kwargs = dict(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Add query_filter only if provided (not as 'filter')
            if query_filter is not None:
                self._ensure_filter_indexes(collection_name, query_filter)
                kwargs["query_filter"] = query_filter

            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold

            results = self.client.query_points(**kwargs)
            return results
        except QdrantCollectionNotFoundError:
            raise
        except Exception as e:
            logger.error("Qdrant query_points error: %s", e, exc_info=True)
            return None

    def get_collection_vector_dim(self, collection_name: str) -> Optional[int]:
        """Fetch and cache the expected vector dimension for a collection."""
        if collection_name in self.collection_dims:
            return self.collection_dims[collection_name]
        try:
            self._ensure_collection_exists(collection_name)
            info = self.client.get_collection(collection_name)
            cfg = getattr(info, "config", None) or {}
            params = getattr(cfg, "params", None) or {}
            vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
            dim = None
            if hasattr(vectors, "size"):
                dim = vectors.size
            elif isinstance(vectors, dict):
                if "size" in vectors:
                    dim = vectors.get("size")
                elif "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
                    dim = vectors["content_vector"].get("size")
            if dim is None:
                dim = getattr(Config.Model, "EMBEDDING_DIM", None) or 768
            dim = int(dim)
            self.collection_dims[collection_name] = dim
            logger.info(f"Collection '{collection_name}' expects dim={dim}")
            return dim
        except QdrantCollectionNotFoundError:
            raise
        except Exception as e:
            logger.error("Could not fetch vector dim for collection '%s': %s", collection_name, e, exc_info=True)
            raise ValueError(f"qdrant_collection_unavailable: collection '{collection_name}'") from e

    def retrieve(
            self,
            collection_name: str,
            query: str,
            filter_profile: str = None,
            top_k: int = 50,
            score_threshold: float = 0.10,  # ✅ CHANGED FROM 0.15
            document_ids: Optional[List[str]] = None,
            source_files: Optional[List[str]] = None,
            doc_types: Optional[List[str]] = None,
            section_titles: Optional[List[str]] = None,
            section_paths: Optional[List[str]] = None,
            page_numbers: Optional[List[int]] = None,
            min_confidence: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using Qdrant's native search."""
        try:
            self._ensure_profile(filter_profile)
            target_dim = self.get_collection_vector_dim(collection_name)
            model = get_model(required_dim=target_dim)
            q_dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
            if target_dim and q_dim and target_dim != q_dim:
                logger.warning(f"Embedding dim {q_dim} does not match collection dim {target_dim}; using model regardless")
            query_vector = model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()
            if target_dim and len(query_vector) != target_dim:
                raise ValueError(f"Embedding dim {len(query_vector)} does not match collection dim {target_dim}")
        except QdrantCollectionNotFoundError:
            raise
        except Exception as err:
            logger.warning("Dense embedding failed; falling back to sparse retrieval: %s", err)
            return self.sparse_retrieve(
                collection_name=collection_name,
                query=query,
                profile_id=filter_profile,
                top_k=max(10, int(top_k * 0.6)),
                filters={
                    "document_ids": document_ids,
                    "source_files": source_files,
                    "doc_types": doc_types,
                    "section_titles": section_titles,
                    "section_paths": section_paths,
                    "page_numbers": page_numbers,
                    "min_confidence": min_confidence,
                },
            )

        logger.info(f"Searching collection '{collection_name}' for query: {query[:100]}")

        query_filter = self._build_filter(
            str(filter_profile),
            document_ids=document_ids,
            source_files=source_files,
            doc_types=doc_types,
            section_titles=section_titles,
            section_paths=section_paths,
            page_numbers=page_numbers,
            min_confidence=min_confidence,
        )

        results = self.run_search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="content_vector",
            score_threshold=score_threshold
        )

        if not results or not getattr(results, "points", []):
            logger.warning(f"No results found in collection '{collection_name}'")
            return []

        points = results.points or []
        logger.info("Qdrant returned %d hits", len(points))
        logger.info("Top scores: %s", [p.score for p in points[:3]])

        chunks: List[RetrievedChunk] = []
        for pt in points:
            payload = pt.payload or {}
            text = payload.get("text", "")
            snippet = text[:120].replace("\n", " ")
            logger.debug("Hit score=%.4f snippet=%s", pt.score, snippet)
            source = payload.get("source_file") or payload.get("source") or "unknown"
            chunk = RetrievedChunk(
                id=str(pt.id),
                text=text,
                score=float(pt.score),
                source=source,
                metadata=payload
            )
            chunks.append(chunk)

        return chunks

    def hybrid_retrieve(
            self,
            collection_name: str,
            query: str,
            profile_id: str,
            top_k: int = 50,
            document_ids: Optional[List[str]] = None,
            source_files: Optional[List[str]] = None,
            doc_types: Optional[List[str]] = None,
            section_titles: Optional[List[str]] = None,
            section_paths: Optional[List[str]] = None,
            page_numbers: Optional[List[int]] = None,
            min_confidence: Optional[float] = None,
            score_threshold: float = 0.05,
    ) -> List[RetrievedChunk]:
        """Hybrid dense + sparse retrieval with reciprocal rank fusion."""
        self._ensure_profile(profile_id)
        query_filter = self._build_filter(
            profile_id,
            document_ids=document_ids,
            source_files=source_files,
            doc_types=doc_types,
            section_titles=section_titles,
            section_paths=section_paths,
            page_numbers=page_numbers,
            min_confidence=min_confidence,
        )

        dense_vector = None
        try:
            target_dim = self.get_collection_vector_dim(collection_name)
            model = get_model(required_dim=target_dim)
            dense_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32).tolist()
            if target_dim and len(dense_vector) != target_dim:
                logger.error(
                    "Embedding dim %s does not match collection dim %s for collection '%s'",
                    len(dense_vector),
                    target_dim,
                    collection_name,
                )
                dense_vector = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dense embedding unavailable; using sparse-only retrieval: %s", exc)

        dense_results = None
        if dense_vector is not None:
            dense_results = self.run_search(
                collection_name=collection_name,
                query_vector=dense_vector,
                query_filter=query_filter,
                limit=top_k,
                vector_name="content_vector",
                score_threshold=score_threshold,
            )
        sparse_results = None
        try:
            sparse_vector = self._build_sparse_query(query)
            sparse_results = self.run_search(
                collection_name=collection_name,
                query_vector=sparse_vector,
                query_filter=query_filter,
                limit=top_k,
                vector_name="keywords_vector",
                score_threshold=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sparse retrieval failed: %s", exc)

        dense_chunks = self._points_to_chunks(dense_results, method="dense")
        sparse_chunks = self._points_to_chunks(sparse_results, method="sparse")

        if not dense_chunks and not sparse_chunks:
            return self._lexical_fallback_retrieve(
                collection_name=collection_name,
                query=query,
                profile_id=profile_id,
                filters={
                    "document_ids": document_ids,
                    "source_files": source_files,
                    "doc_types": doc_types,
                    "section_titles": section_titles,
                    "section_paths": section_paths,
                    "page_numbers": page_numbers,
                    "min_confidence": min_confidence,
                },
                top_k=top_k,
            )

        merged = self._rrf_merge(dense_chunks, sparse_chunks)
        return merged[:top_k]

    def sparse_retrieve(
            self,
            collection_name: str,
            query: str,
            profile_id: str,
            top_k: int = 30,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Lexical-style retrieval using sparse vectors."""
        filters = filters or {}
        query_filter = self._build_filter(
            profile_id,
            document_ids=filters.get("document_ids"),
            source_files=filters.get("source_files"),
            doc_types=filters.get("doc_types"),
            section_titles=filters.get("section_titles"),
            section_paths=filters.get("section_paths"),
            page_numbers=filters.get("page_numbers"),
            min_confidence=filters.get("min_confidence"),
        )
        sparse_vector = self._build_sparse_query(query)
        results = self.run_search(
            collection_name=collection_name,
            query_vector=sparse_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="keywords_vector",
            score_threshold=None,
        )
        chunks = self._points_to_chunks(results, method="sparse")
        if chunks:
            return chunks
        return self._lexical_fallback_retrieve(
            collection_name=collection_name,
            query=query,
            profile_id=profile_id,
            filters=filters,
            top_k=top_k,
        )

    def metadata_retrieve(
            self,
            collection_name: str,
            profile_id: str,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 20,
    ) -> List[RetrievedChunk]:
        """Retrieve chunks using only metadata filters (no vector scoring)."""
        filters = filters or {}
        query_filter = self._build_filter(
            profile_id,
            document_ids=filters.get("document_ids"),
            source_files=filters.get("source_files"),
            doc_types=filters.get("doc_types"),
            section_titles=filters.get("section_titles"),
            section_paths=filters.get("section_paths"),
            page_numbers=filters.get("page_numbers"),
            min_confidence=filters.get("min_confidence"),
        )
        self._ensure_filter_indexes(collection_name, query_filter)
        try:
            scroll = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=limit,
            )
        except Exception as exc:
            logger.debug("Metadata-only retrieval failed: %s", exc)
            return []

        points = []
        if hasattr(scroll, "points"):
            points = scroll.points or []
        elif isinstance(scroll, tuple) and scroll:
            points = scroll[0] or []

        chunks: List[RetrievedChunk] = []
        for pt in points:
            payload = pt.payload or {}
            chunks.append(
                RetrievedChunk(
                    id=str(pt.id),
                    text=payload.get("text", ""),
                    score=0.05,
                    metadata=payload,
                    source=payload.get("source_file") or payload.get("source"),
                    method="metadata",
                )
            )
        return chunks

    def _lexical_fallback_retrieve(
            self,
            collection_name: str,
            query: str,
            profile_id: str,
            filters: Optional[Dict[str, Any]] = None,
            top_k: int = 20,
    ) -> List[RetrievedChunk]:
        """Fallback retrieval using metadata scroll + token overlap ranking."""
        filters = filters or {}
        query_filter = self._build_filter(
            profile_id,
            document_ids=filters.get("document_ids"),
            source_files=filters.get("source_files"),
            doc_types=filters.get("doc_types"),
            section_titles=filters.get("section_titles"),
            page_numbers=filters.get("page_numbers"),
            min_confidence=filters.get("min_confidence"),
        )
        self._ensure_filter_indexes(collection_name, query_filter)
        limit = int(getattr(Config.Retrieval, "METADATA_FALLBACK_LIMIT", 200))
        min_score = float(getattr(Config.Retrieval, "METADATA_FALLBACK_MIN_SCORE", 0.02))
        try:
            scroll = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=limit,
            )
        except Exception as exc:
            logger.debug("Metadata fallback scroll failed: %s", exc)
            return []

        points = []
        if hasattr(scroll, "points"):
            points = scroll.points or []
        elif isinstance(scroll, tuple) and scroll:
            points = scroll[0] or []

        query_tokens = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower()))
        if not query_tokens:
            return []

        ranked: List[RetrievedChunk] = []
        for pt in points:
            payload = pt.payload or {}
            text = payload.get("text", "") or ""
            tokens = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
            if not tokens:
                continue
            overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
            if overlap < min_score:
                continue
            ranked.append(
                RetrievedChunk(
                    id=str(pt.id),
                    text=text,
                    score=float(overlap),
                    metadata=payload,
                    source=payload.get("source_file") or payload.get("source"),
                    method="metadata_lexical",
                )
            )

        ranked.sort(key=lambda c: float(c.score), reverse=True)
        return ranked[:top_k]

    def expand_with_neighbors(
            self,
            collection_name: str,
            seed_chunks: List[RetrievedChunk],
            profile_id: Optional[str],
            window: int = 1,
            max_new: int = 6
    ) -> List[RetrievedChunk]:
        """
        Pull immediate neighbor chunks (same document_id, adjacent chunk_index)
        to improve local context continuity.
        """
        if not seed_chunks or window <= 0:
            return seed_chunks

        seen = set()
        expanded = []

        for chunk in seed_chunks:
            meta = chunk.metadata or {}
            doc_id = meta.get("document_id")
            idx = meta.get("chunk_index")
            if doc_id is None or idx is None:
                expanded.append(chunk)
                continue

            key = (str(doc_id), str(idx))
            seen.add(key)
            expanded.append(chunk)

            if len(expanded) >= len(seed_chunks) + max_new:
                continue

            neighbor_filters = []

            # Prefer exact prev/next chunk ids to avoid schema/type issues on chunk_index.
            neighbor_ids = []
            for key_name in ("prev_chunk_id", "next_chunk_id"):
                neighbor_val = meta.get(key_name)
                if neighbor_val:
                    neighbor_ids.append(str(neighbor_val))
            if neighbor_ids:
                must = [
                    {"key": "document_id", "match": {"value": str(doc_id)}},
                    {"key": "chunk_id", "match": {"any": neighbor_ids}},
                ]
                if profile_id is not None:
                    must.append({"key": "profile_id", "match": {"value": str(profile_id)}})
                neighbor_filters.append({"must": must})

            # Fallback to chunk_index equality (not range) to avoid 400s when stored as keyword.
            try:
                idx_int = int(idx)
            except Exception:
                idx_int = None
            if idx_int is not None and window > 0:
                neighbor_values = []
                for delta in range(-window, window + 1):
                    if delta == 0:
                        continue
                    candidate = idx_int + delta
                    if candidate < 0:
                        continue
                    neighbor_values.extend([candidate, str(candidate)])
                # Deduplicate while preserving order
                seen_values = set()
                neighbor_values = [v for v in neighbor_values if not (v in seen_values or seen_values.add(v))]
                if neighbor_values:
                    must = [
                        {"key": "document_id", "match": {"value": str(doc_id)}},
                        {"key": "chunk_index", "match": {"any": neighbor_values}},
                    ]
                    if profile_id is not None:
                        must.append({"key": "profile_id", "match": {"value": str(profile_id)}})
                    neighbor_filters.append({"must": must})

            if not neighbor_filters:
                continue

            for scroll_filter in neighbor_filters:
                try:
                    scroll = self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=scroll_filter,
                        with_payload=True,
                        with_vectors=False,
                        limit=max(window * 4, 2),  # small fetch per seed
                    )
                except Exception as exc:
                    logger.debug("Neighbor fetch failed for doc %s idx %s: %s", doc_id, idx, exc)
                    continue

                points = []
                if hasattr(scroll, "points"):
                    points = scroll.points or []
                elif isinstance(scroll, tuple) and scroll:
                    points = scroll[0] or []

                for pt in points:
                    payload = pt.payload or {}
                    neighbor_idx = payload.get("chunk_index")
                    neighbor_key = None
                    if neighbor_idx is not None:
                        neighbor_key = (str(payload.get("document_id", doc_id)), str(neighbor_idx))
                    if neighbor_key and neighbor_key in seen:
                        continue
                    text = payload.get("text") or ""
                    if not text:
                        continue
                    if neighbor_key:
                        seen.add(neighbor_key)
                    neighbor_score = max(float(chunk.score) - 0.05, 0.0)
                    neighbor_source = (
                        payload.get("source_file")
                        or payload.get("source")
                        or chunk.source
                        or "unknown"
                    )
                    expanded.append(
                        RetrievedChunk(
                            id=str(pt.id),
                            text=text,
                            score=neighbor_score,
                            source=neighbor_source,
                            metadata=payload
                        )
                    )
                    if len(expanded) >= len(seed_chunks) + max_new:
                        break

        return expanded

    def get_profile_context(
            self,
            collection_name: str,
            profile_id: str,
            max_points: int = 400,
            refresh_seconds: int = 300
    ) -> ProfileContextSnapshot:
        """Build lightweight context from existing embeddings to guide vague queries."""
        cache_key = (collection_name, str(profile_id))
        now = time.time()
        cached = self.profile_context_cache.get(cache_key)
        if cached and (now - cached.last_updated) < refresh_seconds:
            return cached

        filter_ = {"must": [{"key": "profile_id", "match": {"value": str(profile_id)}}]}
        collected_points = []
        next_offset = None
        batch_size = min(120, max_points)

        try:
            while len(collected_points) < max_points:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_,
                    offset=next_offset,
                    limit=min(batch_size, max_points - len(collected_points)),
                    with_payload=True,
                    with_vectors=False
                )

                if hasattr(scroll_result, "points"):
                    batch = scroll_result.points or []
                    next_offset = getattr(scroll_result, "next_page_offset", None)
                elif isinstance(scroll_result, tuple):
                    batch = scroll_result[0] if len(scroll_result) > 0 else []
                    next_offset = scroll_result[1] if len(scroll_result) > 1 else None
                else:
                    batch = []
                    next_offset = None

                if not batch:
                    break

                collected_points.extend(batch)

                if not next_offset:
                    break
        except Exception as e:
            logger.warning(f"Failed to build profile context for {profile_id}: {e}")
            snapshot = ProfileContextSnapshot([], [], 0, now)
            self.profile_context_cache[cache_key] = snapshot
            return snapshot

        token_counts: Counter = Counter()
        doc_hints: List[str] = []
        seen_hints = set()

        for pt in collected_points:
            payload = pt.payload or {}
            text = payload.get("text") or ""
            if text:
                token_counts.update(self.preprocessor.tokenize(text))

            for hint_key in ("source_file", "document_id", "section"):
                hint_val = payload.get(hint_key)
                if hint_val:
                    hint_val = str(hint_val)
                    if hint_val not in seen_hints:
                        doc_hints.append(hint_val)
                        seen_hints.add(hint_val)

        top_keywords = [w for w, _ in token_counts.most_common(40)]
        snapshot = ProfileContextSnapshot(
            top_keywords=top_keywords,
            document_hints=doc_hints[:12],
            total_chunks=len(collected_points),
            last_updated=now
        )
        self.profile_context_cache[cache_key] = snapshot
        return snapshot


class HybridReranker:
    """Reranks retrieved chunks using BM25 + vector scores with dynamic weighting."""

    def __init__(self, alpha: float = 0.7, cross_encoder: Optional[CrossEncoder] = None):
        """Initialize reranker."""
        self.alpha = alpha
        self.preprocessor = TextPreprocessor()
        self.cross_encoder = cross_encoder

    def adjust_alpha(self, query: str) -> float:
        """Dynamically adjust alpha based on query characteristics."""
        words = query.lower().split()

        question_words = {'what', 'why', 'how', 'explain', 'describe', 'understand'}
        if any(qw in words for qw in question_words):
            return 0.75

        if any(re.search(r'\d+', word) for word in words) or len(words) <= 3:
            return 0.6

        return self.alpha

    def rerank(
            self,
            chunks: List[RetrievedChunk],
            query: str,
            top_k: int = 10,
            use_cross_encoder: bool = True
    ) -> List[RetrievedChunk]:
        """Rerank chunks using hybrid BM25 + vector scoring."""
        if not chunks:
            return []

        try:
            alpha = self.adjust_alpha(query)
            logger.info(f"Using alpha={alpha:.2f} for reranking")

            texts = [chunk.text for chunk in chunks]
            tokenized_corpus = [self.preprocessor.tokenize(text) for text in texts]
            tokenized_query = self.preprocessor.tokenize(query)

            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float64)

            vector_scores = np.array([float(chunk.score) for chunk in chunks], dtype=np.float64)

            if len(vector_scores) > 1:
                v_min, v_max = vector_scores.min(), vector_scores.max()
                b_min, b_max = bm25_scores.min(), bm25_scores.max()

                v_range = v_max - v_min if v_max > v_min else 1.0
                b_range = b_max - b_min if b_max > b_min else 1.0

                vector_scores_norm = (vector_scores - v_min) / v_range
                bm25_scores_norm = (bm25_scores - b_min) / b_range
            else:
                vector_scores_norm = np.ones_like(vector_scores)
                bm25_scores_norm = np.ones_like(bm25_scores)

            hybrid_scores = (
                    alpha * vector_scores_norm +
                    (1 - alpha) * bm25_scores_norm
            )

            top_n_for_ce = min(top_k * 2, len(chunks))
            sorted_indices = np.argsort(hybrid_scores)[::-1][:top_n_for_ce]

            candidate_chunks = [chunks[idx] for idx in sorted_indices]

            if use_cross_encoder and self.cross_encoder and len(candidate_chunks) > 1:
                try:
                    pairs = [[query, chunk.text] for chunk in candidate_chunks]
                    ce_scores = self.cross_encoder.predict(pairs)

                    for i, score in enumerate(ce_scores):
                        candidate_chunks[i].score = float(score)

                    candidate_chunks.sort(key=lambda c: c.score, reverse=True)
                    logger.info(f"Applied CrossEncoder reranking to {len(candidate_chunks)} chunks")

                except Exception as e:
                    logger.warning(f"CrossEncoder reranking failed: {e}")
                    for i, chunk in enumerate(candidate_chunks):
                        chunk.score = float(hybrid_scores[sorted_indices[i]])
            else:
                for i, chunk in enumerate(candidate_chunks):
                    chunk.score = float(hybrid_scores[sorted_indices[i]])

            reranked_chunks = candidate_chunks[:top_k]

            logger.info(f"Reranked to {len(reranked_chunks)} chunks (alpha={alpha:.2f})")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Error in reranking: {e}", exc_info=True)
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


class ContextBuilder:
    """Builds formatted context for LLM with source citations."""

    @staticmethod
    def build_source_hints(chunks: List[RetrievedChunk]) -> str:
        """Create a compact source map to bias model attention to top evidence."""
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks[:5], 1):
            meta = chunk.metadata or {}
            source_name = chunk.source or meta.get('source_file', f"doc_{chunk.id[:8]}")
            page = meta.get('page')
            section = meta.get('section')
            score = round(float(chunk.score), 3)
            parts = [f"{i}) {source_name}", f"score={score}"]
            if page is not None:
                parts.append(f"page={page}")
            if section:
                parts.append(f"section={section}")
            lines.append(" | ".join(parts))
        return "SOURCE MAP:\n" + "\n".join(lines) + "\n"

    @staticmethod
    def build_context(chunks: List[RetrievedChunk], max_chunks: int = 3) -> str:
        """Build context string with source citations."""
        if not chunks:
            return ""

        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            normalized = ' '.join(chunk.text.split())
            if normalized not in seen_texts and normalized.strip():
                seen_texts.add(normalized)
                unique_chunks.append(chunk)

        selected_chunks = unique_chunks[:max_chunks]

        context_parts = []
        source_map = ContextBuilder.build_source_hints(selected_chunks)
        if source_map:
            context_parts.append(source_map)
        for i, chunk in enumerate(selected_chunks, 1):
            # Use source_file directly from metadata as fallback
            source_name = chunk.source or chunk.metadata.get('source_file', f"doc_{chunk.id[:8]}")
            context_parts.append(
                f"[SOURCE: {source_name}]\n{chunk.text}\n[/SOURCE]"
            )

        return "\n".join(context_parts)

    @staticmethod
    def extract_sources(chunks: List[RetrievedChunk], max_sources: int = 3) -> List[Dict[str, Any]]:
        """Extract source information for response metadata."""
        sources = []
        for i, chunk in enumerate(chunks[:max_sources], 1):
            meta = chunk.metadata or {}
            source_name = chunk.source or meta.get("source_file") or meta.get("filename") or "Document"
            source_name = os.path.basename(str(source_name))
            source_name = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", source_name).strip() or "Document"
            sources.append({
                'source_id': i,
                'source_name': source_name,
                'excerpt': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'relevance_score': round(float(chunk.score), 3)
            })
        return sources


class DomainPromptAdapter:
    """Creates lightweight, in-context adapters to steer local LLM responses."""

    @staticmethod
    def build_adapter(
        profile_context: Dict[str, Any],
        query: str,
        sentiment: Optional[str] = None,
        empathy_level: Optional[str] = None,
    ) -> str:
        if not profile_context:
            return ""

        keywords = profile_context.get("keywords") or []
        hints = profile_context.get("hints") or []
        sampled = profile_context.get("sampled_chunks", 0)

        adapter_lines = []
        if keywords:
            adapter_lines.append(
                f"Domain keywords to preserve and prefer in wording: {', '.join(keywords[:12])}."
            )
        if hints:
            adapter_lines.append(f"Relevant documents/sections to anchor on: {', '.join(hints[:6])}.")
        if sampled:
            adapter_lines.append(f"Context built from {sampled} profile chunks; avoid generic responses.")

        adapter_lines.append(
            "Favor the terminology above when answering; align synonyms in the question to these domain terms."
        )
        adapter_lines.append(
            f"If the user query is vague ('{query}'), proactively ground the answer in the domain cues above."
        )
        if sentiment or empathy_level:
            empathy = empathy_level or ("supportive" if sentiment in {"frustrated", "negative", "confusion"} else "neutral")
            adapter_lines.append(
                f"Use a {empathy} tone; acknowledge the user's sentiment if evident before presenting facts."
            )
        return "\n".join(adapter_lines)


class AnswerabilityDetector:
    """Detects if a question can be answered from provided context."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def check_answerability(self, query: str, context: str, has_chunks: bool = False) -> Tuple[bool, str]:
        """
        Check if the query can be answered from the context.

        If we already have retrieved chunks, bias toward answering to avoid premature
        "cannot answer" responses when relevant evidence exists.
        """
        if has_chunks and context.strip():
            return True, "Context present from retrieved chunks"

        prompt = f"""You are an answerability classifier. Determine if the USER QUESTION can be answered using ONLY the information in the DOCUMENT CONTEXT.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

Respond with ONLY one of these formats:
- If answerable: "ANSWERABLE: <brief reason>"
- If not answerable: "NOT_ANSWERABLE: <what information is missing>"

Your response:"""

        try:
            response = self.llm_client.generate(prompt, max_retries=2)
            response = response.strip()

            if response.startswith("ANSWERABLE"):
                return True, response.replace("ANSWERABLE:", "").strip()
            elif response.startswith("NOT_ANSWERABLE"):
                return False, response.replace("NOT_ANSWERABLE:", "").strip()
            else:
                return True, "Classification unclear"

        except Exception as e:
            logger.warning(f"Answerability check failed: {e}")
            return True, "Check failed"


class MetricsTracker:
    """Lightweight metrics sink for usage and quality signals."""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.retention_days = 120  # keep ~4 months of daily metrics

    @staticmethod
    def _day(ts: Optional[float] = None) -> str:
        return datetime.utcfromtimestamp(ts or time.time()).strftime("%Y-%m-%d")

    def record(
            self,
            model_name: str,
            subscription_id: str,
            profile_id: str,
            query_type: str,
            context_found: bool,
            grounded: bool,
            cached: bool,
            processing_time: float,
            retrieval_stats: Optional[Dict[str, Any]] = None
    ):
        if not self.redis:
            return

        day = self._day()
        base_key = f"rag:metrics:daily:{day}"
        model_key = f"rag:metrics:model:{day}"
        profile_key = f"rag:metrics:profile:{day}"

        stats = retrieval_stats or {}
        hits = float(stats.get("initial_retrieved", 0))
        final_ctx = float(stats.get("final_context", 0))

        pipe = self.redis.pipeline()
        pipe.hincrby(base_key, "total", 1)
        pipe.hincrby(base_key, f"type:{query_type}", 1)
        if context_found:
            pipe.hincrby(base_key, "context_found", 1)
        if grounded:
            pipe.hincrby(base_key, "grounded", 1)
        if cached:
            pipe.hincrby(base_key, "cache_hits", 1)
        pipe.hincrbyfloat(base_key, "sum_latency", float(processing_time or 0.0))
        pipe.hincrbyfloat(base_key, "sum_hits", hits)
        pipe.hincrbyfloat(base_key, "sum_final_ctx", final_ctx)
        pipe.hincrby(model_key, model_name or "unknown", 1)
        pipe.hincrby(profile_key, _slug(profile_id), 1)
        pipe.expire(base_key, self.retention_days * 86400)
        pipe.expire(model_key, self.retention_days * 86400)
        pipe.expire(profile_key, self.retention_days * 86400)
        try:
            pipe.execute()
        except Exception as exc:
            logger.debug("Metrics write failed: %s", exc)

    def summary(self, days: int = 7) -> Dict[str, Any]:
        if not self.redis:
            return {"available": False, "reason": "redis_unavailable"}

        days = max(1, min(days, 90))
        total = Counter()
        model_usage: Counter = Counter()
        profile_usage: Counter = Counter()

        for offset in range(days):
            day = self._day(time.time() - offset * 86400)
            base_key = f"rag:metrics:daily:{day}"
            model_key = f"rag:metrics:model:{day}"
            profile_key = f"rag:metrics:profile:{day}"

            data = self.redis.hgetall(base_key) or {}
            for k, v in data.items():
                try:
                    if k.startswith("sum_"):
                        total[k] += float(v)
                    else:
                        total[k] += int(v)
                except Exception:
                    continue

            for k, v in (self.redis.hgetall(model_key) or {}).items():
                try:
                    model_usage[k] += int(v)
                except Exception:
                    continue
            for k, v in (self.redis.hgetall(profile_key) or {}).items():
                try:
                    profile_usage[k] += int(v)
                except Exception:
                    continue

        grand_total = int(total.get("total", 0))
        grounded = int(total.get("grounded", 0))
        context_found = int(total.get("context_found", 0))
        cache_hits = int(total.get("cache_hits", 0))
        not_answerable = int(total.get("type:not_answerable", 0))
        no_results = int(total.get("type:no_results", 0))

        # Derived rates
        def _pct(num: float, denom: float) -> float:
            return round((num / denom) * 100, 2) if denom else 0.0

        avg_latency = round(total.get("sum_latency", 0.0) / grand_total, 3) if grand_total else 0.0
        avg_hits = round(total.get("sum_hits", 0.0) / grand_total, 3) if grand_total else 0.0
        avg_ctx = round(total.get("sum_final_ctx", 0.0) / max(grounded, 1), 3)

        return {
            "available": True,
            "window_days": days,
            "totals": {
                "total_queries": grand_total,
                "grounded_answers": grounded,
                "context_found": context_found,
                "cache_hits": cache_hits,
                "not_answerable": not_answerable,
                "no_results": no_results,
            },
            "rates": {
                "accuracy_pct": _pct(grounded, grand_total),
                "context_found_pct": _pct(context_found, grand_total),
                "cache_hit_pct": _pct(cache_hits, grand_total),
                "not_answerable_pct": _pct(not_answerable, grand_total),
                "no_results_pct": _pct(no_results, grand_total),
            },
            "averages": {
                "latency_seconds": avg_latency,
                "retrieval_hits": avg_hits,
                "context_chunks_used": avg_ctx,
            },
            "usage": {
                "models": dict(model_usage.most_common(10)),
                "profiles": dict(profile_usage.most_common(10)),
            }
        }


class PromptBuilder:
    """Builds structured prompts with strict grounding and citation requirements."""

    # In dw_newron.py, replace PromptBuilder.build_qa_prompt method (around line 1700)

    @staticmethod
    def build_qa_prompt(
            query: str,
            context: str,
            persona: str,
            conversation_summary: str = "",
            domain_guidance: str = "",
            feedback_memory: str = "",
            retrieval_brief: str = "",
            evidence_plan: str = "",
            intent: str = "",
            intent_type: str = "",
            required_evidence: Optional[Dict[str, Any]] = None,
            output_format_preference: str = "",
            sentiment: str = "",
            instructions: Optional[Dict[str, Any]] = None,
            companion_opener: str = "",
            style_directives: Optional[Dict[str, Any]] = None,
            template_guidance: str = "",
            template_outline: str = "",
            continuity_notes: str = "",
    ) -> str:
        """Build a structured QA prompt with grounding and citation requirements."""
        instructions = instructions or {}
        style_directives = style_directives or {}
        convo_block = f"\nPRIOR CONVERSATION SUMMARY:\n{conversation_summary}\n" if conversation_summary else ""
        domain_block = f"\nDOMAIN ADAPTER:\n{domain_guidance}\n" if domain_guidance else ""
        feedback_block = f"\nRECENT FEEDBACK:\n{feedback_memory}\n" if feedback_memory else ""
        retrieval_block = f"\nRETRIEVAL NOTES:\n{retrieval_brief}\n" if retrieval_brief else ""
        evidence_block = f"\nINTERNAL EVIDENCE PLAN:\n{evidence_plan}\n" if evidence_plan else ""
        continuity_block = f"\nCONTINUITY NOTES:\n{continuity_notes}\n" if continuity_notes else ""

        instruction_lines = []
        if intent:
            instruction_lines.append(f"Detected intent: {intent}.")
        if intent_type:
            instruction_lines.append(f"Intent type: {intent_type}.")
        if sentiment and sentiment != "neutral" and not companion_opener:
            instruction_lines.append(
                f"Detected sentiment: {sentiment}. Keep the tone supportive without adding a generic preamble."
            )
        if output_format_preference:
            instruction_lines.append(f"Preferred output format: {output_format_preference}.")
        if required_evidence:
            instruction_lines.append(f"Evidence requirements: {json.dumps(required_evidence, ensure_ascii=True)}.")
        if instructions:
            instruction_lines.append(f"User instructions: {instructions}.")
            fields = instructions.get("fields") if isinstance(instructions, dict) else None
            if fields:
                instruction_lines.append(f"Required fields: {fields}.")
        if companion_opener:
            instruction_lines.append(f"Required opener (use verbatim as the first sentence): {companion_opener}")
        if style_directives:
            tone = style_directives.get("tone")
            humor = style_directives.get("humor_level")
            if tone:
                instruction_lines.append(f"Tone target: {tone}.")
            if humor:
                instruction_lines.append("Humor allowed: keep it mild and professional; avoid jokes in sensitive domains.")
        if template_guidance:
            instruction_lines.append(f"Response template guidance: {template_guidance}")
        instruction_block = ""
        if instruction_lines:
            instruction_block = "\nRESPONSE DIRECTIVES:\n" + "\n".join(instruction_lines) + "\n"
        outline_block = ""
        if template_outline:
            outline_block = f"\nOUTPUT FORMAT (STRICT):\n{template_outline}\n"

        prompt = f"""You are a {persona}. Deliver a clear, reader-friendly answer that feels human and intentional, not robotic. Avoid canned phrases and generic preambles. Do not mention retrieval or internal system behavior. If an opener is provided, use it verbatim as the first sentence; otherwise start directly with the answer content. Use natural phrasing and proper punctuation.

DOCUMENT CONTEXT:
{context}
{convo_block}
{domain_block}
{retrieval_block}
{feedback_block}
{evidence_block}
{continuity_block}
{instruction_block}
{outline_block}

USER QUESTION: {query}

GROUNDING & CITATION RULES (MANDATORY):
1) Use ONLY the document context above; if something is missing, say so plainly.
2) EVERY factual claim must cite immediately with [SOURCE-X]; multiple sources -> [SOURCE-1, SOURCE-3].
3) Use precise facts (names, numbers, dates) but paraphrase; do NOT quote passages unless explicitly asked.
4) If sources disagree, briefly note the discrepancy with citations.
5) Do not invent, pad, or generalize beyond the provided text.

DOCUMENT FOCUS DECISION FLOW (MANDATORY):
- If the question names a document, ID, or section, anchor the answer there.
- Otherwise infer the most relevant document/section from the context and source map.
- If multiple documents are relevant, default to the single most relevant document.
- Include multiple documents only when comparison or aggregation is clearly requested.
- Ask a clarification question only when ambiguity across unrelated documents would change the answer, and list concrete options.

ANSWER STYLE (MANDATORY):
- If an opener is provided, use it verbatim as the first sentence and do not add additional greetings or acknowledgements.
- Do NOT begin with generic phrases like "I reviewed the documents", "Here are the most relevant passages", or "Based on the passages".
- Match the response template guidance above; if none, use a concise overview, Evidence bullets, then a brief Takeaway.
- Comparison intent => include a Comparison table and a Key Takeaways section.
- How-to intent => numbered steps, with prerequisites if relevant.
- Troubleshooting intent => Symptoms, Likely Causes, Fix Steps.
- Extraction/listing intent => table plus a short Notes section.
- Summary intent => headings with bullets.
- Analysis intent => include an Analysis section with implications/risks/decisions.
- Tone: warm, confident colleague; honor any tone or empathy instructions explicitly requested.
- Bullets or steps must be full sentences that stitch facts together (who/what/when/where/why/how) with inline citations.
- Cite using [SOURCE-X] only; these will be rewritten into user-friendly source labels without section/page references.
- Never expose internal identifiers (chunk_id, document_id, Qdrant ids, file paths, system names).
- If the question is about an entity not present, say so and list what the documents do cover (with citations), then stop.
- Use conversation summary only when it directly helps answer the current question; otherwise ignore it.

Provide the answer now with citations inline after each claim."""

        return prompt

class ConversationSummarizer:
    """Summarizes the last few turns to keep context tight."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def summarize(self, conversation_text: str) -> str:
        if not conversation_text:
            return ""
        prompt = f"""Summarize the following conversation turns into 3-5 concise bullets capturing user intent and assistant answers. Do NOT invent details.

CONVERSATION:
{conversation_text}

SUMMARY:"""
        try:
            summary = self.llm_client.generate(prompt, max_retries=2, backoff=0.5)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Conversation summarization failed: {e}")
            return ""


class EnterpriseRAGSystem:
    """
    Enhanced RAG system with query reformulation, conversational context,
    cross-encoder reranking, and grounding.
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            profile_id: Optional[str] = None,
            backend_override: Optional[str] = None,
            model_path: Optional[str] = None
    ):
        """Initialize the RAG system with lazy-loaded components."""
        try:
            # Initialize LLM client (Ollama by default, Gemini when requested)
            self.llm_client = create_llm_client(
                model_name,
                backend_override=backend_override,
                model_path=model_path
            )
            self.model_name = getattr(self.llm_client, "model_name", model_name or "default")

            # Initialize other components
            qdrant_client = get_qdrant_client()
            embedding_override = resolve_embedding_model_for_generation(model_name)
            model = get_model_by_name(embedding_override) if embedding_override else get_model()
            cross_encoder = get_cross_encoder()

            self.model = model
            self.client = qdrant_client
            self.retriever = QdrantRetriever(qdrant_client, model)
            self.reranker = HybridReranker(alpha=0.7, cross_encoder=cross_encoder)
            self.context_builder = ContextBuilder()
            self.intelligent_context_builder = IntelligentContextBuilder(
                max_context_chunks=getattr(Config.Retrieval, "MAX_CONTEXT_CHUNKS", 7)
            )
            self.query_understanding = QueryUnderstanding()
            self.hybrid_retriever = HybridRetriever(
                qdrant_client,
                model,
                config=HybridRetrieverConfig(
                    topk_dense=int(getattr(Config.Retrieval, "TOPK_DENSE", 50)),
                    hybrid_alpha=float(getattr(Config.Retrieval, "HYBRID_ALPHA", 0.75)),
                ),
            )
            self.reranker_v2 = Reranker(
                cross_encoder=cross_encoder,
                llm_client=self.llm_client,
                config=RerankerConfig(top_k=int(getattr(Config.Retrieval, "TOPK_RERANK", 20))),
            )
            self.context_assembler = ContextAssembler(
                max_tokens=int(getattr(Config.Retrieval, "MAX_CONTEXT_TOKENS", 2048)),
                max_chunks=int(getattr(Config.Retrieval, "FINAL_CONTEXT_CHUNKS", 8)),
            )
            self.fast_query_analyzer = QueryAnalyzer()
            self.evidence_constraints = EvidenceConstraints()
            self.hybrid_ranker = HybridRanker(HybridRankerConfig())
            self.retrieval_quality = RetrievalQualityScorer(
                threshold_low=float(getattr(Config.Retrieval, "RETRIEVAL_QUALITY_THRESH_LOW", 0.45)),
                threshold_high=float(getattr(Config.Retrieval, "RETRIEVAL_QUALITY_THRESH_HIGH", 0.75)),
                budget_ms=int(getattr(Config.Retrieval, "RETRIEVAL_GUARD_BUDGET_MS", 40)),
            )
            self.fallback_repair = FallbackRepair(
                max_attempts=int(getattr(Config.Retrieval, "RETRIEVAL_FALLBACK_MAX_ATTEMPTS", 1)),
                rewrite_enabled=bool(getattr(Config.Retrieval, "RETRIEVAL_FALLBACK_REWRITE", True)),
            )
            self.prompt_builder = PromptBuilder()
            self.greeting_handler = GreetingHandler()
            self.query_reformulator = QueryReformulator(self.llm_client)
            self.query_intelligence = QueryIntelligence(self.llm_client)
            self.answerability_detector = AnswerabilityDetector(self.llm_client)
            self.evidence_planner = EvidencePlanner(self.llm_client)
            self.answer_verifier = AnswerVerifier()
            self.confidence_scorer = ConfidenceScorer()
            self.retrieval_confidence = RetrievalConfidenceScorer()
            self.learning_signals = LearningSignalStore()
            self.verification_agent = VerificationAgent()
            self.clarification_agent = ClarificationAgent()
            self.retriever_manager = RetrieverManager(
                qdrant_retriever=self.retriever,
                hybrid_retriever=self.hybrid_retriever,
                verification_agent=self.verification_agent,
                chunk_factory=RetrievedChunk,
                max_workers=int(getattr(Config.Execution, "RETRIEVER_MAX_WORKERS", 4)),
            )
            self.model_arbitration = ModelArbitrationLayer(
                create_llm_client,
                max_workers=int(getattr(Config.ModelArbitration, "MAX_PARALLEL", 3)),
            )
            self.post_processor = PostProcessor(self.answer_verifier, self.confidence_scorer)
            self.bad_answer_evaluator = BadAnswerEvaluator(
                EvalConfig(
                    eval_budget_ms=int(getattr(Config.Quality, "EVAL_BUDGET_MS", 40)),
                    high_confidence_threshold=float(getattr(Config.Quality, "HIGH_CONFIDENCE_THRESHOLD", 0.75)),
                    lex_support_th=float(getattr(Config.Quality, "LEX_SUPPORT_TH", 0.18)),
                    supported_ratio_th=float(getattr(Config.Quality, "SUPPORTED_RATIO_TH", 0.60)),
                    critical_supported_ratio_th=float(getattr(Config.Quality, "CRITICAL_SUPPORTED_RATIO_TH", 0.75)),
                    overall_score_th=float(getattr(Config.Quality, "OVERALL_SCORE_TH", 0.72)),
                )
            )
            self.auto_repair_engine = AutoRepairEngine(
                RepairConfig(
                    max_attempts=int(getattr(Config.Quality, "MAX_REPAIR_ATTEMPTS", 2)),
                    repair_budget_ms=int(getattr(Config.Quality, "REPAIR_BUDGET_MS", 500)),
                )
            )
            # Initialize Redis client for storing conversation history and feedback.
            redis_client = get_redis_client()
            self.redis_cache = RedisJsonCache(
                redis_client,
                default_ttl=int(getattr(Config.Retrieval, "KG_RETRIEVAL_CACHE_TTL_SECONDS", 240)),
            )
            self.companion_classifier = CompanionClassifier(redis_client)

            # Initialize conversation history backed by Redis. Avoid instantiating
            # a non-redis version first since it would immediately be overwritten.
            self.conversation_history = ConversationHistory(
                max_turns=int(getattr(Config.Chat, "MAX_HISTORY_TURNS", 6)),
                redis_client=redis_client,
            )
            self.conversation_summarizer = ConversationSummarizer(self.llm_client)
            self.feedback_memory = ChatFeedbackMemory(max_items=12, redis_client=redis_client)
            self.kg_store = None
            self.graph_guided_retriever = None
            try:
                self.kg_store = Neo4jStore()
                self.graph_guided_retriever = GraphGuidedRetriever(
                    neo4j_store=self.kg_store,
                    cache=self.redis_cache,
                    probe_limit=int(getattr(Config.Retrieval, "KG_PROBE_LIMIT", 20)),
                    probe_timeout_ms=int(getattr(Config.Retrieval, "KG_PROBE_TIMEOUT_MS", 80)),
                    cache_ttl_seconds=int(getattr(Config.Retrieval, "KG_PROBE_TTL_SECONDS", 1200)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("KG retriever unavailable; continuing without graph guidance: %s", exc)
            self._warm_up_llm()

            logger.info("EnterpriseRAGSystem initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EnterpriseRAGSystem: {e}")
            raise

    # In dw_newron.py, line ~1800 in preprocess_query method

    def preprocess_query(
            self,
            query: str,
            user_id: str,
            namespace: str,
            use_reformulation: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Preprocess query with normalization and optional reformulation."""
        metadata = {
            'original_query': query,
            'corrections': [],
            'reformulated': False,
            'expanded': False,
            'intent': self._detect_intent(query)
        }

        processed_query = re.sub(r"\s+", " ", query or "").strip()
        metadata['normalized'] = processed_query

        if use_reformulation and self._is_query_vague(processed_query):
            conv_context = self.conversation_history.get_context(
                namespace,
                user_id,
                max_turns=int(getattr(Config.Chat, "CONTEXT_TURNS", 4)),
            )
            reformulated = self.query_reformulator.reformulate(processed_query, conv_context)
            if reformulated and reformulated != processed_query:
                processed_query = reformulated
                metadata['reformulated'] = True
            elif conv_context:
                processed_query = f"{processed_query} ; context: {conv_context}"

        expanded = self._expand_query_terms(processed_query)
        if expanded and expanded != processed_query:
            processed_query = expanded
            metadata['expanded'] = True

        return processed_query, metadata

    @staticmethod
    def _is_retrieval_sufficient(chunks: List[RetrievedChunk], min_hits: int = 3, min_score: float = 0.18) -> bool:
        """Decide if current retrieval is good enough to stop trying fallbacks."""
        if not chunks:
            return False
        if len(chunks) >= min_hits:
            return True
        top_score = float(chunks[0].score)

        return top_score >= min_score

    @staticmethod
    def _is_query_vague(query: str) -> bool:
        """Heuristic to detect short or underspecified questions."""
        tokens = query.split()
        if len(tokens) <= 3:
            return True
        meaningful_tokens = [t for t in tokens if len(t) > 3]
        return len(meaningful_tokens) <= 2

    @staticmethod
    def _detect_intent(query: str) -> str:
        q = query.lower()
        if any(word in q for word in ["summary", "summarize", "overview", "brief"]):
            return "summary"
        if any(word in q for word in ["compare", "difference", "vs", "versus", "comparison"]):
            return "comparison"
        if any(word in q for word in ["list", "extract", "identify", "show", "find"]):
            return "extraction"
        if any(word in q for word in ["why", "cause", "impact", "implication", "analyze", "evaluate", "reason"]):
            return "reasoning"
        if any(word in q for word in ["how", "steps", "procedure", "configure", "setup"]):
            return "procedural"
        if any(word in q for word in ["error", "issue", "fail", "troubleshoot", "bug"]):
            return "troubleshooting"
        return "factual"

    @staticmethod
    def _expand_query_terms(query: str) -> str:
        synonyms = {
            "install": ["setup", "configure"],
            "error": ["failure", "issue", "exception"],
            "policy": ["guideline", "rule", "standard"],
            "process": ["procedure", "workflow"],
        }
        additions = []
        lower_q = query.lower()
        for term, syns in synonyms.items():
            if term in lower_q:
                additions.extend(syns)
        if additions:
            return query + " " + " ".join(additions)
        return query

    @staticmethod
    def _contextualize_query(query: str, profile_context: ProfileContextSnapshot) -> Tuple[str, Dict[str, Any]]:
        """Blend the user query with profile-specific hints to guide retrieval."""
        if not profile_context or not (profile_context.top_keywords or profile_context.document_hints):
            return query, {}

        keywords = profile_context.top_keywords[:8]
        hints = profile_context.document_hints[:3]
        extras = []
        if hints:
            extras.append("related to " + ", ".join(hints))
        if keywords:
            extras.append("keywords: " + ", ".join(keywords))

        contextual_query = " ; ".join([query] + extras)
        return contextual_query, {"profile_keywords_used": keywords, "profile_hints_used": hints}

    @staticmethod
    def _clean_label(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        base = os.path.basename(text)
        base = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", base)
        base = re.sub(r"[_\-]+", " ", base)
        base = re.sub(r"\s+", " ", base).strip()
        return base or text

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}

    @classmethod
    def _source_label_from_chunk(cls, chunk: RetrievedChunk) -> str:
        meta = chunk.metadata or {}
        raw = (
            chunk.source
            or meta.get("source_file")
            or meta.get("document_name")
            or meta.get("document_title")
            or meta.get("document_id")
        )
        label = cls._clean_label(raw)
        if label:
            return label
        return f"Document {chunk.id[:8]}"

    @staticmethod
    def _format_citation_label(source: Dict[str, Any]) -> str:
        name = str(source.get("source_name") or "").strip()
        if not name:
            return ""
        return f"[Source: {name}]"

    @classmethod
    def _rewrite_citations(cls, answer: str, sources: List[Dict[str, Any]]) -> str:
        if not answer:
            return ""
        def _replace(match: re.Match) -> str:
            idx_raw = match.group(1)
            if not idx_raw or not idx_raw.isdigit():
                return match.group(0)
            idx = int(idx_raw) - 1
            if idx < 0 or idx >= len(sources):
                return match.group(0)
            label = cls._format_citation_label(sources[idx])
            return label or match.group(0)

        return re.sub(r"\[SOURCE-(\d+)\]", _replace, answer)

    @staticmethod
    def _answer_relevance(answer: str, query: str) -> float:
        if not answer or not query:
            return 0.0
        answer_tokens = set(re.findall(r"[a-z0-9]{3,}", answer.lower()))
        query_tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        if not answer_tokens or not query_tokens:
            return 0.0
        return len(answer_tokens & query_tokens) / max(len(query_tokens), 1)

    @staticmethod
    def _extract_required_attributes(query: str, intent_type: str) -> List[str]:
        return extract_required_attributes(query, intent_type)

    @classmethod
    def _filter_chunks_by_intent(
        cls,
        chunks: List[RetrievedChunk],
        required_attributes: List[str],
        entities: List[str],
        intent_type: str,
    ) -> List[RetrievedChunk]:
        return list(filter_chunks_by_intent(chunks, required_attributes, entities, intent_type))

    @staticmethod
    def _extract_answer_requirements(query: str, intent_type: str) -> List[Tuple[str, re.Pattern]]:
        return extract_answer_requirements(query, intent_type)

    @staticmethod
    def _validate_answer_requirements(
        chunks: List[RetrievedChunk],
        requirements: List[Tuple[str, re.Pattern]],
    ) -> List[str]:
        return validate_answer_requirements(chunks, requirements)

    @staticmethod
    def _build_intent_miss_response(entity: str, intent_label: str) -> str:
        return build_intent_miss_response(entity, intent_label)

    @classmethod
    def _section_label_from_chunk(cls, chunk: RetrievedChunk) -> str:
        meta = chunk.metadata or {}
        raw = meta.get("section_title") or meta.get("section")
        return cls._clean_label(raw)

    @classmethod
    def _collect_focus_options(
        cls,
        chunks: List[RetrievedChunk],
        profile_context: Optional[Dict[str, Any]] = None,
        *,
        limit: int = 4,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        doc_map: Dict[str, Dict[str, Any]] = {}
        for chunk in (chunks or [])[:25]:
            label = cls._source_label_from_chunk(chunk)
            entry = doc_map.setdefault(
                label,
                {
                    "label": label,
                    "max_score": 0.0,
                    "total_score": 0.0,
                    "sections": set(),
                },
            )
            score = max(float(getattr(chunk, "score", 0.0) or 0.0), 0.0)
            entry["max_score"] = max(entry["max_score"], score)
            entry["total_score"] += score
            section = cls._section_label_from_chunk(chunk)
            if section:
                entry["sections"].add(section)

        if not doc_map and profile_context:
            for hint in (profile_context.get("hints") or [])[:limit]:
                label = cls._clean_label(hint)
                if not label:
                    continue
                doc_map.setdefault(
                    label,
                    {
                        "label": label,
                        "max_score": 0.0,
                        "total_score": 0.0,
                        "sections": set(),
                    },
                )

        doc_options = sorted(
            doc_map.values(),
            key=lambda d: (float(d.get("max_score", 0.0)), float(d.get("total_score", 0.0)), d.get("label", "")),
            reverse=True,
        )[:limit]

        section_set: Set[str] = set()
        for option in doc_options:
            sections = sorted(option.get("sections") or [])[:4]
            option["sections"] = sections
            section_set.update(sections)

        section_options = sorted(section_set)[:6]
        return doc_options, section_options

    @classmethod
    def _query_mentions_focus(
        cls,
        query: str,
        doc_options: List[Dict[str, Any]],
        section_options: List[str],
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        filters = metadata_filters or {}
        if filters.get("section_titles"):
            return True
        q_lower = (query or "").lower()
        q_tokens = cls._tokenize(query)

        for option in doc_options[:4]:
            label = option.get("label") or ""
            if label and label.lower() in q_lower:
                return True
            if cls._tokenize(label) & q_tokens:
                return True

        for section in section_options[:4]:
            if section.lower() in q_lower:
                return True
            if cls._tokenize(section) & q_tokens:
                return True

        return False

    @classmethod
    def _doc_relatedness(cls, left: str, right: str) -> float:
        left_tokens = cls._tokenize(left)
        right_tokens = cls._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return len(overlap) / max(len(union), 1)

    @classmethod
    def _should_request_clarification(
        cls,
        *,
        query: str,
        doc_options: List[Dict[str, Any]],
        section_options: List[str],
        metadata_filters: Optional[Dict[str, Any]],
        preprocessing_metadata: Optional[Dict[str, Any]],
    ) -> bool:
        if not doc_options or len(doc_options) < 2:
            return False

        pre = preprocessing_metadata or {}
        if pre.get("target_document_id"):
            return False

        if cls._query_mentions_focus(query, doc_options, section_options, metadata_filters):
            return False

        intent = (
            (pre.get("query_analysis") or {}).get("intent")
            or pre.get("intent")
            or ""
        ).lower()
        if intent == "comparison":
            return False

        top = float(doc_options[0].get("max_score", 0.0) or 0.0)
        second = float(doc_options[1].get("max_score", 0.0) or 0.0)
        close_scores = False
        if top <= 0 and second <= 0:
            close_scores = True
        else:
            ratio = second / max(top, 1e-6)
            close_scores = ratio >= 0.88 or abs(top - second) <= 0.02

        unrelated = cls._doc_relatedness(doc_options[0].get("label", ""), doc_options[1].get("label", "")) < 0.25
        generic = cls._is_query_vague(query) or len((query or "").split()) <= 6

        return bool(close_scores and unrelated and generic)

    @staticmethod
    def _format_options(options: List[str], limit: int = 3) -> str:
        cleaned = [opt for opt in (options or []) if opt][:limit]
        return ", ".join(cleaned)

    @staticmethod
    def _truncate_excerpt(text: str, max_len: int = 220) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= max_len:
            return cleaned
        trimmed = cleaned[:max_len].rsplit(" ", 1)[0]
        return (trimmed or cleaned[:max_len]).rstrip() + "..."

    @classmethod
    def _build_evidence_response(
        cls,
        *,
        query: str,
        sources: List[Dict[str, Any]],
        doc_options: Optional[List[Dict[str, Any]]] = None,
        section_options: Optional[List[str]] = None,
        max_items: int = 3,
    ) -> str:
        if not sources:
            doc_labels = [opt.get("label") for opt in (doc_options or []) if opt.get("label")]
            doc_list = cls._format_options(doc_labels)
            if doc_list:
                return f"To answer accurately, choose a document like {doc_list}."
            return "To answer accurately, specify the document you want me to use."

        lines = [
            "I couldn't find enough evidence in the available documents to answer that precisely.",
        ]

        doc_labels = [opt.get("label") for opt in (doc_options or []) if opt.get("label")]
        doc_list = cls._format_options(doc_labels)
        if doc_list:
            lines.append(f"If you want, tell me which document to check (e.g., {doc_list}).")
        else:
            lines.append("If there is a specific document you want me to review, point me to it and I'll recheck.")

        return " ".join(lines)

    @classmethod
    def _build_clarification_question(
        cls,
        *,
        query: str,
        doc_options: List[Dict[str, Any]],
        section_options: List[str],
        profile_context: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        filters = metadata_filters or {}
        explicit_sections = filters.get("section_titles") or []
        if explicit_sections:
            section_name = cls._clean_label(explicit_sections[0])
            section_list = cls._format_options(section_options)
            if section_list:
                return (
                    f"I can focus on the '{section_name}' section. "
                    f"If you meant a different section, choose one: {section_list}."
                )
            return f"I can focus on the '{section_name}' section."

        doc_labels = [opt.get("label") for opt in (doc_options or []) if opt.get("label")]
        if not doc_labels and profile_context:
            doc_labels = [cls._clean_label(h) for h in (profile_context.get("hints") or []) if cls._clean_label(h)]

        doc_list = cls._format_options(doc_labels)
        section_list = cls._format_options(section_options)
        if doc_list and section_list:
            return (
                f"This could refer to multiple documents: {doc_list}. "
                f"Which one should I use? You can also name a section like: {section_list}."
            )
        if doc_list:
            return f"This could refer to multiple documents: {doc_list}. Which one should I use?"
        if section_list:
            return f"Which section should I use: {section_list}?"
        return "Please specify a document or section so I can answer precisely."


    def extract_person_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract person's name from query to filter by specific document.
        """
        import re

        if not query:
            return None

        query = query.strip()

        patterns = [
            # education/skills/experience of Name
            r'\b(?:education|skills?|experience|background|qualification|details?)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',

            # Name's education / skills
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[\'’]s\s+(?:education|skills?|experience)',

            # tell me about Name
            r'\btell\s+me\s+about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})'
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def find_document_id_by_name(
            self,
            collection_name: str,
            person_name: str,
            profile_id: str
    ) -> str | None:
        def _tokenize(name: str) -> List[str]:
            import re
            return [t for t in re.findall(r"[a-z0-9]+", name.lower()) if t]

        def _field_score(value: Optional[str], tokens: List[str], weight: int = 1) -> int:
            if not value or not tokens:
                return 0
            import re
            text = str(value).lower()
            full = " ".join(tokens).strip()
            score = 0
            if full and re.search(rf"\\b{re.escape(full)}\\b", text):
                score += 3 * weight
            for tok in tokens:
                if re.search(rf"\\b{re.escape(tok)}\\b", text):
                    score += 1 * weight
            return score

        def _best_doc_from_metadata(tokens: List[str], max_points: int = 700) -> Optional[str]:
            if not tokens:
                return None
            filter_ = {
                "must": [{"key": "profile_id", "match": {"value": str(profile_id)}}]
            }
            best_doc = None
            best_score = 0
            seen_scores: Dict[str, int] = {}
            next_offset = None
            batch_size = 120
            processed = 0
            try:
                while processed < max_points:
                    scroll_result = self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=filter_,
                        offset=next_offset,
                        limit=batch_size,
                        with_payload=True,
                        with_vectors=False,
                    )
                    if hasattr(scroll_result, "points"):
                        batch = scroll_result.points or []
                        next_offset = getattr(scroll_result, "next_page_offset", None)
                    elif isinstance(scroll_result, tuple):
                        batch = scroll_result[0] if len(scroll_result) > 0 else []
                        next_offset = scroll_result[1] if len(scroll_result) > 1 else None
                    else:
                        batch = []
                        next_offset = None

                    if not batch:
                        break

                    for pt in batch:
                        payload = pt.payload or {}
                        doc_id = payload.get("document_id")
                        if not doc_id:
                            continue
                        score = 0
                        score += _field_score(payload.get("source_file"), tokens, weight=3)
                        score += _field_score(payload.get("doc_name"), tokens, weight=3)
                        score += _field_score(payload.get("document_title"), tokens, weight=3)
                        score += _field_score(payload.get("title"), tokens, weight=2)
                        score += _field_score(payload.get("file_name"), tokens, weight=2)
                        score += _field_score(payload.get("source"), tokens, weight=1)
                        if score <= 0:
                            continue
                        prev = seen_scores.get(doc_id, 0)
                        if score > prev:
                            seen_scores[doc_id] = score
                        if score > best_score:
                            best_score = score
                            best_doc = doc_id
                    processed += len(batch)

                    if not next_offset:
                        break
            except Exception as exc:  # noqa: BLE001
                logger.debug("Metadata name lookup failed: %s", exc)

            if not best_doc:
                return None
            min_score = 4 if len(tokens) >= 2 else 2
            return best_doc if best_score >= min_score else None

        try:
            self.retriever.ensure_collection_exists(collection_name)
            tokens = _tokenize(person_name)

            metadata_match = _best_doc_from_metadata(tokens)
            if metadata_match:
                return metadata_match

            query_vector = None
            try:
                query_vector = self.model.encode(
                    person_name,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).astype(np.float32).tolist()
            except Exception as embed_exc:  # noqa: BLE001
                logger.debug("Embedding name lookup skipped: %s", embed_exc)

            if not query_vector:
                return None

            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="profile_id",
                        match=MatchValue(value=str(profile_id))
                    )
                ]
            )

            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="content_vector",
                query_filter=qdrant_filter,
                limit=10,
                with_payload=True,
                with_vectors=False
            )

            if not results or not results.points:
                return None

            for pt in results.points:
                payload = pt.payload or {}
                text = payload.get("text", "").lower()
                source = payload.get("source_file", "").lower()

                for part in tokens:
                    if part and (part in text or part in source):
                        return payload.get("document_id")

            return None

        except QdrantCollectionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error finding document by name '{person_name}': {e}")
            return None

    @staticmethod
    def _boost_chunks_for_person(
            chunks: List[RetrievedChunk],
            person_name: str,
            max_boost: float = 0.6,
    ) -> None:
        import re

        tokens = [t for t in re.findall(r"[a-z0-9]+", (person_name or "").lower()) if t]
        if not tokens:
            return

        full = " ".join(tokens).strip()

        def _score_field(value: Optional[str], weight: int) -> int:
            if not value:
                return 0
            text = str(value).lower()
            score = 0
            if full and re.search(rf"\\b{re.escape(full)}\\b", text):
                score += 3 * weight
            for tok in tokens:
                if re.search(rf"\\b{re.escape(tok)}\\b", text):
                    score += 1 * weight
            return score

        for chunk in chunks:
            meta = chunk.metadata or {}
            score = 0
            score += _score_field(meta.get("source_file"), 3)
            score += _score_field(meta.get("doc_name"), 3)
            score += _score_field(meta.get("document_title"), 3)
            score += _score_field(meta.get("title"), 2)
            score += _score_field(meta.get("section_title") or meta.get("section"), 1)
            score += _score_field(meta.get("source"), 1)
            if not score and chunk.text:
                score += _score_field(chunk.text[:200], 1)
            if score:
                boost = min(max_boost, 0.08 * score)
                chunk.score = float(chunk.score) + boost

    def _rank_docs_for_entity(
            self,
            collection_name: str,
            profile_id: str,
            entity_value: str,
            entity_type: str,
            max_points: int = 600,
    ) -> tuple[List[tuple[str, int]], List[str]]:
        import re

        tokens = [t for t in re.findall(r"[a-z0-9]+", (entity_value or "").lower()) if t]
        if not tokens:
            return [], []

        full = " ".join(tokens).strip()
        filter_ = {"must": [{"key": "profile_id", "match": {"value": str(profile_id)}}]}
        doc_scores: Dict[str, int] = {}
        section_hits: Dict[str, int] = {}
        next_offset = None
        batch_size = 120
        processed = 0

        def _score_field(value: Optional[str], weight: int) -> int:
            if not value:
                return 0
            text = str(value).lower()
            score = 0
            if full and re.search(rf"\\b{re.escape(full)}\\b", text):
                score += 3 * weight
            for tok in tokens:
                if re.search(rf"\\b{re.escape(tok)}\\b", text):
                    score += 1 * weight
            return score

        section_weight = 3 if entity_type in {"event", "place"} else 2
        doc_weight = 3 if entity_type in {"person", "product"} else 2

        try:
            while processed < max_points:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_,
                    offset=next_offset,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=False,
                )
                if hasattr(scroll_result, "points"):
                    batch = scroll_result.points or []
                    next_offset = getattr(scroll_result, "next_page_offset", None)
                elif isinstance(scroll_result, tuple):
                    batch = scroll_result[0] if len(scroll_result) > 0 else []
                    next_offset = scroll_result[1] if len(scroll_result) > 1 else None
                else:
                    batch = []
                    next_offset = None

                if not batch:
                    break

                for pt in batch:
                    payload = pt.payload or {}
                    doc_id = payload.get("document_id")
                    if not doc_id:
                        continue

                    score = 0
                    score += _score_field(payload.get("source_file"), doc_weight)
                    score += _score_field(payload.get("doc_name"), doc_weight)
                    score += _score_field(payload.get("document_title"), doc_weight)
                    score += _score_field(payload.get("title"), doc_weight)
                    section_title = payload.get("section_title") or payload.get("section")
                    score += _score_field(section_title, section_weight)
                    score += _score_field(payload.get("source"), 1)
                    if score == 0 and payload.get("text"):
                        score += _score_field(str(payload.get("text"))[:180], 1)

                    if score > 0:
                        prev = doc_scores.get(str(doc_id), 0)
                        if score > prev:
                            doc_scores[str(doc_id)] = score
                        if section_title:
                            sec_score = _score_field(section_title, section_weight)
                            if sec_score > 0:
                                section_hits[str(section_title)] = max(section_hits.get(str(section_title), 0), sec_score)

                processed += len(batch)
                if not next_offset:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("Entity doc ranking failed (%s=%s): %s", entity_type, entity_value, exc)

        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        sections = [s for s, _ in sorted(section_hits.items(), key=lambda kv: kv[1], reverse=True)]
        return ranked, sections

    def _resolve_entity_focus(
            self,
            collection_name: str,
            profile_id: str,
            metadata_filters: Dict[str, Any],
            person_name: Optional[str] = None,
            intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        entity_inputs: List[tuple[str, str, bool]] = []
        if person_name:
            entity_inputs.append(("person", person_name, True))
        for name in metadata_filters.get("person_names") or []:
            entity_inputs.append(("person", str(name), False))
        for name in metadata_filters.get("product_names") or []:
            entity_inputs.append(("product", str(name), False))
        for name in metadata_filters.get("place_names") or []:
            entity_inputs.append(("place", str(name), False))
        for name in metadata_filters.get("event_names") or []:
            entity_inputs.append(("event", str(name), False))

        seen = set()
        unique_inputs: List[tuple[str, str, bool]] = []
        for etype, value, primary in entity_inputs:
            key = f"{etype}:{value.lower().strip()}"
            if key and key not in seen:
                seen.add(key)
                unique_inputs.append((etype, value.strip(), primary))

        if not unique_inputs:
            return {"document_ids": [], "section_titles": [], "entities": []}

        doc_scores: Dict[str, int] = {}
        section_titles: List[str] = []
        entities: List[Dict[str, Any]] = []

        for etype, value, primary in unique_inputs:
            ranked_docs, sections = self._rank_docs_for_entity(
                collection_name=collection_name,
                profile_id=profile_id,
                entity_value=value,
                entity_type=etype,
            )
            if ranked_docs:
                for doc_id, score in ranked_docs[:4]:
                    doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), score + (2 if primary else 0))
            if sections:
                section_titles.extend(sections[:3])
            entities.append({"type": etype, "value": value, "ranked_docs": ranked_docs[:3]})

        ranked_doc_ids = [doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)]
        top_score = doc_scores.get(ranked_doc_ids[0], 0) if ranked_doc_ids else 0
        second_score = doc_scores.get(ranked_doc_ids[1], 0) if len(ranked_doc_ids) > 1 else 0
        min_score = 5 if any(len(e[1].split()) >= 2 for e in unique_inputs) else 3
        confidence_gap = top_score >= max(int(second_score * 1.15), min_score)

        strict = False
        if ranked_doc_ids and top_score >= min_score:
            if intent == "comparison" and len(ranked_doc_ids) > 1:
                strict = True
            elif confidence_gap:
                strict = True

        return {
            "document_ids": ranked_doc_ids[:3] if strict else [],
            "section_titles": list(dict.fromkeys(section_titles)),
            "entities": entities,
            "strict": strict,
        }

    @staticmethod
    def _boost_chunks_for_entities(
            chunks: List[RetrievedChunk],
            entity_values: List[str],
            max_boost: float = 0.6,
    ) -> None:
        import re

        if not entity_values:
            return

        normalized = []
        for value in entity_values:
            tokens = [t for t in re.findall(r"[a-z0-9]+", (value or "").lower()) if t]
            if tokens:
                normalized.append(tokens)

        if not normalized:
            return

        def _score_field(value: Optional[str], tokens: List[str]) -> int:
            if not value:
                return 0
            text = str(value).lower()
            full = " ".join(tokens).strip()
            score = 0
            if full and re.search(rf"\\b{re.escape(full)}\\b", text):
                score += 3
            for tok in tokens:
                if re.search(rf"\\b{re.escape(tok)}\\b", text):
                    score += 1
            return score

        for chunk in chunks:
            meta = chunk.metadata or {}
            score = 0
            for tokens in normalized:
                score = max(score, _score_field(meta.get("source_file"), tokens) * 3)
                score = max(score, _score_field(meta.get("doc_name"), tokens) * 3)
                score = max(score, _score_field(meta.get("document_title"), tokens) * 3)
                score = max(score, _score_field(meta.get("title"), tokens) * 2)
                score = max(score, _score_field(meta.get("section_title") or meta.get("section"), tokens))
                score = max(score, _score_field(meta.get("source"), tokens))
                if score == 0 and chunk.text:
                    score = max(score, _score_field(chunk.text[:180], tokens))
            if score:
                boost = min(max_boost, 0.06 * score)
                chunk.score = float(chunk.score) + boost

    @staticmethod
    def _shorten_query_for_ack(query: str, max_tokens: int = 10) -> str:
        tokens = (query or "").split()
        if len(tokens) <= max_tokens:
            return query.strip()
        return " ".join(tokens[:max_tokens]).strip() + "..."

    @staticmethod
    def _needs_acknowledgement(answer: str) -> bool:
        if not answer:
            return True
        prefix = answer.strip().lower()
        starters = (
            "got it", "sure", "absolutely", "of course", "here's", "here is",
            "based on", "from the documents", "i can", "happy to help",
            "i reviewed", "thanks for", "appreciate",
            "you're asking", "you’re asking", "you'd like", "you’d like",
            "it sounds like", "i understand", "i hear you",
            "okay", "alright", "sorry", "glad",
        )
        if prefix.startswith("##"):
            return True
        return not any(prefix.startswith(s) for s in starters)

    def _apply_acknowledgement(
            self,
            answer: str,
            query: str = "",
            intent: str = "",
            sentiment: str = "",
            *,
            opener: Optional[str] = None,
            style_directives: Optional[Dict[str, Any]] = None,
            follow_up: bool = False,
            force: bool = False,
    ) -> str:
        if not answer:
            return answer
        cleaned = answer.strip()
        style_directives = style_directives or {}
        if opener is None:
            opener = generate_opener(
                intent=intent or "factual",
                sentiment=sentiment or "neutral",
                follow_up=follow_up,
                style_directives=style_directives,
                query=query,
            )
        if opener and cleaned.lower().startswith(opener.lower()):
            return cleaned
        if contains_banned_opener(cleaned):
            cleaned = self._strip_first_sentence(cleaned)
        if not force and not self._needs_acknowledgement(cleaned):
            return cleaned
        if not opener:
            return cleaned
        return f"{opener}\n\n{cleaned}".strip()

    @staticmethod
    def _strip_first_sentence(text: str) -> str:
        if not text:
            return text
        parts = re.split(r"(?<=[.!?])\\s+", text.strip(), maxsplit=1)
        if len(parts) == 1:
            return text.strip()
        return parts[1].strip()

    @staticmethod
    def _relax_metadata_filters(metadata_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        relaxed = dict(metadata_filters or {})
        for key in ("section_titles", "page_numbers", "min_confidence"):
            relaxed.pop(key, None)
        if not relaxed.get("document_ids") and not relaxed.get("source_files"):
            relaxed.pop("doc_types", None)
        return relaxed

    def _filter_chunks_by_query_relevance(
            self,
            query: str,
            chunks: List[RetrievedChunk],
            *,
            min_overlap: float,
            keep_top: int,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        scored: List[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            meta = chunk.metadata or {}
            meta_text = " ".join(
                str(value)
                for value in [
                    meta.get("source_file"),
                    meta.get("doc_name"),
                    meta.get("document_title"),
                    meta.get("title"),
                    meta.get("section_title") or meta.get("section"),
                ]
                if value
            )
            overlap_text = _token_overlap_score(query, chunk.text or "")
            overlap_meta = _token_overlap_score(query, meta_text)
            relevance = max(overlap_text, overlap_meta)
            meta["query_relevance"] = round(relevance, 4)
            chunk.metadata = meta
            chunk.score = float(chunk.score) + (relevance * 0.35)
            scored.append((relevance, chunk))

        kept = [chunk for relevance, chunk in scored if relevance >= min_overlap]
        if len(kept) < min(keep_top, len(scored)):
            scored_sorted = sorted(scored, key=lambda item: item[0], reverse=True)
            for relevance, chunk in scored_sorted:
                if chunk in kept:
                    continue
                kept.append(chunk)
                if len(kept) >= min(keep_top, len(scored_sorted)):
                    break

        kept = sorted(kept, key=lambda c: float(c.score), reverse=True)
        return kept

    def retrieve_with_priorities(
            self,
            query: str,
            user_id: str,
            profile_id: str,
            collection_name: str,
            subscription_id: str,
            namespace: str,
            top_k_retrieval: int = 50,
            query_analysis_override: Optional[Any] = None,
            profile_context_override: Optional[Dict[str, Any]] = None,
            strict_relevance: bool = False,
            disable_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval with mandatory profile filters, query rewriting, and intent detection.
        """
        if not profile_id:
            raise ValueError("profile_id is required for retrieval")

        if profile_context_override:
            profile_context_data = profile_context_override
        else:
            profile_context = self.retriever.get_profile_context(collection_name, profile_id)
            profile_context_data = {
                "keywords": profile_context.top_keywords[:12],
                "hints": profile_context.document_hints[:6],
                "sampled_chunks": profile_context.total_chunks
            }

        person_name = self.extract_person_name_from_query(query)
        target_document_id = None
        target_document_ids: List[str] = []

        if person_name:
            logger.info(f" Detected person-specific query for: '{person_name}'")
            target_document_id = self.find_document_id_by_name(collection_name, person_name, profile_id)

            if target_document_id:
                logger.info(f" Will filter results to document: {target_document_id}")
            else:
                logger.warning(f" Could not find document for '{person_name}' - will search all docs")

        is_vague = self._is_query_vague(query)
        attempt_records = []

        primary_query, primary_metadata = self.preprocess_query(query, user_id, namespace, use_reformulation=True)
        primary_metadata["vague_query"] = is_vague
        primary_metadata["profile_context"] = profile_context_data
        primary_metadata["person_name"] = person_name
        primary_metadata["target_document_id"] = target_document_id
        primary_metadata["strict_relevance"] = bool(strict_relevance)

        query_understanding = self.query_understanding.analyze(primary_query, profile_context_data)
        query_analysis = query_analysis_override or self.query_intelligence.analyze(primary_query, profile_context_data)

        kg_probe = None
        kg_cache_hit = False
        kg_future = None
        kg_executor = None
        retrieval_cache_hit = False
        if self.graph_guided_retriever:
            try:
                kg_probe = self.graph_guided_retriever.get_cached_probe(
                    tenant=str(subscription_id or profile_id),
                    collection=collection_name,
                    query=primary_query,
                )
                if kg_probe:
                    kg_cache_hit = True
                else:
                    kg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    kg_future = kg_executor.submit(
                        self.graph_guided_retriever.probe,
                        tenant=str(subscription_id or profile_id),
                        collection=collection_name,
                        query=primary_query,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("KG probe scheduling failed: %s", exc)

        primary_metadata["intent"] = query_understanding.intent
        primary_metadata["query_understanding"] = {
            "intent": query_understanding.intent,
            "sub_queries": query_understanding.sub_queries,
            "expanded_queries": query_understanding.expanded_queries,
            "metadata_filters": query_understanding.metadata_filters,
            "explicit_hints": query_understanding.explicit_hints,
        }
        primary_metadata["query_analysis"] = {
            "intent": query_analysis.intent,
            "sub_queries": query_analysis.sub_queries,
            "expanded_query": query_analysis.expanded_query,
            "expansion_terms": query_analysis.expansion_terms,
            "metadata_filters": query_analysis.metadata_filters,
            "used_llm": query_analysis.used_llm,
            "sentiment": query_analysis.sentiment,
            "instructions": query_analysis.instructions,
        }

        metadata_filters = {**(query_analysis.metadata_filters or {}), **(query_understanding.metadata_filters or {})}

        doc_types = metadata_filters.get("doc_types")
        section_titles = metadata_filters.get("section_titles")
        page_numbers = metadata_filters.get("page_numbers")
        min_confidence = metadata_filters.get("min_confidence")
        source_files = metadata_filters.get("source_files")

        entity_focus = self._resolve_entity_focus(
            collection_name=collection_name,
            profile_id=profile_id,
            metadata_filters=metadata_filters,
            person_name=person_name,
            intent=query_analysis.intent,
        )
        primary_metadata["entity_focus"] = entity_focus
        if entity_focus.get("document_ids"):
            target_document_ids = list(entity_focus.get("document_ids") or [])
            target_document_id = target_document_ids[0] if target_document_ids else target_document_id
            logger.info(
                "Entity focus documents: %s",
                target_document_ids,
            )

        if entity_focus.get("section_titles"):
            section_titles = list(dict.fromkeys((section_titles or []) + list(entity_focus.get("section_titles") or [])))

        strict_doc_focus = bool(target_document_id or target_document_ids)

        broad_multiplier = float(getattr(Config.Retrieval, "BROAD_RECALL_MULTIPLIER", 1.5))
        broad_threshold = float(getattr(Config.Retrieval, "BROAD_RECALL_THRESHOLD", 0.02))
        broad_top_k = max(int(top_k_retrieval), int(top_k_retrieval * broad_multiplier))

        def _annotate_chunks(chunks: List[RetrievedChunk], variant: str) -> None:
            for chunk in chunks:
                meta = chunk.metadata or {}
                methods = set(meta.get("methods") or [])
                if chunk.method:
                    methods.add(chunk.method)
                meta["methods"] = list(methods) if methods else meta.get("methods", [])
                variants = meta.get("query_variants") or []
                if variant and variant not in variants:
                    variants.append(variant)
                meta["query_variants"] = variants
                chunk.metadata = meta

        def _merge_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
            merged: Dict[str, RetrievedChunk] = {}
            for chunk in chunks:
                meta = chunk.metadata or {}
                key = meta.get("chunk_id") or chunk.id
                if key in merged:
                    existing = merged[key]
                    existing_meta = existing.metadata or {}
                    incoming_meta = meta
                    methods = set(existing_meta.get("methods") or [])
                    methods.update(incoming_meta.get("methods") or [])
                    if chunk.method:
                        methods.add(chunk.method)
                    if methods:
                        existing_meta["methods"] = list(methods)
                    variants = set(existing_meta.get("query_variants") or [])
                    variants.update(incoming_meta.get("query_variants") or [])
                    if variants:
                        existing_meta["query_variants"] = list(variants)
                    existing.metadata = existing_meta
                    if float(chunk.score) > float(existing.score):
                        existing.score = chunk.score
                        if chunk.text:
                            existing.text = chunk.text
                else:
                    merged[key] = chunk
            return list(merged.values())

        def _candidate_to_chunk(candidate: Any) -> RetrievedChunk:
            meta = candidate.metadata or {}
            meta["hybrid_vector_score"] = candidate.vector_score
            meta["hybrid_lexical_score"] = candidate.lexical_score
            if candidate.boosts:
                meta["hybrid_boosts"] = candidate.boosts
            return RetrievedChunk(
                id=str(candidate.id),
                text=candidate.text,
                score=float(candidate.score),
                metadata=meta,
                source=candidate.source,
                method=candidate.method,
            )

        query_variants = []
        for variant in query_understanding.query_variants + (query_analysis.query_variants or []):
            if variant not in query_variants:
                query_variants.append(variant)
        if not query_variants:
            query_variants = [primary_query]

        if kg_future and not kg_probe:
            try:
                if kg_future.done():
                    kg_probe = kg_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.debug("KG probe completion failed: %s", exc)

        doc_id_filter = target_document_ids or ([target_document_id] if target_document_id else None)
        kg_doc_ids = []
        kg_section_paths: List[str] = []
        if kg_probe and kg_probe.document_ids:
            kg_doc_ids = list(kg_probe.document_ids)[: int(getattr(Config.Retrieval, "KG_DOC_FILTER_LIMIT", 8))]
            kg_section_paths = list(kg_probe.section_paths or [])
            if not doc_id_filter:
                doc_id_filter = kg_doc_ids
        filter_payload = {
            "document_ids": doc_id_filter,
            "source_files": source_files,
            "doc_types": doc_types,
            "section_titles": section_titles,
            "section_paths": kg_section_paths,
            "page_numbers": page_numbers,
            "min_confidence": min_confidence,
            "document_hints": metadata_filters.get("document_hints"),
        }
        if kg_probe:
            primary_metadata["kg_probe"] = {
                "source": kg_probe.source,
                "doc_hits": len(kg_probe.document_ids or []),
                "section_hits": len(kg_probe.section_paths or []),
                "cache_hit": bool(kg_cache_hit),
            }

        normalized_query = query_analysis.normalized_query or primary_query
        cache_key = f"retrieval:{subscription_id}:{collection_name}:{hash_query(normalized_query)}:{query_analysis.intent or 'factual'}"
        cacheable = not strict_doc_focus and not any([doc_types, section_titles, page_numbers, min_confidence, source_files])
        if self.redis_cache and cacheable and not strict_relevance and not disable_cache:
            cached_payload = self.redis_cache.get_json(cache_key, feature="retrieval")
            if cached_payload:
                retrieval_cache_hit = True
                cached_chunks = []
                for item in cached_payload.get("chunks", []):
                    cached_chunks.append(
                        RetrievedChunk(
                            id=str(item.get("id")),
                            text=item.get("text", ""),
                            score=float(item.get("score", 0.0)),
                            metadata=item.get("metadata") or {},
                            source=item.get("source"),
                            method=item.get("method", "cache"),
                        )
                    )
                if cached_chunks:
                    attempt_records.append(
                        {
                            "label": "retrieval_cache",
                            "query": primary_query,
                            "hits": len(cached_chunks),
                            "top_score": round(float(cached_chunks[0].score), 4),
                            "document_filter": doc_id_filter,
                        }
                    )
                    cache_metrics = get_metrics_store()
                    if cache_metrics.available:
                        cache_metrics.record(
                            counters={
                                "retrieval_cache_hit": 1,
                                "kgprobe_cache_hit": 1 if kg_cache_hit else 0,
                                "kgprobe_cache_miss": 0 if kg_cache_hit else 1,
                            }
                        )
                    if kg_executor:
                        kg_executor.shutdown(wait=False)
                    return {
                        "chunks": cached_chunks,
                        "query": primary_query,
                        "metadata": primary_metadata,
                        "attempts": attempt_records,
                        "selected_strategy": "cache",
                        "profile_context": profile_context_data,
                    }

        dynamic_multiplier = 1.0 + min(1.5, (len(query_variants) * 0.25) + (len(primary_query.split()) / 20))
        dynamic_top_k = max(12, int(top_k_retrieval * dynamic_multiplier))

        query_analysis_payload = {
            "intent": query_analysis.intent,
            "sub_queries": query_analysis.sub_queries,
            "expanded_query": query_analysis.expanded_query,
            "metadata_filters": query_analysis.metadata_filters,
            "sentiment": query_analysis.sentiment,
            "instructions": query_analysis.instructions,
        }
        query_understanding_payload = {
            "intent": query_understanding.intent,
            "sub_queries": query_understanding.sub_queries,
            "expanded_queries": query_understanding.expanded_queries,
            "metadata_filters": query_understanding.metadata_filters,
            "explicit_hints": query_understanding.explicit_hints,
        }

        agent_memory = AgentMemory()
        parallel_plan = self.retriever_manager.run(
            query=primary_query,
            profile_id=profile_id,
            collection_name=collection_name,
            query_analysis=query_analysis_payload,
            query_understanding=query_understanding_payload,
            metadata_filters=filter_payload,
            top_k=dynamic_top_k,
            memory=agent_memory,
            evidence_plan=None,
            profile_context=profile_context_data,
        )
        chunks = parallel_plan.chunks or []
        if kg_future and not kg_probe:
            try:
                if kg_future.done():
                    kg_probe = kg_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.debug("KG probe failed: %s", exc)
        if kg_executor:
            kg_executor.shutdown(wait=False)

        if kg_probe and chunks:
            doc_set = set(kg_probe.document_ids or [])
            section_set = set([str(s).strip().lower() for s in (kg_probe.section_paths or [])])
            for chunk in chunks:
                meta = chunk.metadata or {}
                doc_id = str(meta.get("document_id") or "")
                if doc_id and doc_id in doc_set:
                    chunk.score = float(chunk.score) + 0.12
                    meta["kg_boost"] = True
                section_path = (meta.get("section_path") or meta.get("section_title") or "").strip().lower()
                if section_path and section_path in section_set:
                    chunk.score = float(chunk.score) + 0.08
                    meta["kg_section_boost"] = True
                chunk.metadata = meta
            chunks = sorted(chunks, key=lambda c: float(c.score), reverse=True)
        attempt_records.extend(
            [
                {
                    "label": attempt.label,
                    "query": attempt.query,
                    "hits": attempt.hits,
                    "top_score": attempt.top_score,
                    "document_filter": doc_id_filter,
                }
                for attempt in parallel_plan.attempts
            ]
        )

        if not chunks:
            fallback_limit = int(getattr(Config.Retrieval, "METADATA_FALLBACK_LIMIT", 200))
            fallback_filters = filter_payload if strict_doc_focus else {}
            lexical_chunks = self.retriever._lexical_fallback_retrieve(
                collection_name=collection_name,
                query=primary_query,
                profile_id=profile_id,
                filters=fallback_filters,
                top_k=min(40, fallback_limit),
            )
            if lexical_chunks:
                attempt_records.append(
                    {
                        "label": "metadata_lexical_fallback",
                        "query": primary_query,
                        "hits": len(lexical_chunks),
                        "top_score": round(float(lexical_chunks[0].score), 4) if lexical_chunks else 0.0,
                        "document_filter": None,
                    }
                )
                chunks = lexical_chunks

        if (
            not chunks
            and not strict_doc_focus
            and (target_document_id or doc_types or section_titles or page_numbers or min_confidence or source_files)
        ):
            fallback_candidates = self.hybrid_retriever.retrieve(
                collection_name=collection_name,
                query=primary_query,
                profile_id=profile_id,
                query_id=namespace,
                top_k=broad_top_k,
                filters={"document_ids": None},
                explicit_hints={},
            )
            fallback_chunks = [_candidate_to_chunk(c) for c in fallback_candidates]
            _annotate_chunks(fallback_chunks, primary_query)
            attempt_records.append(
                {
                    "label": "hybrid_no_filters",
                    "query": primary_query,
                    "hits": len(fallback_chunks),
                    "top_score": round(float(fallback_chunks[0].score), 4) if fallback_chunks else 0.0,
                    "document_filter": None,
                }
            )
            chunks = _merge_chunks(fallback_chunks)

        if person_name and chunks:
            self._boost_chunks_for_person(chunks, person_name)
        if chunks:
            entity_values: List[str] = []
            for key in ("product_names", "place_names", "event_names"):
                for value in metadata_filters.get(key) or []:
                    if value:
                        entity_values.append(str(value))
            if entity_values:
                self._boost_chunks_for_entities(chunks, entity_values)
            min_overlap = float(getattr(Config.Retrieval, "MIN_QUERY_OVERLAP", 0.06))
            keep_top = int(getattr(Config.Retrieval, "RELEVANCE_KEEP_TOP_K", 12))
            effective_min = min_overlap * (0.6 if strict_doc_focus else 1.0)
            if strict_relevance:
                effective_min = max(min_overlap * 1.4, min_overlap + 0.04)
            chunks = self._filter_chunks_by_query_relevance(
                primary_query,
                chunks,
                min_overlap=effective_min,
                keep_top=keep_top,
            )

        logger.info(
            "Retrieval plan completed",
            extra={
                "query_id": namespace,
                "strategy": "parallel_hybrid",
                "attempts": attempt_records,
                "candidate_count": len(chunks),
                "top_score": round(float(chunks[0].score), 4) if chunks else 0.0,
                "top_k": dynamic_top_k,
                "threshold": broad_threshold,
            },
        )

        if self.redis_cache and cacheable and chunks:
            cached_chunks_payload = []
            for chunk in chunks[: max(10, int(top_k_retrieval))]:
                cached_chunks_payload.append(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "score": float(chunk.score),
                        "metadata": chunk.metadata or {},
                        "source": chunk.source,
                        "method": chunk.method,
                    }
                )
            self.redis_cache.set_json(
                cache_key,
                stamp_cache_payload({"chunks": cached_chunks_payload}),
                feature="retrieval",
                ttl=int(getattr(Config.Retrieval, "KG_RETRIEVAL_CACHE_TTL_SECONDS", 240)),
            )

        cache_metrics = get_metrics_store()
        if cache_metrics.available:
            counters = {}
            if cacheable and not retrieval_cache_hit:
                counters["retrieval_cache_miss"] = 1
            if self.graph_guided_retriever:
                counters["kgprobe_cache_hit"] = 1 if kg_cache_hit else 0
                counters["kgprobe_cache_miss"] = 0 if kg_cache_hit else 1
            if counters:
                cache_metrics.record(counters=counters)

        return {
            "chunks": chunks,
            "query": primary_query,
            "metadata": primary_metadata,
            "attempts": attempt_records,
            "selected_strategy": "parallel_hybrid",
            "profile_context": profile_context_data,
        }


    def _warm_up_llm(self):
        """Warm LLM backend so first user calls do not fail cold."""
        warm_fn = getattr(self.llm_client, "warm_up", None)
        if callable(warm_fn):
            try:
                warm_fn()
            except Exception as e:
                logger.warning(f"LLM warm-up skipped due to error: {e}")

    def answer_question(
            self,
            query: str,
            profile_id: str,
            subscription_id: str,
            user_id: str,
            persona: str = "professional document analysis assistant",
            top_k_retrieval: int = int(getattr(Config.Retrieval, "TOPK_DENSE", 50)),
            top_k_rerank: int = int(getattr(Config.Retrieval, "TOPK_RERANK", 20)),
            final_k: int = int(getattr(Config.Retrieval, "FINAL_CONTEXT_CHUNKS", 8)),
            session_id: Optional[str] = None,
            new_session: bool = False,
            disable_answer_cache: bool = False,
            force_refresh: bool = False,
            request_id: Optional[str] = None,
            index_version: Optional[str] = None,
            tools: Optional[List[str]] = None,
            use_tools: bool = False,
            tool_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Main method to answer questions using enhanced RAG pipeline."""
        start_time = time.time()
        telemetry = telemetry_store() if METRICS_V2_ENABLED else None
        metrics_store = get_metrics_store()
        tool_list = tools or []
        use_tooling = bool(use_tools or tool_list)
        global _FIRST_METRICS_REQUEST
        is_cold_start = False
        if _FIRST_METRICS_REQUEST:
            is_cold_start = True
            _FIRST_METRICS_REQUEST = False

        def _sanitize_response(payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return sanitize_user_payload(payload)
            except Exception:
                return payload

        try:
            if not profile_id:
                raise ValueError("profile_id is required for retrieval")
            try:
                collection_name = _resolve_collection_name(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"code": "qdrant_collection_unresolved", "message": str(exc)}},
                ) from exc
            logger.info(
                "Resolved Qdrant collection",
                extra={
                    "collection_name": collection_name,
                    "profile_id": profile_id,
                    "subscription_id": subscription_id,
                    "request_id": request_id,
                    "session_id": session_id,
                },
            )
            embedding_override = resolve_embedding_model_for_generation(self.model_name)
            suffix_map = getattr(Config.ModelArbitration, "INDEX_SUFFIX_MAP", {}) or {}
            suffix = None
            if embedding_override:
                suffix = suffix_map.get(embedding_override) or suffix_map.get(self.model_name)
            if suffix:
                candidate_collection = f"{collection_name}__{suffix}"
                try:
                    self.retriever.ensure_collection_exists(candidate_collection)
                    collection_name = candidate_collection
                except Exception:
                    self.retriever.ensure_collection_exists(collection_name)
            else:
                self.retriever.ensure_collection_exists(collection_name)
            namespace = _build_namespace(subscription_id, profile_id, self.model_name, session_id)
            metrics = get_metrics_tracker()

            effective_new_session = new_session or force_refresh
            if effective_new_session:
                logger.info("Resetting conversation context for new session: %s", session_id or "default")
                self.conversation_history.clear_history(namespace, user_id)
                try:
                    self.feedback_memory.clear(namespace, user_id)
                except Exception as feedback_exc:
                    logger.debug("Feedback memory clear failed: %s", feedback_exc)

            # Quick collection diagnostics to avoid silent empty searches
            try:
                stats = self.client.count(collection_name=collection_name, exact=False)
                total_points = getattr(stats, "count", 0)
                logger.info(f"Collection '{collection_name}' point count: {total_points}")
                if total_points == 0:
                    logger.warning(f"Collection '{collection_name}' is empty; retrieval will return no results")
            except Exception as diag_exc:
                logger.warning(f"Could not count collection '{collection_name}': {diag_exc}")

            feedback_prefix = ""
            original_query = query
            strict_relevance = False
            repair_mode = False
            sentiment_seed_query = query
            if self.greeting_handler.is_negative_feedback(query):
                apology = "Sorry about that - I'll broaden the search and tighten the grounding."
                repair_mode = True
                if self.greeting_handler._has_question_cue(query):
                    feedback_prefix = apology
                else:
                    last_question = self.conversation_history.get_last_user_message(namespace, user_id)
                    if last_question:
                        feedback_prefix = apology
                        query = last_question
                        sentiment_seed_query = original_query
                    else:
                        response_text = (
                            f"{apology} Please share the question again so I can fix it accurately."
                        )
                        self.conversation_history.add_turn(namespace, user_id, original_query, response_text)
                        return _sanitize_response({
                            "response": response_text,
                            "sources": [],
                            "user_id": user_id,
                            "collection": collection_name,
                            "request_id": request_id,
                            "index_version": index_version,
                            "context_found": False,
                            "query_type": "feedback_negative",
                            "grounded": False
                        })

            def _apply_feedback_prefix(text: str) -> str:
                if not feedback_prefix:
                    return text
                cleaned = (text or "").strip()
                if cleaned.lower().startswith(feedback_prefix.lower()):
                    return text
                return f"{feedback_prefix}\n\n{cleaned}" if cleaned else feedback_prefix

            if self.greeting_handler.is_positive_feedback(query):
                response_text = (
                    "Happy to help. Got it - I'll keep this preference for this session. "
                    "If you want me to dig into another document or topic, just let me know."
                )
                self.conversation_history.add_turn(namespace, user_id, query, response_text)
                return _sanitize_response({
                    "response": response_text,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "positive_feedback",
                    "grounded": True
                })

            if self.greeting_handler.is_greeting(query):
                greeting_response = (
                    f"Hi! I'm your {persona}. I can search your documents and answer questions. "
                    f"What would you like to explore?"
                )
                self.conversation_history.add_turn(namespace, user_id, query, greeting_response)

                return _sanitize_response({
                    "response": greeting_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "greeting",
                    "grounded": True
                })

            if self.greeting_handler.is_farewell(query):
                farewell_response = "Thanks for chatting. If you need anything else, come back anytime."
                self.conversation_history.clear_history(namespace, user_id)

                return _sanitize_response({
                    "response": farewell_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "farewell",
                    "grounded": True
                })

            logger.info(f"Processing query for collection '{collection_name}': {query[:100]}")

            profile_context = self.retriever.get_profile_context(collection_name, profile_id)
            profile_context_data = {
                "keywords": profile_context.top_keywords[:12],
                "hints": profile_context.document_hints[:6],
                "sampled_chunks": profile_context.total_chunks,
            }
            pre_query_analysis = self.query_intelligence.analyze(query, profile_context_data)
            conversation_context = self.conversation_history.get_context(
                namespace,
                user_id,
                max_turns=int(getattr(Config.Chat, "CONTEXT_TURNS", 4)),
            )
            is_follow_up = bool(conversation_context) and not effective_new_session
            companion_result = self.companion_classifier.classify(
                user_query=sentiment_seed_query,
                last_turns_summary=conversation_context,
                intent_from_query_intelligence=pre_query_analysis.intent,
                session_id=session_id,
            )
            companion_sentiment = companion_result.sentiment
            companion_style = dict(companion_result.style_directives or {})
            companion_intent = companion_result.intent_expected
            if companion_sentiment == "frustrated":
                repair_mode = True
            if repair_mode:
                companion_style["tone"] = "supportive"
                companion_style["humor_level"] = 0
                feedback_prefix = ""
            clarification = self.clarification_agent.decide(
                query,
                conversation_context=conversation_context,
                query_analysis={
                    "intent": pre_query_analysis.intent,
                    "metadata_filters": pre_query_analysis.metadata_filters,
                },
                profile_context=profile_context_data,
            )
            if clarification.should_clarify:
                clarification_response = clarification.question or "Which document or section should I focus on?"
                self.conversation_history.add_turn(namespace, user_id, query, clarification_response)
                return _sanitize_response({
                    "response": clarification_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "clarification",
                    "grounded": True,
                    "clarification_reason": clarification.reason,
                })

            if repair_mode:
                top_k_retrieval = max(top_k_retrieval, int(top_k_retrieval * 1.5))
                top_k_rerank = max(top_k_rerank, int(top_k_rerank * 1.5))
                pre_query_analysis.metadata_filters = self._relax_metadata_filters(
                    pre_query_analysis.metadata_filters
                )

            retrieval_start = time.time()
            retrieval_plan = self.retrieve_with_priorities(
                query=query,
                user_id=user_id,
                profile_id=profile_id,
                collection_name=collection_name,
                subscription_id=subscription_id,
                namespace=namespace,
                top_k_retrieval=top_k_retrieval,
                query_analysis_override=pre_query_analysis,
                profile_context_override=profile_context_data,
                strict_relevance=strict_relevance,
                disable_cache=repair_mode,
            )

            # Use the retrieval plan output directly to avoid diverging queries and
            # ensure reranking operates on the best available candidates.
            processed_query = retrieval_plan.get("query", query)
            preprocessing_metadata = retrieval_plan.get("metadata", {})
            retrieved_chunks = retrieval_plan.get("chunks") or []

            fast_analysis = self.fast_query_analyzer.analyze(processed_query)
            preprocessing_metadata["fast_query_analysis"] = fast_analysis.as_dict()
            evidence_requirements = EvidenceRequirements.from_required_evidence(fast_analysis.required_evidence)
            retrieved_chunks = self.hybrid_ranker.rank(
                processed_query,
                retrieved_chunks,
                evidence_requirements,
                intent_type=fast_analysis.intent_type,
            )
            retrieval_quality = self.retrieval_quality.evaluate(
                processed_query,
                retrieved_chunks,
                evidence_requirements,
            )
            logger.info(
                "Retrieval guardrail",
                extra={
                    "query_id": request_id,
                    "intent_type": fast_analysis.intent_type,
                    "output_format": fast_analysis.output_format_preference,
                    "explicitness": round(fast_analysis.explicitness_score, 3),
                    "evidence_constraints": fast_analysis.required_evidence,
                    "retrieval_quality": retrieval_quality.score,
                    "quality_breakdown": retrieval_quality.breakdown,
                    "quality_low": retrieval_quality.is_low,
                },
            )

            retrieval_attempts = retrieval_plan.get("attempts", [])
            selected_strategy = retrieval_plan.get("selected_strategy", "direct_qdrant")
            profile_context_data = retrieval_plan.get("profile_context", {})
            mrr_score = 0.0
            target_doc_id = preprocessing_metadata.get("target_document_id")
            if target_doc_id:
                for rank, chunk in enumerate(retrieved_chunks, start=1):
                    doc_id = (chunk.metadata or {}).get("document_id")
                    if doc_id and str(doc_id) == str(target_doc_id):
                        mrr_score = 1.0 / rank
                        break
            retrieval_hits = 1.0 if retrieved_chunks else 0.0
            retrieval_latency_ms = (time.time() - retrieval_start) * 1000
            top_scores = [float(chunk.score) for chunk in retrieved_chunks[:5]] if retrieved_chunks else []
            avg_topk_score = (sum(top_scores) / len(top_scores)) if top_scores else 0.0
            score_gap = (top_scores[0] - avg_topk_score) if top_scores else 0.0

            retrieval_hit_rate = None
            gold_refs = preprocessing_metadata.get("gold_doc_ids") or preprocessing_metadata.get("gold_docs")
            if gold_refs:
                retrieved_doc_ids = {
                    str((chunk.metadata or {}).get("document_id")) for chunk in retrieved_chunks if (chunk.metadata or {}).get("document_id")
                }
                retrieval_hit_rate = 1.0 if any(str(ref) in retrieved_doc_ids for ref in gold_refs) else 0.0
            retrieval_metric_kwargs = {
                "avg_topk_score": avg_topk_score,
                "score_gap": score_gap,
                "retrieval_hit_rate": retrieval_hit_rate,
                "retrieval_latency_ms": retrieval_latency_ms,
            }

            def _candidate_to_chunk(candidate: Any) -> RetrievedChunk:
                meta = candidate.metadata or {}
                meta["hybrid_vector_score"] = candidate.vector_score
                meta["hybrid_lexical_score"] = candidate.lexical_score
                if candidate.boosts:
                    meta["hybrid_boosts"] = candidate.boosts
                return RetrievedChunk(
                    id=str(candidate.id),
                    text=candidate.text,
                    score=float(candidate.score),
                    metadata=meta,
                    source=candidate.source,
                    method=candidate.method,
                )

            metadata_filters = (preprocessing_metadata.get("query_analysis") or {}).get("metadata_filters") or {}
            fallback_filters = {
                "document_ids": metadata_filters.get("document_ids")
                or ([preprocessing_metadata.get("target_document_id")] if preprocessing_metadata.get("target_document_id") else None),
                "source_files": metadata_filters.get("source_files"),
                "doc_types": metadata_filters.get("doc_types"),
                "section_titles": metadata_filters.get("section_titles"),
                "page_numbers": metadata_filters.get("page_numbers"),
                "min_confidence": metadata_filters.get("min_confidence"),
                "document_hints": metadata_filters.get("document_hints")
                or (fast_analysis.required_evidence.get("section_hints") if fast_analysis.required_evidence else None),
            }

            def _fallback_retrieve(rewrite_query: str, top_k: int) -> List[RetrievedChunk]:
                candidates = self.hybrid_retriever.retrieve(
                    collection_name=collection_name,
                    query=rewrite_query,
                    profile_id=profile_id,
                    query_id=namespace,
                    top_k=top_k,
                    filters=fallback_filters,
                    explicit_hints={},
                )
                return [_candidate_to_chunk(c) for c in candidates]

            if retrieval_quality.is_low and not use_tooling:
                fallback_result = self.fallback_repair.repair(
                    query=processed_query,
                    chunks=retrieved_chunks,
                    requirements=evidence_requirements,
                    quality=retrieval_quality,
                    retrieve_fn=_fallback_retrieve,
                    top_k=top_k_retrieval,
                )
                if fallback_result.used_fallback:
                    retrieved_chunks = fallback_result.chunks
                    logger.info(
                        "Retrieval fallback triggered",
                        extra={
                            "query_id": request_id,
                            "reason": fallback_result.reason,
                            "attempts": fallback_result.attempts,
                            "hits": len(retrieved_chunks),
                        },
                    )
                    retrieval_attempts.append(
                        {
                            "label": "guardrail_repair",
                            "query": processed_query,
                            "hits": len(retrieved_chunks),
                            "top_score": round(float(retrieved_chunks[0].score), 4) if retrieved_chunks else 0.0,
                            "document_filter": fallback_filters.get("document_ids") or None,
                        }
                    )
                    retrieval_quality = self.retrieval_quality.evaluate(
                        processed_query,
                        retrieved_chunks,
                        evidence_requirements,
                    )

            required_attributes = self._extract_required_attributes(processed_query, fast_analysis.intent_type)
            intent_filtered_chunks = self._filter_chunks_by_intent(
                retrieved_chunks,
                required_attributes,
                fast_analysis.entities,
                fast_analysis.intent_type,
            )

            answer_requirements = self._extract_answer_requirements(processed_query, fast_analysis.intent_type)
            missing_requirements = self._validate_answer_requirements(intent_filtered_chunks, answer_requirements)
            if fast_analysis.intent_type == "comparison" and fast_analysis.comparison_entities:
                missing_entities = []
                for entity in fast_analysis.comparison_entities:
                    entity_lower = entity.lower()
                    if not any(entity_lower in (chunk.text or "").lower() for chunk in intent_filtered_chunks):
                        missing_entities.append(entity)
                if missing_entities:
                    missing_requirements.append("comparison_entities")

            intent_repair_attempted = False
            if (not intent_filtered_chunks or missing_requirements) and not use_tooling:
                intent_repair_attempted = True
                repair_top_k = max(top_k_retrieval, int(top_k_retrieval * 1.25))
                repaired_chunks = _fallback_retrieve(processed_query, repair_top_k)
                repaired_chunks = self._filter_chunks_by_intent(
                    repaired_chunks,
                    required_attributes,
                    fast_analysis.entities,
                    fast_analysis.intent_type,
                )
                missing_requirements = self._validate_answer_requirements(repaired_chunks, answer_requirements)
                if repaired_chunks:
                    retrieval_attempts.append(
                        {
                            "label": "intent_repair_retry",
                            "query": processed_query,
                            "hits": len(repaired_chunks),
                            "top_score": round(float(repaired_chunks[0].score), 4) if repaired_chunks else 0.0,
                            "document_filter": fallback_filters.get("document_ids") or None,
                        }
                    )
                intent_filtered_chunks = repaired_chunks

            if not intent_filtered_chunks and not use_tooling:
                entity_name = (
                    (preprocessing_metadata.get("person_name") or "")
                    or (fast_analysis.entities[0] if fast_analysis.entities else "")
                )
                intent_label = ""
                if required_attributes:
                    if "education" in required_attributes:
                        intent_label = "education"
                    elif "experience" in required_attributes:
                        intent_label = "experience"
                    elif "skills" in required_attributes:
                        intent_label = "skills"
                    elif "certification" in required_attributes:
                        intent_label = "certifications"
                    elif "contact" in required_attributes:
                        intent_label = "contact"
                response_text = self._build_intent_miss_response(entity_name, intent_label or "relevant")
                response_text = response_text + " If you want me to recheck a specific document, just tell me which one."
                self.conversation_history.add_turn(namespace, user_id, query, response_text)
                return _sanitize_response({
                    "response": response_text,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,
                    "query_type": "clarification_needed",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,
                })

            if missing_requirements and not use_tooling:
                entity_name = (
                    (preprocessing_metadata.get("person_name") or "")
                    or (fast_analysis.entities[0] if fast_analysis.entities else "")
                )
                intent_label = "education" if "education" in (required_attributes or []) else "details"
                response_text = self._build_intent_miss_response(entity_name, intent_label)
                response_text = response_text + " I can recheck if you point me to a specific document or section."
                self.conversation_history.add_turn(namespace, user_id, query, response_text)
                return _sanitize_response({
                    "response": response_text,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,
                    "query_type": "clarification_needed",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,
                })

            if intent_repair_attempted:
                retrieved_chunks = intent_filtered_chunks
            else:
                retrieved_chunks = intent_filtered_chunks or retrieved_chunks

            retrieval_quality = self.retrieval_quality.evaluate(
                processed_query,
                retrieved_chunks,
                evidence_requirements,
            )

            def _record_request_metrics(
                *,
                query_type: str,
                answer_text: str,
                context_text: str,
                context_found: bool,
                grounded: bool,
                has_citations: bool,
                processing_seconds: float,
                prompt_text: Optional[str] = None,
                tool_successes: int = 0,
                document_ids: Optional[List[str]] = None,
                confidence: Optional[float] = None,
                support_score: Optional[float] = None,
                coverage_score: Optional[float] = None,
                numeric_support_rate: Optional[float] = None,
                avg_topk_score: Optional[float] = None,
                score_gap: Optional[float] = None,
                retrieval_hit_rate: Optional[float] = None,
                retrieval_latency_ms: Optional[float] = None,
                rerank_latency_ms: Optional[float] = None,
                context_latency_ms: Optional[float] = None,
            ) -> None:
                if not metrics_store.available:
                    return
                prompt_tokens = _approx_token_count(prompt_text or query)
                completion_tokens = _approx_token_count(answer_text)
                # Embedding similarity is sampled; token overlap is the low-cost fallback.
                sentence_similarity = _semantic_similarity(query, processed_query or query)
                semantic_preservation = _semantic_similarity(context_text, answer_text)
                answer_relevance = _semantic_similarity(query, answer_text)
                # Grounding blends retrieval context presence with citation signals.
                context_grounding = 0.5 * float(bool(context_found)) + 0.5 * float(bool(has_citations))
                # Hallucination risk inversely tracks preservation + grounding.
                hallucination_risk = max(0.0, min(1.0, 1.0 - ((semantic_preservation + context_grounding) / 2.0)))
                response_consistency = 1.0 if has_citations else (0.6 if grounded else 0.2)
                temperature = float(getattr(Config.LLM, "TEMPERATURE", 0.2))
                temperature_sensitivity = max(0.0, min(1.0, 1.0 - temperature))
                latency_ms = processing_seconds * 1000

                counters = {
                    "requests_total": 1,
                    "retrieval_total": 1,
                    "retrieval_hits": retrieval_hits,
                    "retrieval_grounded": 1 if grounded else 0,
                }
                if query_type == "document_qa":
                    counters["queries_answered"] = 1
                elif query_type in {"clarification_needed", "closest_answer", "evidence_only"}:
                    counters["queries_followup"] = 1
                else:
                    counters["queries_refused"] = 1
                if query_type != "error":
                    counters["requests_success"] = 1
                if use_tooling:
                    counters["requests_with_tools"] = 1
                    if tool_successes == 0:
                        counters["llm_without_tool_fallback"] = 1

                distributions = {}
                if query_type in {"error"}:
                    distributions["failure_type"] = {query_type: 1}

                values = {
                    "prompt_tokens": float(prompt_tokens),
                    "completion_tokens": float(completion_tokens),
                    "sentence_transformation_similarity_score": sentence_similarity,
                    "semantic_preservation_score": semantic_preservation,
                    "answer_faithfulness_score": semantic_preservation,
                    "answer_relevance_score": answer_relevance,
                    "context_grounding_score": context_grounding,
                    "hallucination_risk_score": hallucination_risk,
                    "response_consistency_score": response_consistency,
                    "temperature_sensitivity_score": temperature_sensitivity,
                    "mean_reciprocal_rank": mrr_score,
                    "request_latency_ms": latency_ms,
                }
                if confidence is not None:
                    values["answer_confidence_score"] = float(confidence)
                if support_score is not None:
                    values["evidence_support_score"] = float(support_score)
                if coverage_score is not None:
                    values["citation_coverage_score"] = float(coverage_score)
                if numeric_support_rate is not None:
                    values["numeric_support_rate"] = float(numeric_support_rate)
                if avg_topk_score is not None:
                    values["retrieval_avg_topk_score"] = float(avg_topk_score)
                if score_gap is not None:
                    values["retrieval_score_gap"] = float(score_gap)
                if retrieval_hit_rate is not None:
                    values["retrieval_hit_rate"] = float(retrieval_hit_rate)
                if retrieval_latency_ms is not None:
                    values["retrieval_latency_ms"] = float(retrieval_latency_ms)
                if rerank_latency_ms is not None:
                    values["rerank_latency_ms"] = float(rerank_latency_ms)
                if context_latency_ms is not None:
                    values["context_build_latency_ms"] = float(context_latency_ms)
                if is_cold_start:
                    values["cold_start_latency_ms"] = latency_ms
                else:
                    values["warm_start_latency_ms"] = latency_ms

                metrics_store.record(
                    counters=counters,
                    values=values,
                    distributions=distributions,
                    histograms={
                        "request_latency_ms": latency_ms,
                        "cold_start_latency_ms": latency_ms if is_cold_start else None,
                        "warm_start_latency_ms": latency_ms if not is_cold_start else None,
                    },
                    model_id=self.model_name,
                )
                if document_ids:
                    for doc_id in set(document_ids):
                        metrics_store.record(
                            counters=counters,
                            values=values,
                            distributions=distributions,
                            histograms={"request_latency_ms": latency_ms},
                            document_id=str(doc_id),
                            model_id=self.model_name,
                        )
            if telemetry:
                telemetry.increment("retrieval_requests_count")
                telemetry.observe("retrieval_latency_ms", retrieval_latency_ms)
                telemetry.observe("retrieval_topk", len(retrieved_chunks))
                approx_tokens = 0
                doc_ids = []
                for chunk in retrieved_chunks:
                    text = getattr(chunk, "text", "") or ""
                    approx_tokens += len(str(text).split())
                    meta = getattr(chunk, "metadata", {}) or {}
                    doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("docId")
                    if doc_id:
                        doc_ids.append(str(doc_id))
                telemetry.observe("retrieval_context_tokens", approx_tokens)
                if retrieved_chunks:
                    telemetry.increment("retrieval_hits")
                telemetry.set_gauge("last_retrieval_time", time.time())
                for doc_id in set(doc_ids):
                    telemetry.record_doc_metric(doc_id, "last_retrieval_time", time.time())

            # Cache lookup (per subscription/profile/query + recent memory fingerprint)
            cache = get_redis_client() if (ENABLE_ANSWER_CACHE and not disable_answer_cache) else None
            cache_key = None
            conversation_context_for_cache = self.conversation_history.get_context(
                namespace,
                user_id,
                max_turns=int(getattr(Config.Chat, "CONTEXT_TURNS", 4)),
            )
            memory_fingerprint = "noctx"
            if conversation_context_for_cache:
                memory_fingerprint = hashlib.md5(conversation_context_for_cache.encode("utf-8")).hexdigest()
            session_fingerprint = _slug(session_id) if session_id else "default"
            profile_context_fingerprint = "noprof"
            try:
                profile_context_fingerprint = hashlib.md5(
                    json.dumps(profile_context_data or {}, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            except Exception as fp_exc:
                logger.debug("Failed to fingerprint profile context: %s", fp_exc)
            if cache:
                cache_key = (
                    f"rag:{ANSWER_CACHE_VERSION}:"
                    f"sub={_slug(subscription_id)}:"  # ✅ Added subscription
                    f"col={_slug(collection_name)}:"
                    f"prof={_slug(profile_id)}:"
                    f"usr={hashlib.md5(user_id.encode('utf-8')).hexdigest()[:8]}:"  # ✅ Added user
                    f"sess={session_fingerprint}:"
                    f"model={_slug(self.model_name)}:"
                    f"strat={selected_strategy}:"
                    f"mem={memory_fingerprint[:8]}:"  # ✅ Shortened to avoid key bloat
                    f"q={hashlib.md5(processed_query.encode('utf-8')).hexdigest()}"
                )
                cached = cache.get(cache_key)
                if cached:
                    try:
                        cached_obj = json.loads(cached)
                        if cached_obj.get("cache_version") and cached_obj["cache_version"] != ANSWER_CACHE_VERSION:
                            raise ValueError("Cache version mismatch")
                        logger.info("Cache hit for query; ignoring because answer cache is disabled")
                    except Exception:
                        logger.warning("Failed to parse cached answer; ignoring")
            logger.info(f"Preprocessed query: {processed_query}")

            if not retrieved_chunks and not use_tooling:
                analysis_payload = preprocessing_metadata.get("query_analysis") or {}
                intent = analysis_payload.get("intent") or ""
                sentiment = companion_sentiment or analysis_payload.get("sentiment") or "neutral"
                doc_options, section_options = self._collect_focus_options([], profile_context_data)
                metadata_filters = (preprocessing_metadata.get("query_analysis") or {}).get("metadata_filters") or {}
                clarification = self._build_clarification_question(
                    query=query,
                    doc_options=doc_options,
                    section_options=section_options,
                    profile_context=profile_context_data,
                    metadata_filters=metadata_filters,
                )
                clarification = self._apply_acknowledgement(
                    clarification,
                    query=query,
                    intent=intent,
                    sentiment=sentiment,
                    style_directives=companion_style,
                    follow_up=is_follow_up,
                )
                clarification = _apply_feedback_prefix(clarification)
                self.conversation_history.add_turn(namespace, user_id, query, clarification)

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="clarification_needed",
                        context_found=False,
                        grounded=False,
                        cached=False,
                        processing_time=time.time() - start_time,
                        retrieval_stats={"initial_retrieved": 0, "final_context": 0},
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (clarification_needed) failed: %s", metric_exc)

                _record_request_metrics(
                    query_type="clarification_needed",
                    answer_text=clarification,
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query,
                    **retrieval_metric_kwargs,
                )
                self.learning_signals.record_low_confidence(
                    query=query,
                    context="",
                    answer=clarification,
                    reason="no_results_clarification",
                    metadata={
                        "profile_id": profile_id,
                        "subscription_id": subscription_id,
                        "processed_query": processed_query,
                    },
                )

                return _sanitize_response({
                    "response": clarification,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,
                    "query_type": "clarification_needed",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,
                    "processing_time": time.time() - start_time,
                })

            # Boost relevance for recently cited documents to honor session context
            recent_docs = set(self.conversation_history.get_recent_doc_ids(namespace, user_id))
            if recent_docs:
                for chunk in retrieved_chunks:
                    doc_id = (chunk.metadata or {}).get("document_id")
                    if doc_id and doc_id in recent_docs:
                        label = self._source_label_from_chunk(chunk)
                        if _token_overlap_score(processed_query, label) >= 0.05:
                            chunk.score = float(chunk.score) + 0.08

            stage_timings: Dict[str, float] = {}

            def _build_context_from_chunks(
                chunks: List[RetrievedChunk],
                rerank_top_k_override: Optional[int] = None,
                rerank_query: Optional[str] = None,
                use_heavy_rerank: bool = True,
                intent_type: str = "factual",
            ) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], str, List[Dict[str, Any]]]:
                rerank_start = time.time()
                rerank_query = rerank_query or processed_query
                rerank_top_k = rerank_top_k_override or top_k_rerank
                if use_heavy_rerank and getattr(Config.Retrieval, "RERANKER_ENABLED", True):
                    reranked = self.reranker_v2.rerank(
                        query=rerank_query,
                        candidates=chunks,
                        top_k=rerank_top_k,
                    )
                else:
                    reranked = sorted(chunks, key=lambda c: float(c.score), reverse=True)
                stage_timings["rerank_ms"] = (time.time() - rerank_start) * 1000

                if use_heavy_rerank and getattr(Config.Retrieval, "USE_ADJACENT_EXPANSION", False):
                    neighbor_window = getattr(Config.Retrieval, "NEIGHBOR_WINDOW", 2) or 2
                    neighbor_max_new = getattr(Config.Retrieval, "NEIGHBOR_MAX_NEW", 10) or 10
                    reranked = self.retriever.expand_with_neighbors(
                        collection_name=collection_name,
                        seed_chunks=reranked,
                        profile_id=profile_id,
                        window=int(neighbor_window),
                        max_new=int(neighbor_max_new)
                    )
                    reranked = sorted(reranked, key=lambda c: float(c.score), reverse=True)

                context_start = time.time()
                candidate_dicts = [
                    {
                        "text": chunk.text,
                        "score": float(chunk.score),
                        "metadata": chunk.metadata or {},
                    }
                    for chunk in reranked
                ]
                context_result = self.context_assembler.build(candidate_dicts, intent_type=intent_type)
                stage_timings["context_ms"] = (time.time() - context_start) * 1000
                ctx_text = context_result.context_text
                ctx_sources = context_result.sources

                finals: List[RetrievedChunk] = []
                for item in context_result.selected_chunks:
                    meta = item.get("metadata") or {}
                    chunk_id = meta.get("chunk_id") or meta.get("id") or meta.get("point_id") or ""
                    finals.append(
                        RetrievedChunk(
                            id=str(chunk_id),
                            text=item.get("text") or "",
                            score=float(item.get("score", 0.0)),
                            metadata=meta,
                            source=meta.get("source_file") or meta.get("source"),
                            method="context",
                        )
                    )

                logger.info(
                    "Built context with %s chunks, %s chars",
                    len(finals),
                    len(ctx_text or ""),
                )
                return reranked, finals, ctx_text, ctx_sources

            use_heavy_rerank = bool(getattr(Config.Retrieval, "RERANKER_ENABLED", True)) and (
                repair_mode
                or (
                    retrieval_quality.is_low
                    and bool(getattr(Config.Retrieval, "RETRIEVAL_RERANK_ON_LOW_QUALITY", True))
                )
                or (
                    fast_analysis.high_stakes
                    and bool(getattr(Config.Retrieval, "RETRIEVAL_RERANK_ON_HIGH_STAKES", True))
                )
            )
            reranked_chunks, final_chunks, context, context_sources = _build_context_from_chunks(
                retrieved_chunks,
                use_heavy_rerank=use_heavy_rerank,
                intent_type=fast_analysis.intent_type,
            )
            if context_sources:
                safe_sources = [
                    {
                        "document": src.get("source_name"),
                        "section": src.get("section"),
                        "page": src.get("page"),
                    }
                    for src in context_sources[:5]
                ]
                logger.info(
                    "Selected context sources",
                    extra={
                        "query_id": request_id,
                        "sources": safe_sources,
                    },
                )
            retrieval_metric_kwargs.update(
                {
                    "rerank_latency_ms": stage_timings.get("rerank_ms"),
                    "context_latency_ms": stage_timings.get("context_ms"),
                }
            )

            metadata_filters = (preprocessing_metadata.get("query_analysis") or {}).get("metadata_filters") or {}
            doc_options, section_options = self._collect_focus_options(
                final_chunks or retrieved_chunks,
                profile_context_data,
            )
            evidence_threshold = float(getattr(Config.Retrieval, "RETRIEVAL_EVIDENCE_MIN_COVERAGE", 0.6))
            if fast_analysis.explicitness_score >= 0.6:
                evidence_threshold = max(
                    evidence_threshold,
                    float(getattr(Config.Retrieval, "RETRIEVAL_EVIDENCE_STRICT_COVERAGE", 0.75)),
                )
            if fast_analysis.intent_type == "comparison":
                evidence_threshold = max(evidence_threshold, 0.7)
            evidence_status = self.evidence_constraints.evaluate(
                final_chunks,
                evidence_requirements,
                min_ratio=evidence_threshold,
            )
            if not evidence_status.satisfied and not use_tooling:
                intent_hint = fast_analysis.intent_type
                sentiment_hint = companion_sentiment or (preprocessing_metadata.get("query_analysis") or {}).get("sentiment") or "neutral"
                missing = ", ".join(evidence_status.missing) if evidence_status.missing else "required evidence"
                if fast_analysis.explicitness_score >= 0.6:
                    clarification = (
                        f"I couldn't find evidence for {missing} in the documents I searched. "
                        "Please point me to the document or section that contains it."
                    )
                else:
                    clarification = self._build_evidence_response(
                        query=query,
                        sources=context_sources or [],
                        doc_options=doc_options,
                        section_options=section_options,
                    )
                clarification = self._apply_acknowledgement(
                    clarification,
                    query=query,
                    intent=intent_hint,
                    sentiment=sentiment_hint,
                    style_directives=companion_style,
                    follow_up=is_follow_up,
                )
                clarification = _apply_feedback_prefix(clarification)
                self.conversation_history.add_turn(namespace, user_id, query, clarification)
                processing_time = time.time() - start_time
                _record_request_metrics(
                    query_type="clarification_needed",
                    answer_text=clarification,
                    context_text=context or "",
                    context_found=bool(final_chunks),
                    grounded=False,
                    has_citations=False,
                    processing_seconds=processing_time,
                    prompt_text=processed_query,
                    **retrieval_metric_kwargs,
                )
                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="clarification_needed",
                        context_found=bool(final_chunks),
                        grounded=False,
                        cached=False,
                        processing_time=processing_time,
                        retrieval_stats={
                            "initial_retrieved": len(retrieved_chunks),
                            "after_rerank": len(reranked_chunks),
                            "final_context": len(final_chunks),
                        },
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (clarification_needed) failed: %s", metric_exc)
                return _sanitize_response({
                    "response": clarification,
                    "sources": context_sources or [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": bool(final_chunks),
                    "query_type": "clarification_needed",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,
                    "processing_time": round(processing_time, 2),
                })

            tool_successes = 0
            tool_failures = 0
            if use_tooling and tool_list:
                try:
                    import asyncio
                    from src.tools.base import registry
                except Exception as tool_import_exc:  # noqa: BLE001
                    logger.warning("Tool registry not available: %s", tool_import_exc)
                    tool_list = []
                tool_chunks: List[str] = []
                for tool_name in tool_list:
                    payload = {
                        "input": {"query": query, "context": context},
                        "context": {"profile_id": profile_id, "subscription_id": subscription_id},
                        "options": {"requested_by": "rag_pipeline"},
                    }
                    extra_input = (tool_inputs or {}).get(tool_name) if tool_inputs else None
                    if isinstance(extra_input, dict):
                        payload["input"].update(extra_input)
                    elif extra_input is not None:
                        payload["input"]["value"] = extra_input
                    tool_start = time.time()
                    try:
                        tool_resp = asyncio.run(
                            registry.invoke(tool_name, payload, correlation_id=request_id)
                        )
                        if tool_resp.get("status") == "success":
                            tool_successes += 1
                            tool_result = tool_resp.get("result") or {}
                            snippet = json.dumps(tool_result, default=str)
                            tool_chunks.append(f"[{tool_name}] {snippet[:800]}")
                            for src in tool_resp.get("sources") or []:
                                meta = src.get("metadata") or {}
                                meta["tool_output"] = True
                                src["metadata"] = meta
                                context_sources.append(src)
                        else:
                            logger.warning("Tool %s returned status=%s", tool_name, tool_resp.get("status"))
                            tool_failures += 1
                        if metrics_store.available:
                            tool_latency_ms = (time.time() - tool_start) * 1000
                            metrics_store.record(
                                counters={
                                    "tool_calls": 1,
                                    "tool_success": 1 if tool_resp.get("status") == "success" else 0,
                                    "tool_failure": 0 if tool_resp.get("status") == "success" else 1,
                                },
                                values={"tool_latency_ms": tool_latency_ms},
                                histograms={"tool_latency_ms": tool_latency_ms},
                                distributions={"tool_usage": {tool_name: 1}},
                                model_id=self.model_name,
                                tool=tool_name,
                            )
                    except Exception as tool_exc:  # noqa: BLE001
                        logger.warning("Tool %s failed: %s", tool_name, tool_exc)
                        tool_failures += 1
                        if metrics_store.available:
                            tool_latency_ms = (time.time() - tool_start) * 1000
                            metrics_store.record(
                                counters={
                                    "tool_calls": 1,
                                    "tool_failure": 1,
                                },
                                values={"tool_latency_ms": tool_latency_ms},
                                histograms={"tool_latency_ms": tool_latency_ms},
                                distributions={"tool_usage": {tool_name: 1}},
                                model_id=self.model_name,
                                tool=tool_name,
                            )
                if tool_chunks:
                    context = (context or "") + "\n\n".join(tool_chunks)

            conversation_context = self.conversation_history.get_context(
                namespace,
                user_id,
                max_turns=int(getattr(Config.Chat, "SUMMARY_TURNS", 6)),
            )
            summary_relevance = _token_overlap_score(query, conversation_context)
            if summary_relevance < 0.05:
                conversation_context = ""
            conversation_summary = self.conversation_summarizer.summarize(conversation_context)
            analysis_payload = preprocessing_metadata.get("query_analysis") or {}
            intent = analysis_payload.get("intent") or ""
            sentiment = companion_sentiment or analysis_payload.get("sentiment") or "neutral"
            instructions = analysis_payload.get("instructions") or {}
            intent_map = {
                "instruction/how-to": "how-to",
                "numeric_lookup": "extraction",
                "field_extraction": "extraction",
                "summarization": "summary",
                "deep_analysis": "analysis",
                "comparison": "comparison",
                "troubleshooting": "troubleshooting",
            }
            intent_override = intent_map.get(fast_analysis.intent_type, intent or fast_analysis.intent_type)
            intent = intent_override or companion_intent or intent
            if fast_analysis.output_format_preference == "table":
                instructions = dict(instructions)
                instructions.setdefault("use_table", True)
                instructions.setdefault("format", "table")
            elif fast_analysis.output_format_preference == "steps":
                instructions = dict(instructions)
                instructions.setdefault("step_by_step", True)
                instructions.setdefault("format", "steps")
            elif fast_analysis.output_format_preference == "bullets":
                instructions = dict(instructions)
                instructions.setdefault("use_bullets", True)
                instructions.setdefault("format", "bullets")
            elif fast_analysis.output_format_preference == "narrative":
                instructions = dict(instructions)
                instructions.setdefault("style", "narrative")
            if required_attributes and "education" in required_attributes:
                instructions = dict(instructions)
                instructions.setdefault("use_bullets", True)
                education_fields = []
                for req_name, _pattern in (answer_requirements or []):
                    if req_name == "degree":
                        education_fields.append("Degree")
                    elif req_name == "institution":
                        education_fields.append("Institution")
                    elif req_name == "year":
                        education_fields.append("Year")
                if not education_fields:
                    education_fields = ["Degree", "Institution"]
                instructions.setdefault("fields", education_fields)
            if repair_mode:
                instructions = dict(instructions)
                instructions.setdefault("analysis", True)
                instructions.setdefault("brevity", "detailed")
            companion_opener = generate_opener(
                intent=intent,
                sentiment=sentiment,
                follow_up=is_follow_up,
                style_directives=companion_style,
                query=sentiment_seed_query or query,
            )
            if repair_mode and feedback_prefix:
                companion_opener = feedback_prefix
                feedback_prefix = ""
            template = ResponseTemplateSelector.select(intent, instructions)
            template_guidance = template.render_guidance()
            template_outline = template.render_outline()
            if repair_mode:
                template_guidance = (
                    (template_guidance or "")
                    + " STRICT: Only state claims supported by context. If not supported, say insufficient evidence."
                ).strip()
            empathy_level = "supportive" if instructions.get("empathy") or sentiment in {"frustrated", "negative", "confusion"} else "neutral"
            adapter_text = DomainPromptAdapter.build_adapter(
                profile_context_data,
                query,
                sentiment=sentiment,
                empathy_level=empathy_level,
            )
            feedback_text = self.feedback_memory.build_feedback_context(namespace, user_id, limit=5)
            continuity_notes = ""
            recent_doc_ids = set(self.conversation_history.get_recent_doc_ids(namespace, user_id))
            if recent_doc_ids:
                recent_labels = []
                for chunk in final_chunks[:8]:
                    meta = chunk.metadata or {}
                    doc_id = meta.get("document_id")
                    if doc_id and doc_id in recent_doc_ids:
                        label = self._source_label_from_chunk(chunk)
                        if label and label not in recent_labels:
                            recent_labels.append(label)
                if recent_labels:
                    continuity_notes = (
                        "Previously discussed documents: "
                        + ", ".join(recent_labels[:4])
                        + ". Reuse prior citations when relevant and avoid repeating full excerpts."
                    )

            # ✅ FIXED: Skip answerability check when we have retrieved chunks
            if final_chunks and context.strip():
                # We have context - let LLM decide if it can answer
                is_answerable = True
                answerability_reason = "Context available from retrieved chunks"
                logger.info("✅ Skipping answerability check - have valid context")
            else:
                # Only check when no chunks retrieved
                is_answerable, answerability_reason = self.answerability_detector.check_answerability(
                    query, context, has_chunks=bool(final_chunks)
                )

            # Answerability is recorded for telemetry, but it should not block responses.

            retrieval_brief = (
                f"strategy={selected_strategy}; processed_query=\"{processed_query}\"; "
                f"context_chunks={len(final_chunks)}; attempts={len(retrieval_attempts)}"
            )
            retrieval_brief += (
                f"; intent_type={fast_analysis.intent_type}; "
                f"quality={retrieval_quality.score}; evidence_coverage={round(evidence_status.coverage_ratio, 3)}"
            )
            if context_sources:
                top_sources = ", ".join(str(src.get('source_name')) for src in context_sources[:3])
                retrieval_brief += f"; top_sources={top_sources}"
            if getattr(Config.Retrieval, "USE_ADJACENT_EXPANSION", False):
                retrieval_brief += "; adjacent_expansion=on"
                if conversation_summary:
                    retrieval_brief += "; convo_summary=on"

            def _generate_answer(
                ctx_text: str,
                ctx_sources: List[Dict[str, Any]],
                ctx_chunks: List[RetrievedChunk],
                *,
                template_guidance_override: Optional[str] = None,
                template_outline_override: Optional[str] = None,
                instructions_override: Optional[Dict[str, Any]] = None,
                intent_override: Optional[str] = None,
            ) -> Tuple[str, str, str, Dict[str, Any]]:
                evidence_plan = self.evidence_planner.plan(processed_query, ctx_text)
                evidence_plan_text = json.dumps(evidence_plan, ensure_ascii=True) if evidence_plan else ""
                effective_guidance = template_guidance_override if template_guidance_override is not None else template_guidance
                effective_outline = template_outline_override if template_outline_override is not None else template_outline
                effective_instructions = instructions_override if instructions_override is not None else instructions
                effective_intent = intent_override if intent_override is not None else intent
                prompt_text = self.prompt_builder.build_qa_prompt(
                    query=query,
                    context=ctx_text,
                    persona=persona,
                    conversation_summary=conversation_summary,
                    domain_guidance=adapter_text,
                    feedback_memory=feedback_text,
                    retrieval_brief=retrieval_brief,
                    evidence_plan=evidence_plan_text,
                    intent=effective_intent,
                    intent_type=fast_analysis.intent_type,
                    required_evidence=fast_analysis.required_evidence,
                    output_format_preference=fast_analysis.output_format_preference,
                    sentiment=sentiment,
                    instructions=effective_instructions,
                    companion_opener=companion_opener,
                    style_directives=companion_style,
                    template_guidance=effective_guidance,
                    template_outline=effective_outline,
                    continuity_notes=continuity_notes,
                )
                candidate_models = []
                if getattr(Config.ModelArbitration, "ENABLED", True):
                    for entry in getattr(Config.ModelArbitration, "MODELS", []) or []:
                        if isinstance(entry, dict) and entry.get("name"):
                            candidate_models.append(
                                ModelCandidate(
                                    name=str(entry.get("name")),
                                    backend=entry.get("backend"),
                                    weight=float(entry.get("weight", 1.0)),
                                )
                            )
                responses = self.model_arbitration.generate_candidates(
                    prompt_text,
                    primary_model=self.model_name,
                    candidate_models=candidate_models,
                    temperature=float(getattr(Config.LLM, "TEMPERATURE", 0.2)),
                )
                candidate_pairs = [(resp.model, resp.text) for resp in responses if resp.text]
                post_result = self.post_processor.select_best(
                    candidate_pairs or [(self.model_name, "")],
                    sources=ctx_sources,
                    retrieved_chunks=ctx_chunks,
                    confidence_threshold=float(getattr(Config.Retrieval, "MIN_CONFIDENCE_TO_ANSWER", 0.62)),
                    opener=companion_opener,
                )
                response = post_result.answer
                if not (response or "").strip():
                    response = self._build_evidence_response(query=query, sources=ctx_sources)
                if response and response.strip().lower().startswith(query.strip().lower()):
                    trimmed = response[len(query):].lstrip(" :.-\n\t")
                    if trimmed:
                        response = trimmed
                verification = post_result.verification
                retrieval_conf = self.retrieval_confidence.score(ctx_chunks, ctx_sources)
                confidence = post_result.confidence
                breakdown = {
                    **retrieval_conf.breakdown,
                    "support_score": verification.support_score,
                    "coverage_score": verification.coverage_score,
                    "numeric_support_rate": verification.numeric_support_rate,
                }
                return response, prompt_text, evidence_plan_text, {
                    "verification": verification,
                    "confidence": confidence,
                    "confidence_breakdown": breakdown,
                    "candidate_scores": [
                        {
                            "model": score.model,
                            "confidence": score.confidence,
                            "grounded": score.grounded,
                        }
                        for score in post_result.candidate_scores
                    ],
                }

            def _generate_answer_guarded(
                ctx_text: str,
                ctx_sources: List[Dict[str, Any]],
                ctx_chunks: List[RetrievedChunk],
                *,
                template_guidance_override: Optional[str] = None,
                template_outline_override: Optional[str] = None,
                instructions_override: Optional[Dict[str, Any]] = None,
                intent_override: Optional[str] = None,
            ) -> Tuple[str, str, str, Dict[str, Any]]:
                answer_text, prompt_text, evidence_plan_text, scoring_obj = _generate_answer(
                    ctx_text,
                    ctx_sources,
                    ctx_chunks,
                    template_guidance_override=template_guidance_override,
                    template_outline_override=template_outline_override,
                    instructions_override=instructions_override,
                    intent_override=intent_override,
                )
                if not contains_banned_opener(answer_text):
                    return answer_text, prompt_text, evidence_plan_text, scoring_obj
                stripped = self._strip_first_sentence(answer_text)
                strict_instructions = dict(instructions_override or instructions)
                strict_instructions.setdefault("no_generic_openers", True)
                strict_instructions.setdefault("start_directly", True)
                retry_answer, retry_prompt, retry_evidence, retry_scoring = _generate_answer(
                    ctx_text,
                    ctx_sources,
                    ctx_chunks,
                    template_guidance_override=template_guidance_override,
                    template_outline_override=template_outline_override,
                    instructions_override=strict_instructions,
                    intent_override=intent_override or intent,
                )
                if contains_banned_opener(retry_answer):
                    retry_answer = self._strip_first_sentence(retry_answer)
                return (
                    retry_answer or stripped,
                    retry_prompt or prompt_text,
                    retry_evidence or evidence_plan_text,
                    retry_scoring or scoring_obj,
                )

            sources = context_sources or self.context_builder.extract_sources(final_chunks)

            def _build_verification_sources(
                ctx_sources: List[Dict[str, Any]],
                ctx_chunks: List[RetrievedChunk],
            ) -> List[Dict[str, Any]]:
                if not ctx_sources:
                    return []
                text_by_id = {}
                for chunk in ctx_chunks:
                    meta = chunk.metadata or {}
                    chunk_id = meta.get("chunk_id")
                    if chunk_id:
                        text_by_id[str(chunk_id)] = chunk.text or ""
                verification_sources = []
                for idx, src in enumerate(ctx_sources):
                    src_copy = dict(src)
                    chunk_id = src.get("chunk_id")
                    full_text = text_by_id.get(str(chunk_id)) if chunk_id else None
                    if not full_text and idx < len(ctx_chunks):
                        full_text = ctx_chunks[idx].text or ""
                    if full_text:
                        src_copy["excerpt"] = full_text
                    verification_sources.append(src_copy)
                return verification_sources

            verification_sources = _build_verification_sources(sources, final_chunks)
            answer, prompt, evidence_plan_text, scoring = _generate_answer_guarded(
                context,
                verification_sources,
                final_chunks,
            )
            answer = self._rewrite_citations(answer, verification_sources)
            verification = scoring["verification"]
            confidence = scoring["confidence"]
            confidence_breakdown = scoring["confidence_breakdown"]
            logger.info(
                "Confidence scored",
                extra={
                    "query_id": request_id,
                    "confidence": round(confidence, 4),
                    "breakdown": confidence_breakdown,
                },
            )

            confidence_threshold = float(
                getattr(Config.Retrieval, "MIN_CONFIDENCE_TO_ANSWER", getattr(Config.Retrieval, "CONFIDENCE_THRESHOLD", 0.55))
            )
            low_confidence_retry = False
            doc_focus_retry = False
            doc_focus_document: Optional[str] = None
            if confidence < confidence_threshold and not use_tooling:
                evidence_payload = {}
                if evidence_plan_text:
                    try:
                        evidence_payload = json.loads(evidence_plan_text)
                    except Exception:
                        evidence_payload = {}
                if evidence_payload:
                    evidence_plan = self.retriever_manager.run(
                        query=processed_query,
                        profile_id=profile_id,
                        collection_name=collection_name,
                        query_analysis=preprocessing_metadata.get("query_analysis") or {},
                        query_understanding=preprocessing_metadata.get("query_understanding") or {},
                        metadata_filters=(preprocessing_metadata.get("query_analysis") or {}).get("metadata_filters") or {},
                        top_k=max(top_k_retrieval, int(top_k_retrieval * 1.2)),
                        memory=AgentMemory(),
                        evidence_plan=evidence_payload,
                        profile_context=profile_context_data,
                    )
                    if evidence_plan.chunks:
                        retrieval_attempts.append(
                            {
                                "label": "evidence_plan_retry",
                                "query": processed_query,
                                "hits": len(evidence_plan.chunks),
                                "top_score": round(float(evidence_plan.chunks[0].score), 4) if evidence_plan.chunks else 0.0,
                                "document_filter": None,
                            }
                        )
                        retrieved_chunks = evidence_plan.chunks
                        reranked_chunks, final_chunks, context, context_sources = _build_context_from_chunks(
                            evidence_plan.chunks,
                            use_heavy_rerank=True,
                            intent_type=fast_analysis.intent_type,
                        )
                        sources = context_sources or self.context_builder.extract_sources(final_chunks)
                        verification_sources = _build_verification_sources(sources, final_chunks)
                        answer, prompt, evidence_plan_text, scoring = _generate_answer_guarded(
                            context, verification_sources, final_chunks
                        )
                        answer = self._rewrite_citations(answer, verification_sources)
                        verification = scoring["verification"]
                        confidence = scoring["confidence"]
                        confidence_breakdown = scoring["confidence_breakdown"]
                        low_confidence_retry = True

                if confidence < confidence_threshold:
                    fallback_top_k = max(top_k_retrieval * 2, len(retrieved_chunks) + 5)
                    fallback_chunks = self.retriever.hybrid_retrieve(
                        collection_name=collection_name,
                        query=processed_query,
                        profile_id=profile_id,
                        top_k=fallback_top_k,
                        document_ids=None,
                        doc_types=None,
                        section_titles=None,
                        page_numbers=None,
                        min_confidence=None,
                        score_threshold=None,
                    )
                    if fallback_chunks:
                        retrieval_attempts.append(
                            {
                                "label": "low_confidence_broad_retry",
                                "query": processed_query,
                                "hits": len(fallback_chunks),
                                "top_score": round(float(fallback_chunks[0].score), 4) if fallback_chunks else 0.0,
                                "document_filter": None,
                            }
                        )
                        retrieved_chunks = fallback_chunks
                        reranked_chunks, final_chunks, context, context_sources = _build_context_from_chunks(
                            fallback_chunks,
                            use_heavy_rerank=True,
                            intent_type=fast_analysis.intent_type,
                        )
                        sources = context_sources or self.context_builder.extract_sources(final_chunks)
                        verification_sources = _build_verification_sources(sources, final_chunks)
                        answer, prompt, evidence_plan_text, scoring = _generate_answer_guarded(
                            context, verification_sources, final_chunks
                        )
                        answer = self._rewrite_citations(answer, verification_sources)
                        verification = scoring["verification"]
                        confidence = scoring["confidence"]
                        confidence_breakdown = scoring["confidence_breakdown"]
                        low_confidence_retry = True

            metadata_filters = (preprocessing_metadata.get("query_analysis") or {}).get("metadata_filters") or {}
            doc_options, section_options = self._collect_focus_options(
                final_chunks or retrieved_chunks,
                profile_context_data,
            )

            if intent == "comparison":
                min_compare_docs = int(getattr(Config.Retrieval, "MIN_COMPARISON_DOCS", 2))
                doc_labels = []
                for chunk in (final_chunks or retrieved_chunks):
                    label = self._source_label_from_chunk(chunk)
                    if label:
                        doc_labels.append(label)
                distinct_docs = list(dict.fromkeys(doc_labels))
                if len(distinct_docs) < min_compare_docs:
                    doc_list = self._format_options([opt.get("label") for opt in doc_options if opt.get("label")])
                    if doc_list:
                        clarification = (
                            "You asked for a comparison, but I only have one relevant document so far. "
                            f"Which other document should I compare it to? Options: {doc_list}."
                        )
                    else:
                        clarification = (
                            "You asked for a comparison, but I only have one relevant document so far. "
                            "Which other document should I compare it to?"
                        )
                    clarification = self._apply_acknowledgement(
                        clarification,
                        query=query,
                        intent=intent,
                        sentiment=sentiment,
                        style_directives=companion_style,
                        follow_up=is_follow_up,
                    )
                    clarification = _apply_feedback_prefix(clarification)
                    self.conversation_history.add_turn(namespace, user_id, query, clarification)
                    processing_time = time.time() - start_time
                    _record_request_metrics(
                        query_type="clarification_needed",
                        answer_text=clarification,
                        context_text=context or "",
                        context_found=bool(final_chunks),
                        grounded=False,
                        has_citations=False,
                        processing_seconds=processing_time,
                        prompt_text=prompt,
                        confidence=confidence,
                        support_score=verification.support_score,
                        coverage_score=verification.coverage_score,
                        numeric_support_rate=verification.numeric_support_rate,
                        **retrieval_metric_kwargs,
                    )
                    try:
                        metrics.record(
                            model_name=self.model_name,
                            subscription_id=subscription_id,
                            profile_id=profile_id,
                            query_type="clarification_needed",
                            context_found=bool(final_chunks),
                            grounded=False,
                            cached=False,
                            processing_time=processing_time,
                            retrieval_stats={
                                "initial_retrieved": len(retrieved_chunks),
                                "after_rerank": len(reranked_chunks),
                                "final_context": len(final_chunks),
                            },
                        )
                    except Exception as metric_exc:
                        logger.debug("Metrics record (clarification_needed) failed: %s", metric_exc)
                    return _sanitize_response({
                        "response": clarification,
                        "sources": sources,
                        "user_id": user_id,
                        "collection": collection_name,
                        "request_id": request_id,
                        "index_version": index_version,
                        "context_found": bool(final_chunks),
                        "query_type": "clarification_needed",
                        "preprocessing": preprocessing_metadata,
                        "retrieval_attempts": retrieval_attempts,
                        "selected_strategy": selected_strategy,
                        "profile_context": profile_context_data,
                        "grounded": False,
                        "confidence": round(confidence, 3),
                        "confidence_breakdown": confidence_breakdown,
                        "verification": {
                            "citations_valid": verification.citations_valid,
                            "invalid_citations": verification.invalid_citations,
                            "missing_citations": verification.missing_citations,
                            "unsupported_sentences": verification.unsupported_sentences,
                            "support_score": verification.support_score,
                            "coverage_score": verification.coverage_score,
                            "numeric_support_rate": verification.numeric_support_rate,
                            "overall_grounded": verification.overall_grounded,
                        },
                        "processing_time": round(processing_time, 2),
                        "retrieval_notes": retrieval_brief,
                        "doc_focus_retry": doc_focus_retry,
                        "doc_focus_document": doc_focus_document,
                    })

            if (
                confidence < confidence_threshold
                and not verification.overall_grounded
                and doc_options
                and len(doc_options) > 1
            ):
                focus_label = doc_options[0].get("label")
                focused_chunks = [
                    chunk for chunk in retrieved_chunks
                    if self._source_label_from_chunk(chunk) == focus_label
                ] if focus_label else []
                if focused_chunks and len(focused_chunks) < len(retrieved_chunks):
                    retrieval_attempts.append(
                        {
                            "label": "low_confidence_doc_focus_retry",
                            "query": processed_query,
                            "hits": len(focused_chunks),
                            "top_score": round(float(focused_chunks[0].score), 4) if focused_chunks else 0.0,
                            "document_filter": focus_label,
                        }
                    )
                    doc_focus_retry = True
                    doc_focus_document = focus_label
                    if focus_label and f"doc_focus={focus_label}" not in retrieval_brief:
                        retrieval_brief += f"; doc_focus={focus_label}"
                    reranked_chunks, final_chunks, context, context_sources = _build_context_from_chunks(
                        focused_chunks,
                        use_heavy_rerank=True,
                        intent_type=fast_analysis.intent_type,
                    )
                    sources = context_sources or self.context_builder.extract_sources(final_chunks)
                    verification_sources = _build_verification_sources(sources, final_chunks)
                    answer, prompt, evidence_plan_text, scoring = _generate_answer_guarded(
                        context, verification_sources, final_chunks
                    )
                    answer = self._rewrite_citations(answer, verification_sources)
                    verification = scoring["verification"]
                    confidence = scoring["confidence"]
                    confidence_breakdown = scoring["confidence_breakdown"]
                    doc_options, section_options = self._collect_focus_options(
                        final_chunks or focused_chunks,
                        profile_context_data,
                    )

            if confidence < confidence_threshold and (final_chunks or retrieved_chunks):
                # Tighten focus to the top document + section for higher precision.
                focus_source = (final_chunks or retrieved_chunks)[0]
                focus_meta = focus_source.metadata or {}
                focus_doc_id = focus_meta.get("document_id")
                focus_section = focus_meta.get("section_title") or focus_meta.get("section")
                focused_chunks = [
                    chunk
                    for chunk in (retrieved_chunks or [])
                    if (focus_doc_id and (chunk.metadata or {}).get("document_id") == focus_doc_id)
                ]
                if focus_section:
                    focused_chunks = [
                        chunk
                        for chunk in focused_chunks
                        if (chunk.metadata or {}).get("section_title") == focus_section
                        or (chunk.metadata or {}).get("section") == focus_section
                    ]
                if focused_chunks:
                    retrieval_attempts.append(
                        {
                            "label": "low_confidence_doc_section_focus",
                            "query": processed_query,
                            "hits": len(focused_chunks),
                            "top_score": round(float(focused_chunks[0].score), 4) if focused_chunks else 0.0,
                            "document_filter": focus_doc_id,
                        }
                    )
                    reranked_chunks, final_chunks, context, context_sources = _build_context_from_chunks(
                        focused_chunks,
                        use_heavy_rerank=True,
                        intent_type=fast_analysis.intent_type,
                    )
                    sources = context_sources or self.context_builder.extract_sources(final_chunks)
                    verification_sources = _build_verification_sources(sources, final_chunks)
                    answer, prompt, evidence_plan_text, scoring = _generate_answer_guarded(
                        context, verification_sources, final_chunks
                    )
                    answer = self._rewrite_citations(answer, verification_sources)
                    verification = scoring["verification"]
                    confidence = scoring["confidence"]
                    confidence_breakdown = scoring["confidence_breakdown"]

            needs_clarification = (
                confidence < confidence_threshold
                and not verification.overall_grounded
                and self._should_request_clarification(
                    query=query,
                    doc_options=doc_options,
                    section_options=section_options,
                    metadata_filters=metadata_filters,
                    preprocessing_metadata=preprocessing_metadata,
                )
            )

            if needs_clarification:
                clarification = self._build_clarification_question(
                    query=query,
                    doc_options=doc_options,
                    section_options=section_options,
                    profile_context=profile_context_data,
                    metadata_filters=metadata_filters,
                )
                self.conversation_history.add_turn(namespace, user_id, query, clarification)
                self.learning_signals.record_low_confidence(
                    query=query,
                    context=context or "",
                    answer=answer,
                    reason="low_confidence_clarification",
                    metadata={
                        "profile_id": profile_id,
                        "subscription_id": subscription_id,
                        "confidence": confidence,
                        "confidence_breakdown": confidence_breakdown,
                        "doc_focus_document": doc_focus_document,
                        "query_type": "clarification_needed",
                    },
                )
                processing_time = time.time() - start_time
                _record_request_metrics(
                    query_type="clarification_needed",
                    answer_text=clarification,
                    context_text=context or "",
                    context_found=bool(final_chunks),
                    grounded=False,
                    has_citations=False,
                    processing_seconds=processing_time,
                    prompt_text=prompt,
                    confidence=confidence,
                    support_score=verification.support_score,
                    coverage_score=verification.coverage_score,
                    numeric_support_rate=verification.numeric_support_rate,
                    **retrieval_metric_kwargs,
                )
                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="clarification_needed",
                        context_found=bool(final_chunks),
                        grounded=False,
                        cached=False,
                        processing_time=processing_time,
                        retrieval_stats={
                            "initial_retrieved": len(retrieved_chunks),
                            "after_rerank": len(reranked_chunks),
                            "final_context": len(final_chunks),
                        },
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (clarification_needed) failed: %s", metric_exc)
                return _sanitize_response({
                    "response": clarification,
                    "sources": sources,
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": bool(final_chunks),
                    "query_type": "clarification_needed",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,
                    "confidence": round(confidence, 3),
                    "confidence_breakdown": confidence_breakdown,
                    "verification": {
                        "citations_valid": verification.citations_valid,
                        "invalid_citations": verification.invalid_citations,
                        "missing_citations": verification.missing_citations,
                        "unsupported_sentences": verification.unsupported_sentences,
                        "support_score": verification.support_score,
                        "coverage_score": verification.coverage_score,
                        "numeric_support_rate": verification.numeric_support_rate,
                        "overall_grounded": verification.overall_grounded,
                    },
                    "processing_time": round(processing_time, 2),
                    "retrieval_notes": retrieval_brief,
                    "doc_focus_retry": doc_focus_retry,
                    "doc_focus_document": doc_focus_document,
                })

            min_support_score = float(getattr(Config.Retrieval, "MIN_SUPPORT_SCORE", 0.15))
            relevance = self._answer_relevance(answer, processed_query or query)
            if (
                confidence < confidence_threshold
                and (not verification.overall_grounded)
                and verification.support_score < min_support_score
                and relevance < 0.12
            ):
                fallback_response = self._build_evidence_response(
                    query=query,
                    sources=verification_sources or sources,
                    doc_options=doc_options,
                    section_options=section_options,
                )
                fallback_response = self._apply_acknowledgement(
                    fallback_response,
                    query=query,
                    intent=intent,
                    sentiment=sentiment,
                    style_directives=companion_style,
                    follow_up=is_follow_up,
                )
                fallback_response = _apply_feedback_prefix(fallback_response)
                self.conversation_history.add_turn(namespace, user_id, query, fallback_response)
                processing_time = time.time() - start_time
                _record_request_metrics(
                    query_type="closest_answer",
                    answer_text=fallback_response,
                    context_text=context or "",
                    context_found=bool(final_chunks),
                    grounded=False,
                    has_citations=False,
                    processing_seconds=processing_time,
                    prompt_text=prompt,
                    confidence=confidence,
                    support_score=verification.support_score,
                    coverage_score=verification.coverage_score,
                    numeric_support_rate=verification.numeric_support_rate,
                    **retrieval_metric_kwargs,
                )
                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="closest_answer",
                        context_found=bool(final_chunks),
                        grounded=False,
                        cached=False,
                        processing_time=processing_time,
                        retrieval_stats={
                            "initial_retrieved": len(retrieved_chunks),
                            "after_rerank": len(reranked_chunks),
                            "final_context": len(final_chunks),
                        },
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (closest_answer) failed: %s", metric_exc)
                return _sanitize_response({
                    "response": fallback_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": bool(final_chunks),
                    "query_type": "closest_answer",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,
                    "confidence": round(confidence, 3),
                    "confidence_breakdown": confidence_breakdown,
                    "verification": {
                        "citations_valid": verification.citations_valid,
                        "invalid_citations": verification.invalid_citations,
                        "missing_citations": verification.missing_citations,
                        "unsupported_sentences": verification.unsupported_sentences,
                        "support_score": verification.support_score,
                        "coverage_score": verification.coverage_score,
                        "numeric_support_rate": verification.numeric_support_rate,
                        "overall_grounded": verification.overall_grounded,
                    },
                    "processing_time": round(processing_time, 2),
                    "retrieval_notes": retrieval_brief,
                })

            if confidence < confidence_threshold and not verification.overall_grounded and relevance < 0.12:
                evidence_response = self._build_evidence_response(
                    query=query,
                    sources=sources,
                    doc_options=doc_options,
                    section_options=section_options,
                )
                evidence_response = self._apply_acknowledgement(
                    evidence_response,
                    query=query,
                    intent=intent,
                    sentiment=sentiment,
                    style_directives=companion_style,
                    follow_up=is_follow_up,
                )
                evidence_response = _apply_feedback_prefix(evidence_response)
                self.conversation_history.add_turn(namespace, user_id, query, evidence_response)
                self.learning_signals.record_low_confidence(
                    query=query,
                    context=context or "",
                    answer=answer,
                    reason="evidence_only_response",
                    metadata={
                        "profile_id": profile_id,
                        "subscription_id": subscription_id,
                        "confidence": confidence,
                        "confidence_breakdown": confidence_breakdown,
                        "doc_focus_document": doc_focus_document,
                        "query_type": "evidence_only",
                    },
                )
                processing_time = time.time() - start_time
                _record_request_metrics(
                    query_type="evidence_only",
                    answer_text=evidence_response,
                    context_text=context or "",
                    context_found=bool(final_chunks),
                    grounded=False,
                    has_citations=bool(re.search(r'\[SOURCE-\d+\]', evidence_response)),
                    processing_seconds=processing_time,
                    prompt_text=prompt,
                    confidence=confidence,
                    support_score=verification.support_score,
                    coverage_score=verification.coverage_score,
                    numeric_support_rate=verification.numeric_support_rate,
                    **retrieval_metric_kwargs,
                )
                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="evidence_only",
                        context_found=bool(final_chunks),
                        grounded=False,
                        cached=False,
                        processing_time=processing_time,
                        retrieval_stats={
                            "initial_retrieved": len(retrieved_chunks),
                            "after_rerank": len(reranked_chunks),
                            "final_context": len(final_chunks),
                        },
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (evidence_only) failed: %s", metric_exc)
                return _sanitize_response({
                    "response": evidence_response,
                    "sources": sources,
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": bool(final_chunks),
                    "query_type": "evidence_only",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,
                    "confidence": round(confidence, 3),
                    "confidence_breakdown": confidence_breakdown,
                    "verification": {
                        "citations_valid": verification.citations_valid,
                        "invalid_citations": verification.invalid_citations,
                        "missing_citations": verification.missing_citations,
                        "unsupported_sentences": verification.unsupported_sentences,
                        "support_score": verification.support_score,
                        "coverage_score": verification.coverage_score,
                        "numeric_support_rate": verification.numeric_support_rate,
                        "overall_grounded": verification.overall_grounded,
                    },
                    "processing_time": round(processing_time, 2),
                    "retrieval_notes": retrieval_brief,
                    "doc_focus_retry": doc_focus_retry,
                    "doc_focus_document": doc_focus_document,
                })

            quality_eval = None
            repair_outcome = None
            if getattr(Config.Quality, "ENABLED", True):
                eval_chunks = final_chunks or retrieved_chunks
                try:
                    retrieval_conf_score = float(self.retrieval_confidence.score(eval_chunks, sources).confidence)
                except Exception:
                    retrieval_conf_score = 0.0

                quality_eval = self.bad_answer_evaluator.evaluate(
                    query=query,
                    answer=answer,
                    retrieved_chunks=eval_chunks,
                    retrieval_confidence=retrieval_conf_score,
                    query_intent=intent,
                )

                if getattr(Config.Quality, "AUTO_REPAIR_ENABLED", True) and quality_eval.bad:
                    metadata_filters = (preprocessing_metadata.get("query_analysis") or {}).get("metadata_filters") or {}
                    query_analysis_payload = preprocessing_metadata.get("query_analysis") or {}
                    query_understanding_payload = preprocessing_metadata.get("query_understanding") or {}

                    def _repair_clarification() -> Optional[str]:
                        needs = self._should_request_clarification(
                            query=query,
                            doc_options=doc_options,
                            section_options=section_options,
                            metadata_filters=metadata_filters,
                            preprocessing_metadata=preprocessing_metadata,
                        )
                        if not needs:
                            return None
                        return self._build_clarification_question(
                            query=query,
                            doc_options=doc_options,
                            section_options=section_options,
                            profile_context=profile_context_data,
                            metadata_filters=metadata_filters,
                        )

                    def _repair_refusal() -> str:
                        return (
                            "I don't have enough evidence in the provided documents to answer that reliably. "
                            "If you can share a specific document or section, I can try again."
                        )

                    def _repair_retrieve(repair_query: str, filters: Dict[str, Any], top_k: int) -> Dict[str, Any]:
                        repair_plan = self.retriever_manager.run(
                            query=repair_query,
                            profile_id=profile_id,
                            collection_name=collection_name,
                            query_analysis=query_analysis_payload,
                            query_understanding=query_understanding_payload,
                            metadata_filters=filters,
                            top_k=top_k,
                            memory=AgentMemory(),
                            evidence_plan=None,
                            profile_context=profile_context_data,
                        )
                        attempts = [
                            {
                                "label": f"repair_{attempt.label}",
                                "query": attempt.query,
                                "hits": attempt.hits,
                                "top_score": attempt.top_score,
                                "document_filter": filters.get("document_ids"),
                            }
                            for attempt in (repair_plan.attempts or [])
                        ]
                        return {"chunks": repair_plan.chunks, "attempts": attempts, "query": repair_query}

                    def _repair_context(
                        chunks: List[RetrievedChunk],
                        rerank_k: int,
                        rerank_query: str,
                    ) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], str, List[Dict[str, Any]]]:
                        return _build_context_from_chunks(
                            chunks,
                            rerank_top_k_override=rerank_k,
                            rerank_query=rerank_query,
                            use_heavy_rerank=True,
                            intent_type=fast_analysis.intent_type,
                        )

                    def _template_for_intent(intent_value: str) -> Tuple[str, str]:
                        intent_value = intent_value or intent
                        forced_instructions: Dict[str, Any] = {}
                        if intent_value == "comparison":
                            forced_instructions["use_table"] = True
                        elif intent_value in {"procedural", "how-to"}:
                            forced_instructions["step_by_step"] = True
                        elif intent_value == "summary":
                            forced_instructions["use_bullets"] = True
                        template_obj = ResponseTemplateSelector.select(intent_value, forced_instructions)
                        guidance = template_obj.render_guidance()
                        outline = template_obj.render_outline()
                        guidance = (
                            guidance
                            + " STRICT: Only state claims supported by context. If not supported, say insufficient evidence."
                        ).strip()
                        return guidance, outline

                    def _repair_generate(
                        ctx_text: str,
                        ctx_sources: List[Dict[str, Any]],
                        ctx_chunks: List[RetrievedChunk],
                        template_guidance_override: str,
                        template_outline_override: str,
                        strict_grounding: bool,
                    ) -> Tuple[str, str, str, Dict[str, Any]]:
                        return _generate_answer_guarded(
                            ctx_text,
                            ctx_sources,
                            ctx_chunks,
                            template_guidance_override=template_guidance_override,
                            template_outline_override=template_outline_override,
                            intent_override=quality_eval.intent_expected or intent,
                        )

                    def _eval_wrapper(
                        new_answer: str,
                        eval_chunks: List[RetrievedChunk],
                        retrieval_conf: Optional[float],
                        intent_expected: Optional[str],
                    ):
                        return self.bad_answer_evaluator.evaluate(
                            query=query,
                            answer=new_answer,
                            retrieved_chunks=eval_chunks,
                            retrieval_confidence=retrieval_conf,
                            query_intent=intent_expected or intent,
                        )

                    def _retrieval_confidence_fn(
                        eval_chunks: List[RetrievedChunk],
                        eval_sources: List[Dict[str, Any]],
                    ) -> float:
                        try:
                            return float(self.retrieval_confidence.score(eval_chunks, eval_sources).confidence)
                        except Exception:
                            return 0.0

                    repair_outcome = self.auto_repair_engine.run(
                        query=query,
                        processed_query=processed_query,
                        intent_expected=quality_eval.intent_expected or intent,
                        evaluation=quality_eval,
                        retrieved_chunks=eval_chunks,
                        retrieval_confidence=retrieval_conf_score,
                        metadata_filters=metadata_filters,
                        top_k_retrieval=top_k_retrieval,
                        top_k_rerank=top_k_rerank,
                        retrieve_fn=_repair_retrieve,
                        build_context_fn=_repair_context,
                        build_verification_sources_fn=_build_verification_sources,
                        generate_fn=_repair_generate,
                        rewrite_citations_fn=self._rewrite_citations,
                        eval_fn=_eval_wrapper,
                        retrieval_confidence_fn=_retrieval_confidence_fn,
                        template_for_intent_fn=_template_for_intent,
                        clarification_fn=_repair_clarification,
                        refusal_fn=_repair_refusal,
                    )
                    if repair_outcome and repair_outcome.updated:
                        answer = repair_outcome.answer
                        scoring = repair_outcome.scoring or scoring
                        prompt = scoring.get("prompt") or prompt
                        evidence_plan_text = scoring.get("evidence_plan_text") or evidence_plan_text
                        verification = scoring.get("verification", verification)
                        confidence = scoring.get("confidence", confidence)
                        confidence_breakdown = scoring.get("confidence_breakdown", confidence_breakdown)
                        final_chunks = list(repair_outcome.chunks or final_chunks)
                        retrieved_chunks = list(repair_outcome.chunks or retrieved_chunks)
                        reranked_chunks = list(repair_outcome.chunks or reranked_chunks)
                        sources = repair_outcome.sources or sources
                        context = repair_outcome.context or context
                        if repair_outcome.eval_result:
                            quality_eval = repair_outcome.eval_result
                        if repair_outcome.scoring.get("retrieval_attempts"):
                            retrieval_attempts.extend(repair_outcome.scoring.get("retrieval_attempts"))

                if quality_eval:
                    emit_quality_telemetry(
                        {
                            "query_id": request_id,
                            "intent_expected": quality_eval.intent_expected or intent,
                            "supported_ratio": quality_eval.supported_ratio,
                            "critical_supported_ratio": quality_eval.critical_supported_ratio,
                            "coverage_ratio": quality_eval.coverage_ratio,
                            "structure": {
                                "has_table": bool(quality_eval.structure and quality_eval.structure.has_table),
                                "has_steps": bool(quality_eval.structure and quality_eval.structure.has_steps),
                                "has_headings": bool(quality_eval.structure and quality_eval.structure.has_headings),
                            },
                            "robot_opener": quality_eval.robot_opener,
                            "relevance_score": quality_eval.relevance_score,
                            "query_overlap": quality_eval.query_overlap,
                            "overall_score": quality_eval.overall_score,
                            "bad_flag": quality_eval.bad,
                            "severity": quality_eval.severity,
                            "repair_attempt_count": repair_outcome.attempts if repair_outcome else 0,
                            "eval_latency_ms": quality_eval.eval_latency_ms,
                            "repair_latency_ms": repair_outcome.elapsed_ms if repair_outcome else 0.0,
                        }
                    )
                if repair_outcome and not repair_outcome.updated:
                    if repair_outcome.clarification:
                        clarification = self._apply_acknowledgement(
                            repair_outcome.clarification,
                            query=query,
                            intent=intent,
                            sentiment=sentiment,
                            style_directives=companion_style,
                            follow_up=is_follow_up,
                        )
                        clarification = _apply_feedback_prefix(clarification)
                        self.conversation_history.add_turn(namespace, user_id, query, clarification)
                        processing_time = time.time() - start_time
                        _record_request_metrics(
                            query_type="clarification_needed",
                            answer_text=clarification,
                            context_text=context or "",
                            context_found=False,
                            grounded=False,
                            has_citations=False,
                            processing_seconds=processing_time,
                            prompt_text=prompt,
                            confidence=confidence,
                            support_score=verification.support_score,
                            coverage_score=verification.coverage_score,
                            numeric_support_rate=verification.numeric_support_rate,
                            **retrieval_metric_kwargs,
                        )
                        return _sanitize_response({
                            "response": clarification,
                            "sources": [],
                            "user_id": user_id,
                            "collection": collection_name,
                            "request_id": request_id,
                            "index_version": index_version,
                            "context_found": False,
                            "query_type": "clarification_needed",
                            "preprocessing": preprocessing_metadata,
                            "retrieval_attempts": retrieval_attempts,
                            "selected_strategy": selected_strategy,
                            "profile_context": profile_context_data,
                            "grounded": False,
                            "confidence": round(confidence, 3),
                            "confidence_breakdown": confidence_breakdown,
                            "verification": {
                                "citations_valid": verification.citations_valid,
                                "invalid_citations": verification.invalid_citations,
                                "missing_citations": verification.missing_citations,
                                "unsupported_sentences": verification.unsupported_sentences,
                                "support_score": verification.support_score,
                                "coverage_score": verification.coverage_score,
                                "numeric_support_rate": verification.numeric_support_rate,
                                "overall_grounded": verification.overall_grounded,
                            },
                            "processing_time": round(processing_time, 2),
                            "retrieval_notes": retrieval_brief,
                        })
                    if repair_outcome.refusal:
                        refusal = self._apply_acknowledgement(
                            repair_outcome.refusal,
                            query=query,
                            intent=intent,
                            sentiment=sentiment,
                            style_directives=companion_style,
                            follow_up=is_follow_up,
                        )
                        refusal = _apply_feedback_prefix(refusal)
                        self.conversation_history.add_turn(namespace, user_id, query, refusal)
                        processing_time = time.time() - start_time
                        _record_request_metrics(
                            query_type="refusal",
                            answer_text=refusal,
                            context_text=context or "",
                            context_found=False,
                            grounded=False,
                            has_citations=False,
                            processing_seconds=processing_time,
                            prompt_text=prompt,
                            confidence=confidence,
                            support_score=verification.support_score,
                            coverage_score=verification.coverage_score,
                            numeric_support_rate=verification.numeric_support_rate,
                            **retrieval_metric_kwargs,
                        )
                        return _sanitize_response({
                            "response": refusal,
                            "sources": [],
                            "user_id": user_id,
                            "collection": collection_name,
                            "request_id": request_id,
                            "index_version": index_version,
                            "context_found": False,
                            "query_type": "refusal",
                            "preprocessing": preprocessing_metadata,
                            "retrieval_attempts": retrieval_attempts,
                            "selected_strategy": selected_strategy,
                            "profile_context": profile_context_data,
                            "grounded": False,
                            "confidence": round(confidence, 3),
                            "confidence_breakdown": confidence_breakdown,
                            "verification": {
                                "citations_valid": verification.citations_valid,
                                "invalid_citations": verification.invalid_citations,
                                "missing_citations": verification.missing_citations,
                                "unsupported_sentences": verification.unsupported_sentences,
                                "support_score": verification.support_score,
                                "coverage_score": verification.coverage_score,
                                "numeric_support_rate": verification.numeric_support_rate,
                                "overall_grounded": verification.overall_grounded,
                            },
                            "processing_time": round(processing_time, 2),
                            "retrieval_notes": retrieval_brief,
                        })

            answer = self._apply_acknowledgement(
                answer,
                query=query,
                intent=intent,
                sentiment=sentiment,
                opener=companion_opener,
                style_directives=companion_style,
                follow_up=is_follow_up,
            )
            answer = _apply_feedback_prefix(answer)
            self.conversation_history.add_turn(namespace, user_id, query, answer)
            final_doc_ids = [
                (chunk.metadata or {}).get("document_id") for chunk in final_chunks
            ]
            final_doc_ids = [d for d in final_doc_ids if d]
            if final_doc_ids:
                self.conversation_history.add_sources(namespace, user_id, final_doc_ids)
            self.feedback_memory.add_feedback(namespace, user_id, query, answer, sources)

            has_citations = bool(re.search(r'\[SOURCE-\d+\]', answer))

            processing_time = time.time() - start_time

            # Construct the full response object up front so it can be cached
            response_obj = {
                "response": answer,
                "sources": sources,
                "user_id": user_id,
                "collection": collection_name,
                "request_id": request_id,
                "index_version": index_version,
                "context_found": True,
                "query_type": "document_qa",
                "num_sources": len(sources),
                "source_doc_ids": final_doc_ids,
                "preprocessing": preprocessing_metadata,
                "processed_query": processed_query,
                "answerability": {
                    "is_answerable": is_answerable,
                    "reason": answerability_reason
                },
                "grounded": bool(verification.overall_grounded),
                "has_citations": has_citations,
                "confidence": round(confidence, 3),
                "confidence_breakdown": confidence_breakdown,
                "verification": {
                    "citations_valid": verification.citations_valid,
                    "invalid_citations": verification.invalid_citations,
                    "missing_citations": verification.missing_citations,
                    "unsupported_sentences": verification.unsupported_sentences,
                    "support_score": verification.support_score,
                    "coverage_score": verification.coverage_score,
                    "numeric_support_rate": verification.numeric_support_rate,
                    "overall_grounded": verification.overall_grounded,
                },
                "evidence_plan_used": bool(evidence_plan_text),
                "retrieval_attempts": retrieval_attempts,
                "selected_strategy": selected_strategy,
                "profile_context": profile_context_data,
                "processing_time": round(processing_time, 2),
                "retrieval_notes": retrieval_brief,
                "model_name": self.model_name,
                "persona": persona,
                "cache_version": ANSWER_CACHE_VERSION,
                "context_fingerprint": memory_fingerprint,
                "profile_context_fingerprint": profile_context_fingerprint,
                "retrieval_stats": {
                    "initial_retrieved": len(retrieved_chunks),
                    "after_rerank": len(reranked_chunks),
                    "final_context": len(final_chunks)
                },
                "low_confidence_retry": low_confidence_retry,
                "doc_focus_retry": doc_focus_retry,
                "doc_focus_document": doc_focus_document,
                "force_refresh": force_refresh,
                "disable_answer_cache": disable_answer_cache,
            }

            high_conf_threshold = float(getattr(Config.Retrieval, "HIGH_CONFIDENCE_THRESHOLD", 0.75))
            if confidence >= high_conf_threshold and verification.overall_grounded and has_citations:
                self.learning_signals.record_high_quality(
                    query=query,
                    context=context or "",
                    answer=answer,
                    sources=sources,
                    metadata={
                        "profile_id": profile_id,
                        "subscription_id": subscription_id,
                        "document_ids": final_doc_ids,
                        "confidence": confidence,
                    },
                )
            elif confidence < confidence_threshold:
                self.learning_signals.record_low_confidence(
                    query=query,
                    context=context or "",
                    answer=answer,
                    reason="low_confidence",
                    metadata={
                        "profile_id": profile_id,
                        "subscription_id": subscription_id,
                        "confidence": confidence,
                        "confidence_breakdown": confidence_breakdown,
                        "verification": {
                            "citations_valid": verification.citations_valid,
                            "missing_citations": verification.missing_citations,
                        },
                    },
                )

            try:
                metrics.record(
                    model_name=self.model_name,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    query_type=response_obj.get("query_type", "document_qa"),
                    context_found=True,
                    grounded=bool(response_obj.get("grounded", True)),
                    cached=False,
                    processing_time=processing_time,
                    retrieval_stats=response_obj.get("retrieval_stats") or {}
                )
            except Exception as metric_exc:
                logger.debug("Metrics record (success) failed: %s", metric_exc)

            _record_request_metrics(
                query_type=response_obj.get("query_type", "document_qa"),
                answer_text=answer,
                context_text=context or "",
                context_found=True,
                grounded=bool(response_obj.get("grounded", True)),
                has_citations=has_citations,
                processing_seconds=processing_time,
                prompt_text=prompt,
                tool_successes=tool_successes,
                document_ids=final_doc_ids,
                confidence=confidence,
                support_score=verification.support_score,
                coverage_score=verification.coverage_score,
                numeric_support_rate=verification.numeric_support_rate,
                **retrieval_metric_kwargs,
            )

            # Persist the response in Redis before returning. Only cache successful
            # answers (document_qa) to improve subsequent response accuracy.
            sanitized_response = _sanitize_response(response_obj)
            if cache and cache_key:
                try:
                    cache.setex(cache_key, ANSWER_CACHE_TTL, json.dumps(sanitized_response))
                except Exception as cache_exc:
                    logger.warning(f"Failed to cache answer: {cache_exc}")

            # Return the constructed response
            return sanitized_response

        except HTTPException:
            raise
        except QdrantCollectionNotFoundError as exc:
            detail = {
                "error": {
                    "code": "qdrant_collection_not_found",
                    "message": str(exc),
                    "details": {
                        "collection_name": exc.collection_name,
                        "available_collections": exc.available_collections,
                    },
                }
            }
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail) from exc
        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            try:
                if telemetry:
                    telemetry.increment("retrieval_failures_count")
            except Exception:
                pass

            error_response = "Sorry, something went wrong on my side. Please try again, and let me know if it keeps happening."

            try:
                metrics = get_metrics_tracker()
                metrics.record(
                    model_name=getattr(self, "model_name", "unknown"),
                    subscription_id=subscription_id if 'subscription_id' in locals() else "unknown",
                    profile_id=profile_id if 'profile_id' in locals() else "unknown",
                    query_type="error",
                    context_found=False,
                    grounded=False,
                    cached=False,
                    processing_time=time.time() - start_time,
                    retrieval_stats={}
                )
            except Exception as metric_exc:
                logger.debug("Metrics record (error) failed: %s", metric_exc)

            if "_record_request_metrics" in locals():
                extra_kwargs = retrieval_metric_kwargs if "retrieval_metric_kwargs" in locals() else {}
                _record_request_metrics(
                    query_type="error",
                    answer_text=error_response,
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query if "processed_query" in locals() else query,
                    **extra_kwargs,
                )
            if "self" in locals():
                try:
                    self.learning_signals.record_failure(
                        query=query,
                        reason="error",
                        metadata={
                            "profile_id": profile_id if "profile_id" in locals() else None,
                            "subscription_id": subscription_id if "subscription_id" in locals() else None,
                            "error": str(e),
                        },
                    )
                except Exception:
                    pass

            return _sanitize_response({
                "response": error_response,
                "sources": [],
                "user_id": user_id,
                "collection": collection_name if 'collection_name' in locals() else profile_id,
                "request_id": request_id,
                "index_version": index_version,
                "context_found": False,
                "query_type": "error",
                "error": str(e),
                "grounded": False,
                "processing_time": time.time() - start_time
            })


# Global RAG system instance (lazy initialization)
_RAG_SYSTEM = None
_RAG_MODEL = None
_RAG_PROFILE = None
_RAG_BACKEND = None
_RAG_MODEL_PATH = None
_LLM_CLIENTS: dict[tuple[str, str, str | None], Any] = {}
_LLM_SEMAPHORE: Optional[threading.Semaphore] = None


class _LLMClientWrapper:
    """Wrap LLM clients to enforce concurrency limits without changing callers."""

    def __init__(self, client, semaphore: threading.Semaphore):
        self._client = client
        self._semaphore = semaphore

    def __getattr__(self, item):
        return getattr(self._client, item)

    def generate(self, *args, **kwargs):
        with self._semaphore:
            return self._client.generate(*args, **kwargs)


def create_llm_client(
        model_name: Optional[str] = None,
        backend_override: Optional[str] = None,
        model_path: Optional[str] = None
):
    """Factory to select LLM backend based on requested model name or env."""
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        _LLM_SEMAPHORE = threading.Semaphore(getattr(Config.LLM, "MAX_CONCURRENCY", 2))
    name = (model_name or "").lower()
    backend = (backend_override or os.getenv("LLM_BACKEND", "")).lower().strip()
    resolved_backend = backend
    if not resolved_backend:
        resolved_backend = "gemini" if name.startswith("gemini") else "ollama"
    cache_key = (resolved_backend, model_name or "", model_path)
    if cache_key in _LLM_CLIENTS:
        return _LLM_CLIENTS[cache_key]

    if resolved_backend in {"openai", "openai_compatible", "local_http"}:
        client = OpenAICompatibleClient(model_name)
    elif resolved_backend == "gemini" or name.startswith("gemini"):
        client = GeminiClient(model_name)
    elif resolved_backend == "unsloth" or model_path:
        from src.finetune.llm_client import UnslothLLMClient

        target = model_path or model_name
        client = UnslothLLMClient(target)
    elif resolved_backend == "ollama":
        client = OllamaClient(model_name)
    else:
        client = OllamaClient(model_name)

    wrapped = _LLMClientWrapper(client, _LLM_SEMAPHORE)
    _LLM_CLIENTS[cache_key] = wrapped
    return wrapped
    # default to local Ollama for plug-and-play usage
    return OllamaClient(model_name)


def get_metrics_tracker() -> MetricsTracker:
    """Singleton metrics tracker."""
    global _METRICS_TRACKER
    if _METRICS_TRACKER is None:
        _METRICS_TRACKER = MetricsTracker(get_redis_client())
    return _METRICS_TRACKER


def get_rag_system(
        model_name: Optional[str] = None,
        profile_id: Optional[str] = None,
        backend_override: Optional[str] = None,
        model_path: Optional[str] = None
) -> EnterpriseRAGSystem:
    """Get or create the RAG system instance (singleton with lazy loading)."""
    global _RAG_SYSTEM, _RAG_MODEL, _RAG_PROFILE, _RAG_BACKEND, _RAG_MODEL_PATH
    needs_new = (
        _RAG_SYSTEM is None
        or (model_name and model_name != _RAG_MODEL)
        or (_RAG_PROFILE != profile_id)
        or (_RAG_BACKEND != backend_override)
        or (_RAG_MODEL_PATH != model_path)
    )
    if needs_new:
        try:
            _RAG_SYSTEM = EnterpriseRAGSystem(
                model_name=model_name,
                profile_id=profile_id,
                backend_override=backend_override,
                model_path=model_path
            )
            _RAG_MODEL = model_name
            _RAG_PROFILE = profile_id
            _RAG_BACKEND = backend_override
            _RAG_MODEL_PATH = model_path
            logger.info("RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    return _RAG_SYSTEM


def answer_question(
        query: str,
        user_id: str,
        profile_id: str,
        subscription_id: str = "default",
        model_name: str = "llama3.2",
        persona: str = "professional document analysis assistant",
        session_id: Optional[str] = None,
        new_session: bool = False,
        disable_answer_cache: bool = False,
        force_refresh: bool = False,
        request_id: Optional[str] = None,
        index_version: Optional[str] = None,
        tools: Optional[List[str]] = None,
        use_tools: bool = False,
        tool_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for answering questions with enhanced NLU.

    Args:
        query: User's question
        user_id: User identifier
        profile_id: Department/collection name ('finance', 'banking', 'technical')
        model_name: LLM model name (for compatibility)
        persona: Assistant persona

    Returns:
        Response dictionary with enhanced metadata
    """
    resolved_model = resolve_model_for_profile(profile_id, model_name)
    effective_model = resolved_model.model_name or model_name
    rag_system = get_rag_system(
        effective_model,
        profile_id=profile_id,
        backend_override=resolved_model.backend,
        model_path=resolved_model.model_path
    )
    return rag_system.answer_question(
        query=query,
        profile_id=profile_id,
        subscription_id=subscription_id,
        user_id=user_id,
        persona=persona,
        session_id=session_id,
        new_session=new_session,
        disable_answer_cache=disable_answer_cache,
        force_refresh=force_refresh,
        request_id=request_id,
        index_version=index_version,
        tools=tools,
        use_tools=use_tools,
        tool_inputs=tool_inputs,
    )


def debug_collection(profile_id: str, subscription_id: str = "default") -> Dict[str, Any]:
    """
    Debug utility to check collection status with defensive error handling.

    Args:
        profile_id: Profile/department identifier
        subscription_id: Tenant/subscription identifier

    Returns:
        Collection statistics
    """
    try:
        collection_name = build_collection_name(subscription_id)
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(collection_name)

        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )

        sample_points = []
        if isinstance(scroll_result, tuple) and len(scroll_result) > 0:
            points_list = scroll_result[0]
            if points_list:
                sample_points = [
                    {
                        "id": str(p.id),
                        "text_preview": p.payload.get('text', '')[:200] if p.payload else 'No text',
                        "source": p.payload.get('source', 'unknown') if p.payload else 'unknown'
                    }
                    for p in points_list
                ]

        return {
            "collection_name": collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance),
            "sample_points": sample_points,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error debugging collection '{collection_name if 'collection_name' in locals() else profile_id}': {e}", exc_info=True)
        return {
            "collection_name": collection_name if 'collection_name' in locals() else profile_id,
            "error": str(e),
            "status": "error"
        }


def add_domain_terms(terms: List[str]):
    """
    Legacy hook for domain terms; kept for API compatibility.

    Args:
        terms: List of domain-specific terms
    """
    try:
        logger.info("Spell checking is disabled; ignoring %d domain terms", len(terms))
    except Exception as e:
        logger.error(f"Error adding domain terms: {e}")


def clear_conversation_history(
        user_id: str,
        subscription_id: str = "default",
        profile_id: str = "default",
        model_name: str = "",
        session_id: Optional[str] = None
):
    """
    Clear conversation history for a specific user.

    Args:
        user_id: User identifier
        subscription_id: Tenant/subscription identifier
        profile_id: Profile identifier
        model_name: Optional model name to namespace cache keys
    """
    try:
        rag_system = get_rag_system()
        ns = _build_namespace(
            subscription_id,
            profile_id,
            model_name or getattr(rag_system, "model_name", ""),
            session_id=session_id
        )
        rag_system.conversation_history.clear_history(ns, user_id)
        logger.info(f"Cleared conversation history for user {user_id}")
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")


def metrics_summary(days: int = 7) -> Dict[str, Any]:
    """
    Aggregate metrics for the requested window (defaults to 7 days).
    """
    tracker = get_metrics_tracker()
    return tracker.summary(days=days)


class RAGEvaluator:
    """Evaluation utilities for monitoring RAG performance."""

    @staticmethod
    def evaluate_retrieval(
            queries: List[str],
            profile_id: str,
            ground_truth_docs: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using MRR and Precision@K.

        Args:
            queries: List of test queries
            profile_id: Collection name
            ground_truth_docs: List of lists of relevant document IDs for each query

        Returns:
            Dictionary of evaluation metrics
        """
        if len(queries) != len(ground_truth_docs):
            raise ValueError("Queries and ground truth must have same length")

        rag_system = get_rag_system()
        mrr_scores = []
        p_at_1 = []
        p_at_3 = []
        p_at_5 = []

        for query, relevant_docs in zip(queries, ground_truth_docs):
            try:
                chunks = rag_system.retriever.retrieve(
                    collection_name=profile_id,
                    query=query,
                    top_k=10
                )

                retrieved_ids = [chunk.id for chunk in chunks]

                for rank, doc_id in enumerate(retrieved_ids, 1):
                    if doc_id in relevant_docs:
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)

                p_at_1.append(1.0 if retrieved_ids[:1] and retrieved_ids[0] in relevant_docs else 0.0)

                if len(retrieved_ids) >= 3:
                    p_at_3.append(
                        len(set(retrieved_ids[:3]) & set(relevant_docs)) / 3.0
                    )

                if len(retrieved_ids) >= 5:
                    p_at_5.append(
                        len(set(retrieved_ids[:5]) & set(relevant_docs)) / 5.0
                    )

            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")

        return {
            "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
            "precision_at_1": np.mean(p_at_1) if p_at_1 else 0.0,
            "precision_at_3": np.mean(p_at_3) if p_at_3 else 0.0,
            "precision_at_5": np.mean(p_at_5) if p_at_5 else 0.0,
            "num_queries": len(queries)
        }


__all__ = [
    'answer_question',
    'debug_collection',
    'add_domain_terms',
    'clear_conversation_history',
    'RAGEvaluator',
    'get_rag_system',
    'metrics_summary'
]
