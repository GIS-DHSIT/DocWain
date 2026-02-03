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
from dataclasses import dataclass, replace
from collections import deque, Counter
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
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
from src.api.enhanced_retrieval import AdaptiveRetriever, GraphGuidedRetriever
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
from src.api.vector_store import build_collection_name
from src.api.genai_client import generate_text, get_genai_client
from src.finetune import resolve_model_for_profile
from src.metrics.ai_metrics import get_metrics_store
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store
from src.security.response_sanitizer import sanitize_user_payload
from src.quality.bad_answer_evaluator import BadAnswerEvaluator, EvalConfig
from src.quality.auto_repair import AutoRepairEngine, RepairConfig
from src.quality.telemetry import emit_quality_telemetry
from src.rag.grounding import enforce_grounding, filter_chunks_by_query_entity
from src.rag.citations import build_citations, filter_inline_citations, replace_citations_line
from src.rag.context_reasoning import (
    AnswerRenderer,
    AnswerRenderResult,
    ContextAwareQueryAnalyzer,
    EvidencePlanBuilder,
    EvidenceQualityScorer,
    WorkingContextAssembler,
)
from src.rag.doc_inventory import DocInventoryItem, fetch_doc_inventory
from src.rag.intent_scope_orchestrator import retrieve_per_doc
from src.rag.intent_profile_entity_orchestrator import orchestrate_intent_profile_entity
from src.rag.cache_guard_v2 import CacheGuardV2, compute_retrieval_signature
from src.rag.answer_orchestrator_v3 import (
    build_clarification_response,
    build_state as build_orchestration_state_v3,
    dominant_doc_for_lookup,
    enforce_single_doc_filter,
    remove_junk_sections,
)
from src.rag.citation_policy import apply as apply_citation_policy
from src.rag.formatting_rules import apply as apply_formatting_rules
from src.rag.format_enforcer import enforce_response_formatting
from src.rag.table_renderer import render_markdown_table
from src.rag.candidate_profile_extractor import CandidateProfile, extract_candidate_profile
from src.rag.ranking_engine import rank_candidates
from src.rag.intent_classifier import (
    INTENT_COMPARE,
    INTENT_PRODUCTS_SERVICES,
    INTENT_SUMMARIZE,
    INTENT_TOTALS,
    classify_intent,
    has_multi_doc_cues,
)
from src.rag.query_cache import (
    QueryContextFingerprintCache,
    compute_retrieval_fingerprint,
    is_query_answer_consistent,
    normalize_query,
)
from src.rag.grounding_guard import verify_grounding
from src.rag.response_formatter import (
    format_conservative_response,
    format_structured_response,
    format_candidate_profile_response,
    format_multi_candidate_response,
)
from src.rag.output_formatter_resume import ResumeProfileView, format_resume_response
from src.response.formatter import format_response_text
from src.tools.resume_analyzer_v2 import analyze_resume_chunks
from src.tools.resume_router import (
    has_resume_docs,
    is_multi_profile_request,
    match_resume_docs_by_name,
    should_bypass_clarification,
    should_route_resume_analyzer,
    select_resume_docs,
)

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
    return None


def _model_kwargs_for_device(device: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    dtype = _resolve_torch_dtype()
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return kwargs


def _load_sentence_transformer(name: str, device: str) -> SentenceTransformer:
    logger.info("Loading sentence transformer model: %s (device=%s)", name, device)
    model_kwargs = _model_kwargs_for_device(device)
    local_files_only = bool(getattr(Config.Model, "DISABLE_HF", False))
    try:
        if model_kwargs:
            return SentenceTransformer(
                name,
                device=device,
                model_kwargs=model_kwargs,
                local_files_only=local_files_only,
            )
        return SentenceTransformer(name, device=device, local_files_only=local_files_only)
    except Exception as exc:  # noqa: BLE001
        if device == "cpu" and model_kwargs and _is_meta_tensor_error(exc):
            logger.warning(
                "Model load on cpu failed (%s); retrying without model kwargs",
                exc,
            )
            return SentenceTransformer(name, device="cpu", local_files_only=local_files_only)
        if device != "cpu" and _is_meta_tensor_error(exc):
            logger.warning(
                "Model load on %s failed (%s); retrying on cpu for stability",
                device,
                exc,
            )
            cpu_kwargs = _model_kwargs_for_device("cpu")
            try:
                if cpu_kwargs:
                    return SentenceTransformer(
                        name,
                        device="cpu",
                        model_kwargs=cpu_kwargs,
                        local_files_only=local_files_only,
                    )
                return SentenceTransformer(name, device="cpu", local_files_only=local_files_only)
            except Exception as cpu_exc:  # noqa: BLE001
                if cpu_kwargs and _is_meta_tensor_error(cpu_exc):
                    logger.warning(
                        "Model load on cpu failed (%s); retrying without model kwargs",
                        cpu_exc,
                    )
                    return SentenceTransformer(name, device="cpu", local_files_only=local_files_only)
                raise
        raise


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
    _configure_hf_env()
    candidates = []
    for name in getattr(Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", []):
        if name and name not in candidates:
            candidates.append(name)
    if not candidates:
        candidates.append(getattr(Config.Model, "SENTENCE_TRANSFORMERS", "sentence-transformers/all-mpnet-base-v2"))

    device = _embedding_device()
    last_error = None
    for name in candidates:
        try:
            model = _load_sentence_transformer(name, device)
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
        try:
            model = _load_sentence_transformer(model_name, _embedding_device())
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
        _MODEL = _load_model_candidates()

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


@dataclass
class EvidenceChunk:
    chunk_id: str
    text: str
    section: str
    page: Optional[Any]
    doc_name: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class EvidenceSet:
    chunks: List[EvidenceChunk]
    chunk_map: Dict[str, EvidenceChunk]


@dataclass
class EvidenceMap:
    fields: Dict[str, List[Tuple[str, str]]]
    sections: Dict[str, List[Tuple[str, str]]]


@dataclass
class DraftPoint:
    field: str
    value: str
    chunk_ids: List[str]


@dataclass
class AnswerPlan:
    answer_type: str
    requested_fields: List[str]
    required_sections: List[str]
    candidate_entities: List[str]
    must_cite: bool
    draft_points: List[DraftPoint]


@dataclass
class ValidatedPoint:
    field: str
    value: str
    chunk_ids: List[str]
    supported: bool


@dataclass
class ExtractedLineItem:
    name: str
    amount: Decimal
    raw_amount: str
    chunk_id: str


@dataclass
class ExtractedTotal:
    label: str
    amount: Decimal
    raw_amount: str
    chunk_id: str


@dataclass
class StructuredExtraction:
    items: List[ExtractedLineItem]
    totals: List[ExtractedTotal]
    currency: str
    computed: Dict[str, Any]

    @property
    def has_data(self) -> bool:
        return bool(self.items or self.totals)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "items": [
                {"name": item.name, "amount": str(item.amount), "chunk_id": item.chunk_id}
                for item in self.items
            ],
            "totals": [
                {"label": total.label, "amount": str(total.amount), "chunk_id": total.chunk_id}
                for total in self.totals
            ],
            "currency": self.currency,
            "computed": {k: str(v) for k, v in (self.computed or {}).items()},
        }


_PRECISION_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "as",
    "at",
    "from",
    "into",
    "about",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b")
_NUMERIC_RE = re.compile(r"\b\d[\d,./:-]*\b")
_DATE_WORD_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)


def _normalize_entity_text(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _entity_tokens(value: str) -> List[str]:
    normalized = _normalize_entity_text(value)
    if not normalized:
        return []
    return [tok for tok in normalized.split() if tok]


def _tokenize_for_overlap(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    return [tok for tok in tokens if tok not in _PRECISION_STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Keep bullet-like lines intact.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and any(ln.startswith(("-", "*", "•")) for ln in lines):
        return lines
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_key_value_pairs(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not text:
        return pairs
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            continue
        if len(label) > 64 or not re.search(r"[A-Za-z]", label):
            continue
        pairs.append((label, value))
    return pairs


def _extract_pattern_fields(text: str) -> Dict[str, List[str]]:
    fields: Dict[str, List[str]] = {}
    if not text:
        return fields
    emails = _EMAIL_RE.findall(text)
    if emails:
        fields.setdefault("Email", []).extend(sorted(set(emails)))
    phones = _PHONE_RE.findall(text)
    if phones:
        fields.setdefault("Phone", []).extend(sorted(set(phones)))
    numeric = _NUMERIC_RE.findall(text)
    if numeric:
        fields.setdefault("ID", []).extend(sorted(set(numeric)))
    if _DATE_WORD_RE.search(text):
        fields.setdefault("Date", []).extend(sorted(set(_DATE_WORD_RE.findall(text))))
    return fields


def build_evidence_set(chunks: List[RetrievedChunk], max_chunks: int = 15) -> EvidenceSet:
    selected = chunks[:max_chunks]
    evidence_chunks: List[EvidenceChunk] = []
    chunk_map: Dict[str, EvidenceChunk] = {}
    for chunk in selected:
        meta = chunk.metadata or {}
        chunk_id = meta.get("chunk_id") or chunk.id
        section = meta.get("section_title") or meta.get("section_path") or meta.get("section") or ""
        page = meta.get("page") or meta.get("page_start") or meta.get("page_end")
        doc_name = (
            meta.get("file_name")
            or meta.get("filename")
            or meta.get("source_file")
            or meta.get("source")
            or chunk.source
            or ""
        )
        evidence = EvidenceChunk(
            chunk_id=str(chunk_id),
            text=chunk.text or "",
            section=str(section or ""),
            page=page,
            doc_name=str(doc_name or ""),
            metadata=meta,
            score=float(chunk.score),
        )
        evidence_chunks.append(evidence)
        chunk_map[evidence.chunk_id] = evidence
    return EvidenceSet(chunks=evidence_chunks, chunk_map=chunk_map)


def build_evidence_map(evidence_set: EvidenceSet) -> EvidenceMap:
    fields: Dict[str, List[Tuple[str, str]]] = {}
    sections: Dict[str, List[Tuple[str, str]]] = {}
    for chunk in evidence_set.chunks:
        meta_fields = chunk.metadata.get("fields") if isinstance(chunk.metadata, dict) else None
        if isinstance(meta_fields, dict):
            for label, value in meta_fields.items():
                if not label or value is None:
                    continue
                fields.setdefault(str(label), []).append((str(value), chunk.chunk_id))
        for label, value in _extract_key_value_pairs(chunk.text):
            fields.setdefault(label, []).append((value, chunk.chunk_id))
        pattern_fields = _extract_pattern_fields(chunk.text)
        for label, values in pattern_fields.items():
            for value in values:
                fields.setdefault(label, []).append((value, chunk.chunk_id))
        section_key = chunk.section or "Untitled Section"
        for sentence in _split_sentences(chunk.text):
            sections.setdefault(section_key, []).append((sentence, chunk.chunk_id))
    return EvidenceMap(fields=fields, sections=sections)


class IntentType:
    FACT_LOOKUP = "FACT_LOOKUP"
    SECTION_SUMMARY = "SECTION_SUMMARY"
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    TABULAR_SUMMARY = "TABULAR_SUMMARY"
    NUMERIC_AGGREGATION = "NUMERIC_AGGREGATION"
    COMPARISON = "COMPARISON"


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    signals: Dict[str, bool]
    reason: str = ""


class IntentRouter:
    """Deterministic intent router (no LLM calls)."""

    _COMPARISON_PATTERNS = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference\b",
    ]
    _SECTION_PATTERNS = [
        r"\bsummary\b",
        r"\bsummarize\b",
        r"\boverview\b",
        r"\bsection\b",
        r"\bbrief\b",
        r"\bhigh[- ]level\b",
    ]
    _ENTITY_PATTERNS = [
        r"\bextract\b",
        r"\bidentify\b",
        r"\bentities?\b",
        r"\bnames?\b",
        r"\bwho\b",
        r"\bemail\b",
        r"\bphone\b",
        r"\baddress\b",
    ]
    _TABULAR_PATTERNS = [
        r"\btable\b",
        r"\btabular\b",
        r"\bspreadsheet\b",
        r"\bitemi[sz]ed\b",
        r"\bline items?\b",
        r"\bbreakdown\b",
        r"\bfor each\b",
        r"\bper item\b",
        r"\bper product\b",
        r"\bper line\b",
    ]
    _NUMERIC_PATTERNS = [
        r"\btotal(s)?\b",
        r"\bgrand total\b",
        r"\bsubtotal(s)?\b",
        r"\bsum\b",
        r"\baggregate(d|ion)?\b",
        r"\bcombined\b",
        r"\boverall\b",
        r"\bcount\b",
        r"\bhow many\b",
        r"\bhow much\b",
        r"\baverage\b",
        r"\bmean\b",
        r"\bmedian\b",
        r"\bpercent(age)?\b",
        r"\bratio\b",
        r"\bcalculate\b",
        r"\bcalculation\b",
        r"\bcompute\b",
        r"\binvoice\b",
        r"\bamount due\b",
        r"\btax\b",
        r"\bfee\b",
    ]

    @staticmethod
    def _match_any(patterns: List[str], text: str) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def classify(self, query: str) -> IntentDecision:
        q = (query or "").lower()
        has_comparison = self._match_any(self._COMPARISON_PATTERNS, q)
        has_numeric = self._match_any(self._NUMERIC_PATTERNS, q)
        has_tabular = self._match_any(self._TABULAR_PATTERNS, q)
        has_section = self._match_any(self._SECTION_PATTERNS, q)
        has_entity = self._match_any(self._ENTITY_PATTERNS, q)

        signals = {
            "comparison": has_comparison,
            "numeric": has_numeric,
            "tabular": has_tabular,
            "section": has_section,
            "entity": has_entity,
        }

        if has_comparison:
            return IntentDecision(IntentType.COMPARISON, signals, reason="comparison keyword")
        if has_numeric and has_tabular:
            # Prefer numeric aggregation when totals are explicitly requested.
            return IntentDecision(IntentType.NUMERIC_AGGREGATION, signals, reason="numeric+tabular keywords")
        if has_numeric:
            return IntentDecision(IntentType.NUMERIC_AGGREGATION, signals, reason="numeric keyword")
        if has_tabular:
            return IntentDecision(IntentType.TABULAR_SUMMARY, signals, reason="tabular keyword")
        if has_section:
            return IntentDecision(IntentType.SECTION_SUMMARY, signals, reason="summary keyword")
        if has_entity:
            return IntentDecision(IntentType.ENTITY_EXTRACTION, signals, reason="entity keyword")
        return IntentDecision(IntentType.FACT_LOOKUP, signals, reason="default")


_INTENT_ROUTER = IntentRouter()


def detect_intent_class(query: str) -> str:
    return _INTENT_ROUTER.classify(query).intent


_AGGREGATION_PATTERNS = [
    r"\btotal(s)?\b",
    r"\bgrand total\b",
    r"\bsubtotal(s)?\b",
    r"\bsum\b",
    r"\baggregate(d|ion)?\b",
    r"\bcombined\b",
    r"\boverall\b",
    r"\bcount\b",
    r"\bhow many\b",
    r"\bhow much\b",
    r"\baverage\b",
    r"\bmean\b",
    r"\bmedian\b",
    r"\bpercent(age)?\b",
    r"\bratio\b",
    r"\bcalculate\b",
    r"\bcalculation\b",
    r"\bcompute\b",
    r"\bsummary\b",
    r"\bsummarize\b",
    r"\boverview\b",
    r"\binvoice\b",
    r"\bsubtotal\b",
    r"\btax\b",
    r"\bfee\b",
]

_MULTI_DOC_HINTS = [
    r"\bacross\b",
    r"\bacross\s+all\b",
    r"\ball\s+documents?\b",
    r"\ball\s+invoices?\b",
    r"\ball\s+contracts?\b",
    r"\ball\s+receipts?\b",
    r"\bcombined\b",
    r"\boverall\b",
    r"\bcompany[- ]wide\b",
    r"\borganization[- ]wide\b",
    r"\bportfolio\b",
    r"\bentire\b",
    r"\bmultiple\b",
    r"\bvarious\b",
]


def is_aggregation_intent(query: str) -> bool:
    decision = _INTENT_ROUTER.classify(query)
    return decision.intent in {IntentType.NUMERIC_AGGREGATION, IntentType.TABULAR_SUMMARY}


def is_multi_doc_aggregation(query: str) -> bool:
    q = (query or "").lower()
    return any(re.search(pattern, q) for pattern in _MULTI_DOC_HINTS)


def _extract_doc_id(meta: Dict[str, Any]) -> Optional[str]:
    if not meta:
        return None
    return meta.get("document_id") or meta.get("doc_id") or meta.get("docId")


def _top_doc_ids(chunks: List["RetrievedChunk"], max_docs: int = 4) -> List[str]:
    counts: Counter = Counter()
    for chunk in chunks:
        meta = chunk.metadata or {}
        doc_id = _extract_doc_id(meta)
        if doc_id:
            counts[str(doc_id)] += 1
    return [doc_id for doc_id, _ in counts.most_common(max_docs)]


def _merge_retrieved_chunks(
    primary: List["RetrievedChunk"],
    extra: List["RetrievedChunk"],
) -> List["RetrievedChunk"]:
    seen = {chunk.id for chunk in primary if chunk.id}
    merged = list(primary)
    for chunk in extra:
        if not chunk.id or chunk.id in seen:
            continue
        merged.append(chunk)
        seen.add(chunk.id)
    return merged


_NAME_STOPWORDS = {
    "resume",
    "cv",
    "profile",
    "document",
    "documents",
    "certifications",
    "skills",
    "experience",
    "education",
    "about",
    "for",
    "of",
    "on",
    "the",
}


def extract_candidate_entities(query: str) -> List[str]:
    if not query:
        return []
    candidates: List[str] = []
    patterns = [
        r'\b(?:education|skills?|experience|background|qualification|details?)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[\'"]s\s+(?:education|skills?|experience)',
        r'\btell\s+me\s+about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'\b(?:resume|cv|profile|document|record|report|notes?)\s+(?:of|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'["\']([A-Za-z][A-Za-z\-. ]+?)["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                candidates.append(candidate)

    if not candidates:
        for match in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", query):
            if match and match.lower() not in _NAME_STOPWORDS:
                candidates.append(match.strip())

    unique: List[str] = []
    seen = set()
    for cand in candidates:
        cleaned = re.sub(r"[^\w\s\-.]", "", cand).strip()
        if not cleaned:
            continue
        norm = cleaned.lower()
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(cleaned)
    return unique


def infer_requested_fields(query: str, intent: str, evidence_map: EvidenceMap) -> List[str]:
    q = (query or "").lower()
    fields: List[str] = []

    def _add(field_name: str):
        if field_name and field_name not in fields:
            fields.append(field_name)

    if any(term in q for term in ["contact", "reach", "phone", "email"]):
        _add("Email")
        _add("Phone")
    keyword_map: Dict[str, str] = {
        "email": "Email",
        "e-mail": "Email",
        "phone": "Phone",
        "mobile": "Phone",
        "address": "Address",
        "diagnosis": "Diagnosis",
        "admission": "Admission",
        "education": "Education",
        "degree": "Education",
        "skills": "Skills",
        "experience": "Experience",
        "projects": "Projects",
        "certification": "Certifications",
        "certifications": "Certifications",
        "license": "Licenses",
        "licenses": "Licenses",
        "summary": "Summary",
        "dob": "Date of Birth",
        "birth": "Date of Birth",
        "date of birth": "Date of Birth",
        "id": "ID",
        "policy": "Policy Number",
        "invoice": "Invoice",
        "amount": "Amount",
        "total": "Total",
        "date": "Date",
    }
    for key, label in keyword_map.items():
        if key in q:
            _add(label)

    if intent in {IntentType.SECTION_SUMMARY, "SECTION_DUMP"} and not fields:
        for section in evidence_map.sections.keys():
            _add(section)

    if intent in {IntentType.NUMERIC_AGGREGATION, IntentType.TABULAR_SUMMARY}:
        _add("Line Items")
        _add("Subtotal")
        _add("Tax")
        _add("Total")

    if not fields and evidence_map.fields:
        for label in list(evidence_map.fields.keys())[:4]:
            _add(label)

    return fields


_CURRENCY_SYMBOLS = "$"
_AMOUNT_RE = re.compile(rf"(?P<currency>[{_CURRENCY_SYMBOLS}])?\s*(?P<value>\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?|\d+(?:\.\d{{2}})?)")
_CURRENCY_RE = re.compile(rf"[{_CURRENCY_SYMBOLS}]\s*\d")
_TABLE_SPLIT_RE = re.compile(r"\s{2,}|\t|\|")
_TABLE_HEADER_RE = re.compile(r"\b(item|description|product|qty|quantity|unit price|price|amount|total)\b", re.IGNORECASE)
_TOTAL_LABEL_RE = re.compile(
    r"\b(grand total|total due|amount due|total amount|invoice total|subtotal|tax|fees?|total)\b",
    re.IGNORECASE,
)


def _normalize_amount(raw: str) -> Tuple[Optional[Decimal], str]:
    if not raw:
        return None, ""
    symbol_match = re.search(rf"[{_CURRENCY_SYMBOLS}]", raw)
    symbol = symbol_match.group(0) if symbol_match else ""
    cleaned = raw.replace(symbol, "")
    cleaned = cleaned.replace(",", "").strip()
    negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        negative = True
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("-"):
        negative = True
        cleaned = cleaned[1:].strip()
    try:
        value = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None, symbol
    if negative:
        value = value * Decimal("-1")
    return value, symbol


def _format_amount(value: Decimal, symbol: str) -> str:
    if value is None:
        return ""
    try:
        quantized = value.quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        quantized = value
    if symbol:
        return f"{symbol}{quantized:,.2f}"
    return f"{quantized:,.2f}"


def _extract_amounts(line: str) -> List[Tuple[Decimal, str, str]]:
    results: List[Tuple[Decimal, str, str]] = []
    for match in _AMOUNT_RE.finditer(line):
        raw = match.group(0)
        value, symbol = _normalize_amount(raw)
        if value is None:
            continue
        results.append((value, symbol, raw))
    return results


def _looks_like_header(line: str, amounts: List[Tuple[Decimal, str, str]]) -> bool:
    if not line:
        return False
    if _TABLE_HEADER_RE.search(line) and not amounts:
        return True
    return False


def _extract_line_item(line: str) -> Tuple[Optional[str], Optional[Decimal], str]:
    amounts = _extract_amounts(line)
    if not amounts:
        return None, None, ""
    if _TOTAL_LABEL_RE.search(line):
        return None, None, ""
    if _looks_like_header(line, amounts):
        return None, None, ""
    # Choose the last amount as line total
    amount_value, symbol, raw = amounts[-1]
    if not re.search(r"[A-Za-z]", line):
        return None, None, ""
    name = line[: line.rfind(raw)].strip(" -|:\t")
    if _TABLE_SPLIT_RE.search(line):
        parts = [p.strip() for p in _TABLE_SPLIT_RE.split(line) if p.strip()]
        if parts:
            name = parts[0]
    if not name or len(name) < 2:
        return None, None, ""
    return name, amount_value, symbol or ""


def _extract_totals_from_line(line: str) -> List[Tuple[str, Decimal, str]]:
    totals: List[Tuple[str, Decimal, str]] = []
    label_match = _TOTAL_LABEL_RE.search(line)
    if not label_match:
        return totals
    amounts = _extract_amounts(line)
    if not amounts:
        return totals
    amount_value, symbol, raw = amounts[-1]
    label = label_match.group(1).lower()
    label_map = {
        "grand total": "Total",
        "total due": "Total",
        "amount due": "Total Due",
        "total amount": "Total",
        "invoice total": "Total",
        "subtotal": "Subtotal",
        "tax": "Tax",
        "fee": "Fee",
        "fees": "Fees",
        "total": "Total",
    }
    label_display = label_map.get(label, label.title())
    totals.append((label_display, amount_value, symbol or raw))
    return totals


def extract_structured_data(chunks: List["RetrievedChunk"]) -> StructuredExtraction:
    items: List[ExtractedLineItem] = []
    totals: List[ExtractedTotal] = []
    currency_symbol = ""

    for chunk in chunks or []:
        meta = chunk.metadata or {}
        chunk_id = meta.get("chunk_id") or chunk.id
        text = chunk.text or ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for label, amount_value, symbol in _extract_totals_from_line(line):
                if symbol and not currency_symbol:
                    currency_symbol = symbol if symbol in _CURRENCY_SYMBOLS else currency_symbol
                totals.append(
                    ExtractedTotal(
                        label=label,
                        amount=amount_value,
                        raw_amount=str(amount_value),
                        chunk_id=str(chunk_id),
                    )
                )
            name, amount_value, symbol = _extract_line_item(line)
            if name and amount_value is not None:
                if symbol and not currency_symbol:
                    currency_symbol = symbol if symbol in _CURRENCY_SYMBOLS else currency_symbol
                items.append(
                    ExtractedLineItem(
                        name=name,
                        amount=amount_value,
                        raw_amount=str(amount_value),
                        chunk_id=str(chunk_id),
                    )
                )

    # Compute consistency checks
    computed: Dict[str, Any] = {}
    if items:
        items_sum = sum((item.amount for item in items), Decimal("0.00"))
        computed["items_sum"] = items_sum
    primary_total = None
    for preferred in ("Total", "Total Due", "Grand Total"):
        for total in totals:
            if total.label == preferred:
                primary_total = total.amount
                break
        if primary_total is not None:
            break
    if primary_total is None and totals:
        primary_total = totals[0].amount
    if primary_total is not None and computed.get("items_sum") is not None:
        diff = computed["items_sum"] - primary_total
        tolerance = max(Decimal("0.01"), (abs(primary_total) * Decimal("0.01")))
        computed["primary_total"] = primary_total
        computed["difference"] = diff
        computed["matches"] = abs(diff) <= tolerance

    return StructuredExtraction(items=items, totals=totals, currency=currency_symbol, computed=computed)


def _chunk_signal_flags(chunk: "RetrievedChunk") -> Dict[str, bool]:
    meta = chunk.metadata or {}
    text = chunk.text or ""
    chunk_type = (meta.get("chunk_type") or "").lower()
    has_table = chunk_type in {"table", "table_row", "table_header"} or (
        _TABLE_SPLIT_RE.search(text) and _TABLE_HEADER_RE.search(text)
    )
    has_numeric = bool(_TOTAL_LABEL_RE.search(text) or _CURRENCY_RE.search(text) or _NUMERIC_RE.search(text))
    has_summary = chunk_type in {"doc_summary", "section_summary", "summary"} or "summary" in text.lower()
    return {"table": has_table, "numeric": has_numeric, "summary": has_summary}


def select_aggregation_chunks(
    chunks: List["RetrievedChunk"],
    *,
    min_chunks: int = 6,
    max_chunks: int = 18,
) -> List["RetrievedChunk"]:
    if not chunks:
        return []
    max_chunks = max(max_chunks, min_chunks)
    best_by_id: Dict[str, RetrievedChunk] = {}
    ordered_ids: List[str] = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        key = str(meta.get("chunk_id") or chunk.id or "")
        if not key:
            key = str(id(chunk))
        if key not in best_by_id:
            best_by_id[key] = chunk
            ordered_ids.append(key)
        else:
            if float(chunk.score) > float(best_by_id[key].score):
                best_by_id[key] = chunk
    unique_chunks = [best_by_id[key] for key in ordered_ids]

    scored: List[Tuple[float, RetrievedChunk, Dict[str, bool]]] = []
    for chunk in unique_chunks:
        signals = _chunk_signal_flags(chunk)
        score = float(chunk.score)
        if signals["numeric"]:
            score += 0.6
        if signals["table"]:
            score += 0.5
        if signals["summary"]:
            score += 0.15
        scored.append((score, chunk, signals))

    scored.sort(key=lambda item: item[0], reverse=True)
    summary_chunks = [c for _, c, s in scored if s["summary"]]
    line_item_chunks = [c for _, c, s in scored if s["numeric"] or s["table"]]
    other_chunks = [c for _, c, s in scored if not (s["summary"] or s["numeric"] or s["table"])]

    selected: List[RetrievedChunk] = []
    selected_ids: Set[str] = set()

    def _add(chunk: RetrievedChunk) -> None:
        key = str((chunk.metadata or {}).get("chunk_id") or chunk.id or "")
        if key in selected_ids:
            return
        selected.append(chunk)
        selected_ids.add(key)

    summary_target = min(len(summary_chunks), max(1, max_chunks // 4))
    for chunk in summary_chunks[:summary_target]:
        _add(chunk)

    for chunk in line_item_chunks:
        if len(selected) >= max_chunks:
            break
        _add(chunk)

    if len(selected) < min_chunks:
        for chunk in other_chunks:
            if len(selected) >= min_chunks:
                break
            _add(chunk)

    if len(selected) < max_chunks:
        for _, chunk, _ in scored:
            if len(selected) >= max_chunks:
                break
            _add(chunk)

    return selected


_TOTAL_LABEL_ORDER = {
    "Subtotal": 1,
    "Tax": 2,
    "Fee": 3,
    "Fees": 3,
    "Total": 4,
    "Total Due": 5,
}


def _build_sources_from_chunk_ids(evidence_set: EvidenceSet, chunk_ids: List[str]) -> List[Dict[str, Any]]:
    source_order: List[str] = []
    seen: Set[str] = set()
    for chunk_id in chunk_ids:
        if chunk_id in seen:
            continue
        if chunk_id not in evidence_set.chunk_map:
            continue
        seen.add(chunk_id)
        source_order.append(chunk_id)
    sources: List[Dict[str, Any]] = []
    for idx, chunk_id in enumerate(source_order, 1):
        chunk = evidence_set.chunk_map.get(chunk_id)
        if not chunk:
            continue
        doc_name = chunk.doc_name or "Document"
        section = chunk.section or "Section"
        page = chunk.page if chunk.page is not None else "N/A"
        sources.append(
            {
                "source_id": idx,
                "source_name": doc_name,
                "section": section,
                "page": page,
            }
        )
    return sources


def assemble_aggregation_response(
    *,
    header: Optional[str],
    preface_lines: List[str],
    extraction: StructuredExtraction,
    evidence_set: EvidenceSet,
) -> Tuple[str, List[Dict[str, Any]]]:
    lines: List[str] = []
    if header:
        lines.append(header)
    for line in preface_lines:
        if line:
            lines.append(line)

    used_chunk_ids: List[str] = []

    if extraction.items:
        seen_items: Set[Tuple[str, str]] = set()
        for item in extraction.items:
            amount_text = _format_amount(item.amount, extraction.currency)
            item_key = (item.name.strip().lower(), amount_text)
            if item_key in seen_items:
                continue
            seen_items.add(item_key)
            lines.append(f"- {item.name}: {amount_text}")
            used_chunk_ids.append(item.chunk_id)
    else:
        lines.append("- Line items: Only totals were visible in the retrieved pages.")

    if extraction.totals:
        ordered_totals = sorted(
            extraction.totals,
            key=lambda t: _TOTAL_LABEL_ORDER.get(t.label, 99),
        )
        seen_totals: Set[Tuple[str, str]] = set()
        for total in ordered_totals:
            amount_text = _format_amount(total.amount, extraction.currency)
            total_key = (total.label.strip().lower(), amount_text)
            if total_key in seen_totals:
                continue
            seen_totals.add(total_key)
            lines.append(f"- {total.label}: {amount_text}")
            used_chunk_ids.append(total.chunk_id)
    else:
        lines.append("- Totals: No total amounts were visible in the retrieved pages.")

    computed = extraction.computed or {}
    if computed.get("items_sum") is not None:
        items_sum_text = _format_amount(computed["items_sum"], extraction.currency)
        if computed.get("primary_total") is not None:
            total_text = _format_amount(computed["primary_total"], extraction.currency)
            if computed.get("matches"):
                lines.append(f"Note: The sum of listed items is {items_sum_text}, which matches the stated total {total_text}.")
            else:
                diff_text = _format_amount(computed.get("difference", Decimal("0.00")), extraction.currency)
                lines.append(
                    f"Note: The sum of listed items is {items_sum_text}, which differs from the stated total {total_text} by {diff_text}."
                )
        else:
            lines.append(f"Note: The sum of listed items is {items_sum_text}.")

    if not extraction.items or not extraction.totals:
        lines.append(
            "Note: Only amounts visible in the retrieved pages are listed; additional line items or totals may appear elsewhere in the document."
        )

    sources = _build_sources_from_chunk_ids(evidence_set, used_chunk_ids)
    citations_line = build_citations(sources)
    lines.append(citations_line)
    return "\n".join(lines), sources


def _summarize_evidence_for_plan(evidence_map: EvidenceMap, max_items: int = 40) -> str:
    lines: List[str] = []
    lines.append("EVIDENCE FIELDS:")
    count = 0
    for label, items in evidence_map.fields.items():
        for value, chunk_id in items:
            lines.append(f"- {label}: {value} (chunk_id={chunk_id})")
            count += 1
            if count >= max_items:
                break
        if count >= max_items:
            break
    lines.append("EVIDENCE SECTIONS:")
    count = 0
    for section, items in evidence_map.sections.items():
        for sentence, chunk_id in items[:2]:
            lines.append(f"- {section}: {sentence} (chunk_id={chunk_id})")
            count += 1
            if count >= max_items:
                break
        if count >= max_items:
            break
    return "\n".join(lines)


def _parse_answer_plan(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def generate_answer_plan(
    *,
    query: str,
    intent: str,
    evidence_map: EvidenceMap,
    candidate_entities: List[str],
    evidence_set: Optional[EvidenceSet] = None,
    llm_client: Optional[Any] = None,
    max_fields: int = 6,
) -> AnswerPlan:
    requested_fields = infer_requested_fields(query, intent, evidence_map)[:max_fields]
    plan: Optional[AnswerPlan] = None
    if llm_client:
        evidence_summary = _summarize_evidence_for_plan(evidence_map)
        prompt = (
            "You are producing a machine-checkable AnswerPlan. Use ONLY the evidence items below. "
            "Return strict JSON only.\n\n"
            f"USER QUERY: {query}\n"
            f"INTENT: {intent}\n"
            f"CANDIDATE ENTITIES: {candidate_entities}\n"
            f"SUGGESTED FIELDS: {requested_fields}\n\n"
            f"{evidence_summary}\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "answer_type": "FACT_LOOKUP|SECTION_SUMMARY|ENTITY_EXTRACTION|TABULAR_SUMMARY|NUMERIC_AGGREGATION|COMPARISON",\n'
            '  "requested_fields": ["..."],\n'
            '  "required_sections": ["..."],\n'
            '  "candidate_entities": ["..."],\n'
            '  "must_cite": true,\n'
            '  "draft_points": [\n'
            '    {"field": "...", "value": "...", "chunk_ids": ["chunk_id"]}\n'
            "  ]\n"
            "}\n"
            "Rules: draft_points must use only chunk_ids from evidence; value must be copied from evidence items.\n"
        )
        try:
            raw = llm_client.generate(prompt, max_retries=1, backoff=0.2)
            parsed = _parse_answer_plan(raw)
            if isinstance(parsed, dict) and parsed:
                plan = _coerce_answer_plan(parsed, intent, requested_fields, candidate_entities)
        except Exception as exc:  # noqa: BLE001
            logger.debug("AnswerPlan generation failed: %s", exc)

    if plan is None:
        plan = build_deterministic_plan(
            intent=intent,
            requested_fields=requested_fields,
            candidate_entities=candidate_entities,
            evidence_map=evidence_map,
        )
    if evidence_set:
        valid_ids = set(evidence_set.chunk_map.keys())
        filtered_points = []
        for point in plan.draft_points:
            if any(cid in valid_ids for cid in point.chunk_ids):
                filtered_points.append(point)
        plan.draft_points = filtered_points
    return plan


def _coerce_answer_plan(
    parsed: Dict[str, Any],
    fallback_intent: str,
    requested_fields: List[str],
    candidate_entities: List[str],
) -> AnswerPlan:
    answer_type = str(parsed.get("answer_type") or fallback_intent).strip().upper()
    if answer_type not in {
        IntentType.FACT_LOOKUP,
        IntentType.SECTION_SUMMARY,
        IntentType.ENTITY_EXTRACTION,
        IntentType.TABULAR_SUMMARY,
        IntentType.NUMERIC_AGGREGATION,
        IntentType.COMPARISON,
        "SECTION_DUMP",
        "RANKING",
        "FORMAT_REQUEST",
    }:
        answer_type = fallback_intent
    fields = parsed.get("requested_fields") or requested_fields
    if not isinstance(fields, list):
        fields = requested_fields
    fields = [str(f).strip() for f in fields if str(f).strip()]
    sections = parsed.get("required_sections") or []
    if not isinstance(sections, list):
        sections = []
    sections = [str(s).strip() for s in sections if str(s).strip()]
    entities = parsed.get("candidate_entities") or candidate_entities
    if not isinstance(entities, list):
        entities = candidate_entities
    entities = [str(e).strip() for e in entities if str(e).strip()]
    must_cite = bool(parsed.get("must_cite", True))
    raw_points = parsed.get("draft_points") or []
    draft_points: List[DraftPoint] = []
    if isinstance(raw_points, list):
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "").strip()
            value = str(item.get("value") or item.get("value_candidate") or "").strip()
            chunk_ids = item.get("chunk_ids") or item.get("chunk_id") or []
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            if not isinstance(chunk_ids, list):
                chunk_ids = []
            chunk_ids = [str(cid).strip() for cid in chunk_ids if str(cid).strip()]
            if not field or not value or not chunk_ids:
                continue
            draft_points.append(DraftPoint(field=field, value=value, chunk_ids=chunk_ids))
    return AnswerPlan(
        answer_type=answer_type,
        requested_fields=fields,
        required_sections=sections,
        candidate_entities=entities,
        must_cite=must_cite,
        draft_points=draft_points,
    )


def build_deterministic_plan(
    *,
    intent: str,
    requested_fields: List[str],
    candidate_entities: List[str],
    evidence_map: EvidenceMap,
) -> AnswerPlan:
    draft_points: List[DraftPoint] = []
    for field in requested_fields:
        field_norm = field.lower()
        matched = False
        for label, values in evidence_map.fields.items():
            if field_norm in label.lower():
                for value, chunk_id in values[:2]:
                    draft_points.append(DraftPoint(field=field, value=value, chunk_ids=[chunk_id]))
                matched = True
                break
        if matched:
            continue
        for section, items in evidence_map.sections.items():
            if field_norm in section.lower():
                sentence, chunk_id = items[0]
                draft_points.append(DraftPoint(field=field, value=sentence, chunk_ids=[chunk_id]))
                matched = True
                break
        if not matched and intent in {IntentType.SECTION_SUMMARY, "SECTION_DUMP"}:
            for section, items in evidence_map.sections.items():
                sentence, chunk_id = items[0]
                draft_points.append(DraftPoint(field=section, value=sentence, chunk_ids=[chunk_id]))
                break

    return AnswerPlan(
        answer_type=intent,
        requested_fields=requested_fields,
        required_sections=[],
        candidate_entities=candidate_entities,
        must_cite=True,
        draft_points=draft_points,
    )


def _point_supported(
    value: str,
    chunk_text: str,
    candidate_entities: List[str],
    answer_type: str,
) -> bool:
    if not value:
        return False
    text_norm = (chunk_text or "").lower()
    value_norm = value.lower()
    if "not explicitly mentioned" in value_norm:
        return True

    numeric_tokens = _NUMERIC_RE.findall(value)
    email_tokens = _EMAIL_RE.findall(value)
    phone_tokens = _PHONE_RE.findall(value)
    if numeric_tokens or email_tokens or phone_tokens or _DATE_WORD_RE.search(value):
        for token in numeric_tokens + email_tokens + phone_tokens:
            if token and token.lower() not in text_norm:
                return False
        if _DATE_WORD_RE.search(value) and not _DATE_WORD_RE.search(chunk_text):
            return False
        return True

    if answer_type in {IntentType.SECTION_SUMMARY, "SECTION_DUMP"}:
        if value_norm in text_norm:
            return True
        try:
            import difflib
            ratio = difflib.SequenceMatcher(None, value_norm, text_norm).ratio()
            return ratio >= 0.85
        except Exception:
            return False

    value_tokens = _tokenize_for_overlap(value)
    if not value_tokens:
        return False
    chunk_tokens = set(_tokenize_for_overlap(chunk_text))
    overlap = len(set(value_tokens) & chunk_tokens)
    if overlap < 2:
        return False
    if candidate_entities:
        entity_tokens = {tok for name in candidate_entities for tok in _entity_tokens(name)}
        if entity_tokens and not (entity_tokens & chunk_tokens):
            return False
    return True


def validate_answer_plan(
    plan: AnswerPlan,
    evidence_set: EvidenceSet,
) -> List[ValidatedPoint]:
    validated: List[ValidatedPoint] = []
    for point in plan.draft_points:
        supported = False
        for chunk_id in point.chunk_ids:
            chunk = evidence_set.chunk_map.get(chunk_id)
            if not chunk:
                continue
            if _point_supported(point.value, chunk.text, plan.candidate_entities, plan.answer_type):
                supported = True
                break
        if supported:
            validated.append(
                ValidatedPoint(
                    field=point.field,
                    value=point.value,
                    chunk_ids=[cid for cid in point.chunk_ids if cid in evidence_set.chunk_map],
                    supported=True,
                )
            )
        else:
            validated.append(
                ValidatedPoint(
                    field=point.field,
                    value="Not explicitly mentioned in the provided documents.",
                    chunk_ids=[],
                    supported=False,
                )
            )
    return validated


def _format_inline_citations(chunk_ids: List[str], source_map: Dict[str, int]) -> str:
    if not chunk_ids:
        return ""
    ordered: List[int] = []
    seen: Set[int] = set()
    for chunk_id in chunk_ids:
        source_id = source_map.get(chunk_id)
        if source_id is None or source_id in seen:
            continue
        seen.add(source_id)
        ordered.append(source_id)
    if not ordered:
        return ""
    labels = ", ".join(f"SOURCE-{sid}" for sid in ordered)
    return f"[{labels}]"


def _is_list_field(field: str) -> bool:
    return field.lower() in {
        "skills",
        "experience",
        "projects",
        "certifications",
        "licenses",
        "education",
        "contact",
    }


def assemble_response_text(
    *,
    header: Optional[str],
    preface_lines: List[str],
    plan: AnswerPlan,
    validated_points: List[ValidatedPoint],
    evidence_set: EvidenceSet,
) -> Tuple[str, List[Dict[str, Any]]]:
    lines: List[str] = []
    if header:
        lines.append(header)
    for line in preface_lines:
        if line:
            lines.append(line)

    candidate_chunk_ids: Set[str] = set()
    for point in validated_points:
        if point.supported:
            candidate_chunk_ids.update(point.chunk_ids)
    source_order = [chunk.chunk_id for chunk in evidence_set.chunks if chunk.chunk_id in candidate_chunk_ids]
    source_map = {chunk_id: idx + 1 for idx, chunk_id in enumerate(source_order)}

    points_by_field: Dict[str, List[ValidatedPoint]] = {}
    for point in validated_points:
        points_by_field.setdefault(point.field, []).append(point)

    requested_fields = plan.requested_fields or list(points_by_field.keys())
    if requested_fields:
        seen_fields: Set[str] = set()
        deduped: List[str] = []
        for field in requested_fields:
            if field in seen_fields:
                continue
            seen_fields.add(field)
            deduped.append(field)
        requested_fields = deduped
    if not requested_fields:
        requested_fields = ["Answer"]

    for field in requested_fields:
        points = points_by_field.get(field) or []
        supported_points = [p for p in points if p.supported and p.chunk_ids]
        if not supported_points:
            lines.append(f"- {field}: Not explicitly mentioned in the provided documents.")
            continue
        if _is_list_field(field) and len(supported_points) > 1:
            combined_values = []
            combined_chunks: List[str] = []
            for point in supported_points:
                if point.value not in combined_values:
                    combined_values.append(point.value)
                combined_chunks.extend(point.chunk_ids)
            combined_chunks = [cid for cid in combined_chunks if cid in source_map]
            inline = _format_inline_citations(combined_chunks, source_map)
            joined = "; ".join(combined_values)
            lines.append(f"- {field}: {joined} {inline}".rstrip())
            continue
        seen_values: Set[str] = set()
        for point in supported_points:
            value_key = point.value.strip().lower()
            if value_key in seen_values:
                continue
            seen_values.add(value_key)
            inline = _format_inline_citations(point.chunk_ids, source_map)
            lines.append(f"- {field}: {point.value} {inline}".rstrip())

    used_chunk_ids = source_order
    citations: List[str] = []
    sources: List[Dict[str, Any]] = []
    for chunk_id in used_chunk_ids:
        chunk = evidence_set.chunk_map.get(chunk_id)
        if not chunk:
            continue
        source_id = source_map.get(chunk_id)
        if not source_id:
            continue
        doc_name = chunk.doc_name or "Document"
        section = chunk.section or "Section"
        page = chunk.page if chunk.page is not None else "N/A"
        citations.append(f"{doc_name} | {section} | {page}")
        sources.append(
            {
                "source_id": source_id,
                "source_name": doc_name,
                "section": section,
                "page": page,
            }
        )

    citations_line = "Citations: " + "; ".join(citations) if citations else "Citations:"
    lines.append(citations_line)
    return "\n".join(lines), sources


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
        self.last_docs: Dict[Tuple[str, str], Dict[str, str]] = {}
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _history_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:history:{namespace}:{user_id}"

    def _docs_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:recent_docs:{namespace}:{user_id}"

    def _last_doc_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:last_doc:{namespace}:{user_id}"

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

    def _load_last_doc(self, namespace: str, user_id: str):
        key = (namespace, user_id)
        if key in self.last_docs or not self.redis:
            return
        try:
            cached = self.redis.get(self._last_doc_key(namespace, user_id))
        except Exception as exc:
            logger.warning(f"Failed to read last doc from Redis: {exc}")
            return
        if not cached:
            return
        try:
            payload = json.loads(cached)
            if isinstance(payload, dict):
                self.last_docs[key] = {
                    "doc_id": str(payload.get("doc_id") or ""),
                    "doc_name": str(payload.get("doc_name") or ""),
                }
        except Exception as exc:
            logger.warning(f"Failed to parse last doc from Redis: {exc}")

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

    def _persist_last_doc(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            payload = self.last_docs.get((namespace, user_id), {})
            self.redis.setex(self._last_doc_key(namespace, user_id), self.ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Failed to persist last doc to Redis: {exc}")

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

    def get_last_user_message(self, namespace: str, user_id: str) -> Optional[str]:
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories or not self.histories[key]:
            return None
        return self.histories[key][-1].user_message

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

    def set_last_active_document(self, namespace: str, user_id: str, doc_id: Optional[str], doc_name: Optional[str]):
        key = (namespace, user_id)
        self.last_docs[key] = {
            "doc_id": str(doc_id or ""),
            "doc_name": str(doc_name or ""),
        }
        self._persist_last_doc(namespace, user_id)

    def get_last_active_document(self, namespace: str, user_id: str) -> Optional[Dict[str, str]]:
        key = (namespace, user_id)
        self._load_last_doc(namespace, user_id)
        if key not in self.last_docs:
            return None
        payload = self.last_docs.get(key) or {}
        if not payload:
            return None
        return payload

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
        if key in self.last_docs:
            self.last_docs[key].clear()
        if self.redis:
            try:
                self.redis.delete(self._history_key(namespace, user_id))
                self.redis.delete(self._docs_key(namespace, user_id))
                self.redis.delete(self._last_doc_key(namespace, user_id))
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


def _detect_sensitivity(query: str) -> str:
    q = (query or "").lower()
    if any(term in q for term in ["diagnosis", "patient", "medical", "treatment", "lab", "clinical", "hospital"]):
        return "HIGH"
    if any(term in q for term in ["legal", "compliance", "contract", "policy", "audit"]):
        return "MED"
    return "LOW"


def _detect_emotion_hint(query: str, asked_before: bool) -> str:
    q = (query or "")
    lowered = q.lower()
    if "??" in q or "!!" in q or any(term in lowered for term in ["urgent", "asap", "now"]):
        return "URGENT"
    if asked_before or any(term in lowered for term in ["again", "still", "repeat"]):
        return "FRUSTRATED"
    return "NEUTRAL"


class HumanizationComposer:
    """Deterministic microcopy selector with light session memory."""

    def __init__(self, max_history: int = 6):
        self.max_history = max_history
        self._history: Dict[Tuple[str, str], deque] = {}

    def _remember(self, key: Tuple[str, str], line: str) -> None:
        if not line:
            return
        if key not in self._history:
            self._history[key] = deque(maxlen=self.max_history)
        self._history[key].append(line)

    def _recent(self, key: Tuple[str, str]) -> Set[str]:
        if key not in self._history:
            return set()
        return set(self._history[key])

    def _pick(self, candidates: List[str], recent: Set[str], seed: int) -> str:
        if not candidates:
            return ""
        pool = [c for c in candidates if c not in recent] or candidates
        rng = random.Random(seed)
        return rng.choice(pool)

    def compose(
        self,
        *,
        namespace: str,
        user_id: str,
        intent: str,
        sensitivity: str,
        emotion: str,
        missingness: str,
        asked_before: bool,
        confidence: float,
        entity_name: Optional[str],
        seed_text: str,
    ) -> List[str]:
        key = (namespace, user_id)
        recent = self._recent(key)
        seed = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest()[:8], 16)

        ack_candidates: List[str] = []
        include_ack = (
            asked_before
            or emotion in {"FRUSTRATED", "URGENT"}
            or sensitivity == "HIGH"
            or missingness in {"PARTIAL", "FULL"}
        )
        if not include_ack:
            ack_line = ""
        elif asked_before or emotion == "FRUSTRATED":
            ack_candidates = [
                "I double-checked the document for that.",
                "I rechecked the record to be precise.",
            ]
        elif sensitivity == "HIGH":
            ack_candidates = [
                "Here's what the record notes.",
                "Here's what the report states.",
            ]
        elif entity_name:
            ack_candidates = [
                f"Here are the details for {entity_name}.",
                f"Here's what I found for {entity_name}.",
            ]
        else:
            ack_candidates = [
                "Here's what the document states.",
                "Here are the details from the document.",
                "This is what the document notes.",
            ]

        if include_ack and ack_candidates:
            ack_line = self._pick(ack_candidates, recent, seed)

        frame_line = ""
        if missingness in {"PARTIAL", "FULL"}:
            limitation_candidates = [
                "I don't see that detail stated in the provided pages.",
                "That specific detail isn't explicitly stated in the document.",
            ]
            frame_line = self._pick(limitation_candidates, recent, seed + 7)
        elif confidence < 0.5:
            frame_line = "The document provides partial details on this request."

        lines = [ln for ln in [ack_line, frame_line] if ln]
        for line in lines:
            self._remember(key, line)
        return lines


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
        self.source_file_cache: Dict[Tuple[str, str, str], Tuple[float, List[str]]] = {}
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
        subscription_id: Optional[str] = None,
    ) -> Filter:
        self._ensure_profile(profile_id)
        conditions = [
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        ]
        if subscription_id:
            conditions.append(FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))))
        if document_ids:
            conditions.append(FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids])))
        if source_files:
            conditions.append(FieldCondition(key="source_file", match=MatchAny(any=[str(s) for s in source_files])))
        return Filter(must=conditions)

    def _build_sparse_query(self, query: str) -> SparseVector:
        matrix = self._hash_vectorizer.transform([query])
        coo = matrix.tocoo()
        return SparseVector(indices=coo.col.tolist(), values=coo.data.astype(np.float32).tolist())

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
                kwargs["query_filter"] = query_filter

            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold

            results = self.client.query_points(**kwargs)
            return results
        except Exception as e:
            logger.error("Qdrant query_points error: %s", e, exc_info=True)
            return None

    def get_collection_vector_dim(self, collection_name: str) -> Optional[int]:
        """Fetch and cache the expected vector dimension for a collection."""
        if collection_name in self.collection_dims:
            return self.collection_dims[collection_name]
        try:
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
        except Exception as e:
            fallback_dim = getattr(Config.Model, "EMBEDDING_DIM", None) or 768
            logger.warning(
                "Could not fetch vector dim for collection '%s': %s; using fallback dim=%s",
                collection_name,
                e,
                fallback_dim,
            )
            self.collection_dims[collection_name] = int(fallback_dim)
            return int(fallback_dim)

    def retrieve(
            self,
            collection_name: str,
            query: str,
            filter_profile: str = None,
            subscription_id: Optional[str] = None,
            top_k: int = 50,
            score_threshold: float = 0.10  # ✅ CHANGED FROM 0.15
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
        except Exception as err:
            logger.error(f"Failed to embed query for retrieval: {err}", exc_info=True)
            return []

        logger.info(f"Searching collection '{collection_name}' for query: {query[:100]}")

        query_filter = self._build_filter(str(filter_profile), subscription_id=subscription_id)

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
            subscription_id: Optional[str] = None,
            top_k: int = 50,
            document_ids: Optional[List[str]] = None,
            source_files: Optional[List[str]] = None,
            score_threshold: float = 0.05,
    ) -> List[RetrievedChunk]:
        """Hybrid dense + sparse retrieval with reciprocal rank fusion."""
        self._ensure_profile(profile_id)
        query_filter = self._build_filter(
            profile_id,
            document_ids=document_ids,
            source_files=source_files,
            subscription_id=subscription_id,
        )

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
            return []
        sparse_vector = self._build_sparse_query(query)

        dense_results = self.run_search(
            collection_name=collection_name,
            query_vector=dense_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="content_vector",
            score_threshold=score_threshold,
        )
        sparse_results = self.run_search(
            collection_name=collection_name,
            query_vector=sparse_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="keywords_vector",
            score_threshold=None,
        )

        dense_chunks = self._points_to_chunks(dense_results, method="dense")
        sparse_chunks = self._points_to_chunks(sparse_results, method="sparse")

        if not dense_chunks and not sparse_chunks:
            return []

        merged = self._rrf_merge(dense_chunks, sparse_chunks)
        return merged[:top_k]

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

    def collect_document_chunks(
            self,
            collection_name: str,
            profile_id: str,
            subscription_id: Optional[str] = None,
            document_ids: Optional[List[str]] = None,
            max_chunks: int = 200,
            batch_size: int = 120,
    ) -> List[RetrievedChunk]:
        """Scroll all chunks for selected documents to improve aggregation recall."""
        self._ensure_profile(profile_id)
        if not document_ids or max_chunks <= 0:
            return []

        query_filter = self._build_filter(
            profile_id,
            document_ids=document_ids,
            subscription_id=subscription_id,
        )

        collected: List[RetrievedChunk] = []
        next_offset = None
        remaining = int(max_chunks)

        while remaining > 0:
            limit = min(batch_size, remaining)
            try:
                scroll = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    offset=next_offset,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.debug("Document scroll failed for %s: %s", document_ids, exc)
                break

            points = []
            if hasattr(scroll, "points"):
                points = scroll.points or []
                next_offset = getattr(scroll, "next_page_offset", None)
            elif isinstance(scroll, tuple):
                points = scroll[0] if len(scroll) > 0 else []
                next_offset = scroll[1] if len(scroll) > 1 else None
            else:
                points = []
                next_offset = None

            if not points:
                break

            for pt in points:
                payload = pt.payload or {}
                text = payload.get("text") or ""
                if not text:
                    continue
                source = payload.get("source_file") or payload.get("source") or "unknown"
                collected.append(
                    RetrievedChunk(
                        id=str(pt.id),
                        text=text,
                        score=0.01,
                        source=source,
                        metadata=payload,
                        method="doc_scroll",
                    )
                )
                remaining -= 1
                if remaining <= 0:
                    break

            if not next_offset:
                break

        return collected

    def list_source_files(
            self,
            collection_name: str,
            profile_id: str,
            subscription_id: Optional[str] = None,
            max_sources: int = 200,
            max_points: int = 1200,
            cache_ttl: int = 300,
    ) -> List[str]:
        """Return unique source_file values for the collection/profile."""
        self._ensure_profile(profile_id)
        key = (collection_name, str(profile_id), str(subscription_id or ""))
        now = time.time()
        cached = self.source_file_cache.get(key)
        if cached and (now - cached[0]) < cache_ttl:
            return list(cached[1])

        query_filter = self._build_filter(profile_id, subscription_id=subscription_id)
        unique: List[str] = []
        seen = set()
        next_offset = None
        remaining = int(max_points)

        while remaining > 0 and len(unique) < max_sources:
            limit = min(200, remaining)
            try:
                scroll = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    offset=next_offset,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.debug("Source file scroll failed: %s", exc)
                break

            points = []
            if hasattr(scroll, "points"):
                points = scroll.points or []
                next_offset = getattr(scroll, "next_page_offset", None)
            elif isinstance(scroll, tuple):
                points = scroll[0] if len(scroll) > 0 else []
                next_offset = scroll[1] if len(scroll) > 1 else None
            else:
                next_offset = None

            if not points:
                break

            for pt in points:
                payload = pt.payload or {}
                source = payload.get("source_file") or payload.get("source") or payload.get("file_name")
                if not source:
                    continue
                source = str(source)
                if source in seen:
                    continue
                seen.add(source)
                unique.append(source)
                if len(unique) >= max_sources:
                    break

            remaining -= limit
            if not next_offset:
                break

        unique = sorted(unique)
        self.source_file_cache[key] = (now, unique)
        return unique

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
            source_name = chunk.source or meta.get('source_file') or f"Document {i}"
            page = meta.get('page')
            section = meta.get('section_title') or meta.get('section')
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

        seen_keys: Set[Tuple[Any, ...]] = set()
        unique_chunks = []
        for chunk in chunks:
            meta = chunk.metadata or {}
            normalized = ' '.join((chunk.text or "").split())
            if not normalized.strip():
                continue
            chunk_id = meta.get("chunk_id")
            if chunk_id:
                key = ("chunk_id", str(chunk_id))
            else:
                doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("docId")
                page = meta.get("page") or meta.get("page_start") or meta.get("page_end")
                idx = meta.get("chunk_index")
                if doc_id is not None and page is not None and idx is not None:
                    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    key = ("loc_text", str(doc_id), str(page), str(idx), text_hash)
                else:
                    key = None
            if key is not None:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            unique_chunks.append(chunk)

        selected_chunks = unique_chunks[:max_chunks]

        context_parts = []
        source_map = ContextBuilder.build_source_hints(selected_chunks)
        if source_map:
            context_parts.append(source_map)
        for i, chunk in enumerate(selected_chunks, 1):
            # Use source_file directly from metadata as fallback
            source_name = chunk.source or chunk.metadata.get('source_file') or f"Document {i}"
            meta = chunk.metadata or {}
            section = meta.get('section_title') or meta.get('section')
            page = meta.get('page')
            header_parts = [f"SOURCE: {source_name}"]
            if section:
                header_parts.append(f"Section: {section}")
            if page is not None:
                header_parts.append(f"Page: {page}")
            header = " | ".join(header_parts)
            context_parts.append(
                f"[{header}]\n{chunk.text}\n[/SOURCE]"
            )

        return "\n".join(context_parts)

    @staticmethod
    def extract_sources(chunks: List[RetrievedChunk], max_sources: int = 3) -> List[Dict[str, Any]]:
        """Extract source information for response metadata."""
        sources = []
        for i, chunk in enumerate(chunks[:max_sources], 1):
            sources.append({
                'source_id': i,
                'source_name': chunk.source or f"Document {i}",
            })
        return sources


class DomainPromptAdapter:
    """Creates lightweight, in-context adapters to steer local LLM responses."""

    @staticmethod
    def build_adapter(profile_context: Dict[str, Any], query: str) -> str:
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
            profile_id: Optional[str] = None,
            subscription_id: Optional[str] = None,
            redis_client: Optional[Any] = None,
    ) -> str:
        """Build a structured QA prompt with grounding and citation requirements."""
        convo_block = f"\nPRIOR CONVERSATION SUMMARY:\n{conversation_summary}\n" if conversation_summary else ""
        domain_block = f"\nDOMAIN ADAPTER:\n{domain_guidance}\n" if domain_guidance else ""
        feedback_block = f"\nRECENT FEEDBACK:\n{feedback_memory}\n" if feedback_memory else ""
        retrieval_block = f"\nRETRIEVAL NOTES:\n{retrieval_brief}\n" if retrieval_brief else ""

        prompt = f"""Answer questions using the DOCUMENT CONTEXT below.

DOCUMENT CONTEXT:
{context}
{convo_block}
{domain_block}
{retrieval_block}
{feedback_block}

USER QUESTION: {query}

RESPONSE GUIDANCE:
- Base your answer on the document context above.
- Be clear and detailed; use short paragraphs and bullets when helpful.
- If the documents do not contain the answer, say what is missing and what to look for.
- End with a single citations line in this format:
  "Citations: file_name | section | page; file_name | section | page"
- Do not use [SOURCE-*] tags or inline citations.

Provide the answer now."""

        from src.prompting.prompt_builder import inject_persona_prompt

        return inject_persona_prompt(
            prompt,
            persona,
            profile_id=profile_id,
            subscription_id=subscription_id,
            redis_client=redis_client,
        )

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
            model = get_model()
            cross_encoder = get_cross_encoder()

            self.model = model
            self.client = qdrant_client
            self.retriever = QdrantRetriever(qdrant_client, model)
            self.adaptive_retriever = AdaptiveRetriever(qdrant_client, model)
            self.reranker = HybridReranker(alpha=0.7, cross_encoder=cross_encoder)
            self.context_builder = ContextBuilder()
            self.intelligent_context_builder = IntelligentContextBuilder(
                max_context_chunks=getattr(Config.Retrieval, "MAX_CONTEXT_CHUNKS", 7)
            )
            self.query_analyzer = ContextAwareQueryAnalyzer()
            self.evidence_plan_builder = EvidencePlanBuilder()
            self.working_context_assembler = WorkingContextAssembler()
            self.answer_renderer = AnswerRenderer()
            self.evidence_quality = EvidenceQualityScorer()
            self.prompt_builder = PromptBuilder()
            self.greeting_handler = GreetingHandler()
            self.query_reformulator = QueryReformulator(self.llm_client)
            self.answerability_detector = AnswerabilityDetector(self.llm_client)
            # Initialize Redis client for storing conversation history and feedback.
            redis_client = get_redis_client()
            self.redis_client = redis_client

            # Initialize conversation history backed by Redis. Avoid instantiating
            # a non-redis version first since it would immediately be overwritten.
            self.conversation_history = ConversationHistory(max_turns=3, redis_client=redis_client)
            self.conversation_summarizer = ConversationSummarizer(self.llm_client)
            self.feedback_memory = ChatFeedbackMemory(max_items=12, redis_client=redis_client)
            self.humanizer = HumanizationComposer(max_history=6)
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
            conv_context = self.conversation_history.get_context(namespace, user_id, max_turns=2)
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
        if any(word in q for word in ["how", "steps", "procedure", "configure", "setup"]):
            return "procedural"
        if any(word in q for word in ["error", "issue", "fail", "troubleshoot", "bug"]):
            return "troubleshooting"
        if any(word in q for word in ["difference", "compare", "vs", "versus", "comparison"]):
            return "comparison"
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

    @staticmethod
    def _apply_entity_scope(chunks: List[RetrievedChunk], entity_name: str) -> List[RetrievedChunk]:
        if not entity_name:
            return chunks
        entity_tokens = _entity_tokens(entity_name)
        if not entity_tokens:
            return chunks

        def _matches(value: str) -> bool:
            normalized = _normalize_entity_text(value)
            return all(token in normalized for token in entity_tokens)

        filtered: List[RetrievedChunk] = []
        for chunk in chunks:
            meta = chunk.metadata or {}
            meta_values = [
                meta.get("profile_name"),
                meta.get("person_name"),
                meta.get("document_name"),
                meta.get("file_name"),
                meta.get("filename"),
                meta.get("source_file"),
                meta.get("source"),
            ]
            meta_values = [str(v) for v in meta_values if v]
            if meta_values:
                if any(_matches(v) for v in meta_values):
                    filtered.append(chunk)
                continue
            # Fallback to text match with lower rank
            if _matches(chunk.text or ""):
                chunk.score = float(chunk.score) * 0.85
                filtered.append(chunk)
        return filtered

    def find_document_id_by_name(
            self,
            collection_name: str,
            person_name: str,
            profile_id: str,
            subscription_id: Optional[str] = None,
    ) -> str | None:
        try:
            #  Encode name
            query_vector = self.model.encode(
                person_name,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()

            #  Build filter (CORRECT way)
            must_conditions = [
                FieldCondition(
                    key="profile_id",
                    match=MatchValue(value=str(profile_id))
                )
            ]
            if subscription_id:
                must_conditions.append(
                    FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id)))
                )
            qdrant_filter = Filter(must=must_conditions)

            #  Call Qdrant
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="content_vector",  # ✅ REQUIRED
                query_filter=qdrant_filter,
                limit=10,
                with_payload=True,
                with_vectors=False
            )

            #  Validate response
            if not results or not results.points:
                return None

            #  Extract matching document_id
            for pt in results.points:
                payload = pt.payload or {}
                text = payload.get("text", "").lower()
                source = payload.get("source_file", "").lower()

                for part in person_name.lower().split():
                    if part in text or part in source:
                        return payload.get("document_id")

            return None

        except Exception as e:
            logger.error(f"Error finding document by name '{person_name}': {e}")
            return None

    def retrieve_with_priorities(
            self,
            query: str,
            user_id: str,
            profile_id: str,
            collection_name: str,
            namespace: str,
            subscription_id: Optional[str] = None,
            top_k_retrieval: int = 50,
            aggregation_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval with mandatory profile filters, query rewriting, and intent detection.
        """
        if not profile_id:
            raise ValueError("profile_id is required for retrieval")

        primary_query, primary_metadata = self.preprocess_query(query, user_id, namespace, use_reformulation=False)
        is_vague = self._is_query_vague(primary_query)
        attempt_records = []

        profile_context = self.retriever.get_profile_context(collection_name, profile_id)
        profile_context_data = {
            "keywords": profile_context.top_keywords[:12],
            "hints": profile_context.document_hints[:6],
            "sampled_chunks": profile_context.total_chunks
        }

        candidate_entities = extract_candidate_entities(query) or extract_candidate_entities(primary_query)
        person_name = (
            candidate_entities[0]
            if candidate_entities
            else self.extract_person_name_from_query(query) or self.extract_person_name_from_query(primary_query)
        )
        target_document_id = None

        if person_name:
            logger.info(f" Detected person-specific query for: '{person_name}'")
            target_document_id = self.find_document_id_by_name(
                collection_name,
                person_name,
                profile_id,
                subscription_id=subscription_id,
            )

            if target_document_id:
                logger.info(f" Will filter results to document: {target_document_id}")
            else:
                logger.warning(f" Could not find document for '{person_name}' - will search all docs")
        multi_doc_aggregation = aggregation_mode and (
            is_multi_doc_aggregation(primary_query) or is_multi_doc_aggregation(query)
        )
        if multi_doc_aggregation:
            target_document_id = None
        primary_metadata["vague_query"] = is_vague
        primary_metadata["profile_context"] = profile_context_data
        primary_metadata["person_name"] = person_name
        primary_metadata["candidate_entities"] = candidate_entities
        primary_metadata["target_document_id"] = target_document_id
        primary_metadata["aggregation_mode"] = aggregation_mode
        primary_metadata["multi_doc_aggregation"] = multi_doc_aggregation

        chunks = self.retriever.hybrid_retrieve(
            collection_name=collection_name,
            query=primary_query,
            profile_id=profile_id,
            subscription_id=subscription_id,
            top_k=top_k_retrieval,
            document_ids=[target_document_id] if target_document_id else None,
        )
        if person_name:
            chunks = self._apply_entity_scope(chunks, person_name)
        attempt_records.append(
            {
                "label": "hybrid",
                "query": primary_query,
                "hits": len(chunks),
                "top_score": round(float(chunks[0].score), 4) if chunks else 0.0,
                "document_filter": target_document_id,
            }
        )

        if not chunks and target_document_id:
            chunks = self.retriever.hybrid_retrieve(
                collection_name=collection_name,
                query=primary_query,
                profile_id=profile_id,
                subscription_id=subscription_id,
                top_k=top_k_retrieval,
                document_ids=None,
            )
            if person_name:
                chunks = self._apply_entity_scope(chunks, person_name)
            attempt_records.append(
                {
                    "label": "hybrid_no_doc_filter",
                    "query": primary_query,
                    "hits": len(chunks),
                    "top_score": round(float(chunks[0].score), 4) if chunks else 0.0,
                    "document_filter": None,
                }
            )

        return {
            "chunks": chunks,
            "query": primary_query,
            "metadata": primary_metadata,
            "attempts": attempt_records,
            "selected_strategy": "hybrid",
            "profile_context": profile_context_data,
        }

    def _resolve_available_sources(
            self,
            collection_name: str,
            profile_id: str,
            subscription_id: Optional[str] = None,
    ) -> List[str]:
        try:
            return self.retriever.list_source_files(
                collection_name=collection_name,
                profile_id=profile_id,
                subscription_id=subscription_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Source file listing failed: %s", exc)
            return []

    @staticmethod
    def _convert_adaptive_chunks(raw_chunks: List[Dict[str, Any]]) -> List[RetrievedChunk]:
        converted: List[RetrievedChunk] = []
        for item in raw_chunks or []:
            meta = item.get("metadata") or {}
            source = meta.get("source_file") or meta.get("source") or meta.get("file_name")
            converted.append(
                RetrievedChunk(
                    id=str(item.get("id") or ""),
                    text=item.get("text", ""),
                    score=float(item.get("score") or 0.0),
                    metadata=meta,
                    source=source,
                    method=item.get("method", "dense"),
                )
            )
        return converted

    @staticmethod
    def _doc_name_map(chunks: List[RetrievedChunk]) -> Dict[str, str]:
        counts: Dict[str, Dict[str, int]] = {}
        for chunk in chunks:
            meta = chunk.metadata or {}
            doc_id = str(meta.get("document_id") or "")
            if not doc_id:
                continue
            name = (
                meta.get("source_file")
                or meta.get("file_name")
                or meta.get("filename")
                or meta.get("source")
                or chunk.source
                or ""
            )
            if not name:
                continue
            counts.setdefault(doc_id, {})
            counts[doc_id][str(name)] = counts[doc_id].get(str(name), 0) + 1
        mapping = {}
        for doc_id, name_counts in counts.items():
            mapping[doc_id] = max(name_counts.items(), key=lambda kv: kv[1])[0]
        return mapping

    @staticmethod
    def _normalize_name(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
        return re.sub(r"\s+", " ", cleaned)

    def _select_doc_ids(
            self,
            doc_scores: Dict[str, float],
            doc_name_map: Dict[str, str],
            analysis: "AnalyzerOutput",
            last_active_document: Optional[Dict[str, str]],
            max_docs: int = 3,
    ) -> List[str]:
        if not doc_scores:
            return []

        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        if analysis.scope == "multi_doc" or analysis.intent == "compare":
            return [doc_id for doc_id, _ in ranked[:max_docs]]

        if analysis.target_hint:
            hints = [self._normalize_name(h) for h in analysis.target_hint]
            candidates = []
            for doc_id, score in ranked:
                doc_name = self._normalize_name(doc_name_map.get(doc_id, ""))
                if any(hint and hint in doc_name for hint in hints):
                    candidates.append((doc_id, score))
            if candidates:
                return [candidates[0][0]]
            if last_active_document:
                last_doc_name = self._normalize_name(last_active_document.get("doc_name") or "")
                last_doc_id = last_active_document.get("doc_id")
                if last_doc_name and any(hint and hint in last_doc_name for hint in hints):
                    if last_doc_id and last_doc_id in doc_scores:
                        return [last_doc_id]

        return [ranked[0][0]]

    def _select_final_chunks(
            self,
            chunks: List[RetrievedChunk],
            plan: "EvidencePlan",
            doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        if not doc_ids:
            doc_ids = []
        if plan.doc_selection_policy == "multi_doc_balanced" and doc_ids:
            grouped: Dict[str, List[RetrievedChunk]] = {doc_id: [] for doc_id in doc_ids}
            for chunk in chunks:
                meta = chunk.metadata or {}
                doc_id = str(meta.get("document_id") or "")
                if doc_id in grouped:
                    grouped[doc_id].append(chunk)
            for doc_id in grouped:
                grouped[doc_id] = sorted(grouped[doc_id], key=lambda c: float(c.score), reverse=True)
            selected: List[RetrievedChunk] = []
            added = True
            while added and len(selected) < plan.min_chunks:
                added = False
                for doc_id in doc_ids:
                    if grouped.get(doc_id):
                        selected.append(grouped[doc_id].pop(0))
                        added = True
                        if len(selected) >= plan.min_chunks:
                            break
            remaining = []
            for items in grouped.values():
                remaining.extend(items)
            remaining = sorted(remaining, key=lambda c: float(c.score), reverse=True)
            selected.extend(remaining)
            return selected[: max(plan.max_chunks, plan.min_chunks)]

        target = max(plan.min_chunks, min(plan.max_chunks, len(chunks)))
        return chunks[:target]

    def _retrieve_with_plan(
            self,
            *,
            query: str,
            user_id: str,
            profile_id: str,
            subscription_id: str,
            collection_name: str,
            namespace: str,
            analysis: "AnalyzerOutput",
            plan: "EvidencePlan",
            last_active_document: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        processed_query, preprocessing_metadata = self.preprocess_query(query, user_id, namespace, use_reformulation=False)
        attempts = []
        profile_context = self.retriever.get_profile_context(collection_name, profile_id)
        profile_context_data = {
            "keywords": profile_context.top_keywords[:12],
            "hints": profile_context.document_hints[:6],
            "sampled_chunks": profile_context.total_chunks,
        }

        source_files = analysis.target_hint or None
        doc_ids_filter = None
        if last_active_document and last_active_document.get("doc_id"):
            doc_ids_filter = [last_active_document.get("doc_id")]

        selected_strategy = "adaptive_dense" if plan.retrieval_mode == "coverage" else "hybrid_precision"
        initial_chunks: List[RetrievedChunk] = []
        reranked_chunks: List[RetrievedChunk] = []
        final_chunks: List[RetrievedChunk] = []
        selected_doc_ids: List[str] = []

        if plan.retrieval_mode == "coverage":
            top_k = max(50, int(plan.max_chunks * 4))
            retrieve_adaptive = getattr(self.adaptive_retriever, "retrieve_adaptive", None)
            if callable(retrieve_adaptive):
                raw_chunks = retrieve_adaptive(
                    collection_name=collection_name,
                    query=processed_query,
                    profile_id=profile_id,
                    top_k=top_k,
                    document_ids=doc_ids_filter,
                    source_files=source_files,
                    use_expansion=False,
                    use_keyword_boost=False,
                    subscription_id=subscription_id,
                )
                initial_chunks = self._convert_adaptive_chunks(raw_chunks)
                attempts.append(
                    {
                        "label": "adaptive_dense",
                        "query": processed_query,
                        "hits": len(initial_chunks),
                        "top_score": round(float(initial_chunks[0].score), 4) if initial_chunks else 0.0,
                        "document_filter": doc_ids_filter,
                    }
                )
            else:
                logger.warning("AdaptiveRetriever missing retrieve_adaptive; falling back to hybrid retrieval")
                initial_chunks = self.retriever.hybrid_retrieve(
                    collection_name=collection_name,
                    query=processed_query,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    top_k=top_k,
                    document_ids=doc_ids_filter,
                    source_files=source_files,
                )
                attempts.append(
                    {
                        "label": "hybrid_fallback",
                        "query": processed_query,
                        "hits": len(initial_chunks),
                        "top_score": round(float(initial_chunks[0].score), 4) if initial_chunks else 0.0,
                        "document_filter": doc_ids_filter or source_files,
                    }
                )
            if not initial_chunks and (doc_ids_filter or source_files):
                if callable(retrieve_adaptive):
                    raw_chunks = retrieve_adaptive(
                        collection_name=collection_name,
                        query=processed_query,
                        profile_id=profile_id,
                        top_k=top_k,
                        document_ids=None,
                        source_files=None,
                        use_expansion=False,
                        use_keyword_boost=False,
                        subscription_id=subscription_id,
                    )
                    initial_chunks = self._convert_adaptive_chunks(raw_chunks)
                    attempts.append(
                        {
                            "label": "adaptive_dense_no_filter",
                            "query": processed_query,
                            "hits": len(initial_chunks),
                            "top_score": round(float(initial_chunks[0].score), 4) if initial_chunks else 0.0,
                            "document_filter": None,
                        }
                    )
                else:
                    initial_chunks = self.retriever.hybrid_retrieve(
                        collection_name=collection_name,
                        query=processed_query,
                        profile_id=profile_id,
                        subscription_id=subscription_id,
                        top_k=top_k,
                        document_ids=None,
                        source_files=None,
                    )
                    attempts.append(
                        {
                            "label": "hybrid_fallback_no_filter",
                            "query": processed_query,
                            "hits": len(initial_chunks),
                            "top_score": round(float(initial_chunks[0].score), 4) if initial_chunks else 0.0,
                            "document_filter": None,
                        }
                    )

            doc_scores = self.evidence_quality.score_documents(initial_chunks, processed_query)
            doc_name_map = self._doc_name_map(initial_chunks)
            selected_doc_ids = self._select_doc_ids(
                doc_scores, doc_name_map, analysis, last_active_document, max_docs=3
            )

            selected_chunks = [
                chunk
                for chunk in initial_chunks
                if not selected_doc_ids or str((chunk.metadata or {}).get("document_id") or "") in selected_doc_ids
            ]
            extra_chunks = []
            if selected_doc_ids:
                extra_chunks = self.retriever.collect_document_chunks(
                    collection_name=collection_name,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    document_ids=selected_doc_ids,
                    max_chunks=max(plan.max_chunks * 3, 30),
                )
                if extra_chunks and selected_chunks:
                    section_keys = {
                        (c.metadata or {}).get("section_path")
                        or (c.metadata or {}).get("section_title")
                        or (c.metadata or {}).get("section")
                        for c in selected_chunks
                    }
                    section_keys = {s for s in section_keys if s}
                    if section_keys:
                        same_section = [
                            c
                            for c in extra_chunks
                            if ((c.metadata or {}).get("section_path")
                                or (c.metadata or {}).get("section_title")
                                or (c.metadata or {}).get("section")) in section_keys
                        ]
                        other_section = [c for c in extra_chunks if c not in same_section]
                        extra_chunks = same_section + other_section
            merged = _merge_retrieved_chunks(selected_chunks, extra_chunks)
            candidates = merged

            if plan.adjacent_expand and candidates:
                candidates = self.retriever.expand_with_neighbors(
                    collection_name=collection_name,
                    seed_chunks=candidates,
                    profile_id=profile_id,
                    window=1,
                    max_new=10,
                )
            reranked_chunks = candidates
            if plan.rerank_policy != "off":
                reranked_chunks = self.reranker.rerank(
                    chunks=candidates,
                    query=processed_query,
                    top_k=len(candidates),
                    use_cross_encoder=False,
                )
            final_chunks = self._select_final_chunks(reranked_chunks, plan, selected_doc_ids)

        else:
            top_k = max(20, int(plan.max_chunks * 3))
            initial_chunks = self.retriever.hybrid_retrieve(
                collection_name=collection_name,
                query=processed_query,
                profile_id=profile_id,
                subscription_id=subscription_id,
                top_k=top_k,
                document_ids=doc_ids_filter,
                source_files=source_files,
            )
            attempts.append(
                {
                    "label": "hybrid",
                    "query": processed_query,
                    "hits": len(initial_chunks),
                    "top_score": round(float(initial_chunks[0].score), 4) if initial_chunks else 0.0,
                    "document_filter": doc_ids_filter or source_files,
                }
            )
            if not initial_chunks and (doc_ids_filter or source_files):
                initial_chunks = self.retriever.hybrid_retrieve(
                    collection_name=collection_name,
                    query=processed_query,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    top_k=top_k,
                    document_ids=None,
                    source_files=None,
                )
                attempts.append(
                    {
                        "label": "hybrid_no_filter",
                        "query": processed_query,
                        "hits": len(initial_chunks),
                        "top_score": round(float(initial_chunks[0].score), 4) if initial_chunks else 0.0,
                        "document_filter": None,
                    }
                )

            reranked_chunks = initial_chunks
            if plan.rerank_policy != "off" and initial_chunks:
                reranked_chunks = self.reranker.rerank(
                    chunks=initial_chunks,
                    query=processed_query,
                    top_k=min(len(initial_chunks), plan.max_chunks),
                    use_cross_encoder=True,
                )
            if plan.adjacent_expand and reranked_chunks:
                reranked_chunks = self.retriever.expand_with_neighbors(
                    collection_name=collection_name,
                    seed_chunks=reranked_chunks,
                    profile_id=profile_id,
                    window=1,
                    max_new=6,
                )
            if len(reranked_chunks) < plan.min_chunks and initial_chunks:
                extra = [c for c in initial_chunks if c not in reranked_chunks]
                reranked_chunks = reranked_chunks + extra
            final_chunks = self._select_final_chunks(reranked_chunks, plan, doc_ids_filter or [])
            if doc_ids_filter:
                selected_doc_ids = list(doc_ids_filter)

        return {
            "query": processed_query,
            "metadata": preprocessing_metadata,
            "chunks": final_chunks,
            "retrieved_chunks": initial_chunks,
            "reranked_chunks": reranked_chunks,
            "attempts": attempts,
            "selected_strategy": selected_strategy,
            "profile_context": profile_context_data,
            "selected_doc_ids": selected_doc_ids,
        }

    def _retrieve_multi_doc_with_plan(
            self,
            *,
            query: str,
            user_id: str,
            profile_id: str,
            subscription_id: str,
            collection_name: str,
            namespace: str,
            decision: "IntentScopeDecision",
            doc_inventory: List["DocInventoryItem"],
    ) -> Dict[str, Any]:
        processed_query, preprocessing_metadata = self.preprocess_query(query, user_id, namespace, use_reformulation=False)
        attempts: List[Dict[str, Any]] = []
        profile_context = self.retriever.get_profile_context(collection_name, profile_id)
        profile_context_data = {
            "keywords": profile_context.top_keywords[:12],
            "hints": profile_context.document_hints[:6],
            "sampled_chunks": profile_context.total_chunks,
        }

        if isinstance(decision.target_docs, list) and decision.target_docs:
            selected_docs = list(decision.target_docs)
        else:
            selected_docs = list(doc_inventory)

        max_docs = max(1, int(decision.retrieval_plan.max_docs))
        if len(selected_docs) > max_docs:
            selected_docs = selected_docs[:max_docs]
            preprocessing_metadata["doc_limit_applied"] = True

        per_doc_top_k = max(5, int(decision.retrieval_plan.per_doc_top_k))

        def _retrieve(doc: "DocInventoryItem", top_k: int) -> List[RetrievedChunk]:
            doc_id = doc.doc_id or None
            source_file = doc.source_file or None
            chunks = self.retriever.hybrid_retrieve(
                collection_name=collection_name,
                query=processed_query,
                profile_id=profile_id,
                subscription_id=subscription_id,
                top_k=top_k,
                document_ids=[doc_id] if doc_id else None,
                source_files=[source_file] if source_file else None,
            )
            attempts.append(
                {
                    "label": "per_doc_hybrid",
                    "query": processed_query,
                    "hits": len(chunks),
                    "top_score": round(float(chunks[0].score), 4) if chunks else 0.0,
                    "document_filter": doc_id or source_file,
                }
            )
            return chunks

        def _fallback(doc: "DocInventoryItem", remaining: int) -> List[RetrievedChunk]:
            if not doc.doc_id or remaining <= 0:
                return []
            return self.retriever.collect_document_chunks(
                collection_name=collection_name,
                profile_id=profile_id,
                subscription_id=subscription_id,
                document_ids=[doc.doc_id],
                max_chunks=remaining,
            )

        all_chunks, chunks_by_doc, docs_used = retrieve_per_doc(
            doc_inventory=selected_docs,
            retrieve_fn=_retrieve,
            per_doc_top_k=per_doc_top_k,
            min_chunks_per_doc=5,
            fallback_fn=_fallback,
            max_docs=max_docs,
        )

        if decision.retrieval_plan.adjacent_expand and all_chunks:
            expanded_chunks = self.retriever.expand_with_neighbors(
                collection_name=collection_name,
                seed_chunks=all_chunks,
                profile_id=profile_id,
                window=1,
                max_new=10,
            )
            all_chunks = _merge_retrieved_chunks(all_chunks, expanded_chunks)

        if decision.scope == "multi_doc" and len(docs_used) < 2 and len(doc_inventory) > 1:
            remaining_docs = [doc for doc in doc_inventory if (doc.source_file or doc.doc_id) not in docs_used]
            broadened: List[RetrievedChunk] = []
            for doc in remaining_docs[:3]:
                broadened.extend(_fallback(doc, 3))
            if broadened:
                all_chunks = _merge_retrieved_chunks(all_chunks, broadened)
                preprocessing_metadata["broadened_multi_doc"] = True

        selected_doc_ids = [doc.doc_id for doc in selected_docs if doc.doc_id]
        return {
            "query": processed_query,
            "metadata": preprocessing_metadata,
            "chunks": all_chunks,
            "retrieved_chunks": all_chunks,
            "reranked_chunks": all_chunks,
            "attempts": attempts,
            "selected_strategy": "per_doc_hybrid",
            "profile_context": profile_context_data,
            "selected_doc_ids": selected_doc_ids,
            "chunks_by_doc": chunks_by_doc,
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
            top_k_retrieval: int = 80,
            top_k_rerank: int = 25,
            final_k: int = 12,
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
        use_intel_layer = True
        global _FIRST_METRICS_REQUEST
        is_cold_start = False
        if _FIRST_METRICS_REQUEST:
            is_cold_start = True
            _FIRST_METRICS_REQUEST = False

        try:
            if not profile_id:
                raise ValueError("profile_id is required for retrieval")
            if not subscription_id:
                raise ValueError("subscription_id is required for retrieval")
            collection_name = build_collection_name(subscription_id)
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

            if self.greeting_handler.is_positive_feedback(query):
                response_text = format_response_text(
                    "You're welcome! If you want me to dig into another document or topic, just let me know."
                )
                return {
                    "response": response_text,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "positive_feedback",
                    "grounded": True
                }

            if self.greeting_handler.is_greeting(query):
                greeting_response = (
                    f"Hi! I'm your {persona}. I can search your documents and answer questions. "
                    f"What would you like to explore?"
                )
                greeting_response = format_response_text(greeting_response)
                self.conversation_history.add_turn(namespace, user_id, query, greeting_response)

                return {
                    "response": greeting_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "greeting",
                    "grounded": True
                }

            if self.greeting_handler.is_farewell(query):
                farewell_response = format_response_text(
                    "Thanks for chatting. If you need anything else, come back anytime."
                )
                self.conversation_history.clear_history(namespace, user_id)

                return {
                    "response": farewell_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "farewell",
                    "grounded": True
                }

            logger.info(f"Processing query for collection '{collection_name}': {query[:100]}")

            conversation_context = self.conversation_history.get_context(namespace, user_id, max_turns=3)
            last_active_document = self.conversation_history.get_last_active_document(namespace, user_id)
            available_sources = self._resolve_available_sources(
                collection_name=collection_name,
                profile_id=profile_id,
                subscription_id=subscription_id,
            )

            doc_inventory = fetch_doc_inventory(
                qdrant_client=self.client,
                collection_name=collection_name,
                profile_id=profile_id,
                subscription_id=subscription_id,
                redis_client=self.redis_client,
                cache_ttl_seconds=900,
            )
            if not doc_inventory and available_sources:
                doc_inventory = [
                    DocInventoryItem(doc_id="", source_file=source, document_name=source, doc_type="")
                    for source in available_sources
                    if source
                ]

            intel_scope = None
            intel_domain_hint = None
            intel_known_docs = []
            try:
                from src.docwain_intel.doc_type_inferer import infer_doc_domain_from_prompt, infer_doc_type
                from src.docwain_intel.scope_resolver import DocMeta, resolve_scope

                for doc in doc_inventory:
                    hint_text = " ".join(
                        token for token in [doc.doc_type, doc.document_name, doc.source_file] if token
                    )
                    inferred = infer_doc_type(hint_text)
                    intel_known_docs.append(
                        DocMeta(
                            doc_id=doc.doc_id or "",
                            filename=doc.source_file or "",
                            title=doc.document_name or "",
                            doc_domain=inferred.get("doc_domain", "general"),
                            doc_kind=inferred.get("doc_kind", "general_doc"),
                        )
                    )

                intel_scope = resolve_scope(query, intel_known_docs)
                intel_domain_hint = infer_doc_domain_from_prompt(query)

                filtered = doc_inventory
                if intel_scope and intel_scope.matched_docs:
                    match_ids = {
                        doc.doc_id or doc.filename or doc.title
                        for doc in intel_scope.matched_docs
                    }
                    filtered = [
                        doc
                        for doc in doc_inventory
                        if (doc.doc_id in match_ids)
                        or (doc.source_file in match_ids)
                        or (doc.document_name in match_ids)
                    ]
                if intel_domain_hint and intel_domain_hint != "general" and intel_known_docs:
                    allowed = {
                        doc.doc_id or doc.filename or doc.title
                        for doc in intel_known_docs
                        if doc.doc_domain == intel_domain_hint
                    }
                    domain_filtered = [
                        doc
                        for doc in filtered
                        if (doc.doc_id in allowed)
                        or (doc.source_file in allowed)
                        or (doc.document_name in allowed)
                    ]
                    if domain_filtered:
                        filtered = domain_filtered
                if filtered:
                    doc_inventory = filtered
            except Exception as intel_exc:  # noqa: BLE001
                logger.debug("Intel scope resolution skipped: %s", intel_exc)

            orchestration_v3 = build_orchestration_state_v3(
                qdrant_client=self.client,
                collection_name=collection_name,
                profile_id=profile_id,
                subscription_id=subscription_id,
                redis_client=self.redis_client,
                doc_inventory=doc_inventory,
                query_text=query,
            )

            analysis = self.query_analyzer.analyze(
                query=query,
                conversation_history=conversation_context,
                available_sources=available_sources,
                last_active_document=last_active_document,
            )

            resume_clarification_bypass = should_bypass_clarification(query, doc_inventory)
            if orchestration_v3.intent.is_vague and not resume_clarification_bypass and not use_intel_layer:
                clarification_text = build_clarification_response(query, orchestration_v3.digest, orchestration_v3.intent)
                clarification_text = apply_formatting_rules(clarification_text)
                clarification_text = apply_citation_policy(
                    clarification_text,
                    [],
                    scope_type=orchestration_v3.scope.scope_type,
                )
                clarification_text = format_response_text(clarification_text)
                self.conversation_history.add_turn(namespace, user_id, query, clarification_text)
                return {
                    "response": clarification_text,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,
                    "query_type": "clarification",
                    "grounded": True,
                }

            session_context = {
                "conversation_context": conversation_context,
                "recent_doc_ids": self.conversation_history.get_recent_doc_ids(namespace, user_id),
                "last_active_document": last_active_document,
            }
            orchestration = orchestrate_intent_profile_entity(
                subscription_id=subscription_id,
                profile_id=profile_id,
                session_id=session_id,
                query_text=query,
                session_context=session_context,
                doc_inventory=doc_inventory,
                model_id=self.model_name,
            )
            decision = orchestration.decision
            scope_v3 = orchestration_v3.scope
            if scope_v3.scope_type != decision.scope or (
                isinstance(decision.target_docs, list) and scope_v3.target_docs and decision.target_docs != scope_v3.target_docs
            ):
                decision = replace(decision, scope=scope_v3.scope_type, target_docs=scope_v3.target_docs)
                orchestration = replace(
                    orchestration,
                    intent_type=orchestration_v3.intent.intent_type,
                    scope=scope_v3.scope_type,
                    target_docs=scope_v3.target_docs,
                    decision=decision,
                )
            if should_route_resume_analyzer(query_text=query, chunks=[], doc_inventory=doc_inventory):
                matched_docs = match_resume_docs_by_name(query, doc_inventory)
                if matched_docs:
                    decision = replace(decision, scope="single_doc", target_docs=matched_docs[:1])
                    orchestration = replace(
                        orchestration,
                        scope="single_doc",
                        target_docs=matched_docs[:1],
                        decision=decision,
                    )
            if is_multi_profile_request(query) and has_resume_docs(doc_inventory):
                resume_docs = select_resume_docs(doc_inventory=doc_inventory, chunks_by_doc=None)
                if resume_docs:
                    decision = replace(decision, scope="multi_doc", target_docs=resume_docs)
                    orchestration = replace(
                        orchestration,
                        scope="multi_doc",
                        target_docs=resume_docs,
                        decision=decision,
                    )
            table_requested = orchestration.output_requirements.wants_table or (
                orchestration_v3.intent.intent_type == "request_table"
            )

            if decision.scope in {"multi_doc", "targeted_docs"}:
                analysis.scope = "multi_doc"
                analysis.target_hint = []
                analysis.assumptions = []
                analysis.clarification_needed = False
                analysis.clarification_question = None
            elif isinstance(decision.target_docs, list) and decision.target_docs:
                analysis.scope = "single_doc_default"
                analysis.target_hint = [
                    doc.source_file or doc.document_name for doc in decision.target_docs if (doc.source_file or doc.document_name)
                ]
                if decision.assumption_line:
                    analysis.assumptions = [decision.assumption_line]
                analysis.clarification_needed = False
                analysis.clarification_question = None

            plan = self.evidence_plan_builder.build(analysis, query)

            if analysis.clarification_needed and not use_intel_layer:
                clarification_text = analysis.clarification_question or "Which document should I use?"
                clarification_text = format_response_text(clarification_text)
                self.conversation_history.add_turn(namespace, user_id, query, clarification_text)
                return {
                    "response": clarification_text,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,
                    "query_type": "clarification",
                    "grounded": True,
                }

            if analysis.clarification_needed:
                analysis.clarification_needed = False
                analysis.clarification_question = None

            if decision.intent in {"candidate_profile_extract", "multi_candidate_extract", "ranking"}:
                plan.retrieval_mode = "coverage"
                plan.adjacent_expand = True
                plan.rerank_policy = "order_only"
                plan.max_chunks = max(plan.max_chunks, decision.retrieval_plan.per_doc_top_k)
                plan.min_chunks = max(plan.min_chunks, 8)

            retrieval_start = time.time()
            forced_last_active_document = last_active_document
            if decision.scope == "single_doc" and isinstance(decision.target_docs, list) and decision.target_docs:
                target_doc = decision.target_docs[0]
                forced_last_active_document = {
                    "doc_id": target_doc.doc_id,
                    "doc_name": target_doc.source_file or target_doc.document_name,
                }
                analysis.scope = "single_doc_default"
                analysis.target_hint = [
                    target_doc.source_file or target_doc.document_name
                ]
                plan.adjacent_expand = True
            if decision.scope in {"multi_doc", "targeted_docs"}:
                retrieval_plan = self._retrieve_multi_doc_with_plan(
                    query=query,
                    user_id=user_id,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    collection_name=collection_name,
                    namespace=namespace,
                    decision=decision,
                    doc_inventory=doc_inventory,
                )
            else:
                retrieval_plan = self._retrieve_with_plan(
                    query=query,
                    user_id=user_id,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    collection_name=collection_name,
                    namespace=namespace,
                    analysis=analysis,
                    plan=plan,
                    last_active_document=forced_last_active_document,
                )

            if decision.scope == "multi_doc" and len(doc_inventory) > 1:
                chunks_by_doc = retrieval_plan.get("chunks_by_doc") or {}
                if len(chunks_by_doc) < 2 and not retrieval_plan.get("metadata", {}).get("broadened_retry"):
                    boosted_top_k = min(decision.retrieval_plan.per_doc_top_k + 6, 30)
                    if boosted_top_k > decision.retrieval_plan.per_doc_top_k:
                        boosted_plan = replace(decision.retrieval_plan, per_doc_top_k=boosted_top_k)
                        decision = replace(decision, retrieval_plan=boosted_plan)
                        retrieval_plan = self._retrieve_multi_doc_with_plan(
                            query=query,
                            user_id=user_id,
                            profile_id=profile_id,
                            subscription_id=subscription_id,
                            collection_name=collection_name,
                            namespace=namespace,
                            decision=decision,
                            doc_inventory=doc_inventory,
                        )
                        retrieval_plan.setdefault("metadata", {})["broadened_retry"] = True

            processed_query = retrieval_plan.get("query", query)
            preprocessing_metadata = retrieval_plan.get("metadata", {})
            retrieved_chunks = retrieval_plan.get("retrieved_chunks") or []
            reranked_chunks = retrieval_plan.get("reranked_chunks") or []
            final_chunks = retrieval_plan.get("chunks") or []
            retrieval_attempts = retrieval_plan.get("attempts", [])
            selected_strategy = retrieval_plan.get("selected_strategy", "adaptive_dense")
            profile_context_data = retrieval_plan.get("profile_context", {})
            selected_doc_ids = retrieval_plan.get("selected_doc_ids", []) or []

            if decision.scope == "single_doc" and isinstance(decision.target_docs, list) and decision.target_docs:
                target_doc = decision.target_docs[0]
                retrieved_chunks = enforce_single_doc_filter(retrieved_chunks, target_doc)
                reranked_chunks = enforce_single_doc_filter(reranked_chunks, target_doc)
                final_chunks = enforce_single_doc_filter(final_chunks, target_doc)

            final_chunks = remove_junk_sections(final_chunks, query)

            if decision.scope in {"multi_doc", "targeted_docs"}:
                dominant_doc_id = dominant_doc_for_lookup(orchestration_v3.intent, final_chunks)
                if dominant_doc_id:
                    dominant_doc = next(
                        (doc for doc in doc_inventory if doc.doc_id == dominant_doc_id),
                        None,
                    )
                    if dominant_doc:
                        retrieved_chunks = enforce_single_doc_filter(retrieved_chunks, dominant_doc)
                        reranked_chunks = enforce_single_doc_filter(reranked_chunks, dominant_doc)
                        final_chunks = enforce_single_doc_filter(final_chunks, dominant_doc)
                        decision = replace(decision, scope="targeted_docs", target_docs=[dominant_doc])
                        orchestration = replace(
                            orchestration,
                            scope=decision.scope,
                            target_docs=[dominant_doc],
                            decision=decision,
                        )

            retrieval_hits = 1.0 if retrieved_chunks else 0.0
            mrr_score = 0.0

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
                if query_type != "error":
                    counters["requests_success"] = 1
                if use_tooling:
                    counters["requests_with_tools"] = 1
                    if tool_successes == 0:
                        counters["llm_without_tool_fallback"] = 1

                distributions = {}
                if query_type in {"error", "no_results", "not_answerable"}:
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
                retrieval_latency_ms = (time.time() - retrieval_start) * 1000
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

            logger.info(f"Preprocessed query: {processed_query}")

            if not retrieved_chunks and not use_tooling and not use_intel_layer:
                no_results_response = (
                    "I couldn’t find this information in the selected documents. "
                    f"I searched the available documents for: '{query}'. "
                    "Please point me to the document or section that contains the answer."
                )
                no_results_response = format_response_text(no_results_response)
                self.conversation_history.add_turn(namespace, user_id, query, no_results_response)

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="no_results",
                        context_found=False,
                        grounded=False,
                        cached=False,
                        processing_time=time.time() - start_time,
                        retrieval_stats={"initial_retrieved": 0, "final_context": 0}
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (no_results) failed: %s", metric_exc)

                _record_request_metrics(
                    query_type="no_results",
                    answer_text=no_results_response,
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query,
                )

                return {
                    "response": no_results_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,
                    "query_type": "no_results",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": True,
                    "processing_time": time.time() - start_time
                }

            # Build working context and render answer.
            tool_successes = 0
            tool_failures = 0
            tool_chunks: List[RetrievedChunk] = []
            if use_tooling and tool_list:
                try:
                    import asyncio
                    from src.tools.base import registry
                except Exception as tool_import_exc:  # noqa: BLE001
                    logger.warning("Tool registry not available: %s", tool_import_exc)
                    tool_list = []
                for tool_name in tool_list:
                    payload = {
                        "input": {"query": query, "context": ""},
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
                            snippet = json.dumps(tool_result, default=str)[:800]
                            tool_chunks.append(
                                RetrievedChunk(
                                    id=f"tool-{tool_name}-{tool_successes}",
                                    text=snippet,
                                    score=0.01,
                                    metadata={
                                        "source_file": f"tool:{tool_name}",
                                        "document_id": f"tool:{tool_name}",
                                        "section_title": "Tool Output",
                                    },
                                    source=f"tool:{tool_name}",
                                )
                            )
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
                final_chunks = _merge_retrieved_chunks(final_chunks, tool_chunks)

            intent_label = classify_intent(query)
            normalized_query = normalize_query(processed_query or query)
            retrieval_fingerprint = compute_retrieval_fingerprint(final_chunks)
            normalized_query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()

            doc_name_map = self._doc_name_map(final_chunks)
            doc_names = sorted({name for name in doc_name_map.values() if name})
            if not doc_names:
                doc_names = sorted({
                    (chunk.metadata or {}).get("source_file")
                    or (chunk.metadata or {}).get("file_name")
                    or (chunk.metadata or {}).get("filename")
                    or (chunk.metadata or {}).get("source")
                    or chunk.source
                    for chunk in final_chunks
                    if chunk and ((chunk.metadata or {}).get("source_file") or chunk.source)
                })
            doc_id_set = sorted({
                (chunk.metadata or {}).get("document_id")
                or (chunk.metadata or {}).get("doc_id")
                or (chunk.metadata or {}).get("docId")
                for chunk in final_chunks
                if chunk and ((chunk.metadata or {}).get("document_id") or (chunk.metadata or {}).get("doc_id") or (chunk.metadata or {}).get("docId"))
            })
            doc_set = doc_id_set or doc_names

            structured_extraction = extract_structured_data(final_chunks)
            item_evidence = [
                (item.name, item.chunk_id)
                for item in (structured_extraction.items or [])
                if item.name
            ]
            computed_values = [
                str(val)
                for val in (structured_extraction.computed or {}).values()
                if val is not None
            ]

            use_cache = bool(final_chunks) and not disable_answer_cache and not force_refresh and not use_intel_layer
            qcache = QueryContextFingerprintCache.build(get_redis_client()) if use_cache else None
            cache_key = None
            cache_hit = False
            cached_response_obj = None
            cache_reason = "miss"

            retrieval_signature = compute_retrieval_signature(final_chunks)
            cache_guard_v2 = CacheGuardV2(ttl_seconds=600)
            cache_context = cache_guard_v2.build_context(
                subscription_id=subscription_id,
                profile_id=profile_id,
                query_text=processed_query or query,
                intent_type=orchestration_v3.intent.intent_type,
                scope_type=decision.scope,
                target_docs=decision.target_docs if isinstance(decision.target_docs, list) else [],
                entities=orchestration_v3.entities,
                corpus_fingerprint=orchestration_v3.digest.get("corpus_fingerprint") or "",
                model_id=self.model_name,
                retrieval_signature=retrieval_signature,
                is_vague=orchestration_v3.intent.is_vague,
            )

            if qcache:
                cache_key = cache_guard_v2.build_cache_key(cache_context)
                cached_payload = qcache.get(cache_key)
                cache_decision = cache_guard_v2.evaluate_cached_payload(
                    context=cache_context,
                    cache_key=cache_key,
                    cached_payload=cached_payload,
                )
                cache_reason = cache_decision.reason
                if cache_decision.hit and cached_payload:
                    candidate = cached_payload.get("response") or {}
                    if is_query_answer_consistent(normalized_query, candidate.get("response", ""), intent_label):
                        grounding = verify_grounding(
                            candidate.get("response", ""),
                            final_chunks,
                            intent=intent_label,
                            extracted_items=[name for name, _ in item_evidence],
                            computed_values=computed_values,
                            doc_names=doc_names,
                        )
                        if grounding.passed:
                            cache_hit = True
                            cached_response_obj = candidate
                        else:
                            cache_reason = "grounding_failed"
                            logger.warning("grounding_failed_first_pass cache_key_prefix=%s", cache_key[:8])
                            logger.warning("stale_cache_rejected cache_key_prefix=%s", cache_key[:8])
                    else:
                        cache_reason = "intent_mismatch"
                        logger.warning("stale_cache_rejected cache_key_prefix=%s", cache_key[:8])

            logger.info(
                "cache_hit=%s cache_reason=%s cache_key_hash_prefix=%s normalized_query_hash_prefix=%s retrieval_fingerprint_prefix=%s intent=%s docs_in_context=%s",
                cache_hit,
                cache_reason,
                cache_key[:8] if cache_key else "none",
                normalized_query_hash[:8] if normalized_query_hash else "none",
                retrieval_fingerprint[:8] if retrieval_fingerprint else "none",
                intent_label,
                f"count={len(doc_names)} top={doc_names[:3]}",
            )

            if cache_hit and cached_response_obj:
                response_obj = cached_response_obj
                self.conversation_history.add_turn(namespace, user_id, query, response_obj.get("response") or "")
                final_doc_ids = [
                    (chunk.metadata or {}).get("document_id") for chunk in final_chunks
                ]
                final_doc_ids = [d for d in final_doc_ids if d]
                if final_doc_ids:
                    self.conversation_history.add_sources(namespace, user_id, final_doc_ids)
                    doc_name_map = self._doc_name_map(final_chunks)
                    primary_doc_id = max(set(final_doc_ids), key=final_doc_ids.count)
                    self.conversation_history.set_last_active_document(
                        namespace,
                        user_id,
                        primary_doc_id,
                        doc_name_map.get(primary_doc_id),
                    )
                self.feedback_memory.add_feedback(namespace, user_id, query, response_obj.get("response") or "", response_obj.get("sources") or [])
                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type=response_obj.get("query_type", "document_qa"),
                        context_found=bool(response_obj.get("context_found", True)),
                        grounded=bool(response_obj.get("grounded", True)),
                        cached=True,
                        processing_time=time.time() - start_time,
                        retrieval_stats=response_obj.get("retrieval_stats") or {},
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (cache hit) failed: %s", metric_exc)
                return response_obj

            evidence_cap = max(15, len(final_chunks))
            evidence_set = build_evidence_set(final_chunks, max_chunks=evidence_cap)
            working_context = self.working_context_assembler.assemble(
                query=query,
                chunks=final_chunks,
                analysis=analysis,
            )

            multi_doc_scope = decision.scope in {"multi_doc", "targeted_docs"}
            only_one_doc = len(doc_names) == 1
            assumption_line = None

            if analysis.scope == "single_doc_default" and not analysis.target_hint and len(doc_names) > 1:
                doc_counts = Counter()
                for chunk in final_chunks:
                    doc_id = (chunk.metadata or {}).get("document_id")
                    if doc_id:
                        doc_counts[str(doc_id)] += 1
                best_doc_id = doc_counts.most_common(1)[0][0] if doc_counts else None
                assumed_doc = doc_name_map.get(best_doc_id) if best_doc_id else None
                assumed_doc = assumed_doc or (doc_names[0] if doc_names else None)
                if assumed_doc:
                    assumption_line = (
                        f"I used {assumed_doc} because it has the most relevant evidence in this session."
                    )
                    analysis.assumptions = [assumption_line]
                    logger.warning("fallback_scope_applied scope=single_doc_default")
            if not assumption_line and analysis.assumptions:
                assumption_line = analysis.assumptions[0]
            if not assumption_line and decision.assumption_line:
                assumption_line = decision.assumption_line
            if decision.scope == "multi_doc" and len(doc_names) == 1 and doc_names:
                if len(doc_inventory) <= 1:
                    assumption_line = assumption_line or f"Only one document is available ({doc_names[0]}), so I used it."
                else:
                    assumption_line = assumption_line or (
                        f"I only found matching sections in {doc_names[0]}; other documents did not surface matches."
                    )

            if use_intel_layer:
                from collections import Counter

                from src.docwain_intel.doc_type_inferer import infer_doc_domain_from_prompt, infer_doc_type
                from src.docwain_intel.fact_builder import build_fact_cache
                from src.docwain_intel.hr_renderers import render_generic, render_task
                from src.docwain_intel.intent_router import TASK_1, TASK_2, TASK_3, TASK_4, TASK_5, TASK_6, route_intent
                from src.docwain_intel.sanitizer import sanitize_output

                doc_text_map = {}
                for chunk in final_chunks:
                    meta = chunk.metadata or {}
                    doc_name = (
                        meta.get("source_file")
                        or meta.get("file_name")
                        or meta.get("filename")
                        or meta.get("source")
                        or chunk.source
                        or ""
                    )
                    if doc_name:
                        doc_text_map.setdefault(doc_name, []).append(chunk.text or "")

                doc_metadata = []
                for doc in doc_inventory:
                    doc_name = doc.source_file or doc.document_name or doc.doc_id or "Document"
                    text_hint = " ".join(doc_text_map.get(doc_name, [])) or " ".join(
                        token for token in [doc.doc_type, doc.document_name, doc.source_file] if token
                    )
                    inferred = infer_doc_type(text_hint)
                    doc_metadata.append(
                        {
                            "doc_id": doc.doc_id,
                            "doc_name": doc_name,
                            "doc_domain": inferred.get("doc_domain", "general"),
                            "doc_kind": inferred.get("doc_kind", "general_doc"),
                            "confidence": inferred.get("confidence", 0.1),
                        }
                    )

                if not doc_metadata and doc_text_map:
                    for doc_name, texts in doc_text_map.items():
                        inferred = infer_doc_type(" ".join(texts))
                        doc_metadata.append(
                            {
                                "doc_id": "",
                                "doc_name": doc_name,
                                "doc_domain": inferred.get("doc_domain", "general"),
                                "doc_kind": inferred.get("doc_kind", "general_doc"),
                                "confidence": inferred.get("confidence", 0.1),
                            }
                        )

                if doc_metadata:
                    for chunk in final_chunks:
                        meta = chunk.metadata or {}
                        doc_name = (
                            meta.get("source_file")
                            or meta.get("file_name")
                            or meta.get("filename")
                            or meta.get("source")
                            or chunk.source
                        )
                        for doc in doc_metadata:
                            if doc.get("doc_name") == doc_name:
                                meta["doc_domain"] = doc.get("doc_domain")
                                meta["doc_kind"] = doc.get("doc_kind")
                                chunk.metadata = meta
                                break

                domain_hint = infer_doc_domain_from_prompt(query)
                if doc_metadata:
                    domain_counts = Counter([doc.get("doc_domain") for doc in doc_metadata if doc.get("doc_domain")])
                    doc_domain = domain_counts.most_common(1)[0][0] if domain_counts else domain_hint
                else:
                    doc_domain = domain_hint

                fact_cache = build_fact_cache(final_chunks)
                task = route_intent(query)
                if task in {TASK_1, TASK_2, TASK_3, TASK_4, TASK_5, TASK_6}:
                    answer = render_task(task, fact_cache, doc_metadata)
                else:
                    answer = render_generic(doc_domain, fact_cache, doc_metadata)

                answer = format_response_text(answer)
                answer = sanitize_output(answer)

                self.conversation_history.add_turn(namespace, user_id, query, answer)
                final_doc_ids = [
                    (chunk.metadata or {}).get("document_id") for chunk in final_chunks
                ]
                final_doc_ids = [d for d in final_doc_ids if d]
                if final_doc_ids:
                    self.conversation_history.add_sources(namespace, user_id, final_doc_ids)
                    doc_name_map = self._doc_name_map(final_chunks)
                    primary_doc_id = max(set(final_doc_ids), key=final_doc_ids.count)
                    self.conversation_history.set_last_active_document(
                        namespace,
                        user_id,
                        primary_doc_id,
                        doc_name_map.get(primary_doc_id),
                    )
                self.feedback_memory.add_feedback(namespace, user_id, query, answer, [])

                processing_time = time.time() - start_time
                retrieval_brief = (
                    f"strategy={selected_strategy}; processed_query=\"{processed_query}\"; "
                    f"context_chunks={len(final_chunks)}; attempts={len(retrieval_attempts)}"
                )
                context_found_flag = bool(evidence_set.chunks)
                redacted_doc_ids = ["[redacted]"] * len(final_doc_ids)

                response_obj = {
                    "response": answer,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": context_found_flag,
                    "query_type": "document_qa",
                    "num_sources": 0,
                    "source_doc_ids": redacted_doc_ids,
                    "preprocessing": preprocessing_metadata,
                    "processed_query": processed_query,
                    "answerability": {
                        "is_answerable": bool(final_chunks),
                        "reason": "Context available from retrieved chunks" if final_chunks else "No validated context",
                    },
                    "grounded": bool(final_chunks),
                    "has_citations": False,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "processing_time": round(processing_time, 2),
                    "retrieval_notes": retrieval_brief,
                    "model_name": self.model_name,
                    "persona": persona,
                    "cache_version": ANSWER_CACHE_VERSION,
                    "context_fingerprint": "[redacted]",
                    "profile_context_fingerprint": "[redacted]",
                    "retrieval_stats": {
                        "initial_retrieved": len(retrieved_chunks),
                        "after_rerank": len(reranked_chunks),
                        "final_context": len(final_chunks),
                    },
                    "force_refresh": force_refresh,
                    "disable_answer_cache": disable_answer_cache,
                }

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type=response_obj.get("query_type", "document_qa"),
                        context_found=context_found_flag,
                        grounded=bool(response_obj.get("grounded", True)),
                        cached=False,
                        processing_time=processing_time,
                        retrieval_stats=response_obj.get("retrieval_stats") or {},
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (intel) failed: %s", metric_exc)

                _record_request_metrics(
                    query_type=response_obj.get("query_type", "document_qa"),
                    answer_text=answer,
                    context_text=working_context.brief_text(),
                    context_found=context_found_flag,
                    grounded=bool(response_obj.get("grounded", True)),
                    has_citations=False,
                    processing_seconds=processing_time,
                    prompt_text=processed_query,
                    tool_successes=tool_successes,
                    document_ids=final_doc_ids,
                )

                return response_obj

            candidate_profiles_payload = None
            resume_analysis_payload = None
            candidate_intents = {"candidate_profile_extract", "multi_candidate_extract", "ranking", "compare", "summarize"}
            candidate_mode = decision.candidate_mode and decision.intent in candidate_intents
            if candidate_mode:
                chunks_by_doc = retrieval_plan.get("chunks_by_doc") or {}
                if not chunks_by_doc:
                    chunks_by_doc = {}
                    for chunk in final_chunks:
                        doc_name = (
                            (chunk.metadata or {}).get("source_file")
                            or (chunk.metadata or {}).get("file_name")
                            or (chunk.metadata or {}).get("filename")
                            or (chunk.metadata or {}).get("source")
                            or chunk.source
                            or "Document"
                        )
                        chunks_by_doc.setdefault(doc_name, []).append(chunk)

                resume_route = should_route_resume_analyzer(
                    query_text=query,
                    chunks=final_chunks,
                    doc_inventory=doc_inventory,
                )
                used_chunk_ids = []
                if resume_route:
                    force_all_docs = is_multi_profile_request(query)
                    if force_all_docs:
                        resume_docs = select_resume_docs(doc_inventory=doc_inventory, chunks_by_doc=chunks_by_doc)
                        if resume_docs:
                            resume_names = {
                                (doc.source_file or doc.document_name or doc.doc_id or "").lower(): doc
                                for doc in resume_docs
                            }
                            filtered = {}
                            for name, doc_chunks in chunks_by_doc.items():
                                if name.lower() in resume_names:
                                    filtered[name] = doc_chunks
                            if filtered:
                                chunks_by_doc = filtered

                    resume_analysis = analyze_resume_chunks(
                        chunks_by_doc=chunks_by_doc,
                        doc_inventory=doc_inventory,
                        force_all_docs=force_all_docs,
                        query_text=query,
                    )
                    resume_analysis_payload = []
                    for profile in resume_analysis.profiles:
                        payload = dict(profile.__dict__)
                        payload["achievements_awards"] = profile.awards
                        resume_analysis_payload.append(payload)
                    candidate_profiles_payload = resume_analysis_payload
                    resume_views = [
                        ResumeProfileView(
                            candidate_name=profile.candidate_name,
                            source_type=profile.source_type,
                            source_document=profile.source_document,
                            total_years_experience=profile.total_years_experience,
                            experience_confidence=profile.experience_confidence,
                            experience_basis=profile.experience_basis,
                            experience_details=profile.experience_details,
                            experience_summary=profile.experience_summary,
                            technical_skills=profile.technical_skills,
                            functional_skills=profile.functional_skills,
                            certifications=profile.certifications,
                            education=profile.education,
                            awards=profile.awards,
                        )
                        for profile in resume_analysis.profiles
                    ]
                    ranking_lines = None
                    if decision.intent == "ranking" and resume_analysis.profiles:
                        ranked_candidates = rank_candidates(
                            [
                                CandidateProfile(
                                    candidate_name=profile.candidate_name,
                                    total_years_experience=float(profile.total_years_experience or 0),
                                    experience_summary=profile.experience_summary
                                    or profile.experience_details
                                    or "Summary based on retrieved resume sections.",
                                    technical_skills=profile.technical_skills,
                                    functional_skills=profile.functional_skills,
                                    certifications=profile.certifications,
                                    education=profile.education,
                                    achievements_awards=profile.awards,
                                    source_type="resume",
                                    source_document=profile.source_document,
                                    evidence_chunk_ids=[],
                                )
                                for profile in resume_analysis.profiles
                            ],
                            query,
                        )
                        ranking_lines = []
                        ordered_views = []
                        ordered_payload = []
                        payload_by_doc = {
                            (item.get("source_document") or ""): item for item in resume_analysis_payload
                        }
                        for idx, ranked in enumerate(ranked_candidates, start=1):
                            ranking_lines.append(
                                f"{idx}. {ranked.profile.candidate_name or 'Candidate'} — score {ranked.score:.2f}. {ranked.rationale}"
                            )
                            for view in resume_views:
                                if view.source_document == ranked.profile.source_document:
                                    ordered_views.append(view)
                                    break
                            payload = payload_by_doc.get(ranked.profile.source_document)
                            if payload:
                                ordered_payload.append(payload)
                        resume_views = ordered_views or resume_views
                        resume_analysis_payload = ordered_payload or resume_analysis_payload
                        candidate_profiles_payload = resume_analysis_payload
                        if len(resume_views) > 5 and re.search(r"\\btop\\s+5\\b|top five", query.lower()):
                            resume_views = resume_views[:5]
                            candidate_profiles_payload = candidate_profiles_payload[:5]

                    answer_text = format_resume_response(
                        profiles=resume_views,
                        assumption_line=assumption_line,
                        wants_table=False,
                        ranking_lines=ranking_lines,
                    )
                    used_chunk_ids = resume_analysis.used_chunk_ids
                else:
                    profiles = []
                    for doc_name, doc_chunks in chunks_by_doc.items():
                        profile = extract_candidate_profile(chunks=doc_chunks, source_document=doc_name)
                        profiles.append(profile)
                        used_chunk_ids.extend(profile.evidence_chunk_ids or [])

                    candidate_profiles_payload = [profile.__dict__ for profile in profiles]

                    if decision.intent == "ranking":
                        ranked = rank_candidates(profiles, query)
                        ranking_payload = [(rc.profile.__dict__, rc.score, rc.rationale) for rc in ranked]
                        answer_text = format_multi_candidate_response(
                            profiles=[rc.profile.__dict__ for rc in ranked],
                            assumption_line=assumption_line,
                            ranking=ranking_payload,
                        )
                    elif decision.scope == "single_doc" and profiles:
                        answer_text = format_candidate_profile_response(
                            profile=profiles[0].__dict__,
                            assumption_line=assumption_line,
                        )
                    else:
                        answer_text = format_multi_candidate_response(
                            profiles=[profile.__dict__ for profile in profiles],
                            assumption_line=assumption_line,
                            ranking=None,
                        )

                if not used_chunk_ids:
                    used_chunk_ids = [c.chunk_id for c in evidence_set.chunks[:3]]
                sources = _build_sources_from_chunk_ids(evidence_set, used_chunk_ids)
                citations_line = build_citations(sources)
                answer = replace_citations_line(answer_text, citations_line)
                answer = format_response_text(answer)
                render_result = AnswerRenderResult(text=answer, used_chunk_ids=used_chunk_ids)
            else:
                render_result = self.answer_renderer.render(
                    query=query,
                    analysis=analysis,
                    context=working_context,
                )

            if not candidate_mode:
                formatted = format_structured_response(
                    query=query,
                    intent=intent_label,
                    context=working_context,
                    doc_names=doc_names,
                    assumption_line=assumption_line,
                    item_evidence=item_evidence,
                    strict=False,
                    multi_doc=multi_doc_scope,
                    only_one_doc=only_one_doc,
                )
                used_chunk_ids = formatted.used_chunk_ids or render_result.used_chunk_ids or [
                    c.chunk_id for c in evidence_set.chunks[:3]
                ]
                sources = _build_sources_from_chunk_ids(evidence_set, used_chunk_ids)
                citations_line = build_citations(sources)
                answer = replace_citations_line(formatted.text, citations_line)
                answer = format_response_text(answer)

            grounding = verify_grounding(
                answer,
                final_chunks,
                intent=intent_label,
                extracted_items=[name for name, _ in item_evidence],
                computed_values=computed_values,
                doc_names=doc_names,
            )
            if not grounding.passed and candidate_mode:
                logger.warning("grounding_failed_first_pass candidate_mode")
            if not grounding.passed and not candidate_mode:
                logger.warning("grounding_failed_first_pass")
                strict_formatted = format_structured_response(
                    query=query,
                    intent=intent_label,
                    context=working_context,
                    doc_names=doc_names,
                    assumption_line=assumption_line,
                    item_evidence=item_evidence,
                    strict=True,
                    multi_doc=multi_doc_scope,
                    only_one_doc=only_one_doc,
                )
                used_chunk_ids = strict_formatted.used_chunk_ids or used_chunk_ids
                sources = _build_sources_from_chunk_ids(evidence_set, used_chunk_ids)
                citations_line = build_citations(sources)
                answer = replace_citations_line(strict_formatted.text, citations_line)
                answer = format_response_text(answer)
                grounding = verify_grounding(
                    answer,
                    final_chunks,
                    intent=intent_label,
                    extracted_items=[name for name, _ in item_evidence],
                    computed_values=computed_values,
                    doc_names=doc_names,
                )
                if not grounding.passed:
                    logger.warning("grounding_failed_second_pass")
                    conservative = format_conservative_response(
                        intent=intent_label,
                        doc_names=doc_names,
                        include_products_message=not item_evidence and intent_label == INTENT_PRODUCTS_SERVICES,
                    )
                    used_chunk_ids = used_chunk_ids or [c.chunk_id for c in evidence_set.chunks[:3]]
                    sources = _build_sources_from_chunk_ids(evidence_set, used_chunk_ids)
                    citations_line = build_citations(sources)
                    answer = replace_citations_line(conservative.text, citations_line)
                    answer = format_response_text(answer)

            if table_requested:
                table_columns: List[str] = []
                table_records: List[Dict[str, Any]] = []
                table_intro = assumption_line or "Here is the table based on the retrieved sections."

                if candidate_profiles_payload:
                    table_columns = [
                        "Name",
                        "Experience (Years)",
                        "Technical Skills",
                        "Functional Skills",
                        "Certifications",
                        "Education",
                        "Awards",
                        "Source",
                        "Document",
                    ]
                    for profile in candidate_profiles_payload:
                        years_value = profile.get("total_years_experience")
                        years_display = years_value if years_value is not None else ""
                        table_records.append(
                            {
                                "Name": profile.get("candidate_name") or "Candidate",
                                "Experience (Years)": years_display,
                                "Technical Skills": ", ".join(profile.get("technical_skills") or []),
                                "Functional Skills": ", ".join(profile.get("functional_skills") or []),
                                "Certifications": ", ".join(profile.get("certifications") or []),
                                "Education": ", ".join(profile.get("education") or []),
                                "Awards": ", ".join(profile.get("achievements_awards") or []),
                                "Source": profile.get("source_type") or "Resume",
                                "Document": profile.get("source_document") or "",
                            }
                        )
                elif structured_extraction.items:
                    table_columns = ["Product/Service", "Qty", "Unit Price", "Line Total", "Document"]
                    chunk_doc_map = {}
                    for chunk in final_chunks:
                        meta = chunk.metadata or {}
                        chunk_id = meta.get("chunk_id") or getattr(chunk, "id", None)
                        if not chunk_id:
                            continue
                        doc_name = (
                            meta.get("source_file")
                            or meta.get("file_name")
                            or meta.get("filename")
                            or meta.get("source")
                            or chunk.source
                        )
                        chunk_doc_map[str(chunk_id)] = doc_name or ""
                    for item in structured_extraction.items:
                        table_records.append(
                            {
                                "Product/Service": item.name,
                                "Qty": "",
                                "Unit Price": "",
                                "Line Total": item.raw_amount,
                                "Document": chunk_doc_map.get(item.chunk_id, ""),
                            }
                        )
                else:
                    table_columns = ["Document", "Key Points", "Entities", "Totals (if any)"]
                    facts_by_doc: Dict[str, List[str]] = {}
                    totals_by_doc: Dict[str, List[str]] = {}
                    for fact in working_context.key_facts:
                        facts_by_doc.setdefault(fact.doc_name, []).append(f"{fact.label}: {fact.value}")
                    for claim in working_context.numeric_claims:
                        label = (claim.label or "").lower()
                        if any(tok in label for tok in ("total", "subtotal", "amount due", "balance")):
                            totals_by_doc.setdefault(claim.doc_name, []).append(f"{claim.label}: {claim.value}")
                    entity_text = ", ".join(orchestration.entities.people + orchestration.entities.products)
                    for doc_name in doc_names or ["Document"]:
                        key_points = "; ".join((facts_by_doc.get(doc_name) or [])[:3])
                        totals = "; ".join((totals_by_doc.get(doc_name) or [])[:2])
                        table_records.append(
                            {
                                "Document": doc_name,
                                "Key Points": key_points or "No high-signal facts extracted.",
                                "Entities": entity_text,
                                "Totals (if any)": totals,
                            }
                        )

                if not table_columns:
                    table_columns = ["Document", "Note"]
                    for doc_name in doc_names or ["Document"]:
                        table_records.append(
                            {
                                "Document": doc_name,
                                "Note": "No table-ready fields were found in the retrieved sections.",
                            }
                        )

                table_markdown = render_markdown_table(table_columns, table_records)
                answer = enforce_response_formatting(
                    text=answer,
                    wants_table=True,
                    table_markdown=table_markdown,
                    intro=table_intro,
                )
                answer = replace_citations_line(answer, citations_line)
            else:
                answer = enforce_response_formatting(text=answer)

            answer = apply_formatting_rules(
                answer,
                wants_table=table_requested,
            )
            user_requested_evidence = bool(
                re.search(r"\b(show|provide)\s+(evidence|sources|supporting lines)\b", query or "", re.IGNORECASE)
            )
            answer = apply_citation_policy(
                answer,
                sources,
                scope_type=decision.scope,
                user_requested_evidence=user_requested_evidence,
            )
            answer = format_response_text(answer)

            is_answerable = bool(final_chunks)
            answerability_reason = "Context available from retrieved chunks" if is_answerable else "No validated context"

            retrieval_brief = (
                f"strategy={selected_strategy}; processed_query=\"{processed_query}\"; "
                f"context_chunks={len(final_chunks)}; attempts={len(retrieval_attempts)}"
            )

            self.conversation_history.add_turn(namespace, user_id, query, answer)
            final_doc_ids = [
                (chunk.metadata or {}).get("document_id") for chunk in final_chunks
            ]
            final_doc_ids = [d for d in final_doc_ids if d]
            if final_doc_ids:
                self.conversation_history.add_sources(namespace, user_id, final_doc_ids)
                doc_name_map = self._doc_name_map(final_chunks)
                primary_doc_id = max(set(final_doc_ids), key=final_doc_ids.count)
                self.conversation_history.set_last_active_document(
                    namespace,
                    user_id,
                    primary_doc_id,
                    doc_name_map.get(primary_doc_id),
                )
            self.feedback_memory.add_feedback(namespace, user_id, query, answer, sources)

            has_citations = bool(re.search(r'^\s*citations:\s*\S', answer, flags=re.IGNORECASE | re.MULTILINE))

            processing_time = time.time() - start_time
            context_found_flag = bool(evidence_set.chunks)
            redacted_doc_ids = ["[redacted]"] * len(final_doc_ids)

            # Construct the full response object up front so it can be cached
            response_obj = {
                "response": answer,
                "sources": sources,
                "user_id": user_id,
                "collection": collection_name,
                "request_id": request_id,
                "index_version": index_version,
                "context_found": context_found_flag,
                "query_type": "document_qa",
                "num_sources": len(sources),
                "source_doc_ids": redacted_doc_ids,
                "preprocessing": preprocessing_metadata,
                "processed_query": processed_query,
                "answerability": {
                    "is_answerable": is_answerable,
                    "reason": answerability_reason
                },
                "grounded": bool(sources),
                "has_citations": has_citations,
                "retrieval_attempts": retrieval_attempts,
                "selected_strategy": selected_strategy,
                "profile_context": profile_context_data,
                "processing_time": round(processing_time, 2),
                "retrieval_notes": retrieval_brief,
                "model_name": self.model_name,
                "persona": persona,
                "cache_version": ANSWER_CACHE_VERSION,
                "context_fingerprint": "[redacted]",
                "profile_context_fingerprint": "[redacted]",
                "retrieval_stats": {
                    "initial_retrieved": len(retrieved_chunks),
                    "after_rerank": len(reranked_chunks),
                    "final_context": len(final_chunks)
                },
                "force_refresh": force_refresh,
                "disable_answer_cache": disable_answer_cache,
            }

            try:
                metrics.record(
                    model_name=self.model_name,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    query_type=response_obj.get("query_type", "document_qa"),
                    context_found=context_found_flag,
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
                context_text=working_context.brief_text(),
                context_found=context_found_flag,
                grounded=bool(response_obj.get("grounded", True)),
                has_citations=has_citations,
                processing_seconds=processing_time,
                prompt_text=processed_query,
                tool_successes=tool_successes,
                document_ids=final_doc_ids,
            )

            # Persist the response in Redis before returning. Only cache successful
            # answers (document_qa) to improve subsequent response accuracy.
            if (
                qcache
                and cache_key
                and response_obj.get("query_type") == "document_qa"
                and cache_guard_v2.is_cacheable(cache_context)
            ):
                qcache.set(
                    cache_key=cache_key,
                    response_obj=response_obj,
                    normalized_query_hash=normalized_query_hash,
                    retrieval_fingerprint=retrieval_fingerprint,
                    metadata=cache_context.to_metadata(),
                )

            # Return the constructed response
            return response_obj

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            try:
                if telemetry:
                    telemetry.increment("retrieval_failures_count")
            except Exception:
                pass

            error_response = "Sorry, something went wrong on my side. Please try again, and let me know if it keeps happening."
            error_response = format_response_text(error_response)

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
                _record_request_metrics(
                    query_type="error",
                    answer_text=error_response,
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query if "processed_query" in locals() else query,
                )

            return {
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
            }


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
