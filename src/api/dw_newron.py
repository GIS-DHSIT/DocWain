import re
import json
import time
import threading
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque, Counter
from datetime import datetime, timedelta
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVector
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import os
import redis
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from src.api.config import Config
from src.api.enhanced_context_builder import IntelligentContextBuilder
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from sklearn.feature_extraction.text import HashingVectorizer
from src.api.vector_store import build_collection_name
from src.api.genai_client import generate_text, get_genai_client
from src.finetune import resolve_model_for_profile
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize models and clients (lazy loading to avoid startup errors)
_MODEL = None
_CROSS_ENCODER = None
_QDRANT_CLIENT = None
_REDIS_CLIENT = None
_MODEL_CACHE: Dict[int, SentenceTransformer] = {}
_METRICS_TRACKER = None
REDIS_MEMORY_TTL = int(os.getenv("REDIS_MEMORY_TTL", "86400"))
ANSWER_CACHE_TTL = int(os.getenv("RAG_ANSWER_CACHE_TTL", "1800"))
ANSWER_CACHE_VERSION = "v3"
NO_ANSWER_CACHE = os.getenv("NO_ANSWER_CACHE", "true").strip().lower() in {"true", "1", "yes", "on"}
ENABLE_ANSWER_CACHE = False if NO_ANSWER_CACHE else os.getenv("ENABLE_ANSWER_CACHE", "false").strip().lower() == "true"


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


def _coerce_bool(value: Optional[Any], default: Optional[bool] = None) -> bool:
    """Coerce truthy string/env values into a boolean with a sensible default."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


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
    for name in getattr(Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", []):
        if name and name not in candidates:
            candidates.append(name)
    if not candidates:
        candidates.append(getattr(Config.Model, "SENTENCE_TRANSFORMERS", "sentence-transformers/all-mpnet-base-v2"))

    last_error = None
    for name in candidates:
        try:
            logger.info(f"Loading sentence transformer model: {name}")
            model = SentenceTransformer(name)
            dim = model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model '{name}' with dim={dim}")
            if required_dim is None or dim == required_dim:
                return model
            _MODEL_CACHE[dim] = model  # cache for possible future use
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load model '{name}': {e}")
    raise RuntimeError(f"Could not load any sentence transformer model from {candidates}: {last_error}")


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

    def get_context(self, namespace: str, user_id: str, max_turns: int = 2) -> str:
        """Get recent conversation context as formatted string."""
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories or not self.histories[key]:
            return ""

        context_parts = []
        recent_turns = list(self.histories[key])[-max_turns:]

        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")

        return "\n".join(context_parts)

    def get_recent_doc_ids(self, namespace: str, user_id: str) -> List[str]:
        """Return a list of recently cited document IDs for this user."""
        key = (namespace, user_id)
        self._load_recent_docs(namespace, user_id)
        if key not in self.recent_docs:
            return []
        return list(self.recent_docs[key])

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
        'appreciate it', 'much appreciated', 'thx', 'ty', 'tysm'
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

                return text

            except Exception as e:
                logger.warning(f"Ollama attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error(f"All Ollama retries failed")
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
        for attempt in range(1, max_retries + 1):
            try:
                text, response = generate_text(
                    api_key=self.api_key,
                    model=self.model_name,
                    prompt=prompt,
                    generation_config=self.generation_config,
                )
                if text:
                    return text
                logger.warning(f"No text in response: {response}")
                return "I apologize, but I couldn't generate a proper response."
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error(f"All retry attempts failed: {e}")
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
                    return text
                logger.warning("No text in local LLM response: %s", response)
                return "I apologize, but I couldn't generate a proper response."
            except (HTTPError, URLError, ValueError) as e:
                logger.warning("Local LLM attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error("All local LLM retry attempts failed: %s", e)
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
        self, profile_id: str, document_ids: Optional[List[str]] = None, source_files: Optional[List[str]] = None
    ) -> Filter:
        self._ensure_profile(profile_id)
        conditions = [
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        ]
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

        query_filter = self._build_filter(str(filter_profile))

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
            score_threshold: float = 0.05,
    ) -> List[RetrievedChunk]:
        """Hybrid dense + sparse retrieval with reciprocal rank fusion."""
        self._ensure_profile(profile_id)
        query_filter = self._build_filter(profile_id, document_ids=document_ids, source_files=source_files)

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
            sources.append({
                'source_id': i,
                'source_name': chunk.source or f"Document {chunk.id[:8]}",
                'excerpt': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'relevance_score': round(float(chunk.score), 3)
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
            retrieval_brief: str = ""
    ) -> str:
        """Build a structured QA prompt with grounding and citation requirements."""
        convo_block = f"\nPRIOR CONVERSATION SUMMARY:\n{conversation_summary}\n" if conversation_summary else ""
        domain_block = f"\nDOMAIN ADAPTER:\n{domain_guidance}\n" if domain_guidance else ""
        feedback_block = f"\nRECENT FEEDBACK:\n{feedback_memory}\n" if feedback_memory else ""
        retrieval_block = f"\nRETRIEVAL NOTES:\n{retrieval_brief}\n" if retrieval_brief else ""

        prompt = f"""You are a {persona}. Deliver a clear, reader-friendly answer that feels human and intentional, not robotic. Start with a tight summary, then unpack the best evidence.

DOCUMENT CONTEXT:
{context}
{convo_block}
{domain_block}
{retrieval_block}
{feedback_block}

USER QUESTION: {query}

GROUNDING & CITATION RULES (MANDATORY):
1) Use ONLY the document context above; if something is missing, say so plainly.
2) EVERY factual claim must cite immediately with [SOURCE-X]; multiple sources -> [SOURCE-1, SOURCE-3].
3) Prefer quoting exact figures, names, dates, section titles, and short snippets with citations.
4) If sources disagree, briefly note the discrepancy with citations.
5) Do not invent, pad, or generalize beyond the provided text.

ANSWER STYLE (MANDATORY):
- Format: 1-2 sentence overview, then 3-6 concise bullet points with citations, then a one-line takeaway.
- Tone: warm, confident colleague; vary sentence openings; avoid filler and repetition.
- Bullets must be full sentences that stitch facts together (who/what/when/where/why/how) with inline citations.
- If the question is about an entity not present, say so and list what the documents do cover (with citations), then stop.

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
            model = get_model()
            cross_encoder = get_cross_encoder()

            self.model = model
            self.client = qdrant_client
            self.retriever = QdrantRetriever(qdrant_client, model)
            self.reranker = HybridReranker(alpha=0.7, cross_encoder=cross_encoder)
            self.context_builder = ContextBuilder()
            self.intelligent_context_builder = IntelligentContextBuilder(
                max_context_chunks=getattr(Config.Retrieval, "MAX_CONTEXT_CHUNKS", 7)
            )
            self.prompt_builder = PromptBuilder()
            self.greeting_handler = GreetingHandler()
            self.query_reformulator = QueryReformulator(self.llm_client)
            self.answerability_detector = AnswerabilityDetector(self.llm_client)
            # Initialize Redis client for storing conversation history and feedback.
            redis_client = get_redis_client()

            # Initialize conversation history backed by Redis. Avoid instantiating
            # a non-redis version first since it would immediately be overwritten.
            self.conversation_history = ConversationHistory(max_turns=3, redis_client=redis_client)
            self.conversation_summarizer = ConversationSummarizer(self.llm_client)
            self.feedback_memory = ChatFeedbackMemory(max_items=12, redis_client=redis_client)
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

    def find_document_id_by_name(
            self,
            collection_name: str,
            person_name: str,
            profile_id: str
    ) -> str | None:
        try:
            #  Encode name
            query_vector = self.model.encode(
                person_name,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()

            #  Build filter (CORRECT way)
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="profile_id",
                        match=MatchValue(value=str(profile_id))
                    )
                ]
            )

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
            top_k_retrieval: int = 50
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval with mandatory profile filters, query rewriting, and intent detection.
        """
        if not profile_id:
            raise ValueError("profile_id is required for retrieval")

        profile_context = self.retriever.get_profile_context(collection_name, profile_id)
        profile_context_data = {
            "keywords": profile_context.top_keywords[:12],
            "hints": profile_context.document_hints[:6],
            "sampled_chunks": profile_context.total_chunks
        }

        person_name = self.extract_person_name_from_query(query)
        target_document_id = None

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

        chunks = self.retriever.hybrid_retrieve(
            collection_name=collection_name,
            query=primary_query,
            profile_id=profile_id,
            top_k=top_k_retrieval,
            document_ids=[target_document_id] if target_document_id else None,
        )
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
                top_k=top_k_retrieval,
                document_ids=None,
            )
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
    ) -> Dict[str, Any]:
        """Main method to answer questions using enhanced RAG pipeline."""
        start_time = time.time()
        telemetry = telemetry_store() if METRICS_V2_ENABLED else None

        try:
            if not profile_id:
                raise ValueError("profile_id is required for retrieval")
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
                return {
                    "response": "You're welcome! If you want me to dig into another document or topic, just let me know.",
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
                farewell_response = "Thanks for chatting. If you need anything else, come back anytime."
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

            retrieval_start = time.time()
            retrieval_plan = self.retrieve_with_priorities(
                query=query,
                user_id=user_id,
                profile_id=profile_id,
                collection_name=collection_name,
                namespace=namespace,
                top_k_retrieval=top_k_retrieval
            )

            # Use the retrieval plan output directly to avoid diverging queries and
            # ensure reranking operates on the best available candidates.
            processed_query = retrieval_plan.get("query", query)
            preprocessing_metadata = retrieval_plan.get("metadata", {})
            retrieved_chunks = retrieval_plan.get("chunks") or []

            retrieval_attempts = retrieval_plan.get("attempts", [])
            selected_strategy = retrieval_plan.get("selected_strategy", "direct_qdrant")
            profile_context_data = retrieval_plan.get("profile_context", {})
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

            # Cache lookup (per subscription/profile/query + recent memory fingerprint)
            cache = get_redis_client() if (ENABLE_ANSWER_CACHE and not disable_answer_cache) else None
            cache_key = None
            conversation_context_for_cache = self.conversation_history.get_context(namespace, user_id, max_turns=2)
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

            if not retrieved_chunks:
                no_results_response = (
                    f"I couldn't find anything in your documents that answers: '{query}'. "
                    "Try rephrasing or tell me which document or section to focus on."
                )
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

            # Boost relevance for recently cited documents to honor session context
            recent_docs = set(self.conversation_history.get_recent_doc_ids(namespace, user_id))
            if recent_docs:
                for chunk in retrieved_chunks:
                    doc_id = (chunk.metadata or {}).get("document_id")
                    if doc_id and doc_id in recent_docs:
                        chunk.score = float(chunk.score) + 0.1

            reranked_chunks = self.reranker.rerank(
                chunks=retrieved_chunks,
                query=processed_query,
                top_k=top_k_rerank,
                use_cross_encoder=True
            )

            if getattr(Config.Retrieval, "USE_ADJACENT_EXPANSION", False):
                neighbor_window = getattr(Config.Retrieval, "NEIGHBOR_WINDOW", 2) or 2
                neighbor_max_new = getattr(Config.Retrieval, "NEIGHBOR_MAX_NEW", 10) or 10
                reranked_chunks = self.retriever.expand_with_neighbors(
                    collection_name=collection_name,
                    seed_chunks=reranked_chunks,
                    profile_id=profile_id,
                    window=int(neighbor_window),
                    max_new=int(neighbor_max_new)
                )
                reranked_chunks = sorted(reranked_chunks, key=lambda c: float(c.score), reverse=True)

            config_context_limit = getattr(Config.Retrieval, "MAX_CONTEXT_CHUNKS", 7)  # ✅ default 7
            context_chunk_limit = max(final_k or 7, config_context_limit)  # ✅ default 7
            context_chunk_limit = min(context_chunk_limit, len(reranked_chunks))
            final_chunks = reranked_chunks[:context_chunk_limit]

            context_sources: List[Dict[str, Any]] = []
            context = ""
            if final_chunks:
                try:
                    chunk_dicts = [
                        {
                            "text": chunk.text,
                            "score": float(chunk.score),
                            "metadata": chunk.metadata or {}
                        }
                        for chunk in final_chunks
                    ]
                    context, context_sources = self.intelligent_context_builder.build_context(
                        chunks=chunk_dicts,
                        query=processed_query,
                        include_metadata=True
                    )
                except Exception as ctx_exc:
                    logger.warning(f"Enhanced context builder failed; falling back: {ctx_exc}")
                    context = self.context_builder.build_context(
                        chunks=final_chunks,
                        max_chunks=context_chunk_limit
                    )
                    context_sources = self.context_builder.extract_sources(final_chunks)

            logger.info(f"Built context with {len(final_chunks)} chunks, {len(context)} chars")

            conversation_context = self.conversation_history.get_context(namespace, user_id, max_turns=3)
            conversation_summary = self.conversation_summarizer.summarize(conversation_context)
            adapter_text = DomainPromptAdapter.build_adapter(profile_context_data, query)
            feedback_text = self.feedback_memory.build_feedback_context(namespace, user_id, limit=5)

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

            # ✅ FIXED: Only block if we truly have NO context at all
            if not is_answerable and not final_chunks:
                not_answerable_response = (
                    f"I don't have enough detail in the documents to answer that. "
                    "If you can point me to a specific document or section, I can check again."
                )
                self.conversation_history.add_turn(namespace, user_id, query, not_answerable_response)

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="not_answerable",
                        context_found=False,  # ✅ Changed to False (no chunks)
                        grounded=False,
                        cached=False,
                        processing_time=time.time() - start_time,
                        retrieval_stats={
                            "initial_retrieved": len(retrieved_chunks),
                            "final_context": 0  # ✅ Changed to 0 (no final chunks)
                        }
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (not_answerable) failed: %s", metric_exc)

                return {
                    "response": not_answerable_response,
                    "sources": [],  # ✅ Changed to empty (no sources)
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,  # ✅ Changed to False
                    "query_type": "not_answerable",
                    "answerability_reason": answerability_reason,
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,  # ✅ Changed to False (couldn't answer)
                    "processing_time": time.time() - start_time
                }

            retrieval_brief = (
                f"strategy={selected_strategy}; processed_query=\"{processed_query}\"; "
                f"context_chunks={len(final_chunks)}; attempts={len(retrieval_attempts)}"
            )
            if context_sources:
                top_sources = ", ".join(str(src.get('source_name')) for src in context_sources[:3])
                retrieval_brief += f"; top_sources={top_sources}"
            if getattr(Config.Retrieval, "USE_ADJACENT_EXPANSION", False):
                retrieval_brief += "; adjacent_expansion=on"
                if conversation_summary:
                    retrieval_brief += "; convo_summary=on"

            prompt = self.prompt_builder.build_qa_prompt(
                query=query,
                context=context,
                persona=persona,
                conversation_summary=conversation_summary,
                domain_guidance=adapter_text,
                feedback_memory=feedback_text,
                retrieval_brief=retrieval_brief
            )

            answer = self.llm_client.generate(prompt)
            if answer and answer.strip().lower().startswith(query.strip().lower()):
                trimmed = answer[len(query):].lstrip(" :.-\n\t")
                if trimmed:
                    answer = trimmed

            sources = context_sources or self.context_builder.extract_sources(final_chunks)

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
                "grounded": True,
                "has_citations": has_citations,
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
                "force_refresh": force_refresh,
                "disable_answer_cache": disable_answer_cache,
            }

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

            # Persist the response in Redis before returning. Only cache successful
            # answers (document_qa) to improve subsequent response accuracy.
            if cache and cache_key:
                try:
                    cache.setex(cache_key, ANSWER_CACHE_TTL, json.dumps(response_obj))
                except Exception as cache_exc:
                    logger.warning(f"Failed to cache answer: {cache_exc}")

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
