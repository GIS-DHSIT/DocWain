"""Hybrid dense + sparse embedding module for DocWain.

Uses Ollama's HTTP API for dense embeddings (bge-m3) and computes
BM25-style TF-IDF sparse vectors locally with spaCy lemmatization.
"""

from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import math
import os
import re
import threading
from collections import Counter
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

# bge-m3 context window in tokens
_MAX_TOKENS = 8192

# Sparse: keep top N terms by weight
_SPARSE_TOP_K = 256

# Control character pattern (keeps printable + standard whitespace)
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class HybridVector(BaseModel):
    """Container for a hybrid dense + sparse embedding."""

    dense: List[float] = Field(default_factory=list, description="Dense embedding vector")
    sparse: Dict[str, float] = Field(default_factory=dict, description="Sparse TF-IDF term weights")
    text_hash: str = Field(default="", description="SHA-256 hex digest of the input text")
    model: str = Field(default="", description="Model name used for dense embedding")

# ---------------------------------------------------------------------------
# Sparse embedding helpers
# ---------------------------------------------------------------------------

class _SpaCyTokenizer:
    """Thread-safe lazy-loaded spaCy tokenizer for lemmatization."""

    _instance: Optional["_SpaCyTokenizer"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import spacy
        try:
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            logger.debug("spaCy model 'en_core_web_sm' not found; falling back to blank English model")
            self._nlp = spacy.blank("en")

    @classmethod
    def get(cls) -> "_SpaCyTokenizer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def lemmatize(self, text: str) -> List[str]:
        """Return lowercased lemmas, filtering stopwords and punctuation."""
        doc = self._nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 1
        ]

def _compute_idf(doc_freq: int, total_docs: int) -> float:
    """Smoothed inverse document frequency."""
    return math.log((1 + total_docs) / (1 + doc_freq)) + 1.0

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def _preprocess(text: str) -> str:
    """Strip control characters and truncate to ~MAX_TOKENS (rough 4-char estimate)."""
    text = _CTRL_CHAR_RE.sub("", text)
    # Rough token estimate: 1 token ~ 4 chars for English
    max_chars = _MAX_TOKENS * 4
    if len(text) > max_chars:
        text = text[:max_chars]
        logger.debug("Text truncated to %d chars (~%d tokens)", max_chars, _MAX_TOKENS)
    return text.strip()

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

# ---------------------------------------------------------------------------
# HybridEmbedder
# ---------------------------------------------------------------------------

class HybridEmbedder:
    """Produces hybrid dense + sparse embeddings.

    Dense: Ollama HTTP API with bge-m3
    Sparse: BM25-style TF-IDF with spaCy lemmatization
    """

    def __init__(
        self,
        ollama_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        request_timeout: float = 30.0,
    ) -> None:
        self._base_url = (ollama_base_url or _OLLAMA_BASE_URL).rstrip("/")
        self._model = embedding_model or _OLLAMA_EMBEDDING_MODEL
        self._timeout = request_timeout

        # IDF corpus stats (accumulated across batch calls)
        self._doc_count = 0
        self._doc_freq: Counter = Counter()
        self._idf_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Dense embedding
    # ------------------------------------------------------------------

    def embed_dense(self, text: str) -> List[float]:
        """Get dense embedding from Ollama. Returns zero vector on failure."""
        text = _preprocess(text)
        if not text:
            return []

        url = f"{self._base_url}/api/embeddings"
        payload = {"model": self._model, "prompt": text}

        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
            if not embedding:
                logger.debug("Ollama returned empty embedding for model=%s", self._model)
            return embedding
        except requests.ConnectionError:
            logger.warning(
                "Ollama unreachable at %s — returning empty dense vector", self._base_url
            )
            return []
        except requests.Timeout:
            logger.warning("Ollama embedding request timed out after %.1fs", self._timeout)
            return []
        except Exception:
            logger.exception("Unexpected error during dense embedding")
            return []

    # ------------------------------------------------------------------
    # Sparse embedding
    # ------------------------------------------------------------------

    def embed_sparse(self, text: str) -> Dict[str, float]:
        """Compute BM25-style TF-IDF sparse vector using spaCy lemmatization.

        Returns a dict mapping term -> weight, keeping the top SPARSE_TOP_K terms.
        """
        text = _preprocess(text)
        if not text:
            return {}

        tokenizer = _SpaCyTokenizer.get()
        lemmas = tokenizer.lemmatize(text)
        if not lemmas:
            return {}

        # Term frequency (normalized by document length)
        tf_counts = Counter(lemmas)
        doc_len = len(lemmas)
        tf = {term: count / doc_len for term, count in tf_counts.items()}

        # IDF: use accumulated corpus stats if available, else self-IDF
        with self._idf_lock:
            total_docs = max(self._doc_count, 1)
            idf_map = {}
            for term in tf:
                df = self._doc_freq.get(term, 0)
                idf_map[term] = _compute_idf(df, total_docs)

        # TF-IDF weights
        weights = {term: tf[term] * idf_map[term] for term in tf}

        # Keep top-K by weight
        if len(weights) > _SPARSE_TOP_K:
            sorted_terms = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            weights = dict(sorted_terms[:_SPARSE_TOP_K])

        return weights

    def _update_idf_corpus(self, lemma_sets: List[set]) -> None:
        """Update IDF corpus statistics from a batch of documents."""
        with self._idf_lock:
            for lemma_set in lemma_sets:
                self._doc_count += 1
                for term in lemma_set:
                    self._doc_freq[term] += 1

    # ------------------------------------------------------------------
    # Hybrid embedding
    # ------------------------------------------------------------------

    def embed_hybrid(self, text: str) -> HybridVector:
        """Produce a combined dense + sparse embedding."""
        clean = _preprocess(text)
        return HybridVector(
            dense=self.embed_dense(clean),
            sparse=self.embed_sparse(clean),
            text_hash=_text_hash(clean),
            model=self._model,
        )

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    def batch_embed(self, texts: List[str]) -> List[HybridVector]:
        """Embed multiple texts, updating IDF corpus stats across the batch.

        Steps:
        1. Pre-tokenize all texts to build corpus-level IDF.
        2. Compute dense + sparse for each text.
        """
        if not texts:
            return []

        logger.info("Batch embedding %d texts with model=%s", len(texts), self._model)

        # Pre-tokenize for IDF
        tokenizer = _SpaCyTokenizer.get()
        all_lemma_sets: List[set] = []
        for text in texts:
            clean = _preprocess(text)
            lemmas = tokenizer.lemmatize(clean)
            all_lemma_sets.append(set(lemmas))

        # Update IDF corpus
        self._update_idf_corpus(all_lemma_sets)

        results: List[HybridVector] = []
        for i, text in enumerate(texts):
            if (i + 1) % 50 == 0 or (i + 1) == len(texts):
                logger.info("Batch progress: %d / %d", i + 1, len(texts))
            results.append(self.embed_hybrid(text))

        logger.info("Batch embedding complete: %d vectors produced", len(results))
        return results
