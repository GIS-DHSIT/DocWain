from __future__ import annotations

from src.utils.logging_utils import get_logger
import math
import re
from typing import Iterable, List, Sequence

from sklearn.feature_extraction.text import HashingVectorizer

logger = get_logger(__name__)

class LocalCrossEncoder:
    """Lightweight cross-encoder scoring without HuggingFace dependencies."""

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "from", "by", "about", "as", "is", "are", "was", "were", "be", "been", "being",
        "that", "this", "these", "those", "it", "its", "their", "them", "they", "you",
        "your", "we", "our", "us", "i", "me", "my", "he", "she", "his", "her", "not",
        "do", "does", "did", "done", "have", "has", "had", "will", "would", "can", "could",
        "should", "may", "might", "must", "if", "then", "so", "than", "such", "also",
    }

    def __init__(self, *, n_features: int = 4096) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._cached_query = None
        self._cached_q_vec = None
        self._cached_q_tokens: List[str] = []
        self._cached_q_numbers: List[str] = []

    def predict(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        scores: List[float] = []
        for pair in pairs:
            if not pair or len(pair) < 2:
                scores.append(0.0)
                continue
            query, doc = str(pair[0] or ""), str(pair[1] or "")
            scores.append(self._score_pair(query, doc))
        return scores

    def _score_pair(self, query: str, doc: str) -> float:
        if not query or not doc:
            return 0.0

        q_vec, q_tokens, q_numbers = self._prepare_query(query)
        d_text = self._normalize(doc)
        d_tokens = self._tokens_from_text(doc)
        d_numbers = self._extract_numbers(doc)

        # Cosine similarity over hashed TF-IDF-ish features.
        d_vec = self.vectorizer.transform([doc])
        cosine = self._safe_dot(q_vec, d_vec)

        # Token overlap (Jaccard).
        q_set = set(q_tokens)
        d_set = set(d_tokens)
        overlap = len(q_set & d_set) / max(len(q_set), 1)

        # Phrase match boost for longer queries.
        phrase_boost = 0.0
        if len(query.split()) >= 3:
            if self._normalize(query) in d_text:
                phrase_boost = 1.0
            else:
                phrase_boost = 0.3 if self._has_bigram_overlap(q_tokens, d_tokens) else 0.0

        # Numeric alignment boost.
        numeric_boost = 0.0
        if q_numbers and d_numbers:
            matches = len(set(q_numbers) & set(d_numbers))
            numeric_boost = min(1.0, matches / max(len(set(q_numbers)), 1))

        # Dynamic weights: emphasize numeric alignment for numeric queries.
        if q_numbers:
            weights = (0.55, 0.2, 0.15, 0.1)
        else:
            weights = (0.6, 0.25, 0.1, 0.05)

        score = (
            weights[0] * cosine
            + weights[1] * overlap
            + weights[2] * phrase_boost
            + weights[3] * numeric_boost
        )
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return float(score)

    def _prepare_query(self, query: str):
        if query == self._cached_query and self._cached_q_vec is not None:
            return self._cached_q_vec, self._cached_q_tokens, self._cached_q_numbers

        q_vec = self.vectorizer.transform([query])
        q_tokens = self._tokens_from_text(query)
        q_numbers = self._extract_numbers(query)

        self._cached_query = query
        self._cached_q_vec = q_vec
        self._cached_q_tokens = q_tokens
        self._cached_q_numbers = q_numbers
        return q_vec, q_tokens, q_numbers

    @staticmethod
    def _safe_dot(q_vec, d_vec) -> float:
        if q_vec is None or d_vec is None:
            return 0.0
        try:
            prod = q_vec @ d_vec.T
            if hasattr(prod, "toarray"):
                return float(prod.toarray()[0][0])
            if hasattr(prod, "A"):
                return float(prod.A[0][0])
            return float(prod[0][0])
        except Exception:
            return 0.0

    @classmethod
    def _tokens_from_text(cls, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        tokens = [t for t in tokens if t not in cls._STOPWORDS and len(t) > 1]
        return cls._simple_lemmatize(tokens)

    @staticmethod
    def _simple_lemmatize(tokens: Iterable[str]) -> List[str]:
        lemmatized: List[str] = []
        for token in tokens:
            if token.endswith("ing") and len(token) > 4:
                lemmatized.append(token[:-3])
            elif token.endswith("ed") and len(token) > 3:
                lemmatized.append(token[:-2])
            elif token.endswith("s") and len(token) > 3:
                lemmatized.append(token[:-1])
            else:
                lemmatized.append(token)
        return lemmatized

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").lower()).strip()
        return cleaned

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return re.findall(r"\b\d+(?:\.\d+)?\b", text or "")

    @staticmethod
    def _has_bigram_overlap(q_tokens: Sequence[str], d_tokens: Sequence[str]) -> bool:
        if len(q_tokens) < 2 or len(d_tokens) < 2:
            return False
        q_bigrams = {" ".join(q_tokens[i:i + 2]) for i in range(len(q_tokens) - 1)}
        d_bigrams = {" ".join(d_tokens[i:i + 2]) for i in range(len(d_tokens) - 1)}
        return bool(q_bigrams & d_bigrams)
