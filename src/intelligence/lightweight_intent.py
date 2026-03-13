"""Embedding-based lightweight intent detection and field relevance scoring."""

import functools
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors using normalized dot product."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class LightweightIntentDetector:
    """Detects query intent via cosine similarity against intent description embeddings."""

    INTENT_DESCRIPTIONS = {
        "extract": "find specific information, get a value, what is, who is, retrieve details",
        "summarize": "give an overview, summarize, brief description, key points, high-level",
        "compare": "compare, contrast, differences, similarities between items or documents",
        "list": "list all, show all, enumerate, what are the items, display everything",
        "rank": "rank, order by, top, best, most, sort by criteria, highest to lowest",
        "analyze": "analyze, assess, evaluate, why, how does, explain reasoning, deep dive",
        "generate": "write, draft, create, compose, prepare a new document or content",
    }

    def __init__(self, embedding_model):
        """Initialize with an embedding model that supports .encode(text_or_list) -> numpy array."""
        self._model = embedding_model
        self._intent_names = list(self.INTENT_DESCRIPTIONS.keys())
        descriptions = list(self.INTENT_DESCRIPTIONS.values())
        self._intent_embeddings = self._model.encode(descriptions)
        logger.info("LightweightIntentDetector initialized with %d intents", len(self._intent_names))

    def detect(self, query: str) -> tuple[str, float]:
        """Detect the most likely intent for a query.

        Returns:
            Tuple of (intent_name, confidence_score).
        """
        query_embedding = self._model.encode(query)
        best_intent = self._intent_names[0]
        best_score = -1.0

        for i, intent_name in enumerate(self._intent_names):
            score = cosine_similarity(query_embedding, self._intent_embeddings[i])
            if score > best_score:
                best_score = score
                best_intent = intent_name

        logger.debug("Intent detected: %s (confidence=%.3f) for query: %s", best_intent, best_score, query[:80])
        return best_intent, best_score


class FieldRelevanceScorer:
    """Scores field relevance to a query using embedding cosine similarity."""

    def __init__(self, embedding_model):
        """Initialize with an embedding model that supports .encode(text_or_list) -> numpy array."""
        self._model = embedding_model
        self._field_cache: dict[str, np.ndarray] = {}

    @functools.lru_cache(maxsize=512)
    def _get_field_embedding(self, field_name: str):
        """Get cached embedding for a field name."""
        return self._model.encode(field_name)

    def score_fields(self, query: str, available_fields: list[str]) -> dict[str, float]:
        """Score each field's relevance to the query.

        Args:
            query: The user query.
            available_fields: List of field names to score.

        Returns:
            Dict mapping field name to cosine similarity score.
        """
        query_embedding = self._model.encode(query)
        scores = {}

        for field in available_fields:
            field_embedding = self._get_field_embedding(field)
            scores[field] = cosine_similarity(query_embedding, field_embedding)

        logger.debug("Scored %d fields for query: %s", len(scores), query[:80])
        return scores
