"""Embedding-based conversational intent detector for non-document queries."""

from typing import Optional

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ConversationalDetector:
    """Detects conversational (non-document) queries using embedding similarity."""

    CONVERSATIONAL_PROTOTYPES = {
        "greeting": "hello, hi, hey, good morning, greetings, howdy",
        "farewell": "goodbye, bye, see you later, thanks for your help, good night",
        "gratitude": "thank you, thanks, appreciate it, great job, well done",
        "identity": "who are you, what are you, what can you do, tell me about yourself",
        "help": "how do I use this, help me, what commands, how does this work, guide me",
    }

    RESPONSES = {
        "greeting": "Hello! I'm ready to help you explore and understand your documents. What would you like to know?",
        "farewell": "Goodbye! Feel free to come back whenever you need help with your documents.",
        "gratitude": "You're welcome! I'm glad I could help. Let me know if there's anything else you need.",
        "identity": "I'm DocWain, an AI-powered document intelligence assistant. I can answer questions about your uploaded documents, extract information, summarize content, and much more.",
        "help": "You can ask me questions about your uploaded documents. Try things like 'summarize this document', 'what is the total amount?', or 'list all key dates'. I'll search through your documents to find the answer.",
    }

    THRESHOLD = 0.75

    def __init__(self, embedding_model):
        """Initialize with an embedding model that supports .encode(text_or_list) -> numpy array."""
        self._model = embedding_model
        self._prototype_names = list(self.CONVERSATIONAL_PROTOTYPES.keys())
        prototype_texts = list(self.CONVERSATIONAL_PROTOTYPES.values())
        self._prototype_embeddings = self._model.encode(prototype_texts)
        logger.info("ConversationalDetector initialized with %d prototypes", len(self._prototype_names))

    def detect(self, query: str) -> Optional[tuple[str, str]]:
        """Detect if a query is conversational rather than a document query.

        Args:
            query: The user's input query.

        Returns:
            Tuple of (intent_type, response_text) if conversational (score > 0.75).
            None if the query is a document query and should proceed to retrieval.
        """
        query_embedding = self._model.encode(query)
        best_type = None
        best_score = -1.0

        for i, proto_name in enumerate(self._prototype_names):
            score = _cosine_similarity(query_embedding, self._prototype_embeddings[i])
            if score > best_score:
                best_score = score
                best_type = proto_name

        if best_score > self.THRESHOLD:
            response = self.RESPONSES[best_type]
            logger.info(
                "Conversational intent detected: %s (score=%.3f) for query: %s",
                best_type, best_score, query[:80],
            )
            return best_type, response

        logger.debug("No conversational intent detected (best=%.3f) for query: %s", best_score, query[:80])
        return None
