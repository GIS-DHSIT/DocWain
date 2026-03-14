"""Learned field importance classifier for query→field_tag prediction.

A lightweight NumPy-only MLP (1024→64→12 sigmoid) trained self-supervised
on synthetic query templates derived from ``_FIELD_FOCUS_MAP``.  Predicts
which schema fields (skills, education, experience, …) are relevant to a
given query embedding, replacing brittle keyword matching with a learned
representation.

Usage::

    from src.rag_v3.field_classifier import ensure_field_classifier, get_field_classifier

    clf = ensure_field_classifier(embedder)   # train or load once
    probs = clf.predict(query_embedding)       # {"skills": 0.92, "education": 0.04, ...}
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Field taxonomy (12 fields, ordered)
# ---------------------------------------------------------------------------
FIELD_NAMES: List[str] = [
    "skills",
    "education",
    "certifications",
    "experience",
    "summary",
    "contact",
    "achievements",
    "totals",
    "items",
    "parties",
    "clauses",
    "terms",
]

_FIELD_INDEX = {name: i for i, name in enumerate(FIELD_NAMES)}

# ---------------------------------------------------------------------------
# Self-supervised training templates (40+ covering all 12 fields)
# ---------------------------------------------------------------------------
TRAINING_TEMPLATES: List[Tuple[str, Set[str]]] = [
    # Skills
    ("What are the technical skills?", {"skills"}),
    ("List all programming languages and frameworks", {"skills"}),
    ("What technologies does this person know?", {"skills"}),
    ("Which tools and software are mentioned?", {"skills"}),
    ("Tech stack and competencies", {"skills"}),
    # Education
    ("Education background and degrees", {"education"}),
    ("Which university did they attend?", {"education"}),
    ("What is their highest qualification?", {"education"}),
    ("Academic history and GPA", {"education"}),
    # Certifications
    ("What certifications do they have?", {"certifications"}),
    ("Professional certifications and licenses", {"certifications"}),
    ("Is this person AWS certified?", {"certifications"}),
    ("List credentials and certifications", {"certifications"}),
    # Experience
    ("Work experience and employment history", {"experience"}),
    ("How many years of experience?", {"experience"}),
    ("Previous job roles and responsibilities", {"experience"}),
    ("Career history and work timeline", {"experience"}),
    ("What companies have they worked at?", {"experience"}),
    # Summary
    ("Give me a profile summary", {"summary"}),
    ("Professional overview and objective", {"summary"}),
    ("Summarize this candidate's background", {"summary", "experience"}),
    # Contact
    ("Contact information", {"contact"}),
    ("What is their email address?", {"contact"}),
    ("Phone number and LinkedIn profile", {"contact"}),
    ("How can I reach this person?", {"contact"}),
    # Achievements
    ("Awards and accomplishments", {"achievements"}),
    ("Notable achievements in their career", {"achievements"}),
    ("Has this person won any awards?", {"achievements"}),
    # Totals (invoice)
    ("What is the total invoice amount?", {"totals"}),
    ("Show me the balance due", {"totals"}),
    ("Grand total and subtotal", {"totals"}),
    ("How much is owed?", {"totals"}),
    # Items (invoice)
    ("List all line items and products", {"items"}),
    ("What items are on this invoice?", {"items"}),
    ("Product descriptions and quantities", {"items"}),
    # Parties (invoice)
    ("Who is the vendor?", {"parties"}),
    ("Customer and billing information", {"parties"}),
    ("Buyer and seller details", {"parties"}),
    # Clauses (legal)
    ("What are the contract clauses?", {"clauses"}),
    ("Liability and indemnification terms", {"clauses"}),
    ("Key legal provisions", {"clauses"}),
    # Terms (legal)
    ("Payment terms and conditions", {"terms"}),
    ("Contract duration and renewal terms", {"terms"}),
    # Multi-field
    ("Skills and education of this candidate", {"skills", "education"}),
    ("Experience and certifications", {"experience", "certifications"}),
    ("Full profile with skills and experience", {"skills", "experience", "summary"}),
    ("Invoice totals and line items", {"totals", "items"}),
]


# ---------------------------------------------------------------------------
# FieldImportanceClassifier — NumPy-only MLP
# ---------------------------------------------------------------------------

class FieldImportanceClassifier:
    """Two-layer MLP: Linear(1024→64) → ReLU → Linear(64→12) → Sigmoid.

    Trained with binary cross-entropy loss via mini-batch SGD.
    No PyTorch dependency — pure NumPy.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 64,
        n_fields: int = 12,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_fields = n_fields
        self.field_names = FIELD_NAMES[:n_fields]

        # He initialization for ReLU
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_fields) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_fields)

        self._trained = False

    # --- Forward pass ---

    def _forward(self, X: np.ndarray) -> tuple:
        """Forward pass returning (hidden, logits, probs)."""
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        probs = self._sigmoid(logits)
        return h, logits, probs

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    # --- Training ---

    def train(
        self,
        embedder: Any,
        epochs: int = 200,
        lr: float = 0.01,
        templates: Optional[List[Tuple[str, Set[str]]]] = None,
    ) -> List[float]:
        """Train on synthetic templates. Returns per-epoch losses."""
        templates = templates or TRAINING_TEMPLATES
        if not templates:
            log.warning("No training templates provided")
            return []

        # Encode all template queries
        queries = [t[0] for t in templates]
        try:
            X = embedder.encode(queries, normalize_embeddings=True)
            if not isinstance(X, np.ndarray):
                X = np.array(X)
        except Exception:
            log.error("Failed to encode training templates", exc_info=True)
            return []

        # Build label matrix
        Y = np.zeros((len(templates), self.n_fields), dtype=np.float32)
        for i, (_, fields) in enumerate(templates):
            for f in fields:
                if f in _FIELD_INDEX:
                    Y[i, _FIELD_INDEX[f]] = 1.0

        # Update input_dim if embedder produces different size
        if X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)

        losses: List[float] = []
        n = len(X)

        for epoch in range(epochs):
            # Forward
            h, logits, probs = self._forward(X)

            # BCE loss
            eps = 1e-7
            loss = -np.mean(
                Y * np.log(probs + eps) + (1 - Y) * np.log(1 - probs + eps)
            )
            losses.append(float(loss))

            # Backward
            dlogits = (probs - Y) / n  # (n, n_fields)
            dW2 = h.T @ dlogits  # (hidden, n_fields)
            db2 = np.sum(dlogits, axis=0)

            dh = dlogits @ self.W2.T  # (n, hidden)
            dh[h <= 0] = 0  # ReLU gradient

            dW1 = X.T @ dh  # (input, hidden)
            db1 = np.sum(dh, axis=0)

            # SGD update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        self._trained = True
        log.info(
            "Field classifier trained: %d epochs, final loss=%.4f",
            epochs,
            losses[-1] if losses else float("nan"),
        )
        return losses

    # --- Prediction ---

    def predict(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.3,
    ) -> Dict[str, float]:
        """Predict field relevance probabilities for a query embedding.

        Returns a dict of field_name → probability for fields above *threshold*.
        """
        if query_embedding is None:
            return {}

        x = query_embedding.reshape(1, -1)

        # Handle dimension mismatch gracefully
        if x.shape[1] != self.input_dim:
            return {}

        _, _, probs = self._forward(x)
        probs = probs[0]  # shape (n_fields,)

        return {
            self.field_names[i]: float(probs[i])
            for i in range(self.n_fields)
            if probs[i] >= threshold
        }

    # --- Persistence ---

    def save(self, path: str | Path) -> None:
        """Pickle classifier weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_fields": self.n_fields,
            "field_names": self.field_names,
            "_trained": self._trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info("Field classifier saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load classifier weights from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.input_dim = data.get("input_dim", 1024)
        self.hidden_dim = data.get("hidden_dim", 64)
        self.n_fields = data.get("n_fields", 12)
        self.field_names = data.get("field_names", FIELD_NAMES[:self.n_fields])
        self._trained = data.get("_trained", True)
        log.info("Field classifier loaded from %s", path)


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------
_classifier_instance: Optional[FieldImportanceClassifier] = None
_classifier_lock = threading.Lock()

_DEFAULT_MODEL_PATH = Path("models/field_importance_classifier.pkl")


def get_field_classifier() -> Optional[FieldImportanceClassifier]:
    """Return the singleton classifier, or None if not yet initialized."""
    return _classifier_instance


def set_field_classifier(clf: Optional[FieldImportanceClassifier]) -> None:
    """Set the singleton classifier (used in tests)."""
    global _classifier_instance
    _classifier_instance = clf


def ensure_field_classifier(
    embedder: Any,
    model_path: Optional[Path] = None,
) -> FieldImportanceClassifier:
    """Load or train the field classifier singleton.

    Thread-safe: only one thread will train/load at a time.
    """
    global _classifier_instance
    with _classifier_lock:
        if _classifier_instance is not None:
            return _classifier_instance

        path = model_path or _DEFAULT_MODEL_PATH
        clf = FieldImportanceClassifier()

        if path.exists():
            try:
                clf.load(path)
                _classifier_instance = clf
                return clf
            except Exception:
                log.warning("Failed to load field classifier from %s, retraining", path, exc_info=True)

        # Train from scratch
        clf.train(embedder)
        try:
            clf.save(path)
        except Exception:
            log.warning("Failed to save field classifier to %s", path, exc_info=True)

        _classifier_instance = clf
        return clf
