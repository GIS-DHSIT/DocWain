"""
Attention-weighted document type classifier for DPIE.

Classifies documents by learning which lines are most informative
via a learned attention mechanism over line feature vectors.

Architecture:
    1. Each line -> feature vector via LineFeatureEncoder (input)
    2. Learned attention weights determine which lines to focus on
    3. Weighted aggregation produces a single document vector
    4. Softmax classification head maps to document types

No regex anywhere in this module.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DocumentTypeClassifier:
    """Classifies document types by learning which lines are most informative.

    The attention mechanism learns that for resumes, the top lines (name,
    contact) are diagnostic; for invoices, lines with amounts near the bottom
    are diagnostic.  All learned from data -- no hard-coded rules.
    """

    TAXONOMY: List[str] = [
        "resume", "invoice", "purchase_order", "contract", "policy",
        "report", "statement", "presentation", "brochure", "other",
    ]

    def __init__(
        self,
        feature_dim: int = 852,
        num_classes: int = 10,
        attention_dim: int = 64,
    ) -> None:
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.attention_dim = attention_dim

        # Label mapping
        self.label_map: Dict[str, int] = {t: i for i, t in enumerate(self.TAXONOMY)}
        self.index_map: Dict[int, str] = {i: t for i, t in enumerate(self.TAXONOMY)}

        # Xavier initialization
        self.W_attn: np.ndarray = np.random.randn(feature_dim, attention_dim).astype(np.float32) * np.sqrt(2.0 / feature_dim)
        self.v_attn: np.ndarray = np.random.randn(attention_dim, 1).astype(np.float32) * np.sqrt(2.0 / attention_dim)
        self.W_class: np.ndarray = np.random.randn(feature_dim, num_classes).astype(np.float32) * np.sqrt(2.0 / feature_dim)
        self.b_class: np.ndarray = np.zeros(num_classes, dtype=np.float32)

    # -- numerics ----------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / (exp_x.sum() + 1e-12)

    # -- attention ---------------------------------------------------------

    def _compute_attention(self, line_features: np.ndarray) -> np.ndarray:
        """Compute attention weights over lines.

        Args:
            line_features: Shape ``(num_lines, feature_dim)``.

        Returns:
            Attention weights of shape ``(num_lines,)`` summing to 1.
        """
        # projected: (num_lines, attention_dim)
        projected = np.tanh(line_features @ self.W_attn)
        # scores: (num_lines,)
        scores = (projected @ self.v_attn).squeeze(-1)
        return self._softmax(scores)

    def _aggregate(self, line_features: np.ndarray, attention: np.ndarray) -> np.ndarray:
        """Weighted sum of line features.

        Args:
            line_features: Shape ``(num_lines, feature_dim)``.
            attention: Shape ``(num_lines,)``.

        Returns:
            Document vector of shape ``(feature_dim,)``.
        """
        return (line_features * attention[:, None]).sum(axis=0)

    # -- prediction --------------------------------------------------------

    def predict(self, line_features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Predict document type.

        Args:
            line_features: Shape ``(num_lines, feature_dim)``.

        Returns:
            ``(doc_type, confidence, attention_weights)``.
        """
        if line_features.ndim == 1:
            line_features = line_features.reshape(1, -1)

        attention = self._compute_attention(line_features)
        doc_vector = self._aggregate(line_features, attention)
        logits = doc_vector @ self.W_class + self.b_class
        probs = self._softmax(logits)

        best_idx = int(np.argmax(probs))
        doc_type = self.index_map.get(best_idx, "other")
        confidence = float(probs[best_idx])

        return doc_type, confidence, attention

    # -- training ----------------------------------------------------------

    def fit(
        self,
        documents: List[np.ndarray],
        labels: List[str],
        lr: float = 0.001,
        epochs: int = 50,
    ) -> Dict[str, List[float]]:
        """Train the classifier with Adam optimizer and cross-entropy loss.

        Args:
            documents: List of line-feature arrays, each ``(num_lines, feature_dim)``.
            labels: Document type label for each document.
            lr: Learning rate.
            epochs: Number of training epochs.

        Returns:
            Dictionary with ``"loss"`` history per epoch.
        """
        # Encode labels
        label_indices: List[int] = []
        for lbl in labels:
            if lbl not in self.label_map:
                self.label_map[lbl] = len(self.label_map)
                self.index_map[len(self.index_map)] = lbl
            label_indices.append(self.label_map[lbl])

        # Resize classification head if new labels were added
        actual_classes = len(self.label_map)
        if actual_classes > self.num_classes:
            old_W = self.W_class
            old_b = self.b_class
            self.num_classes = actual_classes
            self.W_class = np.zeros((self.feature_dim, actual_classes), dtype=np.float32)
            self.b_class = np.zeros(actual_classes, dtype=np.float32)
            self.W_class[:, :old_W.shape[1]] = old_W
            self.b_class[:old_b.shape[0]] = old_b
            # Xavier init for new columns
            new_cols = actual_classes - old_W.shape[1]
            self.W_class[:, old_W.shape[1]:] = (
                np.random.randn(self.feature_dim, new_cols).astype(np.float32)
                * np.sqrt(2.0 / self.feature_dim)
            )

        # Adam state
        adam_params = self._init_adam()

        loss_history: List[float] = []
        n = len(documents)

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(n)

            for idx in indices:
                line_feat = documents[idx]
                if line_feat.ndim == 1:
                    line_feat = line_feat.reshape(1, -1)
                target = label_indices[idx]

                # Forward
                projected = np.tanh(line_feat @ self.W_attn)
                scores = (projected @ self.v_attn).squeeze(-1)
                attention = self._softmax(scores)
                doc_vector = (line_feat * attention[:, None]).sum(axis=0)
                logits = doc_vector @ self.W_class + self.b_class
                probs = self._softmax(logits)

                loss = -np.log(probs[target] + 1e-12)
                epoch_loss += float(loss)

                # Backward: softmax + CE gradient
                d_logits = probs.copy()
                d_logits[target] -= 1.0

                # Classification head gradients
                d_W_class = np.outer(doc_vector, d_logits)
                d_b_class = d_logits
                d_doc_vector = self.W_class @ d_logits

                # Aggregation gradient
                d_attention = (line_feat * d_doc_vector[None, :]).sum(axis=1)

                # Softmax attention gradient
                d_scores = attention * (d_attention - np.dot(d_attention, attention))

                # Tanh projection gradient
                d_projected = d_scores[:, None] * self.v_attn.T  # (num_lines, attention_dim)
                d_tanh = d_projected * (1.0 - projected ** 2)

                d_W_attn = line_feat.T @ d_tanh
                d_v_attn = projected.T @ d_scores[:, None]

                # Adam update
                grads = {
                    "W_attn": d_W_attn,
                    "v_attn": d_v_attn,
                    "W_class": d_W_class,
                    "b_class": d_b_class,
                }
                self._adam_step(adam_params, grads, lr, epoch * n + int(idx) + 1)

            avg_loss = epoch_loss / max(n, 1)
            loss_history.append(avg_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info("DocumentTypeClassifier epoch %d/%d  loss=%.4f", epoch, epochs, avg_loss)

        return {"loss": loss_history}

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to an ``.npz`` file.

        Args:
            path: File path (should end with ``.npz``).
        """
        np.savez(
            path,
            W_attn=self.W_attn,
            v_attn=self.v_attn,
            W_class=self.W_class,
            b_class=self.b_class,
            label_map=json.dumps(self.label_map),
        )

    def load(self, path: str) -> None:
        """Load model weights from an ``.npz`` file.

        Args:
            path: File path.
        """
        data = np.load(path, allow_pickle=True)
        self.W_attn = data["W_attn"]
        self.v_attn = data["v_attn"]
        self.W_class = data["W_class"]
        self.b_class = data["b_class"]
        self.label_map = json.loads(str(data["label_map"]))
        self.index_map = {v: k for k, v in self.label_map.items()}
        self.feature_dim = self.W_attn.shape[0]
        self.num_classes = self.W_class.shape[1]
        self.attention_dim = self.W_attn.shape[1]

    # -- Adam optimizer helpers --------------------------------------------

    def _init_adam(self) -> Dict[str, Any]:
        """Initialise Adam optimiser state for all parameters."""
        state: Dict[str, Any] = {}
        for name, param in [
            ("W_attn", self.W_attn),
            ("v_attn", self.v_attn),
            ("W_class", self.W_class),
            ("b_class", self.b_class),
        ]:
            state[name] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param),
            }
        return state

    def _adam_step(
        self,
        state: Dict[str, Any],
        grads: Dict[str, np.ndarray],
        lr: float,
        t: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Apply one Adam update to all parameters."""
        params = {
            "W_attn": self.W_attn,
            "v_attn": self.v_attn,
            "W_class": self.W_class,
            "b_class": self.b_class,
        }
        for name in grads:
            g = grads[name]
            s = state[name]
            s["m"] = beta1 * s["m"] + (1 - beta1) * g
            s["v"] = beta2 * s["v"] + (1 - beta2) * (g ** 2)
            m_hat = s["m"] / (1 - beta1 ** t)
            v_hat = s["v"] / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            params[name] -= update

        # Write back (numpy arrays are mutable so this is a safety measure)
        self.W_attn = params["W_attn"]
        self.v_attn = params["v_attn"]
        self.W_class = params["W_class"]
        self.b_class = params["b_class"]
