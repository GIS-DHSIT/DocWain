"""
Section boundary detector for DPIE.

Detects section boundaries using a sliding window over line feature vectors
and transition features between adjacent lines.

Key insight: section boundaries are *transitions*.  The model looks at a
window of 6 lines (3 before + current + 2 after) and computes 12
transition features.  Patterns like "blank line -> short capitalised line
-> long paragraph" are captured as numerical differences rather than regex.

No regex anywhere in this module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SectionBoundaryDetector:
    """Detects section boundaries via context-aware transition classification.

    Uses a two-layer MLP over windowed line features augmented with
    transition features.
    """

    def __init__(
        self,
        feature_dim: int = 852,
        window_before: int = 3,
        window_after: int = 2,
    ) -> None:
        self.feature_dim = feature_dim
        self.window_before = window_before
        self.window_after = window_after
        self.window_size = window_before + 1 + window_after  # 6
        self.transition_dim = 12

        input_dim = feature_dim * self.window_size + self.transition_dim
        hidden_dim = 128
        output_dim = 2  # [not_boundary, is_boundary]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier init
        self.W1: np.ndarray = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W2: np.ndarray = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2: np.ndarray = np.zeros(output_dim, dtype=np.float32)

    # -- transition features -----------------------------------------------

    def _compute_transition_features(
        self,
        prev_features: np.ndarray,
        curr_features: np.ndarray,
    ) -> np.ndarray:
        """Compute 12 transition features between two adjacent lines.

        Args:
            prev_features: Feature vector of the preceding line (852,).
            curr_features: Feature vector of the current line (852,).

        Returns:
            Float32 array of shape ``(12,)``.
        """
        trans = np.zeros(12, dtype=np.float32)

        # Layout feature indices (from LayoutFeatureExtractor.FEATURE_NAMES)
        # 0=char_count, 1=word_count, 4=uppercase_ratio, 7=is_all_caps
        # 8=digit_ratio, 11=ends_with_colon, 12=bullet_char_ratio
        # 14=leading_whitespace, 17=relative_position, 19=preceded_by_blank

        # 1. uppercase_ratio_change
        trans[0] = curr_features[4] - prev_features[4]
        # 2. all_caps_toggle
        trans[1] = abs(curr_features[7] - prev_features[7])
        # 3. char_count_change (normalised)
        trans[2] = (curr_features[0] - prev_features[0]) / max(prev_features[0], 1.0)
        # 4. word_count_change (normalised)
        trans[3] = (curr_features[1] - prev_features[1]) / max(prev_features[1], 1.0)
        # 5. colon_ending_change
        trans[4] = curr_features[11] - prev_features[11]
        # 6. bullet_ratio_change
        trans[5] = curr_features[12] - prev_features[12]
        # 7. leading_ws_change
        trans[6] = curr_features[14] - prev_features[14]
        # 8. digit_ratio_change
        trans[7] = curr_features[8] - prev_features[8]

        # Semantic features start at index 84 (layout=20 + char_pca=64)
        sem_start = 84
        prev_sem = prev_features[sem_start:]
        curr_sem = curr_features[sem_start:]

        # 9. semantic_distance (1 - cosine_similarity)
        prev_norm = np.linalg.norm(prev_sem)
        curr_norm = np.linalg.norm(curr_sem)
        if prev_norm > 1e-8 and curr_norm > 1e-8:
            cos_sim = float(np.dot(prev_sem, curr_sem) / (prev_norm * curr_norm))
            trans[8] = 1.0 - cos_sim
        else:
            trans[8] = 1.0

        # 10. preceded_by_blank
        trans[9] = prev_features[19]
        # 11. relative_position
        trans[10] = curr_features[17]
        # 12. prev_was_short (word_count <= 3)
        trans[11] = 1.0 if prev_features[1] <= 3.0 else 0.0

        return trans

    # -- window construction -----------------------------------------------

    def _build_window(self, all_features: np.ndarray, line_idx: int) -> np.ndarray:
        """Build a windowed feature vector for a single line.

        Args:
            all_features: Shape ``(num_lines, feature_dim)``.
            line_idx: Index of the current line.

        Returns:
            Float32 array of shape ``(feature_dim * 6 + 12,)``.
        """
        n_lines = all_features.shape[0]
        window_vecs: List[np.ndarray] = []

        # Collect window_before + current + window_after
        for offset in range(-self.window_before, self.window_after + 1):
            idx = line_idx + offset
            if 0 <= idx < n_lines:
                window_vecs.append(all_features[idx])
            else:
                window_vecs.append(np.zeros(self.feature_dim, dtype=np.float32))

        # Transition features between previous line and current
        prev_idx = line_idx - 1
        prev_feat = all_features[prev_idx] if prev_idx >= 0 else np.zeros(self.feature_dim, dtype=np.float32)
        curr_feat = all_features[line_idx]
        trans = self._compute_transition_features(prev_feat, curr_feat)

        return np.concatenate(window_vecs + [trans])

    # -- forward -----------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / (exp_x.sum() + 1e-12)

    def _forward(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single forward pass.

        Returns:
            ``(probs, hidden, logits)``
        """
        hidden = np.maximum(0.0, window @ self.W1 + self.b1)  # ReLU
        logits = hidden @ self.W2 + self.b2
        probs = self._softmax(logits)
        return probs, hidden, logits

    # -- prediction --------------------------------------------------------

    def predict_boundaries(
        self,
        all_features: np.ndarray,
        lines: List[str],
    ) -> List[Dict[str, Any]]:
        """Predict section boundaries for a document.

        Args:
            all_features: Shape ``(num_lines, feature_dim)``.
            lines: Corresponding line strings.

        Returns:
            List of dicts with keys ``line_index``, ``is_boundary``,
            ``confidence``, ``heading_text``.
        """
        n_lines = all_features.shape[0]
        results: List[Dict[str, Any]] = []

        for i in range(n_lines):
            window = self._build_window(all_features, i)
            probs, _, _ = self._forward(window)
            is_boundary = bool(probs[1] > 0.5)
            confidence = float(probs[1])

            # First line is always a boundary
            if i == 0:
                is_boundary = True
                confidence = max(confidence, 1.0)

            # Empty content lines are never boundaries
            if i < len(lines) and not lines[i].strip():
                is_boundary = False
                confidence = 0.0

            results.append({
                "line_index": i,
                "is_boundary": is_boundary,
                "confidence": confidence,
                "heading_text": lines[i].strip() if is_boundary and i < len(lines) else None,
            })

        # Post-processing: merge consecutive boundaries (keep highest confidence)
        results = self._merge_consecutive(results)

        return results

    @staticmethod
    def _merge_consecutive(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge runs of consecutive boundary predictions, keeping the highest confidence."""
        if not results:
            return results

        merged = [results[0]]
        for r in results[1:]:
            if r["is_boundary"] and merged[-1]["is_boundary"]:
                # Consecutive boundaries -> keep higher confidence
                if r["confidence"] > merged[-1]["confidence"]:
                    merged[-1] = r
            else:
                merged.append(r)
        return merged

    # -- training ----------------------------------------------------------

    def fit(
        self,
        documents_features: List[np.ndarray],
        boundary_labels: List[List[bool]],
        lr: float = 0.001,
        epochs: int = 30,
    ) -> Dict[str, List[float]]:
        """Train with binary cross-entropy and class-weight balancing.

        Args:
            documents_features: List of per-document feature arrays.
            boundary_labels: List of per-document boundary bool lists.
            lr: Learning rate.
            epochs: Number of training epochs.

        Returns:
            Dictionary with ``"loss"`` history.
        """
        # Compute class weights (boundaries are rare)
        total_pos = sum(sum(1 for b in bl if b) for bl in boundary_labels)
        total_neg = sum(sum(1 for b in bl if not b) for bl in boundary_labels)
        weight_pos = max(total_neg / max(total_pos, 1), 1.0)
        weight_pos = min(weight_pos, 20.0)  # cap

        logger.info("SectionBoundaryDetector: %d pos, %d neg, weight_pos=%.2f",
                     total_pos, total_neg, weight_pos)

        # Adam state
        adam_state = self._init_adam()
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_samples = 0

            doc_order = np.random.permutation(len(documents_features))

            for doc_idx in doc_order:
                feats = documents_features[doc_idx]
                labels = boundary_labels[doc_idx]
                n_lines = feats.shape[0]

                for line_idx in range(n_lines):
                    if line_idx >= len(labels):
                        break

                    window = self._build_window(feats, line_idx)
                    probs, hidden, _ = self._forward(window)

                    target = 1 if labels[line_idx] else 0
                    w = weight_pos if target == 1 else 1.0

                    loss = -w * np.log(probs[target] + 1e-12)
                    epoch_loss += float(loss)
                    n_samples += 1

                    # Backward
                    d_logits = probs.copy()
                    d_logits[target] -= 1.0
                    d_logits *= w

                    d_W2 = np.outer(hidden, d_logits)
                    d_b2 = d_logits.copy()
                    d_h = self.W2 @ d_logits
                    d_relu = d_h * (hidden > 0).astype(np.float32)
                    d_W1 = np.outer(window, d_relu)
                    d_b1 = d_relu.copy()

                    grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
                    t_step = epoch * max(n_samples, 1) + n_samples
                    self._adam_step(adam_state, grads, lr, max(t_step, 1))

            avg_loss = epoch_loss / max(n_samples, 1)
            loss_history.append(avg_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info("SectionBoundaryDetector epoch %d/%d  loss=%.4f", epoch, epochs, avg_loss)

        return {"loss": loss_history}

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to ``.npz``.

        Args:
            path: File path.
        """
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: str) -> None:
        """Load model weights from ``.npz``.

        Args:
            path: File path.
        """
        data = np.load(path, allow_pickle=True)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.input_dim = self.W1.shape[0]
        self.hidden_dim = self.W1.shape[1]

    # -- Adam helpers ------------------------------------------------------

    def _init_adam(self) -> Dict[str, Any]:
        """Initialise Adam optimiser state."""
        state: Dict[str, Any] = {}
        for name, param in [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]:
            state[name] = {"m": np.zeros_like(param), "v": np.zeros_like(param)}
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
        """Apply one Adam update."""
        params = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        for name in grads:
            g = grads[name]
            s = state[name]
            s["m"] = beta1 * s["m"] + (1 - beta1) * g
            s["v"] = beta2 * s["v"] + (1 - beta2) * (g ** 2)
            m_hat = s["m"] / (1 - beta1 ** t)
            v_hat = s["v"] / (1 - beta2 ** t)
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
