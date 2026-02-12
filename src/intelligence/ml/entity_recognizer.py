"""
Span embedding entity recognizer for DPIE.

Detects entities by generating candidate token spans, encoding them
with SentenceTransformer, classifying via a learned MLP, and filtering
overlaps with non-maximum suppression.

Replaces all regex-based NER in ``entity_extractor.py``.

No regex anywhere in this module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Characters that are part of a token (not word-boundary punctuation)
_TOKEN_INNER = frozenset("_@#+")
_TOKEN_CONTINUATION = frozenset("_@#+.-")


class EntityPatternRecognizer:
    """Detects entities via span embedding classification.

    Algorithm:
        1. Generate candidate spans (1-5 tokens) using a sliding window.
        2. Encode each span + its context using SentenceTransformer.
        3. Classify spans using a learned MLP.
        4. Non-maximum suppression removes overlapping detections.
    """

    ENTITY_TYPES: List[str] = [
        "PERSON", "ORGANIZATION", "SKILL", "LOCATION", "DATE",
        "AMOUNT", "EMAIL", "PHONE", "DEGREE", "JOB_TITLE", "NONE",
    ]

    def __init__(self, embedding_model: Any, max_span_tokens: int = 5) -> None:
        self._model = embedding_model
        self.max_span_tokens = max_span_tokens

        self._embedding_dim: Optional[int] = None
        num_types = len(self.ENTITY_TYPES)
        position_features = 8
        input_dim = self.embedding_dim * 2 + position_features  # span + context + 8
        hidden_dim = 128

        # MLP weights
        self.W1: np.ndarray = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W2: np.ndarray = np.random.randn(hidden_dim, num_types).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2: np.ndarray = np.zeros(num_types, dtype=np.float32)

        # Per-type confidence thresholds
        self.thresholds: Dict[str, float] = {t: 0.35 for t in self.ENTITY_TYPES}
        self.thresholds["NONE"] = 1.1  # never select NONE as a detection

        # Label mapping
        self.type_to_idx: Dict[str, int] = {t: i for i, t in enumerate(self.ENTITY_TYPES)}
        self.idx_to_type: Dict[int, str] = {i: t for i, t in enumerate(self.ENTITY_TYPES)}

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding model."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self._model is not None:
            try:
                self._embedding_dim = int(self._model.get_sentence_embedding_dimension())
                return self._embedding_dim
            except Exception:
                pass
        self._embedding_dim = 768
        return self._embedding_dim

    # -- tokenizer (NO REGEX) ----------------------------------------------

    def _tokenize_simple(self, text: str) -> List[Tuple[str, int, int]]:
        """Simple whitespace + punctuation tokenizer -- NO regex.

        Walks through characters, splitting on whitespace and punctuation
        boundaries while keeping email addresses and compound tokens intact.

        Args:
            text: Input text.

        Returns:
            List of ``(token_text, start_char, end_char)`` tuples.
        """
        tokens: List[Tuple[str, int, int]] = []
        i = 0
        n = len(text)

        while i < n:
            # Skip whitespace
            if text[i].isspace():
                i += 1
                continue

            start = i

            if text[i].isalnum() or text[i] in _TOKEN_INNER:
                # Accumulate word-like characters
                while i < n and (text[i].isalnum() or text[i] in _TOKEN_CONTINUATION):
                    # Don't include trailing punctuation
                    if text[i] in (".", "-") and (i + 1 >= n or text[i + 1].isspace()):
                        break
                    i += 1
            else:
                # Single punctuation character as its own token
                i += 1

            token_text = text[start:i]
            if token_text.strip():
                tokens.append((token_text, start, i))

        return tokens

    # -- span generation ---------------------------------------------------

    def _generate_candidate_spans(
        self,
        tokens: List[Tuple[str, int, int]],
    ) -> List[Tuple[int, int, str, int, int]]:
        """Generate all candidate spans of 1 to ``max_span_tokens`` tokens.

        Args:
            tokens: Output of ``_tokenize_simple``.

        Returns:
            List of ``(start_char, end_char, span_text, tok_start_idx, tok_end_idx)``.
        """
        spans: List[Tuple[int, int, str, int, int]] = []
        n = len(tokens)

        for i in range(n):
            for length in range(1, min(self.max_span_tokens, n - i) + 1):
                j = i + length - 1
                start_char = tokens[i][1]
                end_char = tokens[j][2]
                span_text = " ".join(t[0] for t in tokens[i: j + 1])

                # Skip pure punctuation spans
                if not any(c.isalnum() for c in span_text):
                    continue

                spans.append((start_char, end_char, span_text, i, j))

        return spans

    # -- feature computation -----------------------------------------------

    def _compute_span_features(
        self,
        span_text: str,
        context_text: str,
        span_start: int,
        total_length: int,
    ) -> np.ndarray:
        """Compute the 8 position/shape features for a span.

        Args:
            span_text: The span text.
            context_text: Full input text for context embedding.
            span_start: Character offset of the span.
            total_length: Total length of the input text.

        Returns:
            Float32 array of shape ``(8,)``.
        """
        features = np.zeros(8, dtype=np.float32)

        chars = list(span_text)
        words = span_text.split()
        n_chars = len(chars) if chars else 1

        # 1. relative_position
        features[0] = span_start / max(total_length, 1)

        # 2. relative_length
        features[1] = len(span_text) / max(total_length, 1)

        # 3. word_count
        features[2] = float(len(words))

        # 4. uppercase_ratio
        alpha = [c for c in chars if c.isalpha()]
        if alpha:
            features[3] = sum(1 for c in alpha if c.isupper()) / len(alpha)

        # 5. digit_ratio
        features[4] = sum(1 for c in chars if c.isdigit()) / n_chars

        # 6. starts_upper
        for c in span_text:
            if c.isalpha():
                features[5] = 1.0 if c.isupper() else 0.0
                break

        # 7. has_special
        features[6] = 1.0 if any(c in "@./" for c in span_text) else 0.0

        # 8. all_words_capitalized
        if words:
            cap_count = 0
            for w in words:
                first_alpha = ""
                for c in w:
                    if c.isalpha():
                        first_alpha = c
                        break
                if first_alpha and first_alpha.isupper():
                    cap_count += 1
            features[7] = 1.0 if cap_count == len(words) else 0.0

        return features

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text with the embedding model."""
        if self._model is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        try:
            vec = self._model.encode(
                text[:500], normalize_embeddings=True, convert_to_numpy=True,
            )
            return np.asarray(vec, dtype=np.float32).ravel()
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def _build_full_features(
        self,
        span_text: str,
        context_text: str,
        span_start: int,
        total_length: int,
    ) -> np.ndarray:
        """Build the full feature vector for a span.

        Concatenates span embedding (768) + context embedding (768) +
        position features (8) = (1544,).
        """
        span_emb = self._encode_text(span_text)
        context_emb = self._encode_text(context_text[:200])
        pos_features = self._compute_span_features(span_text, context_text, span_start, total_length)
        return np.concatenate([span_emb, context_emb, pos_features])

    # -- classification ----------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / (exp_x.sum() + 1e-12)

    def _classify_span(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify a span via MLP forward pass.

        Args:
            features: Shape ``(input_dim,)``.

        Returns:
            ``(entity_type, confidence)``.
        """
        hidden = np.maximum(0.0, features @ self.W1 + self.b1)  # ReLU
        logits = hidden @ self.W2 + self.b2
        probs = self._softmax(logits)

        best_idx = int(np.argmax(probs))
        entity_type = self.idx_to_type.get(best_idx, "NONE")
        confidence = float(probs[best_idx])

        return entity_type, confidence

    # -- NMS ---------------------------------------------------------------

    @staticmethod
    def _non_maximum_suppression(
        detections: List[Dict[str, Any]],
        overlap_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Remove overlapping detections, keeping the highest confidence.

        Args:
            detections: List of detection dicts with ``start``, ``end``, ``confidence``.
            overlap_threshold: IoU threshold for suppression.

        Returns:
            Filtered detections.
        """
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []

        for det in sorted_dets:
            overlaps = False
            for k in kept:
                # Compute character overlap
                inter_start = max(det["start"], k["start"])
                inter_end = min(det["end"], k["end"])
                inter_len = max(0, inter_end - inter_start)
                min_span_len = min(det["end"] - det["start"], k["end"] - k["start"])

                if min_span_len > 0 and inter_len / min_span_len > overlap_threshold:
                    overlaps = True
                    break

            if not overlaps:
                kept.append(det)

        return kept

    # -- extraction --------------------------------------------------------

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Input text.

        Returns:
            List of entity dicts with keys ``entity``, ``type``, ``start``,
            ``end``, ``confidence``.
        """
        if not text or not text.strip():
            return []

        tokens = self._tokenize_simple(text)
        if not tokens:
            return []

        spans = self._generate_candidate_spans(tokens)
        total_length = len(text)

        detections: List[Dict[str, Any]] = []

        for start_char, end_char, span_text, _, _ in spans:
            features = self._build_full_features(span_text, text, start_char, total_length)
            entity_type, confidence = self._classify_span(features)

            if entity_type == "NONE":
                continue

            threshold = self.thresholds.get(entity_type, 0.35)
            if confidence < threshold:
                continue

            detections.append({
                "entity": span_text,
                "type": entity_type,
                "start": start_char,
                "end": end_char,
                "confidence": confidence,
            })

        return self._non_maximum_suppression(detections)

    def extract_persons(self, text: str) -> List[str]:
        """Extract only PERSON entities from text.

        Convenience method replacing ``_detect_person_name()`` in dw_newron.py.

        Args:
            text: Input text.

        Returns:
            List of person name strings.
        """
        entities = self.extract(text)
        return [e["entity"] for e in entities if e["type"] == "PERSON"]

    # -- training ----------------------------------------------------------

    def fit(
        self,
        examples: List[Dict[str, Any]],
        lr: float = 0.001,
        epochs: int = 30,
    ) -> Dict[str, List[float]]:
        """Train the MLP classifier from labeled entity examples.

        Args:
            examples: List of dicts with ``text`` and ``entities`` keys.
                Each entity has ``span``, ``type``, ``start``, ``end``.
            lr: Learning rate.
            epochs: Number of training epochs.

        Returns:
            Dictionary with ``"loss"`` history.
        """
        # Build training data
        train_X: List[np.ndarray] = []
        train_y: List[int] = []

        for example in examples:
            text = example.get("text", "")
            entities = example.get("entities", [])
            if not text:
                continue

            tokens = self._tokenize_simple(text)
            total_length = len(text)

            # Index labeled spans for fast lookup
            labeled_spans: Dict[Tuple[int, int], str] = {}
            for ent in entities:
                labeled_spans[(ent["start"], ent["end"])] = ent["type"]

            all_spans = self._generate_candidate_spans(tokens)
            positives: List[Tuple[int, int, str, str]] = []
            negatives: List[Tuple[int, int, str]] = []

            for start_char, end_char, span_text, _, _ in all_spans:
                entity_type = labeled_spans.get((start_char, end_char))
                if entity_type:
                    positives.append((start_char, end_char, span_text, entity_type))
                else:
                    negatives.append((start_char, end_char, span_text))

            # Positive examples
            for start_char, end_char, span_text, entity_type in positives:
                features = self._build_full_features(span_text, text, start_char, total_length)
                train_X.append(features)
                train_y.append(self.type_to_idx.get(entity_type, self.type_to_idx["NONE"]))

            # Negative sampling: 3x negatives per positive
            n_neg = min(len(negatives), len(positives) * 3)
            if n_neg > 0 and negatives:
                neg_indices = np.random.choice(len(negatives), size=n_neg, replace=False)
                for ni in neg_indices:
                    start_char, end_char, span_text = negatives[ni]
                    features = self._build_full_features(span_text, text, start_char, total_length)
                    train_X.append(features)
                    train_y.append(self.type_to_idx["NONE"])

        if not train_X:
            logger.warning("No training data for EntityPatternRecognizer")
            return {"loss": []}

        X = np.stack(train_X, axis=0).astype(np.float32)
        y = np.array(train_y, dtype=np.int64)

        # Reinitialise weights for correct input dim
        input_dim = X.shape[1]
        hidden_dim = 128
        num_types = len(self.ENTITY_TYPES)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, num_types).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(num_types, dtype=np.float32)

        # Adam state
        adam_state = self._init_adam()
        n = X.shape[0]
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(n)

            for step, idx in enumerate(indices):
                x = X[idx]
                target = y[idx]

                # Forward
                hidden = np.maximum(0.0, x @ self.W1 + self.b1)
                logits = hidden @ self.W2 + self.b2
                probs = self._softmax(logits)

                loss = -np.log(probs[target] + 1e-12)
                epoch_loss += float(loss)

                # Backward
                d_logits = probs.copy()
                d_logits[target] -= 1.0

                d_W2 = np.outer(hidden, d_logits)
                d_b2 = d_logits.copy()
                d_h = self.W2 @ d_logits
                d_relu = d_h * (hidden > 0).astype(np.float32)
                d_W1 = np.outer(x, d_relu)
                d_b1 = d_relu.copy()

                grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
                t_step = epoch * n + step + 1
                self._adam_step(adam_state, grads, lr, t_step)

            avg_loss = epoch_loss / max(n, 1)
            loss_history.append(avg_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info("EntityPatternRecognizer epoch %d/%d  loss=%.4f", epoch, epochs, avg_loss)

        # Optimize per-type thresholds on training data
        self._optimize_thresholds(X, y)

        return {"loss": loss_history}

    def _optimize_thresholds(self, X: np.ndarray, y: np.ndarray) -> None:
        """Sweep thresholds per entity type to maximize F1 on training data."""
        num_types = len(self.ENTITY_TYPES)

        # Get predictions for all training examples
        all_probs: List[np.ndarray] = []
        for i in range(X.shape[0]):
            hidden = np.maximum(0.0, X[i] @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2
            probs = self._softmax(logits)
            all_probs.append(probs)

        prob_matrix = np.stack(all_probs, axis=0)  # (n, num_types)

        for type_idx in range(num_types):
            type_name = self.idx_to_type.get(type_idx, "NONE")
            if type_name == "NONE":
                continue

            is_positive = (y == type_idx)
            if not np.any(is_positive):
                continue

            type_probs = prob_matrix[:, type_idx]
            best_f1 = 0.0
            best_thresh = 0.35

            for thresh in np.arange(0.1, 0.91, 0.05):
                predicted = type_probs >= thresh
                tp = int(np.sum(predicted & is_positive))
                fp = int(np.sum(predicted & ~is_positive))
                fn = int(np.sum(~predicted & is_positive))

                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(thresh)

            self.thresholds[type_name] = best_thresh

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and thresholds to ``.npz``.

        Args:
            path: File path.
        """
        import json
        np.savez(
            path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            thresholds=json.dumps(self.thresholds),
        )

    def load(self, path: str) -> None:
        """Load model weights and thresholds from ``.npz``.

        Args:
            path: File path.
        """
        import json
        data = np.load(path, allow_pickle=True)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        thresholds_raw = data.get("thresholds")
        if thresholds_raw is not None:
            self.thresholds = json.loads(str(thresholds_raw))

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
