"""
Line-level feature extraction for DPIE.

Encodes each line of a document into an 852-dimensional feature vector
by combining three signal groups:

- Layout features (20 dims): structural geometry computed via pure Python
- Character n-gram features (64 dims): PCA-reduced TF-IDF char n-grams
- Semantic embedding (768 dims): SentenceTransformer encoding

No regex anywhere in this module.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal Group 1: Character N-Gram Vocabulary
# ---------------------------------------------------------------------------


class CharNGramVocab:
    """Learns discriminative character n-grams from labeled lines using chi-squared selection.

    Uses sklearn's ``TfidfVectorizer`` with ``analyzer='char_wb'`` (not regex)
    followed by ``SelectKBest`` with ``chi2`` to keep only the most
    discriminative character-level patterns.

    This *learns* patterns like ALL-CAPS or colon-endings from data rather
    than encoding them as hard-coded rules.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (3, 4), max_features: int = 2000) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self._vectorizer: Optional[Any] = None
        self._selector: Optional[Any] = None
        self._fitted = False

    # -- public API --------------------------------------------------------

    def fit(self, lines: List[str], labels: List[str]) -> "CharNGramVocab":
        """Fit the vocabulary from labeled lines.

        Args:
            lines: Raw text lines.
            labels: Corresponding labels (e.g. line types or document types).

        Returns:
            self for chaining.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import SelectKBest, chi2

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            dtype=np.float32,
        )

        tfidf_matrix = self._vectorizer.fit_transform(lines)

        k = min(self.max_features, tfidf_matrix.shape[1])
        self._selector = SelectKBest(chi2, k=k)
        self._selector.fit(tfidf_matrix, labels)

        self._fitted = True
        logger.info("CharNGramVocab fitted: %d n-grams selected from %d candidates",
                     k, tfidf_matrix.shape[1])
        return self

    def encode(self, line: str) -> np.ndarray:
        """Encode a single line into a dense feature vector.

        Args:
            line: Raw text line.

        Returns:
            Dense vector of shape ``(selected_features,)``, dtype float32.
        """
        if not self._fitted or self._vectorizer is None or self._selector is None:
            return np.zeros(self.max_features, dtype=np.float32)

        tfidf = self._vectorizer.transform([line])
        selected = self._selector.transform(tfidf)
        dense: np.ndarray = np.asarray(selected.todense(), dtype=np.float32).ravel()
        return dense

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Save to disk using pickle."""
        with open(path, "wb") as fh:
            pickle.dump({
                "vectorizer": self._vectorizer,
                "selector": self._selector,
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
                "fitted": self._fitted,
            }, fh)

    def load(self, path: str) -> None:
        """Load from disk."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)  # noqa: S301
        self._vectorizer = data["vectorizer"]
        self._selector = data["selector"]
        self.ngram_range = data["ngram_range"]
        self.max_features = data["max_features"]
        self._fitted = data["fitted"]


# ---------------------------------------------------------------------------
# Signal Group 2: Layout Feature Extractor
# ---------------------------------------------------------------------------

_BULLET_CHARS = frozenset('"-*\u2022\u2023\u25e6\u2043\u2219')


class LayoutFeatureExtractor:
    """Extracts 20 layout geometry features from a text line -- no regex.

    Each feature is a continuous float computed via pure Python character
    inspection (``str.isupper``, ``str.isdigit``, etc.).
    """

    FEATURE_NAMES: List[str] = [
        "char_count",           # 0
        "word_count",           # 1
        "relative_length",      # 2
        "avg_word_length",      # 3
        "uppercase_ratio",      # 4
        "lowercase_ratio",      # 5
        "starts_with_upper",    # 6
        "is_all_caps",          # 7
        "digit_ratio",          # 8
        "has_digits",           # 9
        "numeric_word_ratio",   # 10
        "ends_with_colon",      # 11
        "bullet_char_ratio",    # 12
        "pipe_ratio",           # 13
        "leading_whitespace",   # 14
        "space_ratio",          # 15
        "is_indented",          # 16
        "relative_position",    # 17
        "is_near_top",          # 18
        "preceded_by_blank",    # 19
    ]

    def extract(
        self,
        line: str,
        line_index: int,
        total_lines: int,
        prev_line: str = "",
        next_line: str = "",
    ) -> np.ndarray:
        """Extract 20 layout features for a single line.

        Args:
            line: The text line.
            line_index: Zero-based position in the document.
            total_lines: Total number of lines in the document.
            prev_line: The preceding line (empty at start of document).
            next_line: The following line (empty at end of document).

        Returns:
            Float32 array of shape ``(20,)``.
        """
        features = np.zeros(20, dtype=np.float32)
        if not line:
            return features

        n_chars = len(line)
        words = line.split()
        n_words = len(words)
        stripped = line.rstrip()

        alpha_chars = [c for c in line if c.isalpha()]
        n_alpha = len(alpha_chars)

        # 0: char_count (log-scaled for numerical stability)
        features[0] = float(min(n_chars, 500))

        # 1: word_count
        features[1] = float(min(n_words, 100))

        # 2: relative_length  (vs 80 chars "standard" line)
        features[2] = min(n_chars / 80.0, 5.0) if n_chars > 0 else 0.0

        # 3: avg_word_length
        if n_words > 0:
            features[3] = sum(len(w) for w in words) / n_words
        else:
            features[3] = 0.0

        # 4: uppercase_ratio  (of alpha chars)
        if n_alpha > 0:
            features[4] = sum(1 for c in alpha_chars if c.isupper()) / n_alpha
        # 5: lowercase_ratio
        if n_alpha > 0:
            features[5] = sum(1 for c in alpha_chars if c.islower()) / n_alpha

        # 6: starts_with_upper
        for c in line:
            if c.isalpha():
                features[6] = 1.0 if c.isupper() else 0.0
                break

        # 7: is_all_caps  (all alpha chars are upper, >1 alpha char)
        if n_alpha > 1 and all(c.isupper() for c in alpha_chars):
            features[7] = 1.0

        # 8: digit_ratio
        n_digit = sum(1 for c in line if c.isdigit())
        if n_chars > 0:
            features[8] = n_digit / n_chars

        # 9: has_digits
        features[9] = 1.0 if n_digit > 0 else 0.0

        # 10: numeric_word_ratio  (words that look like numbers)
        if n_words > 0:
            num_words = 0
            for w in words:
                clean = w.strip("$%,.()")
                if clean and all(c.isdigit() or c == "." for c in clean):
                    num_words += 1
            features[10] = num_words / n_words

        # 11: ends_with_colon
        if stripped and stripped[-1] == ":":
            features[11] = 1.0

        # 12: bullet_char_ratio
        lstripped = line.lstrip()
        if lstripped:
            first_nonspace = lstripped[0]
            if first_nonspace in _BULLET_CHARS:
                features[12] = 1.0

        # 13: pipe_ratio
        n_pipe = line.count("|")
        if n_chars > 0:
            features[13] = n_pipe / n_chars

        # 14: leading_whitespace  (normalised)
        leading_ws = len(line) - len(line.lstrip())
        features[14] = min(leading_ws / 20.0, 1.0)

        # 15: space_ratio
        n_space = sum(1 for c in line if c == " ")
        if n_chars > 0:
            features[15] = n_space / n_chars

        # 16: is_indented  (leading whitespace >= 4)
        features[16] = 1.0 if leading_ws >= 4 else 0.0

        # 17: relative_position  (0.0 = top, 1.0 = bottom)
        if total_lines > 1:
            features[17] = line_index / (total_lines - 1)
        else:
            features[17] = 0.0

        # 18: is_near_top  (first 10% of document)
        features[18] = 1.0 if (total_lines < 2 or line_index < total_lines * 0.1) else 0.0

        # 19: preceded_by_blank
        features[19] = 1.0 if (not prev_line or not prev_line.strip()) else 0.0

        return features


# ---------------------------------------------------------------------------
# Signal Group 3: Semantic Line Encoder
# ---------------------------------------------------------------------------


class SemanticLineEncoder:
    """Wraps DocWain's SentenceTransformer for line-level semantic encoding.

    Caches results keyed by normalised line text to avoid redundant GPU/CPU
    work when the same line appears in multiple contexts.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._cache: Dict[str, np.ndarray] = {}
        self._dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding model (typically 768)."""
        if self._dim is not None:
            return self._dim
        if self._model is not None:
            try:
                dim = self._model.get_sentence_embedding_dimension()
                self._dim = int(dim)
                return self._dim
            except Exception:
                pass
        self._dim = 768
        return self._dim

    def encode(self, line: str) -> np.ndarray:
        """Encode a single line.

        Args:
            line: Raw text.

        Returns:
            Normalised embedding of shape ``(embedding_dim,)``.
        """
        key = line.strip().lower()[:200]
        if key in self._cache:
            return self._cache[key]

        if self._model is None:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            self._cache[key] = vec
            return vec

        try:
            vec = self._model.encode(
                line[:500],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            vec = np.asarray(vec, dtype=np.float32).ravel()
        except Exception:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)

        self._cache[key] = vec
        return vec

    def encode_batch(self, lines: List[str]) -> np.ndarray:
        """Batch-encode lines.

        Args:
            lines: List of text lines.

        Returns:
            Array of shape ``(len(lines), embedding_dim)``.
        """
        results: List[np.ndarray] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, line in enumerate(lines):
            key = line.strip().lower()[:200]
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(np.empty(0))  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(line[:500])

        if uncached_texts and self._model is not None:
            try:
                batch_vecs = self._model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=32,
                )
                batch_vecs = np.asarray(batch_vecs, dtype=np.float32)
                for j, idx in enumerate(uncached_indices):
                    vec = batch_vecs[j]
                    key = lines[idx].strip().lower()[:200]
                    self._cache[key] = vec
                    results[idx] = vec
            except Exception:
                for idx in uncached_indices:
                    results[idx] = np.zeros(self.embedding_dim, dtype=np.float32)
        elif uncached_texts:
            for idx in uncached_indices:
                results[idx] = np.zeros(self.embedding_dim, dtype=np.float32)

        return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# Combined Encoder
# ---------------------------------------------------------------------------


class LineFeatureEncoder:
    """Combines layout + character n-gram + semantic features per line.

    Output vector structure (852 dims):
        - ``[0:20]``    layout features (20 dims)
        - ``[20:84]``   PCA-reduced char n-gram features (64 dims)
        - ``[84:852]``  semantic embedding (768 dims)
    """

    LAYOUT_DIM = 20
    CHAR_PCA_DIM = 64
    SEMANTIC_DIM = 768  # default fallback; actual dim from model
    TOTAL_DIM = LAYOUT_DIM + CHAR_PCA_DIM + SEMANTIC_DIM  # 852 default

    @property
    def total_dim(self) -> int:
        """Actual total dimension based on the loaded embedding model."""
        return self.LAYOUT_DIM + self.CHAR_PCA_DIM + self._semantic.embedding_dim

    def __init__(self, sentence_model: Any, char_vocab: Optional[CharNGramVocab] = None) -> None:
        self._layout = LayoutFeatureExtractor()
        self._char_vocab = char_vocab or CharNGramVocab()
        self._semantic = SemanticLineEncoder(sentence_model)
        self._pca: Optional[Any] = None  # sklearn PCA object
        self._pca_fitted = False

    # -- single line -------------------------------------------------------

    def encode_line(
        self,
        line: str,
        line_index: int,
        total_lines: int,
        prev_line: str = "",
        next_line: str = "",
    ) -> np.ndarray:
        """Encode one line into an 852-dim feature vector.

        Args:
            line: Text of the line.
            line_index: Zero-based position.
            total_lines: Total lines in the document.
            prev_line: Preceding line text.
            next_line: Following line text.

        Returns:
            Float32 array of shape ``(852,)``.
        """
        # 1. Layout (20,)
        layout = self._layout.extract(line, line_index, total_lines, prev_line, next_line)

        # 2. Char n-gram  raw -> PCA -> (64,)
        char_raw = self._char_vocab.encode(line)
        char_reduced = self._pca_reduce(char_raw)

        # 3. Semantic (768,)
        semantic = self._semantic.encode(line)

        return np.concatenate([layout, char_reduced, semantic]).astype(np.float32)

    # -- document ----------------------------------------------------------

    def encode_document(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Encode all lines of a document.

        Args:
            text: Full document text.

        Returns:
            Tuple of ``(features, lines)`` where features has shape
            ``(num_lines, 852)`` and lines is the list of non-empty line
            strings.
        """
        raw_lines = text.splitlines()
        # Keep non-blank lines but remember blank positions for prev_line context
        lines: List[str] = []
        line_indices: List[int] = []
        for i, ln in enumerate(raw_lines):
            if ln.strip():
                lines.append(ln)
                line_indices.append(i)

        if not lines:
            return np.zeros((0, self.total_dim), dtype=np.float32), []

        total = len(lines)
        features_list: List[np.ndarray] = []

        for pos, (ln, orig_idx) in enumerate(zip(lines, line_indices)):
            prev = raw_lines[orig_idx - 1] if orig_idx > 0 else ""
            nxt = raw_lines[orig_idx + 1] if orig_idx + 1 < len(raw_lines) else ""
            feat = self.encode_line(ln, pos, total, prev, nxt)
            features_list.append(feat)

        return np.stack(features_list, axis=0), lines

    # -- PCA ---------------------------------------------------------------

    def fit_pca(self, sample_lines: List[str]) -> None:
        """Fit PCA on char n-gram features from sample lines.

        Args:
            sample_lines: Representative lines from the corpus.
        """
        from sklearn.decomposition import PCA

        if not sample_lines:
            return

        raw_vectors = np.stack([self._char_vocab.encode(ln) for ln in sample_lines], axis=0)
        n_components = min(self.CHAR_PCA_DIM, raw_vectors.shape[0], raw_vectors.shape[1])
        if n_components < 1:
            return

        self._pca = PCA(n_components=n_components)
        self._pca.fit(raw_vectors)
        self._pca_fitted = True
        logger.info("PCA fitted: %d components from %d samples", n_components, len(sample_lines))

    # -- persistence -------------------------------------------------------

    def save(self, dir_path: str) -> None:
        """Save encoder state to a directory.

        Args:
            dir_path: Directory path (created if needed).
        """
        os.makedirs(dir_path, exist_ok=True)
        self._char_vocab.save(os.path.join(dir_path, "char_vocab.pkl"))
        if self._pca is not None:
            with open(os.path.join(dir_path, "pca.pkl"), "wb") as fh:
                pickle.dump(self._pca, fh)

    def load(self, dir_path: str) -> None:
        """Load encoder state from a directory.

        Args:
            dir_path: Directory containing saved model files.
        """
        vocab_path = os.path.join(dir_path, "char_vocab.pkl")
        if os.path.exists(vocab_path):
            self._char_vocab.load(vocab_path)

        pca_path = os.path.join(dir_path, "pca.pkl")
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as fh:
                self._pca = pickle.load(fh)  # noqa: S301
            self._pca_fitted = True

    # -- internal ----------------------------------------------------------

    def _pca_reduce(self, raw: np.ndarray) -> np.ndarray:
        """Reduce char n-gram features to 64 dims via PCA.

        Falls back to zero-padded truncation when PCA is not fitted.
        """
        if self._pca_fitted and self._pca is not None:
            try:
                reduced = self._pca.transform(raw.reshape(1, -1))[0]
                result = np.zeros(self.CHAR_PCA_DIM, dtype=np.float32)
                result[: len(reduced)] = reduced
                return result
            except Exception:
                pass

        # Fallback: truncate or zero-pad
        result = np.zeros(self.CHAR_PCA_DIM, dtype=np.float32)
        n = min(len(raw), self.CHAR_PCA_DIM)
        result[:n] = raw[:n]
        return result
