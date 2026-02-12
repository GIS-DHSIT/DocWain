"""
Prototype-based section kind classifier for DPIE.

Classifies sections into a 21-kind taxonomy using dual-signal
prototype embeddings (title + content) and optional NCA metric learning.

Key innovation: a section called "Miscellaneous" but containing skill
keywords will still match ``skills_technical`` because the content signal
pulls it toward that prototype.

Few-shot capable -- needs only 3-5 examples per section kind.

No regex anywhere in this module.
"""
from __future__ import annotations

import logging
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SectionKindClassifier:
    """Classifies sections by proximity to learned prototype embeddings.

    Uses dual-signal representation:
        - **Title embedding**: captures what the section is *called*.
        - **Content embedding**: captures what the section *contains*.

    With Neighborhood Components Analysis (NCA) metric learning when
    sufficient training data (50+ examples) is available.
    """

    TAXONOMY: List[str] = [
        "identity_contact", "summary_objective", "experience", "projects",
        "education", "certifications", "skills_technical", "skills_functional",
        "tools_technologies", "achievements_awards", "publications_patents",
        "leadership_management", "compliance_regulatory", "invoice_metadata",
        "financial_summary", "transactions", "line_items", "parties_addresses",
        "terms_conditions", "totals", "misc",
    ]

    DOMAIN_FILTER: Dict[str, Set[str]] = {
        "resume": {
            "identity_contact", "summary_objective", "experience", "projects",
            "education", "certifications", "skills_technical", "skills_functional",
            "tools_technologies", "achievements_awards", "publications_patents",
            "leadership_management", "misc",
        },
        "invoice": {
            "invoice_metadata", "line_items", "totals", "parties_addresses",
            "terms_conditions", "financial_summary", "misc",
        },
        "contract": {
            "parties_addresses", "terms_conditions", "compliance_regulatory",
            "financial_summary", "misc",
        },
        "report": {
            "summary_objective", "financial_summary", "misc",
        },
    }

    def __init__(
        self,
        embedding_model: Any,
        title_weight: float = 0.6,
        content_weight: float = 0.4,
    ) -> None:
        self._model = embedding_model
        self.title_weight = title_weight
        self.content_weight = content_weight

        # Maps kind -> (title_centroid, content_centroid)
        self.prototypes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Optional NCA metric transform
        self.metric_transform: Optional[np.ndarray] = None

        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the underlying embedding model."""
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

    # -- encoding ----------------------------------------------------------

    def _encode_section(self, title: str, content: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a section into title and content embedding vectors.

        Args:
            title: Section heading text.
            content: Section body text (first 500 chars used).

        Returns:
            ``(title_vec, content_vec)`` each of shape ``(embedding_dim,)``.
        """
        title_text = title.strip() if title and title.strip() else "untitled"
        content_text = content[:500].strip() if content and content.strip() else "empty"

        if self._model is None:
            return (
                np.zeros(self.embedding_dim, dtype=np.float32),
                np.zeros(self.embedding_dim, dtype=np.float32),
            )

        try:
            title_vec = np.asarray(
                self._model.encode(title_text, normalize_embeddings=True, convert_to_numpy=True),
                dtype=np.float32,
            ).ravel()
            content_vec = np.asarray(
                self._model.encode(content_text, normalize_embeddings=True, convert_to_numpy=True),
                dtype=np.float32,
            ).ravel()
        except Exception:
            title_vec = np.zeros(self.embedding_dim, dtype=np.float32)
            content_vec = np.zeros(self.embedding_dim, dtype=np.float32)

        return title_vec, content_vec

    # -- distance ----------------------------------------------------------

    def _combined_distance(
        self,
        title_vec: np.ndarray,
        content_vec: np.ndarray,
        proto_title: np.ndarray,
        proto_content: np.ndarray,
    ) -> float:
        """Compute weighted distance between a section and a prototype.

        If a metric transform is available, vectors are projected first.

        Args:
            title_vec: Section title embedding.
            content_vec: Section content embedding.
            proto_title: Prototype title centroid.
            proto_content: Prototype content centroid.

        Returns:
            Combined distance (lower is closer).
        """
        if self.metric_transform is not None:
            combined = np.concatenate([
                self.title_weight * title_vec,
                self.content_weight * content_vec,
            ])
            combined_proto = np.concatenate([
                self.title_weight * proto_title,
                self.content_weight * proto_content,
            ])
            projected = combined @ self.metric_transform
            projected_proto = combined_proto @ self.metric_transform
            dist = float(np.linalg.norm(projected - projected_proto))
            return dist

        title_dist = self._cosine_distance(title_vec, proto_title)
        content_dist = self._cosine_distance(content_vec, proto_content)
        return self.title_weight * title_dist + self.content_weight * content_dist

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """1 - cosine_similarity, bounded in [0, 2]."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 1.0
        return float(1.0 - np.dot(a, b) / (norm_a * norm_b))

    # -- classification ----------------------------------------------------

    def classify(
        self,
        section_title: str,
        section_content: str,
        doc_type: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Classify a section into the taxonomy.

        Args:
            section_title: Heading text.
            section_content: Body text.
            doc_type: Optional document type to restrict allowed kinds.

        Returns:
            ``(section_kind, confidence)`` tuple.
        """
        if not self.prototypes:
            return "misc", 0.0

        title_vec, content_vec = self._encode_section(section_title, section_content)

        # Determine allowed kinds
        allowed: Optional[Set[str]] = None
        if doc_type and doc_type in self.DOMAIN_FILTER:
            allowed = self.DOMAIN_FILTER[doc_type]

        distances: List[Tuple[str, float]] = []
        for kind, (proto_title, proto_content) in self.prototypes.items():
            if allowed is not None and kind not in allowed:
                continue
            dist = self._combined_distance(title_vec, content_vec, proto_title, proto_content)
            distances.append((kind, dist))

        if not distances:
            return "misc", 0.0

        distances.sort(key=lambda x: x[1])
        best_kind, best_dist = distances[0]

        # Confidence: margin between best and second-best
        if len(distances) >= 2:
            second_dist = distances[1][1]
            margin = second_dist - best_dist
            confidence = float(min(margin / max(best_dist, 1e-6), 1.0))
        else:
            confidence = 1.0 if best_dist < 0.5 else 0.5

        return best_kind, confidence

    # -- training ----------------------------------------------------------

    def fit_from_examples(self, examples: List[Dict[str, str]]) -> None:
        """Fit prototypes from labeled examples.

        Args:
            examples: List of dicts with keys ``title``, ``content``, ``kind``.
        """
        groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

        for ex in examples:
            kind = ex.get("kind", "misc")
            title = ex.get("title", "")
            content = ex.get("content", "")
            title_vec, content_vec = self._encode_section(title, content)
            groups[kind].append((title_vec, content_vec))

        self.prototypes = {}
        for kind, vecs in groups.items():
            title_vecs = np.stack([v[0] for v in vecs], axis=0)
            content_vecs = np.stack([v[1] for v in vecs], axis=0)

            title_centroid = title_vecs.mean(axis=0)
            content_centroid = content_vecs.mean(axis=0)

            # Normalise to unit length
            t_norm = np.linalg.norm(title_centroid)
            c_norm = np.linalg.norm(content_centroid)
            if t_norm > 1e-8:
                title_centroid /= t_norm
            if c_norm > 1e-8:
                content_centroid /= c_norm

            self.prototypes[kind] = (title_centroid, content_centroid)

        logger.info("SectionKindClassifier: fitted %d prototypes from %d examples",
                     len(self.prototypes), len(examples))

    def fit_metric_transform(
        self,
        examples: List[Dict[str, str]],
        projection_dim: int = 128,
    ) -> None:
        """Learn a metric transform using NCA for improved classification.

        Only call this when you have 50+ examples.

        Args:
            examples: Labeled examples (same format as ``fit_from_examples``).
            projection_dim: Output dimensionality of the learned projection.
        """
        if len(examples) < 20:
            logger.warning("Skipping NCA metric learning: only %d examples (need 50+)", len(examples))
            return

        from sklearn.neighbors import NeighborhoodComponentsAnalysis

        X_list: List[np.ndarray] = []
        y_list: List[str] = []

        for ex in examples:
            title_vec, content_vec = self._encode_section(
                ex.get("title", ""), ex.get("content", ""),
            )
            combined = np.concatenate([
                self.title_weight * title_vec,
                self.content_weight * content_vec,
            ])
            X_list.append(combined)
            y_list.append(ex.get("kind", "misc"))

        X = np.stack(X_list, axis=0)

        # NCA needs n_components <= min(n_features, n_samples - 1)
        n_components = min(projection_dim, X.shape[1], X.shape[0] - 1)
        if n_components < 2:
            logger.warning("Insufficient data for NCA (n_components=%d)", n_components)
            return

        nca = NeighborhoodComponentsAnalysis(n_components=n_components, max_iter=100)
        try:
            nca.fit(X, y_list)
            self.metric_transform = nca.components_.T.astype(np.float32)
            logger.info("NCA metric transform fitted: (%d, %d)", *self.metric_transform.shape)
        except Exception as exc:
            logger.warning("NCA fitting failed: %s", exc)
            self.metric_transform = None

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Save prototypes and metric transform.

        Args:
            path: File path.
        """
        data: Dict[str, Any] = {
            "title_weight": self.title_weight,
            "content_weight": self.content_weight,
            "metric_transform": self.metric_transform,
            "prototypes": {},
        }
        for kind, (tv, cv) in self.prototypes.items():
            data["prototypes"][kind] = (tv, cv)

        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    def load(self, path: str) -> None:
        """Load prototypes and metric transform.

        Args:
            path: File path.
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)  # noqa: S301

        self.title_weight = data.get("title_weight", 0.6)
        self.content_weight = data.get("content_weight", 0.4)
        self.metric_transform = data.get("metric_transform")
        self.prototypes = {}
        for kind, (tv, cv) in data.get("prototypes", {}).items():
            self.prototypes[kind] = (np.asarray(tv), np.asarray(cv))
