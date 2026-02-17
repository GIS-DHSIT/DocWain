"""
DPIE integration layer for DocWain.

Provides the ``DPIERegistry`` singleton that manages all DPIE ML models
and exposes drop-in replacement functions for existing regex-based
DocWain pipeline calls.

No regex anywhere in this module.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DPIERegistry:
    """Singleton registry for all DPIE ML models.

    Lazy-loads models on first access.  Auto-trains from Qdrant data
    if saved models do not exist on disk.
    """

    _instance: ClassVar[Optional["DPIERegistry"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._loaded = False
        self._model_dir: str = ""

        # Model references (populated by load / train_and_save)
        self._char_vocab: Optional[Any] = None
        self._line_encoder: Optional[Any] = None
        self._doc_classifier: Optional[Any] = None
        self._section_detector: Optional[Any] = None
        self._section_kind_classifier: Optional[Any] = None
        self._entity_recognizer: Optional[Any] = None
        self._sentence_model: Optional[Any] = None

        # Staleness tracking
        self._last_trained_at: float = 0.0
        self._last_trained_point_count: int = 0
        self._training_in_progress: bool = False

    @classmethod
    def get(cls) -> "DPIERegistry":
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Whether models have been loaded and are ready for inference."""
        return self._loaded

    # -- loading -----------------------------------------------------------

    def load(self, model_dir: str, sentence_model: Any) -> None:
        """Load all saved DPIE models from disk.

        Args:
            model_dir: Directory containing saved model files.
            sentence_model: SentenceTransformer instance for embeddings.

        Raises:
            FileNotFoundError: If any required model file is missing.
        """
        from .ml.line_encoder import CharNGramVocab, LineFeatureEncoder
        from .ml.doc_classifier import DocumentTypeClassifier
        from .ml.section_detector import SectionBoundaryDetector
        from .ml.section_kind_classifier import SectionKindClassifier
        from .ml.entity_recognizer import EntityPatternRecognizer

        self._sentence_model = sentence_model
        self._model_dir = model_dir

        required_files = [
            "char_vocab.pkl",
            "doc_classifier.npz",
            "section_detector.npz",
            "section_kind.pkl",
            "entity_recognizer.npz",
        ]
        for fname in required_files:
            fpath = os.path.join(model_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"DPIE model file missing: {fpath}. "
                    f"Run train_and_save() first or ensure all model files exist in {model_dir}."
                )

        # Load char vocab
        self._char_vocab = CharNGramVocab()
        self._char_vocab.load(os.path.join(model_dir, "char_vocab.pkl"))

        # Load line encoder
        self._line_encoder = LineFeatureEncoder(sentence_model, self._char_vocab)
        self._line_encoder.load(model_dir)

        # Load document classifier
        self._doc_classifier = DocumentTypeClassifier()
        self._doc_classifier.load(os.path.join(model_dir, "doc_classifier.npz"))

        # Load section detector
        self._section_detector = SectionBoundaryDetector()
        self._section_detector.load(os.path.join(model_dir, "section_detector.npz"))

        # Load section kind classifier
        self._section_kind_classifier = SectionKindClassifier(sentence_model)
        self._section_kind_classifier.load(os.path.join(model_dir, "section_kind.pkl"))

        # Load entity recognizer
        self._entity_recognizer = EntityPatternRecognizer(sentence_model)
        self._entity_recognizer.load(os.path.join(model_dir, "entity_recognizer.npz"))

        self._loaded = True
        logger.info("DPIE models loaded from %s", model_dir)

    # -- training ----------------------------------------------------------

    def train_and_save(
        self,
        qdrant_client: Any,
        sentence_model: Any,
        collection_name: str,
        subscription_id: str,
        profile_id: str,
    ) -> Dict[str, Any]:
        """Bootstrap training data from Qdrant and train all DPIE models.

        Args:
            qdrant_client: Qdrant client instance.
            sentence_model: SentenceTransformer instance.
            collection_name: Qdrant collection name.
            subscription_id: Subscription scope.
            profile_id: Profile scope.

        Returns:
            Training statistics (loss curves, data sizes).
        """
        from .ml.line_encoder import CharNGramVocab, LineFeatureEncoder
        from .ml.doc_classifier import DocumentTypeClassifier
        from .ml.section_detector import SectionBoundaryDetector
        from .ml.section_kind_classifier import SectionKindClassifier
        from .ml.entity_recognizer import EntityPatternRecognizer
        from .ml.training_bootstrap import TrainingBootstrap

        self._sentence_model = sentence_model
        model_dir = os.path.join("models", "dpie", subscription_id)
        os.makedirs(model_dir, exist_ok=True)
        self._model_dir = model_dir

        stats: Dict[str, Any] = {}

        # 1. Bootstrap training data
        bootstrap = TrainingBootstrap(qdrant_client, sentence_model, collection_name, subscription_id=subscription_id)
        data = bootstrap.generate_all(profile_id)

        doc_features, doc_labels = data["doc_type"]
        boundary_features, boundary_labels = data["section_boundary"]
        section_kind_examples = data["section_kind"]
        entity_examples = data["entity"]

        stats["data_sizes"] = {
            "doc_type_docs": len(doc_features),
            "boundary_docs": len(boundary_features),
            "section_kind_examples": len(section_kind_examples),
            "entity_examples": len(entity_examples),
        }

        # 2. Train CharNGramVocab
        char_vocab = CharNGramVocab()
        all_lines: List[str] = []
        all_line_labels: List[str] = []
        for feats, label in zip(doc_features, doc_labels):
            # Use line count as proxy: each line gets the doc label
            n_lines = feats.shape[0]
            all_lines.extend(["sample_line"] * n_lines)  # placeholder
            all_line_labels.extend([label] * n_lines)

        # We need actual line text for char n-grams.
        # Re-extract from documents via bootstrap
        line_texts: List[str] = []
        line_text_labels: List[str] = []
        for point_list, label in self._iter_doc_lines(bootstrap, profile_id, doc_labels):
            for line_text in point_list:
                line_texts.append(line_text)
                line_text_labels.append(label)

        if line_texts:
            char_vocab.fit(line_texts, line_text_labels)
        char_vocab.save(os.path.join(model_dir, "char_vocab.pkl"))
        self._char_vocab = char_vocab

        # 3. Fit PCA for LineFeatureEncoder
        line_encoder = LineFeatureEncoder(sentence_model, char_vocab)
        if line_texts:
            sample = line_texts[:min(500, len(line_texts))]
            line_encoder.fit_pca(sample)
        line_encoder.save(model_dir)
        self._line_encoder = line_encoder

        # 4. Train DocumentTypeClassifier
        actual_dim = doc_features[0].shape[1] if doc_features else line_encoder.total_dim
        doc_classifier = DocumentTypeClassifier(feature_dim=actual_dim)
        if doc_features and len(doc_features) >= 2:
            train_stats = doc_classifier.fit(doc_features, doc_labels, epochs=30)
            stats["doc_classifier"] = train_stats
        doc_classifier.save(os.path.join(model_dir, "doc_classifier.npz"))
        self._doc_classifier = doc_classifier

        # 5. Train SectionBoundaryDetector
        section_detector = SectionBoundaryDetector(feature_dim=actual_dim)
        if boundary_features and len(boundary_features) >= 2:
            train_stats = section_detector.fit(boundary_features, boundary_labels, epochs=20)
            stats["section_detector"] = train_stats
        section_detector.save(os.path.join(model_dir, "section_detector.npz"))
        self._section_detector = section_detector

        # 6. Train SectionKindClassifier
        section_kind_cls = SectionKindClassifier(sentence_model)
        if section_kind_examples:
            section_kind_cls.fit_from_examples(section_kind_examples)
            if len(section_kind_examples) >= 50:
                section_kind_cls.fit_metric_transform(section_kind_examples)
        section_kind_cls.save(os.path.join(model_dir, "section_kind.pkl"))
        self._section_kind_classifier = section_kind_cls

        # 7. Train EntityPatternRecognizer
        entity_recognizer = EntityPatternRecognizer(sentence_model)
        if entity_examples and len(entity_examples) >= 5:
            train_stats = entity_recognizer.fit(entity_examples, epochs=20)
            stats["entity_recognizer"] = train_stats
        entity_recognizer.save(os.path.join(model_dir, "entity_recognizer.npz"))
        self._entity_recognizer = entity_recognizer

        self._loaded = True
        self._last_trained_at = time.time()
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            count_result = qdrant_client.count(
                collection_name,
                count_filter=Filter(must=[
                    FieldCondition(key="profile_id", match=MatchValue(value=profile_id)),
                ]),
                exact=False,
            )
            self._last_trained_point_count = count_result.count
        except Exception:
            pass
        logger.info("DPIE models trained and saved to %s", model_dir)
        return stats

    def needs_retrain(
        self,
        qdrant_client: Any,
        collection_name: str,
        profile_id: str,
    ) -> bool:
        """Check if models need retraining based on data growth.

        Returns True if data has grown by 20%+ since last training and
        at least 5 minutes have elapsed.
        """
        if not self._loaded or self._training_in_progress:
            return False
        # Minimum 5-minute cooldown between retrains
        if time.time() - self._last_trained_at < 300:
            return False
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            count_result = qdrant_client.count(
                collection_name,
                count_filter=Filter(must=[
                    FieldCondition(key="profile_id", match=MatchValue(value=profile_id)),
                ]),
                exact=False,
            )
            current_count = count_result.count
        except Exception:
            return False
        if self._last_trained_point_count == 0:
            return False
        growth = (current_count - self._last_trained_point_count) / max(self._last_trained_point_count, 1)
        return growth >= 0.20

    def retrain_async(
        self,
        qdrant_client: Any,
        sentence_model: Any,
        collection_name: str,
        subscription_id: str,
        profile_id: str,
    ) -> None:
        """Retrain DPIE models in a background thread if stale."""
        if self._training_in_progress:
            return

        def _worker() -> None:
            try:
                self._training_in_progress = True
                self.train_and_save(
                    qdrant_client, sentence_model,
                    collection_name, subscription_id, profile_id,
                )
            except Exception as exc:
                logger.warning("DPIE async retrain failed: %s", exc)
            finally:
                self._training_in_progress = False

        thread = threading.Thread(target=_worker, daemon=True, name="dpie-retrain")
        thread.start()

    def ensure_ready(
        self,
        qdrant_client: Any,
        sentence_model: Any,
        collection_name: str,
        subscription_id: str,
        profile_id: str,
    ) -> None:
        """Ensure models are loaded, training if necessary.

        Thread-safe: only one thread will train at a time.

        Args:
            qdrant_client: Qdrant client instance.
            sentence_model: SentenceTransformer instance.
            collection_name: Qdrant collection name.
            subscription_id: Subscription scope.
            profile_id: Profile scope.
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            model_dir = os.path.join("models", "dpie", subscription_id)
            try:
                self.load(model_dir, sentence_model)
                return
            except FileNotFoundError:
                logger.info("DPIE models not found at %s; training...", model_dir)

            self.train_and_save(
                qdrant_client, sentence_model,
                collection_name, subscription_id, profile_id,
            )

    # -- convenience methods -----------------------------------------------

    def classify_document(self, text: str) -> Tuple[str, float]:
        """Classify document type from text.

        Args:
            text: Document text.

        Returns:
            ``(doc_type, confidence)``
        """
        if not self._loaded or self._line_encoder is None or self._doc_classifier is None:
            return "other", 0.0

        line_features, _ = self._line_encoder.encode_document(text)
        if line_features.shape[0] == 0:
            return "other", 0.0

        doc_type, confidence, _ = self._doc_classifier.predict(line_features)
        return doc_type, confidence

    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect section boundaries in text.

        Args:
            text: Document text.

        Returns:
            List of section dicts with ``start_line``, ``end_line``,
            ``heading``, ``confidence``.
        """
        if not self._loaded or self._line_encoder is None or self._section_detector is None:
            return []

        line_features, lines = self._line_encoder.encode_document(text)
        if line_features.shape[0] == 0:
            return []

        predictions = self._section_detector.predict_boundaries(line_features, lines)

        # Convert raw predictions to section spans
        sections: List[Dict[str, Any]] = []
        boundary_indices = [p["line_index"] for p in predictions if p["is_boundary"]]

        for i, start in enumerate(boundary_indices):
            end = boundary_indices[i + 1] - 1 if i + 1 < len(boundary_indices) else len(lines) - 1
            heading = lines[start].strip() if start < len(lines) else ""
            conf = next(
                (p["confidence"] for p in predictions if p["line_index"] == start),
                0.0,
            )
            sections.append({
                "start_line": start,
                "end_line": end,
                "heading": heading,
                "confidence": conf,
            })

        return sections

    def classify_section_kind(
        self,
        title: str,
        content: str,
        doc_type: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Classify a section into the section kind taxonomy.

        Args:
            title: Section heading.
            content: Section body text.
            doc_type: Optional document type for domain filtering.

        Returns:
            ``(section_kind, confidence)``
        """
        if not self._loaded or self._section_kind_classifier is None:
            return "misc", 0.0

        return self._section_kind_classifier.classify(title, content, doc_type)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Input text.

        Returns:
            List of entity dicts with ``entity``, ``type``, ``start``,
            ``end``, ``confidence``.
        """
        if not self._loaded or self._entity_recognizer is None:
            return []

        return self._entity_recognizer.extract(text)

    def detect_person_names(self, text: str) -> List[str]:
        """Extract only person names from text.

        Drop-in replacement for ``_detect_person_name()`` in dw_newron.py.

        Args:
            text: Input text.

        Returns:
            List of person name strings.
        """
        if not self._loaded or self._entity_recognizer is None:
            return []

        return self._entity_recognizer.extract_persons(text)

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _iter_doc_lines(
        bootstrap: Any,
        profile_id: str,
        labels: List[str],
    ) -> List[Tuple[List[str], str]]:
        """Re-extract line texts from documents for char-n-gram training."""
        docs = bootstrap._get_documents_for_profile(profile_id)
        result: List[Tuple[List[str], str]] = []

        for i, (doc_id, chunks) in enumerate(docs.items()):
            if i >= len(labels):
                break
            text = bootstrap._reconstruct_text(chunks)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if lines:
                result.append((lines, labels[i]))

        return result


# ── Drop-in replacement functions ─────────────────────────────────────


def dpie_classify_document_type(
    text_sample: str,
    tables_sample: str,
    filename: str,
    model_name: Optional[str] = None,
) -> Tuple[str, float]:
    """Drop-in replacement for ``classify_document_type()`` in identify.py.

    Uses DPIE if trained; falls back to existing heuristic if not.

    Args:
        text_sample: Representative text from the document.
        tables_sample: Table text (unused by DPIE, kept for compatibility).
        filename: Source filename.
        model_name: LLM model name (unused by DPIE).

    Returns:
        ``(document_type, confidence)``
    """
    registry = DPIERegistry.get()
    if registry.is_loaded:
        doc_type, confidence = registry.classify_document(text_sample)
        if confidence > 0.6:
            return doc_type, confidence

    # Fallback to existing function
    try:
        from src.doc_understanding.identify import classify_document_type as _fallback
        return _fallback(text_sample, tables_sample, filename, model_name)
    except ImportError:
        return "other", 0.5


def dpie_detect_person_name(query: str) -> Optional[str]:
    """Drop-in replacement for ``_detect_person_name()`` in dw_newron.py.

    Uses DPIE if trained; returns ``None`` if not.

    Args:
        query: Input query text.

    Returns:
        First person name found, or ``None``.
    """
    registry = DPIERegistry.get()
    if registry.is_loaded:
        persons = registry.detect_person_names(query)
        if persons:
            return persons[0]
    return None


def dpie_extract_entities(text: str) -> List[Dict[str, Any]]:
    """Drop-in replacement for ``EntityExtractor.extract()`` in kg/entity_extractor.py.

    Returns entity dicts compatible with the existing KG pipeline.

    Args:
        text: Input text.

    Returns:
        List of entity dicts.
    """
    registry = DPIERegistry.get()
    if registry.is_loaded:
        return registry.extract_entities(text)

    # Fallback to existing extractor
    try:
        from src.kg.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        results = extractor.extract(text)
        return [{"entity": e.name, "type": e.type, "confidence": 1.0} for e in results]
    except ImportError:
        return []
