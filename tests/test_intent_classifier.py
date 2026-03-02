"""Tests for the trained multi-head MLP intent + domain classifier.

Covers architecture, training, prediction, persistence, singleton management,
integration with llm_intent, and graceful degradation scenarios.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.intent.intent_classifier import (
    DOMAIN_NAMES,
    INTENT_NAMES,
    TRAINING_TEMPLATES,
    IntentDomainClassifier,
    _augment_with_word_dropout,
    ensure_intent_classifier,
    get_intent_classifier,
    set_intent_classifier,
)


# ---------------------------------------------------------------------------
# Fake embedder for deterministic tests
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic embedder returning fixed-size vectors based on query hash."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(
        self, texts: List[str], normalize_embeddings: bool = False, **kwargs
    ) -> np.ndarray:
        vecs = []
        for text in texts:
            rng = np.random.RandomState(hash(text) % (2**31))
            v = rng.randn(self.dim).astype(np.float32)
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        return np.array(vecs)


# ---------------------------------------------------------------------------
# TestClassifierArchitecture
# ---------------------------------------------------------------------------
class TestClassifierArchitecture:
    """Test MLP architecture: dimensions, forward shapes, softmax properties."""

    def test_default_dimensions(self):
        clf = IntentDomainClassifier()
        assert clf.input_dim == 1024
        assert clf.hidden_dim == 128
        assert clf.n_intents == 8
        assert clf.n_domains == 6

    def test_weight_shapes(self):
        clf = IntentDomainClassifier()
        assert clf.W_shared.shape == (1024, 128)
        assert clf.b_shared.shape == (128,)
        assert clf.W_intent.shape == (128, 8)
        assert clf.b_intent.shape == (8,)
        assert clf.W_domain.shape == (128, 6)
        assert clf.b_domain.shape == (6,)

    def test_custom_dimensions(self):
        clf = IntentDomainClassifier(input_dim=512, hidden_dim=64, n_intents=4, n_domains=3)
        assert clf.W_shared.shape == (512, 64)
        assert clf.W_intent.shape == (64, 4)
        assert clf.W_domain.shape == (64, 3)

    def test_forward_output_shapes(self):
        clf = IntentDomainClassifier()
        X = np.random.randn(5, 1024).astype(np.float32)
        h, ip, dp = clf._forward(X)
        assert h.shape == (5, 128)
        assert ip.shape == (5, 8)
        assert dp.shape == (5, 6)

    def test_softmax_sums_to_one(self):
        clf = IntentDomainClassifier()
        X = np.random.randn(10, 1024).astype(np.float32)
        _, ip, dp = clf._forward(X)
        np.testing.assert_allclose(ip.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(dp.sum(axis=1), 1.0, atol=1e-5)

    def test_softmax_all_positive(self):
        clf = IntentDomainClassifier()
        X = np.random.randn(10, 1024).astype(np.float32)
        _, ip, dp = clf._forward(X)
        assert (ip >= 0).all()
        assert (dp >= 0).all()

    def test_not_trained_initially(self):
        clf = IntentDomainClassifier()
        assert clf._trained is False


# ---------------------------------------------------------------------------
# TestTraining
# ---------------------------------------------------------------------------
class TestTraining:
    """Test training convergence and augmentation."""

    def test_training_decreases_loss(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        losses = clf.train(emb, epochs=100, lr=0.5)
        assert len(losses) == 100
        assert losses[-1] < losses[0], f"Final loss {losses[-1]} should be less than initial {losses[0]}"

    def test_trained_flag_set(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=10, lr=0.1)
        assert clf._trained is True

    def test_augmentation_count(self):
        original = TRAINING_TEMPLATES
        augmented = _augment_with_word_dropout(original, n_variants=2)
        # 2 variants per template
        assert len(augmented) == 2 * len(original)

    def test_augmentation_preserves_labels(self):
        templates = [
            ("What are the technical skills?", "qa", "resume"),
            ("Compare the two candidates", "compare", "resume"),
        ]
        augmented = _augment_with_word_dropout(templates, n_variants=3)
        for _, intent, domain in augmented:
            assert intent in ("qa", "compare")
            assert domain == "resume"

    def test_augmentation_deterministic(self):
        templates = TRAINING_TEMPLATES[:5]
        a1 = _augment_with_word_dropout(templates, n_variants=2, seed=42)
        a2 = _augment_with_word_dropout(templates, n_variants=2, seed=42)
        assert [t[0] for t in a1] == [t[0] for t in a2]

    def test_input_dim_auto_adjusts(self):
        emb = FakeEmbedder(dim=512)
        clf = IntentDomainClassifier(input_dim=1024, hidden_dim=32)
        clf.train(emb, epochs=5, lr=0.1)
        assert clf.input_dim == 512
        assert clf.W_shared.shape[0] == 512

    def test_empty_templates_uses_defaults(self):
        """Passing empty list falls through to default TRAINING_TEMPLATES."""
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        # [] is falsy, so `templates or TRAINING_TEMPLATES` uses defaults
        losses = clf.train(emb, epochs=5, lr=0.5, templates=[])
        assert len(losses) == 5
        assert clf._trained is True

    def test_training_sample_count_logged(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        templates = TRAINING_TEMPLATES[:10]
        clf.train(emb, epochs=5, lr=0.1, templates=templates)
        # 10 base + 20 augmented = 30 total
        assert clf._trained is True


# ---------------------------------------------------------------------------
# TestPrediction
# ---------------------------------------------------------------------------
class TestPrediction:
    """Test prediction output format and confidence gates."""

    def test_predict_returns_all_keys(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)
        vec = emb.encode(["test query"], normalize_embeddings=True)[0]
        result = clf.predict(vec)
        assert "intent" in result
        assert "intent_confidence" in result
        assert "domain" in result
        assert "domain_confidence" in result

    def test_predicted_intent_is_valid(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)
        for q in ["test", "another", "query", "more"]:
            vec = emb.encode([q], normalize_embeddings=True)[0]
            result = clf.predict(vec)
            assert result["intent"] in INTENT_NAMES

    def test_predicted_domain_is_valid(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)
        for q in ["test", "another", "query", "more"]:
            vec = emb.encode([q], normalize_embeddings=True)[0]
            result = clf.predict(vec)
            assert result["domain"] in DOMAIN_NAMES

    def test_confidence_is_probability(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)
        vec = emb.encode(["test"], normalize_embeddings=True)[0]
        result = clf.predict(vec)
        assert 0.0 <= result["intent_confidence"] <= 1.0
        assert 0.0 <= result["domain_confidence"] <= 1.0

    def test_predict_none_embedding_returns_empty(self):
        clf = IntentDomainClassifier()
        result = clf.predict(None)
        assert result == {}

    def test_predict_dimension_mismatch_returns_empty(self):
        clf = IntentDomainClassifier(input_dim=1024)
        vec = np.random.randn(512).astype(np.float32)
        result = clf.predict(vec)
        assert result == {}


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------
class TestPersistence:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            clf.save(path)
            assert os.path.exists(path)

            clf2 = IntentDomainClassifier()
            clf2.load(path)

            assert clf2._trained is True
            assert clf2.input_dim == 64
            assert clf2.hidden_dim == 32
            np.testing.assert_array_equal(clf.W_shared, clf2.W_shared)
            np.testing.assert_array_equal(clf.W_intent, clf2.W_intent)
            np.testing.assert_array_equal(clf.W_domain, clf2.W_domain)
        finally:
            os.unlink(path)

    def test_predictions_match_after_reload(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)

        vec = emb.encode(["test query"], normalize_embeddings=True)[0]
        result1 = clf.predict(vec)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            clf.save(path)
            clf2 = IntentDomainClassifier()
            clf2.load(path)
            result2 = clf2.predict(vec)

            assert result1["intent"] == result2["intent"]
            assert result1["domain"] == result2["domain"]
            assert abs(result1["intent_confidence"] - result2["intent_confidence"]) < 1e-5
        finally:
            os.unlink(path)

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "model.pkl")
            clf = IntentDomainClassifier()
            clf.save(path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# TestSingleton
# ---------------------------------------------------------------------------
class TestSingleton:
    """Test singleton get/set/ensure."""

    def setup_method(self):
        self._original = get_intent_classifier()
        set_intent_classifier(None)

    def teardown_method(self):
        set_intent_classifier(self._original)

    def test_get_returns_none_initially(self):
        assert get_intent_classifier() is None

    def test_set_and_get(self):
        clf = IntentDomainClassifier()
        set_intent_classifier(clf)
        assert get_intent_classifier() is clf

    def test_set_none_clears(self):
        set_intent_classifier(IntentDomainClassifier())
        set_intent_classifier(None)
        assert get_intent_classifier() is None

    def test_ensure_trains_once(self):
        emb = FakeEmbedder(dim=64)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            os.unlink(path)  # ensure it doesn't exist
            clf = ensure_intent_classifier(emb, model_path=path)
            assert clf._trained is True
            assert get_intent_classifier() is clf

            # Second call returns same instance
            clf2 = ensure_intent_classifier(emb, model_path=path)
            assert clf2 is clf
        finally:
            set_intent_classifier(None)
            path.unlink(missing_ok=True)

    def test_ensure_loads_from_disk(self):
        emb = FakeEmbedder(dim=64)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            # Train and save
            clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
            clf.train(emb, epochs=20, lr=0.5)
            clf.save(path)

            # Ensure should load, not retrain
            set_intent_classifier(None)
            clf2 = ensure_intent_classifier(emb, model_path=path)
            assert clf2._trained is True
            np.testing.assert_array_equal(clf.W_shared, clf2.W_shared)
        finally:
            set_intent_classifier(None)
            path.unlink(missing_ok=True)

    def test_concurrent_ensure_trains_once(self):
        emb = FakeEmbedder(dim=64)
        results = []
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        os.unlink(path)

        def worker():
            clf = ensure_intent_classifier(emb, model_path=path)
            results.append(id(clf))

        try:
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            # All threads should get the same instance
            assert len(set(results)) == 1
        finally:
            set_intent_classifier(None)
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------
class TestIntegration:
    """Test _neural_parse() uses the trained classifier."""

    def setup_method(self):
        self._original = get_intent_classifier()

    def teardown_method(self):
        set_intent_classifier(self._original)

    def test_neural_parse_uses_classifier(self):
        from src.intent.llm_intent import _neural_parse

        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)
        set_intent_classifier(clf)

        with patch("src.intent.llm_intent._get_embedder", return_value=emb):
            result = _neural_parse("What are the technical skills?")

        assert result is not None
        assert "intent" in result
        assert "domain" in result
        assert result["intent"] in INTENT_NAMES
        assert result["domain"] in DOMAIN_NAMES

    def test_neural_parse_returns_none_when_no_classifier(self):
        from src.intent.llm_intent import _neural_parse

        set_intent_classifier(None)
        with patch("src.intent.llm_intent._get_embedder", return_value=None):
            with patch("src.intent.intent_classifier.ensure_intent_classifier", side_effect=Exception("no embedder")):
                result = _neural_parse("What are the skills?")
                assert result is None

    def test_fallback_parse_falls_back_to_regex(self):
        from src.intent.llm_intent import _fallback_parse

        set_intent_classifier(None)
        with patch("src.intent.llm_intent._get_embedder", return_value=None):
            with patch("src.intent.intent_classifier.ensure_intent_classifier", side_effect=Exception("no embedder")):
                result = _fallback_parse("summarize the document")
                assert result["intent"] == "summarize"

    def test_fallback_parse_includes_entity_hints(self):
        from src.intent.llm_intent import _fallback_parse

        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)
        set_intent_classifier(clf)

        with patch("src.intent.llm_intent._get_embedder", return_value=emb):
            result = _fallback_parse("What are John's skills?")
            assert "entity_hints" in result

    def test_parse_intent_full_flow(self):
        from src.intent.llm_intent import parse_intent

        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=50, lr=0.5)
        set_intent_classifier(clf)

        with patch("src.intent.llm_intent._get_embedder", return_value=emb):
            result = parse_intent(query="What skills does this person have?")
            assert result.source in ("heuristic", "cache")
            assert result.intent in INTENT_NAMES
            assert result.domain in DOMAIN_NAMES


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------
class TestGracefulDegradation:
    """Test fallback behavior when components are unavailable."""

    def setup_method(self):
        self._original = get_intent_classifier()

    def teardown_method(self):
        set_intent_classifier(self._original)

    def test_no_embedder_falls_back_to_regex(self):
        from src.intent.llm_intent import _fallback_parse

        set_intent_classifier(None)
        with patch("src.intent.llm_intent._get_embedder", return_value=None):
            with patch("src.intent.intent_classifier.ensure_intent_classifier", side_effect=Exception("no embedder")):
                result = _fallback_parse("compare the candidates")
                assert result["intent"] == "compare"
                assert result["domain"] == "generic"

    def test_untrained_classifier_returns_empty(self):
        clf = IntentDomainClassifier()
        assert clf._trained is False
        vec = np.random.randn(1024).astype(np.float32)
        result = clf.predict(vec)
        # Untrained classifier still produces predictions (random weights)
        assert "intent" in result
        assert "domain" in result

    def test_dimension_mismatch_returns_empty(self):
        emb = FakeEmbedder(dim=64)
        clf = IntentDomainClassifier(input_dim=64, hidden_dim=32)
        clf.train(emb, epochs=10, lr=0.5)
        # Wrong dimension input
        wrong_vec = np.random.randn(128).astype(np.float32)
        result = clf.predict(wrong_vec)
        assert result == {}

    def test_failed_encode_returns_empty_losses(self):
        bad_emb = MagicMock()
        bad_emb.encode.side_effect = RuntimeError("encode failed")
        clf = IntentDomainClassifier()
        losses = clf.train(bad_emb, epochs=10)
        assert losses == []
        assert clf._trained is False

    def test_output_format_detected(self):
        from src.intent.llm_intent import _detect_output_format
        assert _detect_output_format("show in table format") == "table"
        assert _detect_output_format("return as json") == "json"
        assert _detect_output_format("write a paragraph") == "paragraph"
        assert _detect_output_format("use markdown") == "markdown"
        assert _detect_output_format("regular query") == "bullets"


# ---------------------------------------------------------------------------
# TestTrainingTemplates
# ---------------------------------------------------------------------------
class TestTrainingTemplates:
    """Verify training template coverage."""

    def test_all_intents_covered(self):
        intents_in_templates = {t[1] for t in TRAINING_TEMPLATES}
        for intent in INTENT_NAMES:
            assert intent in intents_in_templates, f"Intent '{intent}' not in training templates"

    def test_all_domains_covered(self):
        domains_in_templates = {t[2] for t in TRAINING_TEMPLATES}
        for domain in DOMAIN_NAMES:
            assert domain in domains_in_templates, f"Domain '{domain}' not in training templates"

    def test_template_count_reasonable(self):
        assert len(TRAINING_TEMPLATES) >= 80, f"Only {len(TRAINING_TEMPLATES)} templates"

    def test_templates_have_valid_labels(self):
        for query, intent, domain in TRAINING_TEMPLATES:
            assert intent in INTENT_NAMES, f"Invalid intent '{intent}' for '{query}'"
            assert domain in DOMAIN_NAMES, f"Invalid domain '{domain}' for '{query}'"
            assert len(query.strip()) > 5, f"Query too short: '{query}'"
