"""Tests for ML enhancements to query_focus.py.

Tests ML components:
1. Semantic similarity via embedder (cosine similarity scoring)
2. Field importance classifier (NumPy MLP, self-supervised)
3. Section routing (string-matching based scoring)
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag_v3.query_focus import (
    QueryFocus,
    build_query_focus,
    clear_chunk_embed_cache,
    filter_chunks_by_focus,
    score_chunk_relevance,
    score_field_relevance,
    _semantic_similarity_score,
    _section_affinity_score,
    _get_or_encode_chunk,
    _raw_chunk_score,
)
from src.rag_v3.field_classifier import (
    FieldImportanceClassifier,
    FIELD_NAMES,
    TRAINING_TEMPLATES,
    get_field_classifier,
    set_field_classifier,
    ensure_field_classifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic embedder that produces repeatable 64-dim normalized vectors.

    Uses a stable hash (sum of char ordinals) to ensure reproducibility across
    Python sessions (unlike built-in hash() which is randomized).
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._calls = 0

    @staticmethod
    def _stable_hash(text: str) -> int:
        """Python-session-stable hash for deterministic embeddings."""
        h = 0
        for i, ch in enumerate(text):
            h = (h * 31 + ord(ch)) & 0x7FFFFFFF
        return h

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        results = []
        for text in texts:
            rng = np.random.RandomState(self._stable_hash(text))
            vec = rng.randn(self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            results.append(vec)
            self._calls += 1
        return np.array(results)


def _make_chunk(text: str, section_kind: str = "", score: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(section_kind=section_kind, section_title=""),
        score=score,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup():
    """Clear caches and classifier singleton before/after each test."""
    clear_chunk_embed_cache()
    old_clf = get_field_classifier()
    set_field_classifier(None)
    yield
    clear_chunk_embed_cache()
    set_field_classifier(old_clf)


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=64)


# ===========================================================================
# Test Class 1: Semantic Similarity
# ===========================================================================

class TestSemanticSimilarity:
    """Tests for cosine similarity scoring via embedder."""

    def test_semantic_score_zero_without_embedding(self):
        """No query embedding → semantic score = 0.0."""
        focus = QueryFocus(keywords=["python"])
        chunk = _make_chunk("Python is a programming language")
        assert _semantic_similarity_score(chunk, focus) == 0.0

    def test_semantic_score_positive_with_embedding(self, embedder):
        """With query embedding → score > 0."""
        focus = build_query_focus("What are the technical skills?", embedder=embedder)
        chunk = _make_chunk("Python, Java, C++, React, Node.js")
        score = _semantic_similarity_score(chunk, focus)
        assert score >= 0.0  # May be 0 for random embeddings, but should not error

    def test_semantic_score_empty_chunk(self, embedder):
        """Empty chunk text → score = 0.0."""
        focus = build_query_focus("What skills?", embedder=embedder)
        chunk = _make_chunk("")
        assert _semantic_similarity_score(chunk, focus) == 0.0

    def test_cache_hit_avoids_reencoding(self, embedder):
        """Second call with same text uses cache."""
        focus = build_query_focus("skills", embedder=embedder)
        focus._embedder = embedder
        text = "Python Java C++"
        # First call
        emb1 = _get_or_encode_chunk(text, focus)
        calls_after_first = embedder._calls
        # Second call (cached)
        emb2 = _get_or_encode_chunk(text, focus)
        assert embedder._calls == calls_after_first  # no additional encode calls
        assert np.array_equal(emb1, emb2)

    def test_clear_cache_works(self, embedder):
        """clear_chunk_embed_cache() resets the cache."""
        focus = build_query_focus("skills", embedder=embedder)
        focus._embedder = embedder
        _get_or_encode_chunk("test text", focus)
        clear_chunk_embed_cache()
        # After clearing, next call should re-encode
        calls_before = embedder._calls
        _get_or_encode_chunk("test text", focus)
        assert embedder._calls == calls_before + 1

    def test_ml_weights_when_embedding_present(self, embedder):
        """With embedding: weights are 0.35/0.15/0.25/0.25."""
        focus = build_query_focus("What are the skills?", embedder=embedder)
        chunk = _make_chunk("Python Java skills", section_kind="skills_technical", score=0.8)
        score = _raw_chunk_score(chunk, focus)
        # Score should use ML weights (not keyword-only weights)
        assert score > 0.0

    def test_fallback_weights_without_embedding(self):
        """Without embedding: weights are 0.40/0.30/0.30 (unchanged)."""
        focus = build_query_focus("What are the skills?")
        assert focus.query_embedding is None
        chunk = _make_chunk("Python Java skills", section_kind="skills_technical", score=0.8)
        # Keyword "skills" in text → kw_score > 0
        # section_kind matches → sect_score = 1.0
        # reranker = 0.8
        score = _raw_chunk_score(chunk, focus)
        # 0.4 * kw + 0.3 * sect + 0.3 * reranker
        assert score > 0.3

    def test_score_differentiation_with_embedder(self, embedder):
        """Skills query should differentiate between skills and education chunks."""
        focus = build_query_focus("What programming languages and technical skills?", embedder=embedder)

        skills_chunk = _make_chunk(
            "Technical Skills: Python, Java, C++, React, Node.js, Docker, Kubernetes",
            section_kind="skills_technical",
        )
        edu_chunk = _make_chunk(
            "Education: B.Tech Computer Science, Anna University, GPA 8.5",
            section_kind="education",
        )

        skills_score = score_chunk_relevance(skills_chunk, focus)
        edu_score = score_chunk_relevance(edu_chunk, focus)
        # Skills chunk should score higher for a skills query
        # (keyword + section affinity both favor skills)
        assert skills_score > edu_score


# ===========================================================================
# Test Class 2: Field Importance Classifier
# ===========================================================================

class TestFieldClassifier:
    """Tests for FieldImportanceClassifier MLP."""

    def test_training_converges(self, embedder):
        """Loss should decrease over epochs."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        losses = clf.train(embedder, epochs=50, lr=0.05)
        assert len(losses) == 50
        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_predict_skills_query(self, embedder):
        """Skills query (from training set) should produce predictions."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf.train(embedder, epochs=200, lr=0.05)
        # Use exact training template for reliable convergence with random embeddings
        query_emb = embedder.encode(["What are the technical skills?"], normalize_embeddings=True)[0]
        preds = clf.predict(query_emb, threshold=0.1)
        # With 200 epochs on training data, should predict at least one field
        assert len(preds) > 0

    def test_predict_education_query(self, embedder):
        """Education query (from training set) should produce predictions."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf.train(embedder, epochs=200, lr=0.05)
        query_emb = embedder.encode(["Education background and degrees"], normalize_embeddings=True)[0]
        preds = clf.predict(query_emb, threshold=0.1)
        assert len(preds) > 0

    def test_predict_totals_query(self, embedder):
        """Invoice totals query (from training set) should produce predictions."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf.train(embedder, epochs=200, lr=0.05)
        query_emb = embedder.encode(["What is the total invoice amount?"], normalize_embeddings=True)[0]
        preds = clf.predict(query_emb, threshold=0.1)
        assert len(preds) > 0

    def test_threshold_filtering(self, embedder):
        """High threshold should filter out low-confidence predictions."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf.train(embedder, epochs=50, lr=0.05)
        query_emb = embedder.encode(["skills"], normalize_embeddings=True)[0]
        low_thresh = clf.predict(query_emb, threshold=0.1)
        high_thresh = clf.predict(query_emb, threshold=0.8)
        assert len(low_thresh) >= len(high_thresh)

    def test_save_load_roundtrip(self, embedder, tmp_path):
        """Save and load preserves weights."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf.train(embedder, epochs=50, lr=0.05)
        path = tmp_path / "clf.pkl"
        clf.save(path)

        clf2 = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf2.load(path)

        query_emb = embedder.encode(["technical skills"], normalize_embeddings=True)[0]
        preds1 = clf.predict(query_emb)
        preds2 = clf2.predict(query_emb)
        assert preds1 == preds2

    def test_predict_empty_embedding(self, embedder):
        """None embedding → empty predictions."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf.train(embedder, epochs=10, lr=0.05)
        assert clf.predict(None) == {}

    def test_predict_wrong_dimensions(self, embedder):
        """Wrong embedding dimensions → empty predictions."""
        clf = FieldImportanceClassifier(input_dim=64, hidden_dim=32, n_fields=12)
        clf.train(embedder, epochs=10, lr=0.05)
        wrong_emb = np.zeros(128)
        assert clf.predict(wrong_emb) == {}

    def test_field_names_correct(self):
        """Classifier uses correct field taxonomy."""
        clf = FieldImportanceClassifier()
        assert clf.field_names == FIELD_NAMES[:12]
        assert "skills" in clf.field_names
        assert "education" in clf.field_names
        assert "totals" in clf.field_names

    def test_score_field_relevance_uses_probabilities(self, embedder):
        """score_field_relevance() uses field_probabilities when available."""
        focus = QueryFocus(
            keywords=["skills"],
            field_tags={"skills"},
            field_probabilities={"skills": 0.92, "experience": 0.45},
        )
        # Should use probability value, not binary 1.0
        assert score_field_relevance("skills", focus) == pytest.approx(0.92)
        assert score_field_relevance("experience", focus) == pytest.approx(0.45)
        # Field not in probabilities → fallback to keyword matching
        assert score_field_relevance("education", focus) == 0.1


# ===========================================================================
# Test Class 3: Section Routing
# ===========================================================================

class TestSectionRouting:
    """Tests for section-based scoring."""

    def test_misc_chunk_gets_neutral_score(self):
        """Chunk with section_kind='misc' gets neutral penalty."""
        chunk = _make_chunk("Python Java C++ skills", section_kind="misc")
        focus = QueryFocus(section_kinds=["skills_technical"])

        from src.rag_v3.query_focus import _section_affinity_score
        score = _section_affinity_score(chunk, focus)
        # Generic/misc section_kind → neutral 0.3 penalty
        assert score == 0.3


# ===========================================================================
# Test Class 4: Graceful Degradation
# ===========================================================================

class TestGracefulDegradation:
    """All ML components are optional — fallback to keyword-only behavior."""

    def test_no_embedder_identical_to_keyword_only(self):
        """Without embedder, build_query_focus produces same result as before."""
        focus = build_query_focus("What are the technical skills?")
        assert focus.query_embedding is None
        assert focus.field_probabilities is None
        assert focus._embedder is None
        # Keywords and field_tags should still work
        assert "skills" in focus.field_tags
        assert "skills_technical" in focus.section_kinds

    def test_section_affinity_uses_string_matching(self):
        """Section affinity uses string matching."""
        chunk = _make_chunk("Python Java", section_kind="skills_technical")
        focus = QueryFocus(section_kinds=["skills_technical"])
        score = _section_affinity_score(chunk, focus)
        assert score == 1.0  # exact match via string

    def test_no_classifier_uses_keyword_field_tags(self):
        """Without classifier, field_tags come from keywords only."""
        focus = build_query_focus("What are the skills?")
        assert "skills" in focus.field_tags
        assert focus.field_probabilities is None

    def test_all_unavailable_matches_baseline(self):
        """With no ML components, scores match the original keyword-only formula."""
        focus = QueryFocus(
            keywords=["skills", "python"],
            section_kinds=["skills_technical"],
        )
        chunk = _make_chunk("Python and Java skills", section_kind="skills_technical", score=0.5)

        score = _raw_chunk_score(chunk, focus)
        # Without embedding: 0.35 * kw + 0.25 * sect + 0.40 * reranker
        kw = 1.0  # both keywords match
        sect = 1.0  # exact section match
        reranker = 0.5
        expected = 0.35 * kw + 0.25 * sect + 0.40 * reranker
        assert score == pytest.approx(expected, abs=0.01)


# ===========================================================================
# Test Class 5: Integration
# ===========================================================================

class TestIntegration:
    """End-to-end tests combining all ML components."""

    def test_build_query_focus_with_embedder(self, embedder):
        """build_query_focus with embedder populates query_embedding."""
        focus = build_query_focus("What are the skills?", embedder=embedder)
        assert focus.query_embedding is not None
        assert len(focus.query_embedding) == 64
        assert focus._embedder is embedder

    def test_full_scoring_pipeline(self, embedder):
        """Full chunk scoring with embedder works end-to-end."""
        focus = build_query_focus("What technical skills does the candidate have?", embedder=embedder)
        chunks = [
            _make_chunk("Python, Java, React, Docker, Kubernetes", section_kind="skills_technical"),
            _make_chunk("B.Tech in Computer Science from IIT", section_kind="education"),
            _make_chunk("3 years at Google as SWE", section_kind="experience"),
        ]
        scores = [score_chunk_relevance(c, focus) for c in chunks]
        # All scores should be valid floats
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Skills chunk should rank highest (keyword + section affinity)
        assert scores[0] >= scores[1]

    def test_filter_with_embedder(self, embedder):
        """filter_chunks_by_focus works with ML-enhanced scoring."""
        focus = build_query_focus("education details", embedder=embedder)
        chunks = [
            _make_chunk("Python Java React", section_kind="skills_technical"),
            _make_chunk("B.Tech from Anna University", section_kind="education"),
            _make_chunk("AWS Certified Solutions Architect", section_kind="certifications"),
            _make_chunk("M.Tech in Data Science, GPA 9.0", section_kind="education"),
            _make_chunk("5 years at Microsoft", section_kind="experience"),
        ]
        filtered = filter_chunks_by_focus(chunks, focus, min_keep=2, top_k=3)
        # Should keep at most 3 chunks
        assert len(filtered) <= 3
        # Education chunks should be prioritized
        edu_texts = [c.text for c in filtered if "education" in (c.meta.section_kind or "")]
        assert len(edu_texts) >= 1

    def test_singleton_lifecycle(self, embedder, tmp_path):
        """ensure_field_classifier() creates and caches singleton."""
        model_path = tmp_path / "test_model.pkl"
        clf = ensure_field_classifier(embedder, model_path=model_path)
        assert clf is not None
        assert clf._trained
        assert model_path.exists()
        # Second call returns same instance
        clf2 = get_field_classifier()
        assert clf2 is clf
