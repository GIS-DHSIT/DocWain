"""Tests for the hybrid dense + sparse embedding module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.docwain_intel.hybrid_embedding import (
    HybridEmbedder,
    HybridVector,
    _CTRL_CHAR_RE,
    _MAX_TOKENS,
    _preprocess,
    _text_hash,
)


# ---------------------------------------------------------------------------
# 1. HybridVector Pydantic model
# ---------------------------------------------------------------------------

class TestHybridVectorModel:
    def test_default_construction(self):
        vec = HybridVector()
        assert vec.dense == []
        assert vec.sparse == {}
        assert vec.text_hash == ""
        assert vec.model == ""

    def test_populated_construction(self):
        vec = HybridVector(
            dense=[0.1, 0.2, 0.3],
            sparse={"hello": 1.5, "world": 0.8},
            text_hash="abc123",
            model="bge-m3",
        )
        assert len(vec.dense) == 3
        assert vec.sparse["hello"] == 1.5
        assert vec.model == "bge-m3"

    def test_serialization_roundtrip(self):
        vec = HybridVector(dense=[1.0], sparse={"term": 2.0}, text_hash="h", model="m")
        data = vec.model_dump()
        restored = HybridVector(**data)
        assert restored == vec


# ---------------------------------------------------------------------------
# 2. Dense embedding with mocked Ollama
# ---------------------------------------------------------------------------

class TestDenseEmbedding:
    def test_dense_embedding_success(self):
        embedder = HybridEmbedder()
        fake_embedding = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": fake_embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("src.docwain_intel.hybrid_embedding.requests.post", return_value=mock_response) as mock_post:
            result = embedder.embed_dense("Hello world")

        assert result == fake_embedding
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "bge-m3"

    def test_dense_embedding_empty_text(self):
        embedder = HybridEmbedder()
        result = embedder.embed_dense("")
        assert result == []

    def test_dense_embedding_custom_model(self):
        embedder = HybridEmbedder(embedding_model="custom-model")
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5]}
        mock_response.raise_for_status = MagicMock()

        with patch("src.docwain_intel.hybrid_embedding.requests.post", return_value=mock_response):
            embedder.embed_dense("test")

        assert embedder._model == "custom-model"


# ---------------------------------------------------------------------------
# 3. Sparse embedding computation
# ---------------------------------------------------------------------------

class TestSparseEmbedding:
    def test_sparse_embedding_produces_terms(self):
        embedder = HybridEmbedder()
        # Mock spaCy tokenizer
        with patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_tokenizer = MagicMock()
            mock_tokenizer.lemmatize.return_value = ["machine", "learn", "model", "train", "data"]
            mock_get.return_value = mock_tokenizer

            result = embedder.embed_sparse("Machine learning models are trained on data")

        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result.values())

    def test_sparse_embedding_empty_text(self):
        embedder = HybridEmbedder()
        result = embedder.embed_sparse("")
        assert result == {}

    def test_sparse_embedding_weights_are_positive(self):
        embedder = HybridEmbedder()
        with patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_tokenizer = MagicMock()
            mock_tokenizer.lemmatize.return_value = ["python", "code", "function", "variable"]
            mock_get.return_value = mock_tokenizer

            result = embedder.embed_sparse("Python code with functions and variables")

        for weight in result.values():
            assert weight > 0

    def test_sparse_top_k_limit(self):
        """When there are more terms than SPARSE_TOP_K, only top-K are kept."""
        embedder = HybridEmbedder()
        # Create 300 unique lemmas (exceeds 256 limit)
        fake_lemmas = [f"term{i}" for i in range(300)]

        with patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_tokenizer = MagicMock()
            mock_tokenizer.lemmatize.return_value = fake_lemmas
            mock_get.return_value = mock_tokenizer

            result = embedder.embed_sparse("x" * 100)

        assert len(result) <= 256


# ---------------------------------------------------------------------------
# 4. Hybrid embedding combination
# ---------------------------------------------------------------------------

class TestHybridEmbedding:
    def test_hybrid_returns_both_vectors(self):
        embedder = HybridEmbedder()
        fake_dense = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": fake_dense}
        mock_response.raise_for_status = MagicMock()

        with patch("src.docwain_intel.hybrid_embedding.requests.post", return_value=mock_response), \
             patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_tokenizer = MagicMock()
            mock_tokenizer.lemmatize.return_value = ["hello", "world"]
            mock_get.return_value = mock_tokenizer

            result = embedder.embed_hybrid("Hello world")

        assert isinstance(result, HybridVector)
        assert result.dense == fake_dense
        assert len(result.sparse) > 0
        assert result.text_hash != ""
        assert result.model == "bge-m3"

    def test_hybrid_text_hash_deterministic(self):
        embedder = HybridEmbedder()

        with patch("src.docwain_intel.hybrid_embedding.requests.post") as mock_post, \
             patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"embedding": []}
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp
            mock_tokenizer = MagicMock()
            mock_tokenizer.lemmatize.return_value = []
            mock_get.return_value = mock_tokenizer

            v1 = embedder.embed_hybrid("same text")
            v2 = embedder.embed_hybrid("same text")

        assert v1.text_hash == v2.text_hash


# ---------------------------------------------------------------------------
# 5. Batch embedding
# ---------------------------------------------------------------------------

class TestBatchEmbedding:
    def test_batch_embed_multiple_texts(self):
        embedder = HybridEmbedder()
        texts = ["First document", "Second document", "Third document"]

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_response.raise_for_status = MagicMock()

        with patch("src.docwain_intel.hybrid_embedding.requests.post", return_value=mock_response), \
             patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_tokenizer = MagicMock()
            mock_tokenizer.lemmatize.return_value = ["document"]
            mock_get.return_value = mock_tokenizer

            results = embedder.batch_embed(texts)

        assert len(results) == 3
        assert all(isinstance(r, HybridVector) for r in results)

    def test_batch_embed_empty_list(self):
        embedder = HybridEmbedder()
        results = embedder.batch_embed([])
        assert results == []

    def test_batch_updates_idf_corpus(self):
        embedder = HybridEmbedder()
        texts = ["cat dog", "cat bird"]

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5]}
        mock_response.raise_for_status = MagicMock()

        with patch("src.docwain_intel.hybrid_embedding.requests.post", return_value=mock_response), \
             patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_tokenizer = MagicMock()
            # Return different lemmas per call to simulate real tokenization
            mock_tokenizer.lemmatize.side_effect = [
                # Pre-tokenization pass (2 calls)
                ["cat", "dog"],
                ["cat", "bird"],
                # embed_hybrid -> embed_sparse calls (2 calls)
                ["cat", "dog"],
                ["cat", "bird"],
            ]
            mock_get.return_value = mock_tokenizer

            embedder.batch_embed(texts)

        assert embedder._doc_count == 2
        assert embedder._doc_freq["cat"] == 2  # appears in both docs
        assert embedder._doc_freq["dog"] == 1


# ---------------------------------------------------------------------------
# 6. Ollama failure fallback
# ---------------------------------------------------------------------------

class TestOllamaFallback:
    def test_connection_error_returns_empty(self):
        embedder = HybridEmbedder()
        with patch(
            "src.docwain_intel.hybrid_embedding.requests.post",
            side_effect=__import__("requests").ConnectionError("Connection refused"),
        ):
            result = embedder.embed_dense("test text")
        assert result == []

    def test_timeout_returns_empty(self):
        embedder = HybridEmbedder()
        with patch(
            "src.docwain_intel.hybrid_embedding.requests.post",
            side_effect=__import__("requests").Timeout("Request timed out"),
        ):
            result = embedder.embed_dense("test text")
        assert result == []

    def test_http_error_returns_empty(self):
        embedder = HybridEmbedder()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = __import__("requests").HTTPError("500 Server Error")

        with patch("src.docwain_intel.hybrid_embedding.requests.post", return_value=mock_response):
            result = embedder.embed_dense("test text")
        assert result == []

    def test_hybrid_still_returns_sparse_on_ollama_failure(self):
        """When Ollama is down, hybrid should still produce sparse vector."""
        embedder = HybridEmbedder()

        with patch(
            "src.docwain_intel.hybrid_embedding.requests.post",
            side_effect=__import__("requests").ConnectionError("down"),
        ), patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
            mock_tokenizer = MagicMock()
            mock_tokenizer.lemmatize.return_value = ["fallback", "test"]
            mock_get.return_value = mock_tokenizer

            result = embedder.embed_hybrid("fallback test")

        assert result.dense == []
        assert len(result.sparse) > 0
        assert result.text_hash != ""


# ---------------------------------------------------------------------------
# 7. Text truncation and preprocessing
# ---------------------------------------------------------------------------

class TestTextPreprocessing:
    def test_control_chars_stripped(self):
        text = "Hello\x00\x01\x02World\x7f"
        clean = _preprocess(text)
        assert "\x00" not in clean
        assert "\x01" not in clean
        assert "\x7f" not in clean
        assert "HelloWorld" in clean

    def test_truncation_at_max_tokens(self):
        # _MAX_TOKENS * 4 chars is the limit
        long_text = "a" * (_MAX_TOKENS * 4 + 1000)
        result = _preprocess(long_text)
        assert len(result) <= _MAX_TOKENS * 4

    def test_normal_text_unchanged(self):
        text = "Normal English text with spaces."
        result = _preprocess(text)
        assert result == text

    def test_whitespace_stripped(self):
        text = "  spaces around  "
        result = _preprocess(text)
        assert result == "spaces around"

    def test_text_hash_deterministic(self):
        h1 = _text_hash("same input")
        h2 = _text_hash("same input")
        assert h1 == h2

    def test_text_hash_different_for_different_input(self):
        h1 = _text_hash("input one")
        h2 = _text_hash("input two")
        assert h1 != h2


# ---------------------------------------------------------------------------
# 8. Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_sparse_embeddings(self):
        """Multiple threads computing sparse embeddings should not corrupt state."""
        embedder = HybridEmbedder()
        results = [None] * 10
        errors = []

        def worker(idx: int):
            try:
                with patch("src.docwain_intel.hybrid_embedding._SpaCyTokenizer.get") as mock_get:
                    mock_tokenizer = MagicMock()
                    mock_tokenizer.lemmatize.return_value = [f"term{idx}", "shared"]
                    mock_get.return_value = mock_tokenizer
                    results[idx] = embedder.embed_sparse(f"document number {idx}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is not None for r in results)

    def test_concurrent_idf_updates(self):
        """Concurrent batch calls should not corrupt IDF counters."""
        embedder = HybridEmbedder()
        errors = []

        def update_worker(batch_id: int):
            try:
                lemma_sets = [{f"term{batch_id}", "common"} for _ in range(5)]
                embedder._update_idf_corpus(lemma_sets)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 10 batches * 5 docs each = 50 total
        assert embedder._doc_count == 50
        # "common" appears in all 50 docs
        assert embedder._doc_freq["common"] == 50

    def test_spacy_tokenizer_singleton(self):
        """_SpaCyTokenizer.get() returns the same instance across calls."""
        from src.docwain_intel.hybrid_embedding import _SpaCyTokenizer

        # Reset singleton for clean test
        _SpaCyTokenizer._instance = None

        with patch.dict("sys.modules", {"spacy": MagicMock()}) as _:
            import sys
            mock_spacy = sys.modules["spacy"]
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            t1 = _SpaCyTokenizer.get()
            t2 = _SpaCyTokenizer.get()

            assert t1 is t2
            # spacy.load should be called exactly once (singleton)
            mock_spacy.load.assert_called_once()

        # Clean up singleton so it doesn't affect other tests
        _SpaCyTokenizer._instance = None
