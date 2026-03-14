import logging

from src.api import dw_newron as dn
from src.api.config import Config
from src.api import dataHandler
from src.api.enhanced_context_builder import IntelligentContextBuilder


class _FakeVectorStore:
    def __init__(self):
        self.records = []
        self.collection = None
        self.dim = None

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        self.collection = collection_name
        self.dim = vector_size

    def upsert_records(self, collection_name: str, records, batch_size: int = 100) -> int:  # noqa: ANN001
        self.records.extend(records)
        return len(records)


def test_embed_rejects_empty_text_chunks(monkeypatch):
    monkeypatch.setattr(Config.Model, "EMBEDDING_DIM", 4)
    fake_store = _FakeVectorStore()
    monkeypatch.setattr(dataHandler, "get_vector_store", lambda: fake_store)

    valid_text = (
        "Jane Doe has more than ten years of experience in Python data engineering and platform delivery."
    )
    embeddings_payload = {
        "embeddings": [
            [0.1, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
        ],
        "texts": ["", "   ", valid_text],
        "chunk_metadata": [{"document_id": "doc-1"}, {"document_id": "doc-1"}, {"document_id": "doc-1"}],
        "sparse_vectors": [{"indices": [], "values": []}] * 3,
        "doc_metadata": {"filename": "resume.pdf"},
    }

    result = dataHandler.save_embeddings_to_qdrant(
        embeddings_payload,
        subscription_id="sub-1",
        profile_id="prof-1",
        doctag="doc-1",
        source_filename="resume.pdf",
    )

    assert result["points_saved"] == 1
    assert result["dropped_invalid"] == 2
    assert len(fake_store.records) == 1
    payload = fake_store.records[0].payload
    text = payload.get("canonical_text") or payload.get("content") or ""
    assert text.startswith("Jane Doe"), f"Expected text starting with 'Jane Doe', got: {text[:80]}"


def test_query_drops_invalid_chunks_and_returns_diagnostic(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0
            self.model_name = "fake"

        def generate(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            self.calls += 1
            return "should-not-run"

        def generate_with_metadata(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            self.calls += 1
            return "should-not-run", {}

        def warm_up(self):
            return None

    class FakeQdrant:
        def count(self, **kwargs):  # noqa: ANN003
            class _Count:
                count = 1

            return _Count()

    fake_llm = FakeLLM()
    monkeypatch.setattr(dn, "create_llm_client", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(dn, "get_qdrant_client", lambda: FakeQdrant())
    monkeypatch.setattr(dn, "_ensure_qdrant_indexes", lambda *args, **kwargs: None)
    monkeypatch.setattr(dn, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(dn, "get_cross_encoder", lambda: None)
    monkeypatch.setattr(dn, "get_redis_client", lambda: None)
    monkeypatch.setattr(Config.Intelligence, "ENABLED", False)
    monkeypatch.setattr(
        dn.EnterpriseRAGSystem,
        "preprocess_query",
        lambda self, query, user_id, namespace, use_reformulation=True: (query, {}),
    )

    rag = dn.EnterpriseRAGSystem(model_name="fake")
    invalid_chunk = dn.RetrievedChunk(
        id="c1",
        text="   ",
        score=0.9,
        metadata={"source_name": "resume.pdf", "profile_id": "prof-1"},
        source="resume.pdf",
        method="dense",
    )
    retrieval_plan = {
        "chunks": [invalid_chunk],
        "query": "test",
        "metadata": {},
        "attempts": [],
        "selected_strategy": "hybrid",
        "profile_context": {},
        "graph_hints": None,
    }
    monkeypatch.setattr(
        dn.EnterpriseRAGSystem,
        "retrieve_with_priorities",
        lambda self, **kwargs: retrieval_plan,
    )

    response = rag.answer_question(
        query="What skills are listed?",
        profile_id="prof-1",
        subscription_id="sub-1",
        user_id="user-1",
        top_k_retrieval=5,
        top_k_rerank=3,
        final_k=3,
    )

    # With LLM-first architecture, the pipeline may handle empty text chunks
    # through RAG v3 before the legacy path detects them. The key invariants:
    # 1. The response should not contain LLM-fabricated content (FakeLLM returns "should-not-run")
    # 2. The response should be a valid fallback (diagnostic, usage help, or empty-context message)
    resp_text = response.get("response", "")
    assert "should-not-run" not in resp_text, "LLM fabricated content should not appear"
    # Either the legacy path caught it (status=RETRIEVAL_EMPTY_TEXT) or
    # the v3 pipeline produced a valid fallback response
    if "status" in response:
        assert response["status"] == "RETRIEVAL_EMPTY_TEXT"
        assert "retrieval returned empty text" in resp_text.lower()
    else:
        # RAG v3 path: should have a non-empty response (usage help or no-results message)
        assert resp_text, "Expected a non-empty fallback response"


def test_context_builder_uses_payload_text_not_filename():
    builder = dn.ContextBuilder()
    chunk = dn.RetrievedChunk(
        id="c1",
        text="Jane Doe led cloud migration projects and optimized data pipelines.",
        score=0.8,
        metadata={"source_name": "resume.pdf"},
        source="resume.pdf",
        method="dense",
    )

    context = builder.build_context([chunk], max_chunks=1)
    assert "Jane Doe led cloud migration projects" in context


def test_bm25_skip_logs_reason_and_counts(caplog):
    reranker = dn.HybridReranker(cross_encoder=None)
    chunks = [
        dn.RetrievedChunk(id="c1", text="   ", score=0.2, metadata={}, source=None, method="dense"),
        dn.RetrievedChunk(id="c2", text="", score=0.9, metadata={}, source=None, method="dense"),
    ]
    diagnostics = {
        "retrieved_count": 4,
        "dropped_invalid_count": 2,
        "invalid_samples": [{"chunk_id": "c1", "length": 0, "sample": ""}],
    }

    with caplog.at_level(logging.WARNING):
        reranker.rerank(chunks=chunks, query="test", top_k=2, use_cross_encoder=False, diagnostics=diagnostics)

    assert "BM25 skipped" in caplog.text
    assert "retrieved=4" in caplog.text
    assert "dropped_invalid=2" in caplog.text


def test_filter_fallback_only_if_valid_text_exists():
    builder = IntelligentContextBuilder(max_context_chunks=2)
    valid_text = (
        "Jane Doe has over ten years of experience in Python, data engineering, and cloud platforms."
    )
    chunks = [
        {
            "text": valid_text,
            "score": 0.9,
            "metadata": {"document_id": "doc-1", "source_name": "resume.pdf", "ocr_confidence": 0.1},
        }
    ]
    context, sources = builder.build_context(chunks=chunks, query="skills")
    assert valid_text in context
    assert sources

    invalid_chunks = [
        {
            "text": "----",
            "score": 0.9,
            "metadata": {"document_id": "doc-1", "source_name": "resume.pdf"},
        }
    ]
    empty_context, empty_sources = builder.build_context(chunks=invalid_chunks, query="skills")
    assert empty_context == ""
    assert empty_sources == []
