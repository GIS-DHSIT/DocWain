from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.api import dw_newron, rag_state
from src.api.ask_handler import apply_error_contract
from src.api.dw_newron import QdrantRetriever, _build_retrieval_filter_error_response
from src.api.enhanced_context_builder import IntelligentContextBuilder
from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS, ensure_payload_indexes
from src.api.rag_state import AppState


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FakeQdrantIndexing:
    def __init__(self):
        self.payload_schema = {field: {} for field in REQUIRED_PAYLOAD_INDEX_FIELDS if field != "section_id"}
        self.created = []
        self.query_calls = 0

    def get_collection(self, collection_name):  # noqa: ANN001
        _ = collection_name
        return SimpleNamespace(payload_schema=self.payload_schema)

    def create_payload_index(self, collection_name, field_name, field_schema):  # noqa: ANN001
        _ = (collection_name, field_schema)
        self.payload_schema[field_name] = {"data_type": field_schema}
        self.created.append(field_name)

    def query_points(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        self.query_calls += 1
        if "section_id" not in self.payload_schema:
            raise UnexpectedResponse(
                400,
                "Bad Request",
                b'Index required but not found for "section_id" of type [keyword]',
                httpx.Headers(),
            )
        return FakeQueryResult([FakePoint(id="1", score=0.9, payload={"embedding_text": "embed", "content": "content"})])


class FakeEmbedder:
    def encode(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return [0.1, 0.2, 0.3, 0.4]

    def get_sentence_embedding_dimension(self):
        return 4


def test_payload_index_bootstrap_creates_section_id_index():
    client = FakeQdrantIndexing()
    ensure_payload_indexes(client, "col-boot", REQUIRED_PAYLOAD_INDEX_FIELDS, create_missing=True)
    assert "section_id" in client.payload_schema


def test_runtime_autoheal_on_400_missing_index():
    client = FakeQdrantIndexing()
    retriever = QdrantRetriever(client=client, model=FakeEmbedder())
    result = retriever.run_search(
        collection_name="col-autoheal",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        query_filter=None,
        limit=3,
    )
    assert client.query_calls == 2
    assert "section_id" in client.created
    assert result.points


def test_singleton_not_reinitialized_per_request(monkeypatch):
    fake_rag = object()
    state = AppState(
        embedding_model=None,
        reranker=None,
        qdrant_client=None,
        redis_client=None,
        ollama_client=None,
        rag_system=fake_rag,
    )
    rag_state.set_app_state(state)

    def boom(*args, **kwargs):  # noqa: ANN001, ANN002
        raise AssertionError("EnterpriseRAGSystem should not reinitialize")

    monkeypatch.setattr(dw_newron, "EnterpriseRAGSystem", boom)
    assert dw_newron.get_rag_system(model_name="llama3.2") is fake_rag


def test_no_unfiltered_fallback():
    class FakeQdrant:
        def __init__(self):
            self.calls = 0

        def query_points(self, **kwargs):  # noqa: ANN003
            self.calls += 1
            return FakeQueryResult([])

    retriever = QdrantRetriever(client=FakeQdrant(), model=FakeEmbedder())
    chunks = retriever.retrieve(
        collection_name="col",
        query="test",
        subscription_id="sub",
        filter_profile=None,
    )
    assert chunks == []
    assert retriever.client.calls == 0


def test_retrieval_failure_returns_controlled_error():
    response = _build_retrieval_filter_error_response(
        query="test",
        user_id="user",
        collection_name="col",
        request_id="corr-1",
        index_version=None,
        details="missing_index=section_id",
        error_code="RETRIEVAL_INDEX_MISSING",
        documents_searched=["docA.pdf"],
    )
    response, code = apply_error_contract(response, correlation_id="corr-1", state=None)
    assert code == "RETRIEVAL_INDEX_MISSING"
    assert response.get("ok") is False
    msg = response.get("message") or response.get("response") or ""
    assert "section_id" in msg
    assert "unknown" not in msg.lower()


def test_model_consistency_docwain_agent_resolves():
    assert dw_newron._resolve_model_alias("DocWain-Agent") == "qwen3:14b"
    assert dw_newron._resolve_model_alias("gpt-oss") == "qwen3:14b"


def test_text_content_separation():
    payload = {
        "embedding_text": "embedding summary text " * 5,
        "content": "full document content text " * 5,
        "document_id": "doc-1",
    }
    result = FakeQueryResult([FakePoint(id="1", score=0.9, payload=payload)])
    chunks = QdrantRetriever._points_to_chunks(result, method="dense")
    assert chunks[0].text.startswith("embedding summary")

    builder = IntelligentContextBuilder(max_context_chunks=2)
    ctx, sources = builder.build_context(
        chunks=[{"text": chunks[0].text, "score": chunks[0].score, "metadata": chunks[0].metadata}],
        query="test",
    )
    assert "full document content" in ctx
    assert "embedding summary" not in ctx
    assert sources
