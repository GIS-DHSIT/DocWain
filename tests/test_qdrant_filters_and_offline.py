from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from urllib.error import HTTPError

from qdrant_client.models import FieldCondition, Filter

from src.api import dw_newron
from src.api.vector_store import QdrantVectorStore, REQUIRED_PAYLOAD_INDEX_FIELDS, build_qdrant_filter
from src.intelligence.retrieval import run_intelligent_pipeline


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FakeQdrantAll:
    def __init__(self, points):
        self._points = points
        self.payload_schema = {field: {} for field in REQUIRED_PAYLOAD_INDEX_FIELDS}

    def get_collection(self, collection_name):  # noqa: ANN001
        _ = collection_name
        return SimpleNamespace(
            payload_schema=self.payload_schema,
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=4))),
        )

    def create_payload_index(self, collection_name, field_name, field_schema):  # noqa: ANN001
        _ = (collection_name, field_schema)
        self.payload_schema[field_name] = {"data_type": field_schema}

    def query_points(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return FakeQueryResult(self._points)

    def scroll(self, **kwargs):  # noqa: ANN003
        filt = kwargs.get("scroll_filter") or kwargs.get("filter")
        if filt is None:
            return self._points, None
        # Basic profile_id filtering for test isolation
        filtered = []
        for p in self._points:
            payload = p.payload if hasattr(p, "payload") else {}
            keep = True
            if hasattr(filt, "must"):
                for cond in filt.must:
                    if hasattr(cond, "should"):
                        for sub in cond.should:
                            if hasattr(sub, "key") and hasattr(sub, "match"):
                                k = sub.key
                                v = sub.match.value if hasattr(sub.match, "value") else None
                                if k in payload and v is not None and payload[k] != v:
                                    keep = False
            if keep:
                filtered.append(p)
        return filtered, None

    def count(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return SimpleNamespace(count=len(self._points))


class FakeQdrantIndexFailure:
    def get_collection(self, collection_name):  # noqa: ANN001
        _ = collection_name
        return SimpleNamespace(payload_schema={})

    def create_payload_index(self, collection_name, field_name, field_schema):  # noqa: ANN001
        _ = (collection_name, field_name, field_schema)
        raise RuntimeError("Index required but not found for 'profile_id' keyword index")


class FakeEmbedder:
    def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):  # noqa: ANN001
        _ = (text, convert_to_numpy, normalize_embeddings)
        return [0.1, 0.1, 0.1, 0.1]

    def get_sentence_embedding_dimension(self):
        return 4


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):  # noqa: ANN001
        return self.store.get(key)

    def setex(self, key, ttl, value):  # noqa: ANN001, ARG002
        self.store[key] = value

    def delete(self, key):  # noqa: ANN001
        self.store.pop(key, None)


def test_qdrant_filter_uses_profile_id_not_profileId():
    filt = build_qdrant_filter("sub-1", "prof-1")
    keys = []
    for cond in filt.must:
        if isinstance(cond, FieldCondition):
            keys.append(cond.key)
        elif isinstance(cond, Filter):
            for inner in (cond.should or []):
                if isinstance(inner, FieldCondition):
                    keys.append(inner.key)
    assert "profile_id" in keys


def test_filter_chunks_by_profile_scope_fallbacks_when_no_match():
    from src.rag_v3.retrieve import filter_chunks_by_profile_scope
    from src.rag_v3.types import Chunk, ChunkSource

    chunks = [
        Chunk(
            id="c1",
            text="text",
            score=0.1,
            source=ChunkSource(document_name="doc"),
            meta={"profile_id": "p1", "subscription_id": "s1"},
        )
    ]

    filtered = filter_chunks_by_profile_scope(chunks, profile_id="p2", subscription_id="s2")
    assert filtered == []


def test_collection_has_keyword_indexes_for_filter_fields():
    client = FakeQdrantAll(points=[])
    store = QdrantVectorStore(client=client)
    result = store.ensure_payload_indexes("col", REQUIRED_PAYLOAD_INDEX_FIELDS, create_missing=True)
    assert result["missing"] == []
    for field in REQUIRED_PAYLOAD_INDEX_FIELDS:
        assert field in client.payload_schema


def test_profile_isolation_no_cross_profile_chunks():
    points = [
        FakePoint(
            id="1",
            score=0.9,
            payload={
                "subscription_id": "sub-1",
                "profile_id": "prof-A",
                "document_id": "doc-A",
                "doc_domain": "resume",
                "source_name": "A.pdf",
                "text": "Alpha content " * 5,
            },
        ),
        FakePoint(
            id="2",
            score=0.8,
            payload={
                "subscription_id": "sub-1",
                "profile_id": "prof-B",
                "document_id": "doc-B",
                "doc_domain": "resume",
                "source_name": "B.pdf",
                "text": "Beta content " * 5,
            },
        ),
    ]
    response = run_intelligent_pipeline(
        query="summarize the documents",
        subscription_id="sub-1",
        profile_id="prof-A",
        session_id="sess-iso",
        user_id="user",
        redis_client=None,
        qdrant_client=FakeQdrantAll(points),
        embedder=FakeEmbedder(),
    )
    assert response
    text = response.get("response") or ""
    assert "doc-B" not in text
    sources = response.get("sources") or []
    assert all(src.get("document_id") != "doc-B" for src in sources)


def test_no_unfiltered_fallback_on_filter_failure():
    response = run_intelligent_pipeline(
        query="summarize the documents",
        subscription_id="sub-1",
        profile_id="prof-A",
        session_id="sess-fail",
        user_id="user",
        redis_client=None,
        qdrant_client=FakeQdrantIndexFailure(),
        embedder=FakeEmbedder(),
    )
    assert response
    error = (response.get("metadata") or {}).get("error") or {}
    assert error.get("code") == "RETRIEVAL_FILTER_FAILED"
    assert response.get("sources") == []


def test_gemini_429_does_not_change_scope_or_accuracy(monkeypatch):
    class StubGemini:
        def __init__(self):
            self.model_name = "gemini-2.5-flash"
            self.backend = "gemini"

        def generate_with_metadata(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise HTTPError("https://example.com", 429, "Too Many Requests", hdrs=None, fp=None)

        def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise HTTPError("https://example.com", 429, "Too Many Requests", hdrs=None, fp=None)

    points = [
        FakePoint(
            id="1",
            score=0.9,
            payload={
                "subscription_id": "sub-1",
                "profile_id": "prof-A",
                "document_id": "doc-A",
                "doc_domain": "resume",
                "source_name": "A.pdf",
                "text": "Alpha content " * 5,
            },
        ),
        FakePoint(
            id="2",
            score=0.8,
            payload={
                "subscription_id": "sub-1",
                "profile_id": "prof-B",
                "document_id": "doc-B",
                "doc_domain": "resume",
                "source_name": "B.pdf",
                "text": "Beta content " * 5,
            },
        ),
    ]
    fake_qdrant = FakeQdrantAll(points)

    monkeypatch.setattr(dw_newron, "create_llm_client", lambda *args, **kwargs: StubGemini())
    monkeypatch.setattr(dw_newron, "get_qdrant_client", lambda: fake_qdrant)
    monkeypatch.setattr(dw_newron, "get_model", lambda *args, **kwargs: FakeEmbedder())
    monkeypatch.setattr(dw_newron, "get_cross_encoder", lambda *args, **kwargs: None)
    monkeypatch.setattr(dw_newron, "get_redis_client", lambda: FakeRedis())
    monkeypatch.setattr(dw_newron.Config.Intelligence, "ENABLED", False)
    monkeypatch.setattr(
        dw_newron,
        "resolve_model_for_profile",
        lambda *args, **kwargs: SimpleNamespace(model_name="gemini-2.5-flash", backend="gemini", model_path=None),
    )

    dw_newron._RAG_SYSTEM = None
    dw_newron._RAG_MODEL = None
    dw_newron._RAG_PROFILE = None
    dw_newron._RAG_BACKEND = None
    dw_newron._RAG_MODEL_PATH = None

    response = dw_newron.answer_question(
        query="summarize the documents",
        user_id="user",
        profile_id="prof-A",
        subscription_id="sub-1",
        model_name="gemini-2.5-flash",
    )
    text = response.get("response") or ""
    assert "B.pdf" not in text
    sources = response.get("sources") or []
    assert all(src.get("source_name") != "B.pdf" for src in sources)


def test_offline_model_loading_no_hf_calls_in_ask_path(monkeypatch):
    calls = {}

    class SpySentenceTransformer:
        def __init__(self, name, local_files_only=False, **kwargs):  # noqa: ANN001
            calls["local_files_only"] = local_files_only
            self.name = name

        def get_sentence_embedding_dimension(self):
            return 4

    monkeypatch.setattr(dw_newron, "SentenceTransformer", SpySentenceTransformer)
    monkeypatch.setattr(dw_newron.Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", ["local-model"])
    monkeypatch.setattr(dw_newron.Config.Model, "OFFLINE_ONLY", True)

    dw_newron._MODEL = None
    dw_newron._MODEL_CACHE = {}
    dw_newron._MODEL_BY_NAME = {}

    model = dw_newron.get_model(required_dim=4)
    assert calls.get("local_files_only") is True
    assert model.name == "local-model"
