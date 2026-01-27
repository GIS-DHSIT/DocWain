from types import SimpleNamespace

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.api.dw_newron import QdrantRetriever


class FakeClient:
    def __init__(self):
        self.index_calls = []
        self.query_calls = 0

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="col-1")])

    def get_collection(self, collection_name):
        assert collection_name == "col-1"
        vectors = {"content_vector": {"size": 3}}
        params = SimpleNamespace(vectors=vectors)
        config = SimpleNamespace(params=params)
        return SimpleNamespace(config=config)

    def create_payload_index(self, *, collection_name, field_name, field_schema):
        self.index_calls.append(
            {
                "collection_name": collection_name,
                "field_name": field_name,
                "field_schema": field_schema,
            }
        )

    def query_points(self, **_kwargs):
        self.query_calls += 1
        return SimpleNamespace(points=[])


def _build_filter():
    return Filter(
        must=[
            FieldCondition(key="profile_id", match=MatchValue(value="prof-1")),
            FieldCondition(key="doc_type", match=MatchAny(any=["RESUME"])),
        ]
    )


def test_run_search_ensures_doc_type_index_once():
    client = FakeClient()
    retriever = QdrantRetriever(client=client, model=object())

    retriever.run_search(
        collection_name="col-1",
        query_vector=[0.1, 0.2, 0.3],
        query_filter=_build_filter(),
        limit=5,
        vector_name="content_vector",
    )
    retriever.run_search(
        collection_name="col-1",
        query_vector=[0.1, 0.2, 0.3],
        query_filter=_build_filter(),
        limit=5,
        vector_name="content_vector",
    )

    doc_type_calls = [c for c in client.index_calls if c["field_name"] == "doc_type"]
    assert len(doc_type_calls) == 1
    assert doc_type_calls[0]["field_schema"] == "keyword"
    assert client.query_calls == 2
