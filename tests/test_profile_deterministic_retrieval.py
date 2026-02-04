from __future__ import annotations

from typing import Any, Dict, List

from src.orchestrator.response_validator import validate_response_payload
from src.orchestrator.structured_answer import build_payload, select_output_schema
from src.retrieval.deterministic_retrieval import RetrievalPlan, route_query
from src.retrieval.evidence_extractors import extract_contacts
from src.retrieval.profile_document_index import (
    DocumentIndexEntry,
    ProfileDocumentIndex,
    build_profile_document_index,
)
from src.retrieval.profile_evidence import build_profile_evidence_graph


class FakePoint:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.id = payload.get("chunk_id", "chunk")


class FakeQdrantClient:
    def __init__(self, points: List[FakePoint]):
        self.points = points
        self.calls = 0

    def scroll(self, **kwargs):  # noqa: ANN003
        if self.calls > 0:
            return [], None
        self.calls += 1
        return self.points, None


def test_profile_document_index_coverage(monkeypatch):
    points = []
    for idx in range(6):
        doc_id = f"doc-{idx}"
        payload = {
            "subscription_id": "sub",
            "profile_id": "prof",
            "document_id": doc_id,
            "source": {"name": f"Doc_{idx}.pdf"},
            "chunk_count": 2,
        }
        points.extend([FakePoint(payload), FakePoint(payload)])

    monkeypatch.setattr(
        "src.retrieval.profile_document_index.QdrantClient",
        lambda **kwargs: FakeQdrantClient(points),
    )
    monkeypatch.setattr("src.retrieval.profile_document_index.get_redis_client", lambda: None)

    index = build_profile_document_index("sub", "prof")
    assert len(index.document_ids) == 6


def test_contact_extraction_handles_emojis():
    chunk = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "text": "📞 9826110111 | 📧 dev@example.com | https://linkedin.com/in/dev",
        "source_name": "Dev_Resume_IP.pdf",
    }
    contacts = extract_contacts([chunk])
    assert any("9826110111" in item.value for item in contacts["phones"])
    assert any("dev@example.com" == item.value for item in contacts["emails"])
    assert any("linkedin.com/in/dev" in item.value for item in contacts["urls"])


def test_route_query_document_scope_by_source_name():
    docs = {
        "doc-1": DocumentIndexEntry(document_id="doc-1", source_name="Dev_Resume_IP.pdf"),
        "doc-2": DocumentIndexEntry(document_id="doc-2", source_name="Invoice_11.pdf"),
    }
    index = ProfileDocumentIndex(
        subscription_id="sub",
        profile_id="prof",
        document_ids=list(docs.keys()),
        documents=docs,
        total_points=0,
        cached_at=0.0,
    )
    plan = route_query("Find contact info in Dev_Resume_IP.pdf", index)
    assert isinstance(plan, RetrievalPlan)
    assert plan.scope == "DOCUMENT"
    assert plan.target_document_ids == ["doc-1"]


def test_route_query_profile_scope_targets_all_documents():
    docs = {
        "doc-1": DocumentIndexEntry(document_id="doc-1", source_name="A.pdf"),
        "doc-2": DocumentIndexEntry(document_id="doc-2", source_name="B.pdf"),
    }
    index = ProfileDocumentIndex(
        subscription_id="sub",
        profile_id="prof",
        document_ids=list(docs.keys()),
        documents=docs,
        total_points=0,
        cached_at=0.0,
    )
    plan = route_query("Summarize the profile", index)
    assert plan.scope == "PROFILE"
    assert set(plan.target_document_ids) == {"doc-1", "doc-2"}


def test_no_contact_mixing_without_merge():
    corpora = {
        "doc-1": [
            {
                "document_id": "doc-1",
                "chunk_id": "c1",
                "text": "Email: alpha@example.com",
                "source_name": "Alpha.pdf",
            }
        ],
        "doc-2": [
            {
                "document_id": "doc-2",
                "chunk_id": "c2",
                "text": "Email: beta@example.com",
                "source_name": "Beta.pdf",
            }
        ],
    }
    graph = build_profile_evidence_graph(corpora)
    payload = build_payload(
        user_query="List contacts",
        schema_name="extract",
        retrieval_scope="PROFILE",
        target_document_ids=list(corpora.keys()),
        evidence_graph=graph,
    )
    doc_payloads = {doc["document_id"]: doc for doc in payload["documents"]}
    emails_doc1 = doc_payloads["doc-1"]["contacts"]["emails"]
    emails_doc2 = doc_payloads["doc-2"]["contacts"]["emails"]
    assert any(item["value"] == "alpha@example.com" for item in emails_doc1)
    assert any(item["value"] == "beta@example.com" for item in emails_doc2)


def test_no_self_intro_in_output():
    schema = select_output_schema("extract")
    payload = {
        "schema": "extract",
        "documents": [],
        "answer": "I am DocWain and I help with documents.",
    }
    graph = build_profile_evidence_graph({})
    assert not validate_response_payload(payload, schema, graph)
