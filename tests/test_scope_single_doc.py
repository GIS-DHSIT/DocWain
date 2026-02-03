from dataclasses import dataclass

from src.rag.answer_orchestrator_v3 import enforce_single_doc_filter
from src.rag.doc_inventory import DocInventoryItem
from src.rag.entity_detector import detect_entities
from src.rag.intent_router_v3 import classify
from src.rag.scope_resolver_v3 import resolve


@dataclass
class DummyChunk:
    text: str
    metadata: dict


def test_single_doc_scope_resolution_from_filename():
    docs = [
        DocInventoryItem(doc_id="1", source_file="a.pdf", document_name="a.pdf", doc_type="report"),
        DocInventoryItem(doc_id="2", source_file="b.pdf", document_name="b.pdf", doc_type="report"),
    ]
    intent = classify("from a.pdf extract totals", docs)
    entities = detect_entities("from a.pdf extract totals", docs)
    scope = resolve(intent=intent, entities=entities, doc_inventory=docs)

    assert scope.scope_type == "single_doc"
    assert scope.target_docs[0].source_file == "a.pdf"


def test_single_doc_filter_excludes_cross_doc_chunks():
    docs = [
        DocInventoryItem(doc_id="1", source_file="a.pdf", document_name="a.pdf", doc_type="report"),
        DocInventoryItem(doc_id="2", source_file="b.pdf", document_name="b.pdf", doc_type="report"),
    ]
    chunks = [
        DummyChunk(text="A chunk", metadata={"document_id": "1", "source_file": "a.pdf"}),
        DummyChunk(text="B chunk", metadata={"document_id": "2", "source_file": "b.pdf"}),
    ]
    filtered = enforce_single_doc_filter(chunks, docs[0])
    assert len(filtered) == 1
    assert filtered[0].metadata.get("document_id") == "1"
