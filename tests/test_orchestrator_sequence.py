from src.rag.doc_inventory import DocInventoryItem
from src.rag.intent_profile_entity_orchestrator import orchestrate_intent_profile_entity


def test_orchestrator_sequence_and_targeting():
    doc_inventory = [
        DocInventoryItem(doc_id="doc-1", source_file="Resume_Alice.pdf", document_name="Resume_Alice.pdf", doc_type="resume"),
        DocInventoryItem(doc_id="doc-2", source_file="Resume_Bob.pdf", document_name="Resume_Bob.pdf", doc_type="resume"),
    ]
    result = orchestrate_intent_profile_entity(
        subscription_id="sub",
        profile_id="profile",
        session_id="sess",
        query_text="Show Alice Johnson details from Resume_Alice.pdf",
        session_context=None,
        doc_inventory=doc_inventory,
        model_id="llama3.2",
    )
    assert result.sequence == ["intent", "profile", "entities", "scope", "retrieval_plan"]
    assert result.scope == "single_doc"
    assert len(result.target_docs) == 1
    assert result.target_docs[0].source_file == "Resume_Alice.pdf"


def test_orchestrator_defaults_multi_doc_for_summary():
    doc_inventory = [
        DocInventoryItem(doc_id="doc-1", source_file="Invoice_A.pdf", document_name="Invoice_A.pdf", doc_type="invoice"),
        DocInventoryItem(doc_id="doc-2", source_file="Invoice_B.pdf", document_name="Invoice_B.pdf", doc_type="invoice"),
    ]
    result = orchestrate_intent_profile_entity(
        subscription_id="sub",
        profile_id="profile",
        session_id=None,
        query_text="Summarize the invoices",
        session_context=None,
        doc_inventory=doc_inventory,
        model_id="llama3.2",
    )
    assert result.scope == "multi_doc"
    assert len(result.target_docs) == 2
