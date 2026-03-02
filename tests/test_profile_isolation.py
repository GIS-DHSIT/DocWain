from __future__ import annotations

from qdrant_client.models import Filter

from src.doc_understanding.identify import classify_document_type
from src.retrieval.intent_router import IntentResult
from src.retrieval.profile_query import build_grounded_answer
from src.api.vector_store import build_qdrant_filter


def test_profile_filter_includes_subscription_and_profile():
    filt = build_qdrant_filter("sub-1", "profile-1", doc_domain=["invoice"])
    assert isinstance(filt, Filter)
    # build_qdrant_filter wraps subscription_id/profile_id in nested Filter(should=[...])
    # and adds doc_domain as FieldCondition. Collect all keys from all levels.
    all_keys = set()
    for condition in filt.must:
        if hasattr(condition, "key"):
            all_keys.add(condition.key)
        elif hasattr(condition, "should"):
            for sub in condition.should:
                if hasattr(sub, "key"):
                    all_keys.add(sub.key)
    assert "subscription_id" in all_keys or "subscriptionId" in all_keys
    assert "profile_id" in all_keys or "profileId" in all_keys
    assert "doc_domain" in all_keys


def test_document_type_classification_invoice():
    doc_type, confidence = classify_document_type(
        "Invoice #123\nBill To: Example Corp\nTotal Due: $100", "", "invoice_123.pdf"
    )
    assert doc_type == "invoice"
    assert confidence >= 0.5


def test_grounded_answer_redacts_internal_ids():
    intent = IntentResult(intent="qa", target_doc_types=[], constraints={}, need_tables=False, source="test")
    retrieved = [
        {
            "text": "document_id=abc12345 Payment due 01/01/2025",
            "file_name": "invoice.pdf",
            "section_title": "Totals",
            "page_start": 1,
            "page_end": 1,
            "chunk_kind": "section_text",
        }
    ]
    result = build_grounded_answer(query="When is payment due?", intent=intent, retrieved=retrieved, model_name=None)
    assert "document_id" not in result["answer"]
    assert all("document_id" not in citation for citation in result["citations"])
