from __future__ import annotations

from src.rag.doc_inventory import DocInventoryItem
from src.tools.resume_router import is_multi_profile_request, should_bypass_clarification, select_resume_docs


def test_all_profiles_bypass_clarification_and_selects_resume_docs():
    doc_inventory = [
        DocInventoryItem(doc_id="1", source_file="Alice_Resume.pdf", document_name="Alice_Resume.pdf", doc_type="resume"),
        DocInventoryItem(doc_id="2", source_file="Bob_Resume.pdf", document_name="Bob_Resume.pdf", doc_type="resume"),
        DocInventoryItem(doc_id="3", source_file="Invoice_01.pdf", document_name="Invoice_01.pdf", doc_type="invoice"),
    ]
    query = "do this for all profiles"
    assert is_multi_profile_request(query)
    assert should_bypass_clarification(query, doc_inventory)
    selected = select_resume_docs(doc_inventory=doc_inventory, chunks_by_doc=None)
    assert len(selected) == 2
