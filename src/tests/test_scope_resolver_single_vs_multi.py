from src.docwain_intel.scope_resolver import DocMeta, ScopeType, resolve_scope


def test_scope_resolver_single_doc_for_invoice():
    docs = [
        DocMeta(doc_id="1", filename="invoice_123.pdf", title="Invoice 123"),
        DocMeta(doc_id="2", filename="resume_jane.pdf", title="Jane Doe Resume"),
    ]
    scope = resolve_scope("Show details for invoice #123", docs)
    assert scope.scope_type == ScopeType.SINGLE_DOC
    assert scope.matched_docs


def test_scope_resolver_multi_doc_for_top5():
    docs = [
        DocMeta(doc_id="1", filename="doc1.pdf", title="Doc 1"),
        DocMeta(doc_id="2", filename="doc2.pdf", title="Doc 2"),
    ]
    scope = resolve_scope("top 5 candidates", docs)
    assert scope.scope_type == ScopeType.MULTI_DOC
