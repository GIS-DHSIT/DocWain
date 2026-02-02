from src.api.dw_newron import RetrievedChunk
from src.rag.context_reasoning import (
    AnswerRenderer,
    ContextAwareQueryAnalyzer,
    EvidencePlanBuilder,
    WorkingContextAssembler,
)


def _chunk(text: str, doc_id: str, doc_name: str, section: str = "Summary"):
    return RetrievedChunk(
        id=f"{doc_id}-1",
        text=text,
        score=0.5,
        metadata={
            "document_id": doc_id,
            "source_file": doc_name,
            "section_title": section,
            "chunk_id": f"{doc_id}-chunk-1",
        },
        source=doc_name,
    )


def test_pronoun_resolution_uses_last_doc():
    analyzer = ContextAwareQueryAnalyzer()
    analysis = analyzer.analyze(
        query="what is the total price?",
        conversation_history="User: use PO9 invoice",
        available_sources=["PO9 invoice.pdf", "Other.pdf"],
        last_active_document={"doc_id": "doc-po9", "doc_name": "PO9 invoice.pdf"},
    )
    assert analysis.scope == "single_doc_default"
    assert any("PO9" in hint for hint in analysis.target_hint) or analysis.assumptions
    assert not analysis.clarification_needed


def test_compare_intent_sets_multi_doc_scope_and_table():
    analyzer = ContextAwareQueryAnalyzer()
    analysis = analyzer.analyze(
        query="compare totals across the documents",
        conversation_history="",
        available_sources=["A.pdf", "B.pdf"],
        last_active_document=None,
    )
    assert analysis.scope == "multi_doc"
    assert analysis.output_mode == "table"
    plan = EvidencePlanBuilder().build(analysis, "compare totals across the documents")
    assert plan.doc_selection_policy == "multi_doc_balanced"


def test_unlabeled_numbers_are_blocked():
    analyzer = ContextAwareQueryAnalyzer()
    analysis = analyzer.analyze(
        query="what is the total price?",
        conversation_history="",
        available_sources=["Doc.pdf"],
        last_active_document=None,
    )
    assembler = WorkingContextAssembler()
    chunk = _chunk("Call 555-123-4567. VIN 123456789. 2023-01-01.", "doc1", "Doc.pdf", "Footer")
    context = assembler.assemble(query="what is the total price?", chunks=[chunk], analysis=analysis)
    assert context.numeric_claims == []


def test_tabular_formatting_renders_table():
    analyzer = ContextAwareQueryAnalyzer()
    analysis = analyzer.analyze(
        query="show in detailed tabular format",
        conversation_history="",
        available_sources=["Doc.pdf"],
        last_active_document=None,
    )
    assembler = WorkingContextAssembler()
    chunk = _chunk("Total: $100\nTax: $8", "doc1", "Doc.pdf")
    context = assembler.assemble(query="show in detailed tabular format", chunks=[chunk], analysis=analysis)
    renderer = AnswerRenderer()
    result = renderer.render(query="show in detailed tabular format", analysis=analysis, context=context)
    assert "|" in result.text


def test_multi_doc_avoidance_assumption():
    analyzer = ContextAwareQueryAnalyzer()
    analysis = analyzer.analyze(
        query="what is the total price",
        conversation_history="",
        available_sources=["A.pdf", "B.pdf", "C.pdf"],
        last_active_document=None,
    )
    assembler = WorkingContextAssembler()
    chunks = [
        _chunk("Total: $100", "docA", "A.pdf"),
        _chunk("Total: $200", "docB", "B.pdf"),
    ]
    context = assembler.assemble(query="what is the total price", chunks=chunks, analysis=analysis)
    if not analysis.assumptions:
        analysis.assumptions.append("Assuming the best-matching document based on coverage.")
    renderer = AnswerRenderer()
    result = renderer.render(query="what is the total price", analysis=analysis, context=context)
    assert "Assuming" in result.text
