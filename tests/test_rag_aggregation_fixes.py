from __future__ import annotations

from src.api.dw_newron import (
    IntentRouter,
    IntentType,
    RetrievedChunk,
    extract_structured_data,
    assemble_aggregation_response,
    select_aggregation_chunks,
    build_evidence_set,
)
from src.api.enhanced_context_builder import IntelligentContextBuilder


def _chunk(text: str, idx: int, doc_id: str = "doc-1") -> RetrievedChunk:
    return RetrievedChunk(
        id=f"point-{idx}",
        text=text,
        score=0.9 - (idx * 0.01),
        metadata={
            "chunk_id": f"c{idx}",
            "document_id": doc_id,
            "chunk_index": idx,
            "page": max(1, idx // 2 + 1),
            "source_file": "invoice.pdf",
            "section_title": "Invoice",
        },
        source="invoice.pdf",
    )


def test_intent_router_classification():
    router = IntentRouter()
    assert router.classify("What is the total invoice amount?").intent == IntentType.NUMERIC_AGGREGATION
    assert router.classify("Give a breakdown for each product").intent == IntentType.TABULAR_SUMMARY
    assert router.classify("Compare policy A vs policy B").intent == IntentType.COMPARISON
    assert router.classify("Summarize the benefits section").intent == IntentType.SECTION_SUMMARY
    assert router.classify("Extract all email addresses").intent == IntentType.ENTITY_EXTRACTION
    assert router.classify("When is payment due?").intent == IntentType.FACT_LOOKUP


def test_numeric_aggregation_selects_min_chunks():
    chunks = [_chunk(f"Item {i} $ {10 + i}.00", i) for i in range(10)]
    selected = select_aggregation_chunks(chunks, min_chunks=6, max_chunks=10)
    assert len(selected) >= 6


def test_aggregation_context_packs_min_chunks():
    chunks = [_chunk(f"Item {i} $ {10 + i}.00", i) for i in range(8)]
    selected = select_aggregation_chunks(chunks, min_chunks=6, max_chunks=8)
    builder = IntelligentContextBuilder(max_context_chunks=8)
    chunk_dicts = [
        {"text": c.text, "score": c.score, "metadata": c.metadata} for c in selected
    ]
    context, sources = builder.build_context(
        chunks=chunk_dicts,
        query="total invoice",
        intent=IntentType.NUMERIC_AGGREGATION,
        min_chunks=6,
        max_chunks=8,
        preserve_chunks=True,
    )
    assert len(sources) >= 6
    assert len(context) > 0


def test_invoice_regression_structured_extraction():
    chunks = [
        _chunk("Item | Qty | Price | Amount\nLaptop | 1 | $1,000.00 | $1,000.00\nMouse | 2 | $25.00 | $50.00", 0),
        _chunk("Subtotal: $1,050.00\nTax: $105.00\nTotal: $1,155.00", 1),
    ]
    extraction = extract_structured_data(chunks)
    assert extraction.has_data

    evidence_set = build_evidence_set(chunks, max_chunks=10)
    answer, sources = assemble_aggregation_response(
        header="DOCUMENT / INFORMATION",
        preface_lines=[],
        extraction=extraction,
        evidence_set=evidence_set,
    )

    assert "Laptop" in answer
    assert "Total" in answer
    assert "Not found" not in answer
    assert "chunk_id" not in answer.lower()
    assert sources


def test_dedup_preserves_pages_for_similar_tables():
    builder = IntelligentContextBuilder(max_context_chunks=12)
    chunks = []
    base_text = "Line item A with amount $10.00 and extra details to exceed length." * 2
    for idx in range(12):
        chunks.append(
            {
                "text": base_text,
                "score": 0.9 - idx * 0.01,
                "metadata": {
                    "document_id": "doc-1",
                    "chunk_id": f"c{idx}",
                    "chunk_index": idx,
                    "page": idx // 2 + 1,
                    "source_file": "invoice.pdf",
                },
            }
        )
    context, sources = builder.build_context(
        chunks=chunks,
        query="total",
        intent=IntentType.NUMERIC_AGGREGATION,
        min_chunks=6,
        max_chunks=12,
        preserve_chunks=True,
    )
    assert len(sources) >= 4
    assert len(context) > 0
