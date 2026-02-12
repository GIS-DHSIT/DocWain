from __future__ import annotations

import logging

import pytest

from src.api.config import Config
from src.api.vector_store import build_qdrant_filter
from src.api.dw_newron import RetrievedChunk, build_summary_from_chunks
from src.embedding.layout_graph import build_layout_graph
from src.embedding.layout_semantics import build_semantic_payloads
from src.intelligence.formatter import format_facts_response
from src.intelligence.deterministic_router import DeterministicRoute
from src.security.response_sanitizer import sanitize_user_payload
from src.api.pipeline_models import ExtractedDocument, Section, Table
from src.api.enhanced_context_builder import IntelligentContextBuilder


def _make_extracted(text: str, *, table: str | None = None) -> ExtractedDocument:
    sections = [
        Section(
            section_id="s1",
            title="Section 1",
            level=1,
            start_page=1,
            end_page=1,
            text=text,
        )
    ]
    tables = [Table(page=1, text=table, csv=table)] if table else []
    canonical_json = {"pages": [{"page_number": 1, "text": text}]}
    return ExtractedDocument(
        full_text=text,
        sections=sections,
        tables=tables,
        figures=[],
        chunk_candidates=[],
        doc_type="document",
        canonical_json=canonical_json,
    )


@pytest.mark.parametrize(
    "label,text,table",
    [
        ("resume", "John Doe\nSkills: Python, SQL\nExperience: Data Engineer", None),
        ("invoice", "Invoice data\nTotal $1200\nLine item Widget $400", "Item,Qty,Price\nWidget,2,$400"),
        ("medical", "Patient Jane Roe\nDiagnosis: arthritis\nMedication: aspirin", None),
    ],
)
def test_layout_graph_stable(label, text, table):
    extracted = _make_extracted(text, table=table)
    graph = build_layout_graph(extracted, document_id=f"doc-{label}", file_name=f"{label}.pdf")
    assert graph["page_count"] >= 1
    assert graph["pages"]
    assert graph["pages"][0]["blocks"]
    assert "doc_signals" in graph


def test_layout_semantic_chunks_meet_min_required():
    min_required = int(getattr(Config.Retrieval, "MIN_REQUIRED_CHUNKS", 3))
    text = "\n".join([f"Section {i}: detail about skills and results." for i in range(6)])
    extracted = _make_extracted(text)
    semantic = build_semantic_payloads(extracted=extracted, document_id="doc-resume", source_name="resume.pdf")
    assert len(semantic.chunks) >= min_required


def test_qdrant_filter_requires_profile_id():
    with pytest.raises(ValueError):
        build_qdrant_filter(subscription_id="sub", profile_id="")


def test_name_query_filters_to_matching_chunks():
    chunks = [
        RetrievedChunk(id="c1", text="Alice Smith is a data analyst.", score=0.9, metadata={"source_name": "a.pdf"}, source=None, method="dense"),
        RetrievedChunk(id="c2", text="Bob Jones is a nurse.", score=0.8, metadata={"source_name": "b.pdf"}, source=None, method="dense"),
    ]
    summary = build_summary_from_chunks("Alice's summary", chunks)
    assert "Alice" in summary
    assert "Bob" not in summary


def test_no_internal_ids_leak():
    payload = {
        "document_id": "507f1f77bcf86cd799439011",
        "chunk_id": "chunk_123",
        "response": "Document 507f1f77bcf86cd799439011 details",
    }
    cleaned = sanitize_user_payload(payload)
    assert "document_id" not in cleaned
    assert "chunk_id" not in cleaned
    assert "507f1f77bcf86cd799439011" not in (cleaned.get("response") or "")


def test_patient_details_formatting():
    text = "Patient Jane Roe\nDiagnosis: arthritis\nMedication: aspirin"
    extracted = _make_extracted(text)
    semantic = build_semantic_payloads(extracted=extracted, document_id="doc-med", source_name="medical.pdf")
    facts = semantic.entity_facts
    route = DeterministicRoute(
        task_type="extract",
        domain_hint="medical",
        scope="document",
        output_format="bullets",
    )
    response = format_facts_response(
        query="Give patient details",
        route=route,
        facts=facts,
        catalog={"documents": [{"document_id": "doc-med", "source_name": "medical.pdf"}]},
    )
    assert response is not None
    assert "Patient Details" in response


def test_invoice_summary_formatting():
    text = "Widget A $400\nWidget B $800\nTotal $1200"
    extracted = _make_extracted(text, table="Item,Qty,Price\nWidget A,1,$400")
    semantic = build_semantic_payloads(extracted=extracted, document_id="doc-inv", source_name="invoice.pdf")
    facts = semantic.entity_facts
    route = DeterministicRoute(
        task_type="summary",
        domain_hint="invoice",
        scope="document",
        output_format="bullets",
    )
    response = format_facts_response(
        query="Summarize invoice",
        route=route,
        facts=facts,
        catalog={"documents": [{"document_id": "doc-inv", "source_name": "invoice.pdf"}]},
    )
    assert response is not None
    assert "Line Items" in response or "Financial Summary" in response


def test_context_cap_and_second_pass(caplog, monkeypatch):
    chunks = []
    for idx in range(8):
        chunks.append(
            {
                "text": "Detail " * 400,
                "score": 0.9,
                "metadata": {"source_name": f"doc{idx}.pdf", "section_title": "Body", "page": 1},
            }
        )
    monkeypatch.setattr(Config.RagV3, "MAX_CONTEXT_CHARS", 1200)
    builder = IntelligentContextBuilder(max_context_chunks=6)
    with caplog.at_level(logging.INFO):
        context, _sources = builder.build_context(chunks=chunks, query="Provide details")
    assert len(context) <= 1200
    assert "second-pass reduction" in caplog.text
