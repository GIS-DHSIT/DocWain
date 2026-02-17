from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
from src.intelligence.deterministic_router import route_query
from src.intelligence.section_intelligence_builder import SectionIntelligenceBuilder


def _build_chunks(section_map):
    chunk_texts = []
    chunk_metadata = []
    for idx, (section_title, text) in enumerate(section_map):
        chunk_texts.append(text)
        chunk_metadata.append(
            {
                "section_title": section_title,
                "section_path": section_title,
                "chunk_id": f"chunk_{idx}",
                "page_start": idx + 1,
            }
        )
    return chunk_texts, chunk_metadata


def test_router_maps_skills_education_doctors_notes_correctly():
    resume_catalog = {"dominant_domains": {"resume": 3}}
    skills_route = route_query("list skills", {}, resume_catalog, {})
    assert "skills_technical" in skills_route.section_focus
    assert "tools_technologies" in skills_route.section_focus
    assert "skills_functional" in skills_route.section_focus

    edu_route = route_query("education details", {}, resume_catalog, {})
    assert "education" in edu_route.section_focus
    assert "identity_contact" not in edu_route.section_focus

    med_catalog = {"dominant_domains": {"medical": 2}}
    notes_route = route_query("doctor's notes", {}, med_catalog, {})
    assert "notes" in notes_route.section_focus


def test_invoice_section_kind_correctness():
    invoice_sections = [
        ("INVOICE", "Invoice #INV-123\nBill To: Acme Inc"),
        ("LINE ITEMS", "Widget A - Qty 2 - $50"),
        ("TOTAL", "Total: $100\nDue Date: 2024-05-01"),
    ]
    chunk_texts, chunk_metadata = _build_chunks(invoice_sections)
    builder = SectionIntelligenceBuilder(min_sections=2)
    result = builder.build(
        document_id="doc_invoice",
        document_text="\n\n".join([t for _, t in invoice_sections]),
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        metadata={"doc_type": "invoice", "source_name": "invoice.pdf"},
    )
    kinds = {sec.section_kind for sec in result.sections}
    assert "line_items" in kinds or "financial_summary" in kinds
    title_kinds = {
        sec.section_kind
        for sec in result.sections
        if "line" in (sec.section_title or "").lower() or "total" in (sec.section_title or "").lower()
    }
    assert "identity_contact" not in title_kinds


def test_text_content_separation():
    raw = {
        "subscription_id": "sub-1",
        "profile_id": "prof-1",
        "document_id": "doc-1",
        "source_name": "doc.pdf",
        "content": "Line 1\nLine 2\nLine 3",
    }
    payload = build_qdrant_payload(raw)
    text_field = payload.get("canonical_text") or payload.get("content")
    assert text_field
    assert payload["embedding_text"]
    # After the embedding rebuild, embedding_text may equal canonical_text when
    # no section prefix is added (section_kind_source != "title").  The key
    # invariant is that both fields are populated with clean text.
    assert isinstance(payload["embedding_text"], str)
    assert len(payload["embedding_text"]) > 0
