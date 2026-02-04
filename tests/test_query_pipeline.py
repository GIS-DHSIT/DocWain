import re

from src.api import dw_newron as dn


def _chunk(text, metadata=None, score=0.9, chunk_id="c1"):
    return dn.RetrievedChunk(
        id=chunk_id,
        text=text,
        score=score,
        metadata=metadata or {},
        source=None,
        method="dense",
    )


def test_profile_summary_autoscope():
    chunks = [
        _chunk(
            "Muthu - Software Engineer at Acme. Skills: Python, APIs, RAG.",
            metadata={"candidate_name": "Muthu", "doc_type": "RESUME", "document_id": "doc-1"},
            chunk_id="r1",
        ),
        _chunk(
            "Invoice #INV-1029 Total: $1,200 for services.",
            metadata={"doc_type": "INVOICE", "document_id": "doc-2"},
            chunk_id="i1",
        ),
    ]

    summary = dn.build_summary_from_chunks("summarize Muthu's profile", chunks)
    lowered = summary.lower()
    assert "choose one option" not in lowered
    assert "citations:" not in lowered
    assert "invoice" not in lowered
    assert "muthu" in lowered


def test_profile_summary_entity_subset():
    chunks = [
        _chunk(
            "Bharath - Data Analyst at Nimbus. Skills: SQL, dashboards.",
            metadata={"candidate_name": "Bharath", "doc_type": "RESUME", "document_id": "doc-3"},
            chunk_id="b1",
        ),
        _chunk(
            "Muthu - Software Engineer at Acme. Skills: Python, APIs.",
            metadata={"candidate_name": "Muthu", "doc_type": "RESUME", "document_id": "doc-4"},
            chunk_id="m1",
        ),
    ]

    summary = dn.build_summary_from_chunks("summarize Bharath's profile", chunks)
    lowered = summary.lower()
    assert "bharath" in lowered
    assert "muthu" not in lowered
    assert "?" not in summary


def test_domain_gating_hr_excludes_invoice():
    chunks = [
        _chunk(
            "Muthu - Product Manager at Acme. Skills: Roadmaps, Analytics.",
            metadata={"candidate_name": "Muthu", "doc_type": "RESUME", "document_id": "doc-5"},
            chunk_id="r2",
        ),
        _chunk(
            "Invoice for Muthu. Total due: $900.",
            metadata={"candidate_name": "Muthu", "doc_type": "INVOICE", "document_id": "doc-6"},
            chunk_id="i2",
        ),
    ]

    summary = dn.build_summary_from_chunks("summarize Muthu's profile", chunks)
    assert "invoice" not in summary.lower()


def test_no_internal_leaks():
    response = {
        "response": "Profile summary. Citations:\n[SOURCE-1]\nRequest 123e4567-e89b-12d3-a456-426614174000 at /home/user/file.pdf",
        "sources": [
            {
                "document_id": "doc-7",
                "chunk_id": "chunk-9",
                "source_file": "/home/user/file.pdf",
                "source_name": "Resume.pdf",
                "url": "http://localhost/internal",
            }
        ],
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "collection": "internal-collection",
        "context_found": True,
    }

    cleaned = dn.sanitize_response_obj(response)
    assert "citations" not in cleaned["response"].lower()
    assert "123e4567-e89b-12d3-a456-426614174000" not in cleaned["response"]
    assert "/home/user/file.pdf" not in cleaned["response"]
    assert cleaned.get("request_id") is None
    assert all("document_id" not in src for src in cleaned.get("sources", []))
