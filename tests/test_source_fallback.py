"""Tests for source fallback when reranked chunks are empty but document_data is available."""
from __future__ import annotations

import pytest

from src.rag_v3.pipeline import _collect_sources
from src.rag_v3.types import Chunk, ChunkSource


# ---------------------------------------------------------------------------
# Helper: build a Chunk with source metadata
# ---------------------------------------------------------------------------
def _make_chunk(doc_name: str = "resume.pdf", page: int = 1, text: str = "some text", score: float = 0.8) -> Chunk:
    return Chunk(
        id="chunk-1",
        text=text,
        score=score,
        source=ChunkSource(document_name=doc_name, page=page),
        meta={},
    )


# ---------------------------------------------------------------------------
# Inline fallback logic (mirrors what pipeline.py does after _collect_sources)
# ---------------------------------------------------------------------------
def _sources_with_fallback(reranked, document_data):
    """Reproduce the fallback logic added to pipeline.py for unit testing."""
    sources = _collect_sources(reranked)
    if not sources and document_data:
        doc_meta = document_data.get("metadata", {}) if isinstance(document_data, dict) else {}
        doc_name = (doc_meta.get("file_name") or doc_meta.get("source_name") or "Document")
        sources = [{"file_name": doc_name, "page": 1, "snippet": ""}]
    return sources


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSourceFallback:
    """Verify source collection with fallback to document_data metadata."""

    def test_sources_from_chunks_normal(self):
        """When reranked has chunks with sources, _collect_sources returns them normally."""
        chunks = [
            _make_chunk(doc_name="alice_resume.pdf", page=2, text="Alice has 5 years experience"),
            _make_chunk(doc_name="bob_resume.pdf", page=1, text="Bob is a software engineer"),
        ]
        sources = _sources_with_fallback(chunks, document_data=None)
        assert len(sources) == 2
        assert sources[0]["file_name"] == "alice_resume.pdf"
        assert sources[0]["page"] == 2
        assert sources[1]["file_name"] == "bob_resume.pdf"
        assert sources[1]["page"] == 1
        # Snippets should be populated from chunk text
        assert sources[0]["snippet"] != ""
        assert sources[1]["snippet"] != ""

    def test_sources_fallback_to_document_data(self):
        """When reranked is empty but document_data has metadata, sources come from metadata."""
        document_data = {
            "metadata": {
                "file_name": "invoice_2024.pdf",
                "doc_type": "invoice",
            },
            "full_text": "Invoice content here",
        }
        sources = _sources_with_fallback(reranked=[], document_data=document_data)
        assert len(sources) == 1
        assert sources[0]["file_name"] == "invoice_2024.pdf"
        assert sources[0]["page"] == 1
        assert sources[0]["snippet"] == ""

    def test_sources_empty_when_no_data(self):
        """When reranked is empty and no document_data, sources stays empty."""
        sources = _sources_with_fallback(reranked=[], document_data=None)
        assert sources == []

    def test_sources_fallback_with_source_name(self):
        """When document_data has source_name but not file_name, fallback still works."""
        document_data = {
            "metadata": {
                "source_name": "employee_handbook.docx",
            },
        }
        sources = _sources_with_fallback(reranked=[], document_data=document_data)
        assert len(sources) == 1
        assert sources[0]["file_name"] == "employee_handbook.docx"
        assert sources[0]["page"] == 1

    def test_sources_fallback_default_document_name(self):
        """When document_data exists but has no file_name or source_name, default to 'Document'."""
        document_data = {"metadata": {}, "full_text": "Some content"}
        sources = _sources_with_fallback(reranked=[], document_data=document_data)
        assert len(sources) == 1
        assert sources[0]["file_name"] == "Document"

    def test_sources_fallback_no_metadata_key(self):
        """When document_data is a dict but has no 'metadata' key, default name is used."""
        document_data = {"full_text": "Raw text content"}
        sources = _sources_with_fallback(reranked=[], document_data=document_data)
        assert len(sources) == 1
        assert sources[0]["file_name"] == "Document"

    def test_sources_normal_path_ignores_document_data(self):
        """When reranked has chunks, document_data fallback is NOT used even if present."""
        chunks = [_make_chunk(doc_name="report.pdf", page=3)]
        document_data = {
            "metadata": {"file_name": "different_doc.pdf"},
        }
        sources = _sources_with_fallback(chunks, document_data=document_data)
        assert len(sources) == 1
        assert sources[0]["file_name"] == "report.pdf"
        # Should NOT include the document_data source
        assert all(s["file_name"] != "different_doc.pdf" for s in sources)

    def test_sources_fallback_file_name_preferred_over_source_name(self):
        """When both file_name and source_name exist, file_name takes precedence."""
        document_data = {
            "metadata": {
                "file_name": "primary.pdf",
                "source_name": "secondary.pdf",
            },
        }
        sources = _sources_with_fallback(reranked=[], document_data=document_data)
        assert len(sources) == 1
        assert sources[0]["file_name"] == "primary.pdf"
