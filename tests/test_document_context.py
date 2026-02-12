"""Tests for document context assembler (Task 1)."""
from __future__ import annotations

import pytest
from src.rag_v3.types import Chunk, ChunkSource
from src.rag_v3.document_context import (
    DocumentContext,
    assemble_document_contexts,
    render_attributed_evidence,
)


def _make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc1",
    doc_name: str = "Resume.pdf",
    section_kind: str = "",
    doc_domain: str = "resume",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        score=0.9,
        source=ChunkSource(document_name=doc_name, page=1),
        meta={
            "document_id": doc_id,
            "source_name": doc_name,
            "section_kind": section_kind,
            "doc_domain": doc_domain,
        },
    )


class TestAssembleDocumentContexts:
    def test_group_chunks_by_document(self):
        chunks = [
            _make_chunk("c1", "Python, Java", doc_id="doc1", doc_name="Abinaya.pdf"),
            _make_chunk("c2", "5 years at Deloitte", doc_id="doc1", doc_name="Abinaya.pdf"),
            _make_chunk("c3", "AWS, Docker", doc_id="doc2", doc_name="Aadithya.pdf"),
        ]
        contexts = assemble_document_contexts(chunks)
        assert len(contexts) == 2
        names = [ctx.document_name for ctx in contexts]
        assert "Aadithya.pdf" in names
        assert "Abinaya.pdf" in names

    def test_single_document_context(self):
        chunks = [
            _make_chunk("c1", "Python", doc_id="doc1", doc_name="Resume.pdf"),
            _make_chunk("c2", "Java", doc_id="doc1", doc_name="Resume.pdf"),
        ]
        contexts = assemble_document_contexts(chunks)
        assert len(contexts) == 1
        assert contexts[0].document_name == "Resume.pdf"
        assert len(contexts[0].chunks) == 2

    def test_empty_chunks_returns_empty(self):
        assert assemble_document_contexts([]) == []

    def test_document_name_from_source(self):
        chunk = Chunk(
            id="c1", text="test", score=0.9,
            source=ChunkSource(document_name="FromSource.pdf", page=1),
            meta={"document_id": "doc1", "doc_domain": "resume"},
        )
        contexts = assemble_document_contexts([chunk])
        assert contexts[0].document_name == "FromSource.pdf"

    def test_document_name_from_meta(self):
        chunk = Chunk(
            id="c1", text="test", score=0.9,
            source=ChunkSource(document_name="", page=1),
            meta={"document_id": "doc1", "source_name": "FromMeta.pdf", "doc_domain": "invoice"},
        )
        contexts = assemble_document_contexts([chunk])
        # source.document_name is "" (falsy), so falls through to meta
        assert contexts[0].document_name == "FromMeta.pdf"

    def test_section_kinds_extracted(self):
        chunks = [
            _make_chunk("c1", "Python", section_kind="skills_technical"),
            _make_chunk("c2", "5 years", section_kind="experience"),
            _make_chunk("c3", "More skills", section_kind="skills_technical"),
        ]
        contexts = assemble_document_contexts(chunks)
        assert "skills_technical" in contexts[0].section_kinds
        assert "experience" in contexts[0].section_kinds
        assert len(contexts[0].section_kinds) == 2  # deduped


class TestRenderAttributedEvidence:
    def test_render_attributed_evidence_multi_doc(self):
        chunks_a = [_make_chunk("c1", "Python, Java", doc_id="doc1", doc_name="Abinaya.pdf", section_kind="skills")]
        chunks_b = [_make_chunk("c2", "AWS, Docker", doc_id="doc2", doc_name="Aadithya.pdf", section_kind="skills")]
        contexts = [
            DocumentContext(
                document_id="doc1", document_name="Abinaya.pdf",
                doc_domain="resume", chunks=chunks_a, fields={}, section_kinds=["skills"],
            ),
            DocumentContext(
                document_id="doc2", document_name="Aadithya.pdf",
                doc_domain="resume", chunks=chunks_b, fields={}, section_kinds=["skills"],
            ),
        ]
        rendered = render_attributed_evidence(contexts)
        assert "=== Document: Abinaya.pdf (resume) ===" in rendered
        assert "=== Document: Aadithya.pdf (resume) ===" in rendered
        assert "[skills] Python, Java" in rendered
        assert "[skills] AWS, Docker" in rendered

    def test_render_attributed_evidence_single_doc(self):
        chunks = [_make_chunk("c1", "Python, Java", section_kind="skills")]
        contexts = [
            DocumentContext(
                document_id="doc1", document_name="Resume.pdf",
                doc_domain="resume", chunks=chunks, fields={}, section_kinds=["skills"],
            ),
        ]
        rendered = render_attributed_evidence(contexts)
        # Single doc should NOT have the header
        assert "===" not in rendered
        assert "[skills] Python, Java" in rendered
