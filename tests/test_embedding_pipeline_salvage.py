"""Tests for embedding pipeline salvage: prevent valid chunks from being dropped as garbage."""
from __future__ import annotations

import pytest

from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
from src.embedding.pipeline.embed_pipeline import (
    _salvage_chunk_text,
    prepare_embedding_chunks,
)


# ── _is_metadata_garbage: long text exemption ─────────────────────────────


class TestIsMetadataGarbageLongText:
    """Long texts (>500 chars) should NOT be flagged by strong markers alone."""

    def test_long_text_with_incidental_section_marker_passes(self):
        """A 1K-char document that incidentally contains 'Section section_id' should not be garbage."""
        text = (
            "SAP EWM Configuration Guide\n"
            "This document describes the warehouse management setup. "
            "Each Section section_id must be unique within the warehouse structure. "
            "The configuration includes bin types, storage types, and activity areas. "
        ) + ("Additional technical content about warehouse operations. " * 20)
        assert len(text) > 500
        assert _is_metadata_garbage(text) is False

    def test_long_text_with_incidental_start_page_marker_passes(self):
        """A long text mentioning 'start_page 1, end_page' should not be flagged."""
        text = (
            "Document Layout Specification\n"
            "The print layout uses start_page 1, end_page as the pagination range. "
            "This ensures all pages are included in the print job. "
        ) + ("More content about printing and layout configuration details. " * 20)
        assert len(text) > 500
        assert _is_metadata_garbage(text) is False

    def test_long_text_with_chunk_candidate_marker_passes(self):
        """A long text mentioning 'Chunk Candidate text' should not be flagged."""
        text = (
            "Data Processing Pipeline Documentation\n"
            "When reviewing results, each Chunk Candidate text is evaluated for quality. "
            "The evaluation considers semantic relevance, coverage, and diversity. "
        ) + ("Additional documentation content about data processing steps. " * 20)
        assert len(text) > 500
        assert _is_metadata_garbage(text) is False

    def test_short_text_with_strong_marker_still_flagged(self):
        """Short texts with strong markers should still be flagged as garbage."""
        text = "Section section_id abc, chunk_type text, start_page 1, end_page 2"
        assert len(text) < 500
        assert _is_metadata_garbage(text) is True

    def test_extracted_document_repr_always_flagged(self):
        """ExtractedDocument repr prefix should ALWAYS flag regardless of length."""
        short = "Extracted Document (full_text='short resume content here now')"
        assert _is_metadata_garbage(short) is True

        long_repr = "Extracted Document (full_text='" + ("resume content " * 100) + "')"
        assert len(long_repr) > 500
        assert _is_metadata_garbage(long_repr) is True

    def test_extracteddocument_prefix_always_flagged(self):
        """ExtractedDocument( prefix should ALWAYS flag regardless of length."""
        long_repr = "ExtractedDocument(full_text='" + ("resume content " * 100) + "')"
        assert len(long_repr) > 500
        assert _is_metadata_garbage(long_repr) is True

    def test_clean_long_text_passes(self):
        """Normal long document content should pass cleanly."""
        text = (
            "Professional Summary\n"
            "Experienced SAP EWM consultant with over 8 years of expertise in warehouse "
            "management systems, inventory optimization, and supply chain processes. "
            "Skilled in SAP configuration, data migration, and system integration projects "
            "across multiple industries including manufacturing, retail, and logistics.\n\n"
            "Technical Skills\n"
            "- SAP EWM, SAP WM, SAP MM\n"
            "- Warehouse management, inventory control\n"
            "- Python, SQL, ABAP for custom development\n"
        ) + ("Additional experience details and project descriptions. " * 15)
        assert len(text) > 500
        assert _is_metadata_garbage(text) is False


# ── _salvage_chunk_text ───────────────────────────────────────────────────


class TestSalvageChunkText:
    """Verify salvage extracts real content from garbage text."""

    def test_salvages_from_python_repr(self):
        """Extract full_text from ExtractedDocument(full_text='...') repr."""
        text = "Extracted Document (full_text='John Doe\\nSenior SAP Consultant\\nSkills: Python, Java, SAP EWM')"
        result = _salvage_chunk_text(text)
        assert "John Doe" in result
        assert "SAP Consultant" in result
        assert not result.startswith("Extracted Document")

    def test_salvages_from_long_python_repr(self):
        """Extract full_text from a very long ExtractedDocument repr."""
        content = "Gaurav Fegade SAP EWM Consultant\\n" + ("Experience in warehouse management. " * 100)
        text = f"Extracted Document (full_text='{content}', sections=[])"
        result = _salvage_chunk_text(text)
        assert "Gaurav Fegade" in result
        assert "warehouse management" in result
        assert len(result) > 100

    def test_salvages_from_space_delimited_format(self):
        """Extract text from 'Extracted Document full_text ...' format."""
        text = "Extracted Document full_text Professional Summary with extensive experience"
        result = _salvage_chunk_text(text)
        assert "Professional Summary" in result

    def test_returns_empty_for_normal_text(self):
        """Normal text should not be salvaged (nothing to salvage)."""
        result = _salvage_chunk_text("Just a normal sentence about SAP configuration")
        assert result == ""

    def test_returns_empty_for_empty_text(self):
        assert _salvage_chunk_text("") == ""
        assert _salvage_chunk_text(None) == ""

    def test_handles_extracteddocument_no_space(self):
        """Handle ExtractedDocument( without space."""
        text = "ExtractedDocument(full_text='Dev Resume IP Content Here With Skills')"
        result = _salvage_chunk_text(text)
        assert "Dev Resume" in result


# ── prepare_embedding_chunks: salvage instead of drop ─────────────────────


class TestPrepareEmbeddingChunksSalvage:
    """Verify prepare_embedding_chunks salvages text instead of dropping."""

    def test_long_text_with_incidental_marker_not_dropped(self):
        """A long valid chunk that incidentally contains a strong marker should survive."""
        text = (
            "SAP Extended Warehouse Management Configuration\n"
            "Each Section section_id is a unique identifier within the warehouse structure. "
            "The configuration covers storage types, activity areas, and bin management. "
        ) + ("Details about warehouse operations and logistics management processes. " * 20)
        assert len(text) > 500

        chunks = [text]
        meta = [{"section_title": "Config", "section_path": "Config"}]
        result_chunks, result_meta, stats, _rescued = prepare_embedding_chunks(
            chunks, meta,
            subscription_id="sub1", profile_id="prof1",
            document_id="doc1", doc_name="test.pdf",
        )
        assert len(result_chunks) >= 1, "Valid long text should not be dropped"

    def test_extracted_document_repr_salvaged_not_dropped(self):
        """An ExtractedDocument repr should be salvaged and the real text embedded."""
        real_content = (
            "Gaurav Fegade\\nSAP EWM Consultant\\n"
            "Professional Summary\\n"
            "Experienced SAP EWM consultant with over 8 years of expertise. "
            "Skilled in warehouse management systems, inventory optimization. "
        ) + ("Project experience in logistics and supply chain. " * 10)
        repr_text = f"Extracted Document (full_text='{real_content}')"

        chunks = [repr_text]
        meta = [{"section_title": "Resume", "section_path": "Resume"}]
        result_chunks, result_meta, stats, _rescued = prepare_embedding_chunks(
            chunks, meta,
            subscription_id="sub1", profile_id="prof1",
            document_id="doc1", doc_name="Gaurav_Resume.pdf",
        )
        assert len(result_chunks) >= 1, "ExtractedDocument repr should be salvaged, not dropped"
        # The salvaged text should contain the actual content
        combined = " ".join(result_chunks)
        assert "SAP EWM" in combined or "warehouse" in combined

    def test_genuinely_short_chunk_still_dropped(self):
        """A genuinely short chunk (<20 chars) should still be dropped."""
        chunks = ["Hi"]
        meta = [{"section_title": "Short", "section_path": "Short"}]
        result_chunks, result_meta, stats, _rescued = prepare_embedding_chunks(
            chunks, meta,
            subscription_id="sub1", profile_id="prof1",
            document_id="doc1", doc_name="test.pdf",
        )
        assert len(result_chunks) == 0

    def test_normal_text_passes_through(self):
        """Normal clean text should pass through without any salvage needed."""
        text = (
            "Professional Summary\n"
            "Experienced developer with strong Python and cloud infrastructure skills. "
            "Led multiple successful projects in enterprise environments across industries.\n"
            "Technical Skills: Python, Java, AWS, Docker, Kubernetes\n"
        ) + ("More details about professional achievements and project history. " * 5)

        chunks = [text]
        meta = [{"section_title": "Summary", "section_path": "Summary"}]
        result_chunks, result_meta, stats, _rescued = prepare_embedding_chunks(
            chunks, meta,
            subscription_id="sub1", profile_id="prof1",
            document_id="doc1", doc_name="resume.pdf",
        )
        assert len(result_chunks) >= 1

    def test_multiple_chunks_mixed_garbage_and_clean(self):
        """Mix of garbage and clean chunks — garbage salvaged, clean kept."""
        repr_text = "Extracted Document (full_text='Developer with Python and Java skills and cloud experience')"
        clean_text = (
            "Education Background\n"
            "Bachelor of Science in Computer Engineering from State University. "
            "Graduated with honors, specialized in distributed systems and databases.\n"
        ) + ("Additional coursework in machine learning and data science. " * 5)

        chunks = [repr_text, clean_text]
        meta = [
            {"section_title": "Repr", "section_path": "Repr"},
            {"section_title": "Education", "section_path": "Education"},
        ]
        result_chunks, result_meta, stats, _rescued = prepare_embedding_chunks(
            chunks, meta,
            subscription_id="sub1", profile_id="prof1",
            document_id="doc1", doc_name="resume.pdf",
        )
        # At minimum the clean chunk should survive
        assert len(result_chunks) >= 1
