"""Tests for DPIE-driven document type classification in DocumentIntelligence.infer_type()
and DPIE-driven section detection in DocumentExtractor."""

from unittest.mock import MagicMock, patch

import pytest

from src.api.dw_document_extractor import DocumentExtractor, DocumentIntelligence
from src.api.pipeline_models import ChunkCandidate

# Patch target: the function does `from src.intelligence.dpie_integration import DPIERegistry`
# so we patch the class at its source module.
_DPIE_PATCH = "src.intelligence.dpie_integration.DPIERegistry"


class TestDPIEDocTypeClassification:
    """Verify DPIE ML classifier is used as Strategy 1 in infer_type(), with
    heuristic fallback when DPIE is unavailable or low-confidence."""

    def test_dpie_classifies_resume(self):
        """When DPIE is loaded and returns high confidence, infer_type() should
        return the DPIE-predicted type directly."""
        mock_registry = MagicMock()
        mock_registry.is_loaded = True
        mock_registry.classify_document.return_value = ("resume", 0.92)

        with patch(_DPIE_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry

            result = DocumentIntelligence.infer_type(
                tables=[],
                figures=[],
                sections=[],
                full_text="A" * 100,  # >50 chars
                filename_hint="candidate_cv.pdf",
            )

        assert result == "resume"
        mock_registry.classify_document.assert_called_once()
        # Verify text is truncated to 5000 chars
        call_arg = mock_registry.classify_document.call_args[0][0]
        assert len(call_arg) <= 5000

    def test_dpie_fallback_to_heuristic_when_not_loaded(self):
        """When DPIE is not loaded (is_loaded=False), infer_type() should skip
        DPIE and fall through to heuristic logic without calling classify_document."""
        mock_registry = MagicMock()
        mock_registry.is_loaded = False

        with patch(_DPIE_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry

            result = DocumentIntelligence.infer_type(
                tables=[],
                figures=[],
                sections=[],
                full_text="Some document text here that is reasonably long enough.",
                filename_hint="report.pdf",
            )

        # classify_document should NOT have been called
        mock_registry.classify_document.assert_not_called()
        # Heuristic should still return a valid string
        assert isinstance(result, str)
        assert len(result) > 0
        # For a .pdf with no tables/figures/sections, heuristic returns "report"
        assert result == "report"

    def test_dpie_low_confidence_falls_back(self):
        """When DPIE returns low confidence (<0.4), infer_type() should ignore
        DPIE result and fall back to heuristic logic."""
        mock_registry = MagicMock()
        mock_registry.is_loaded = True
        mock_registry.classify_document.return_value = ("other", 0.2)

        with patch(_DPIE_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry

            result = DocumentIntelligence.infer_type(
                tables=[],
                figures=[],
                sections=[],
                full_text="A" * 100,
                filename_hint="data.pptx",
            )

        # DPIE was called but its result was rejected (low confidence)
        mock_registry.classify_document.assert_called_once()
        # Heuristic should kick in: .pptx -> "presentation"
        assert result == "presentation"

    def test_dpie_other_type_falls_back(self):
        """When DPIE returns 'other' even with high confidence, fall back to heuristic."""
        mock_registry = MagicMock()
        mock_registry.is_loaded = True
        mock_registry.classify_document.return_value = ("other", 0.85)

        with patch(_DPIE_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry

            result = DocumentIntelligence.infer_type(
                tables=[],
                figures=[],
                sections=[],
                full_text="A" * 100,
                filename_hint="notes.docx",
            )

        # "other" with high confidence still falls back because doc_type == "other"
        mock_registry.classify_document.assert_called_once()
        # Heuristic for .docx -> "document"
        assert result == "document"

    def test_dpie_import_error_falls_back(self):
        """When DPIE import fails (module not available), infer_type() should
        gracefully fall back to heuristic without crashing."""
        with patch.dict("sys.modules", {"src.intelligence.dpie_integration": None}):
            result = DocumentIntelligence.infer_type(
                tables=[],
                figures=[],
                sections=[],
                full_text="A" * 100,
                filename_hint="slides.pptx",
            )

        # Should not crash, heuristic returns "presentation" for .pptx
        assert result == "presentation"

    def test_dpie_exception_falls_back(self):
        """When DPIE raises an unexpected exception, infer_type() should
        gracefully fall back to heuristic."""
        mock_registry = MagicMock()
        mock_registry.is_loaded = True
        mock_registry.classify_document.side_effect = RuntimeError("model corrupted")

        with patch(_DPIE_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry

            result = DocumentIntelligence.infer_type(
                tables=[],
                figures=[],
                sections=[],
                full_text="A" * 100,
                filename_hint="report.pdf",
            )

        # Exception caught, heuristic runs: .pdf no tables/figures -> "report"
        assert result == "report"

    def test_dpie_short_text_skips_classification(self):
        """When full_text is too short (<=50 chars), DPIE should be skipped
        even if loaded, and heuristic logic runs."""
        mock_registry = MagicMock()
        mock_registry.is_loaded = True

        with patch(_DPIE_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry

            result = DocumentIntelligence.infer_type(
                tables=[],
                figures=[],
                sections=[],
                full_text="Short",
                filename_hint="doc.pdf",
            )

        # DPIE not called because text is too short
        mock_registry.classify_document.assert_not_called()
        assert result == "report"  # .pdf heuristic


# Patch target for section detection: _detect_sections_dpie does
#   from src.intelligence.dpie_integration import DPIERegistry
# inside the method body (lazy import), so we patch at the source module.
_DPIE_SECTION_PATCH = "src.intelligence.dpie_integration.DPIERegistry"


class TestDPIESectionDetection:
    """Verify DPIE ML section detector is used in DocumentExtractor,
    with _is_heading() regex fallback when DPIE is unavailable."""

    def _make_extractor(self):
        """Create a DocumentExtractor with mocked doc-intel to avoid side effects."""
        with patch("src.api.dw_document_extractor.DocumentIntelligence"):
            extractor = DocumentExtractor()
        return extractor

    def test_detect_sections_dpie_returns_results_when_loaded(self):
        """_detect_sections_dpie returns DPIE section results when registry is loaded."""
        extractor = self._make_extractor()

        mock_registry = MagicMock()
        mock_registry.is_loaded = True
        mock_registry.detect_sections.return_value = [
            {"start_line": 0, "end_line": 5, "heading": "Education", "confidence": 0.9},
            {"start_line": 6, "end_line": 12, "heading": "Experience", "confidence": 0.85},
        ]

        with patch(_DPIE_SECTION_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry
            result = extractor._detect_sections_dpie("Line 1\nLine 2\nLine 3\n" * 100)

        assert len(result) == 2
        assert result[0]["heading"] == "Education"
        assert result[1]["heading"] == "Experience"
        mock_registry.detect_sections.assert_called_once()
        # Verify text is truncated to 10000 chars
        call_arg = mock_registry.detect_sections.call_args[0][0]
        assert len(call_arg) <= 10000

    def test_detect_sections_dpie_returns_empty_when_not_loaded(self):
        """_detect_sections_dpie returns empty list when DPIE is not loaded."""
        extractor = self._make_extractor()

        mock_registry = MagicMock()
        mock_registry.is_loaded = False

        with patch(_DPIE_SECTION_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry
            result = extractor._detect_sections_dpie("Some text here")

        assert result == []
        mock_registry.detect_sections.assert_not_called()

    def test_detect_sections_dpie_returns_empty_on_import_error(self):
        """_detect_sections_dpie returns empty list when DPIE import fails."""
        extractor = self._make_extractor()

        with patch.dict("sys.modules", {"src.intelligence.dpie_integration": None}):
            result = extractor._detect_sections_dpie("Some text here")

        assert result == []

    def test_detect_sections_dpie_returns_empty_on_exception(self):
        """_detect_sections_dpie returns empty list when DPIE raises an exception."""
        extractor = self._make_extractor()

        mock_registry = MagicMock()
        mock_registry.is_loaded = True
        mock_registry.detect_sections.side_effect = RuntimeError("model broken")

        with patch(_DPIE_SECTION_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry
            result = extractor._detect_sections_dpie("Some text here")

        assert result == []

    def test_detect_sections_dpie_returns_empty_for_empty_text(self):
        """_detect_sections_dpie returns empty list for empty input text."""
        extractor = self._make_extractor()

        mock_registry = MagicMock()
        mock_registry.is_loaded = True

        with patch(_DPIE_SECTION_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry
            result = extractor._detect_sections_dpie("")

        assert result == []
        mock_registry.detect_sections.assert_not_called()

    def test_apply_dpie_sections_updates_chunk_metadata(self):
        """_apply_dpie_sections updates section_title and section_id on chunk
        candidates whose line ranges overlap with DPIE-detected sections."""
        extractor = self._make_extractor()

        # Simulate a 3-page document where each page contributes some lines.
        # full_text has page markers that build a line-to-page map.
        full_text = (
            "\n--- Page 1 ---\n"
            "Name: John Doe\n"
            "Email: john@example.com\n"
            "\n--- Page 2 ---\n"
            "Education\n"
            "BSc Computer Science\n"
            "University of Testing\n"
            "\n--- Page 3 ---\n"
            "Experience\n"
            "Senior Developer at Acme Corp\n"
            "Built testing frameworks\n"
        )

        chunks = [
            ChunkCandidate(text="Name: John Doe", page=1, section_title="Introduction", section_id="old_s1", chunk_type="text"),
            ChunkCandidate(text="Email: john@example.com", page=1, section_title="Introduction", section_id="old_s1", chunk_type="text"),
            ChunkCandidate(text="BSc Computer Science", page=2, section_title="Introduction", section_id="old_s1", chunk_type="text"),
            ChunkCandidate(text="University of Testing", page=2, section_title="Introduction", section_id="old_s1", chunk_type="text"),
            ChunkCandidate(text="Senior Developer at Acme Corp", page=3, section_title="Introduction", section_id="old_s1", chunk_type="text"),
            ChunkCandidate(text="Built testing frameworks", page=3, section_title="Introduction", section_id="old_s1", chunk_type="text"),
        ]

        # Lines in full_text (0-indexed, excluding blank/page-marker lines):
        # Line 0: "Name: John Doe" (page 1)
        # Line 1: "Email: john@example.com" (page 1)
        # Line 2: "Education" (page 2)
        # Line 3: "BSc Computer Science" (page 2)
        # Line 4: "University of Testing" (page 2)
        # Line 5: "Experience" (page 3)
        # Line 6: "Senior Developer at Acme Corp" (page 3)
        # Line 7: "Built testing frameworks" (page 3)
        dpie_sections = [
            {"start_line": 0, "end_line": 1, "heading": "Contact Info", "confidence": 0.9},
            {"start_line": 2, "end_line": 4, "heading": "Education", "confidence": 0.88},
            {"start_line": 5, "end_line": 7, "heading": "Experience", "confidence": 0.92},
        ]

        extractor._apply_dpie_sections(chunks, dpie_sections, full_text)

        # Page 1 chunks should now have "Contact Info"
        assert chunks[0].section_title == "Contact Info"
        assert chunks[1].section_title == "Contact Info"
        # Page 2 chunks should have "Education"
        assert chunks[2].section_title == "Education"
        assert chunks[3].section_title == "Education"
        # Page 3 chunks should have "Experience"
        assert chunks[4].section_title == "Experience"
        assert chunks[5].section_title == "Experience"

        # section_id should be updated (not the old value)
        assert chunks[0].section_id != "old_s1"
        assert chunks[4].section_id != "old_s1"

    def test_apply_dpie_sections_no_op_when_empty(self):
        """_apply_dpie_sections does nothing when dpie_sections is empty."""
        extractor = self._make_extractor()

        chunks = [
            ChunkCandidate(text="Hello world", page=1, section_title="Original", section_id="orig_id", chunk_type="text"),
        ]
        extractor._apply_dpie_sections(chunks, [], "Hello world\n")

        assert chunks[0].section_title == "Original"
        assert chunks[0].section_id == "orig_id"

    def test_apply_dpie_sections_handles_none_page(self):
        """_apply_dpie_sections skips chunks with page=None gracefully."""
        extractor = self._make_extractor()

        chunks = [
            ChunkCandidate(text="No page info", page=None, section_title="Intro", section_id="s1", chunk_type="text"),
        ]
        dpie_sections = [
            {"start_line": 0, "end_line": 2, "heading": "Section A", "confidence": 0.8},
        ]
        # Should not raise
        extractor._apply_dpie_sections(chunks, dpie_sections, "No page info\n")
        # Chunk with page=None should be left unchanged
        assert chunks[0].section_title == "Intro"

    def test_is_heading_still_works_as_fallback(self):
        """_is_heading() regex continues to work correctly as fallback."""
        # Single uppercase word
        assert DocumentExtractor._is_heading("EDUCATION") is True
        # Numbered heading
        assert DocumentExtractor._is_heading("1.2. Project Overview") is True
        # All caps multi-word
        assert DocumentExtractor._is_heading("WORK EXPERIENCE") is True
        # Normal text is not a heading
        assert DocumentExtractor._is_heading("This is a regular sentence.") is False
        # Empty / None
        assert DocumentExtractor._is_heading("") is False
        assert DocumentExtractor._is_heading(None) is False


# Patch target for SectionChunker: the function does
#   from src.intelligence.dpie_integration import DPIERegistry
# inside _infer_sections_from_text(), so we patch at the source module.
_DPIE_CHUNKER_PATCH = "src.intelligence.dpie_integration.DPIERegistry"


class TestDPIESectionChunking:
    """Verify DPIE ML section detection is used as Strategy 1 in
    _infer_sections_from_text(), with regex heading fallback when
    DPIE is unavailable."""

    def test_infer_sections_uses_dpie_when_available(self):
        """When DPIE is loaded and returns sections, _infer_sections_from_text()
        should create Section objects from DPIE boundaries."""
        from src.embedding.chunking.section_chunker import _infer_sections_from_text

        text = (
            "Contact Information\n"
            "Name: Alice Smith\n"
            "Email: alice@example.com\n"
            "Education\n"
            "BSc Computer Science from MIT\n"
            "Graduated 2020\n"
            "Experience\n"
            "Software Engineer at Acme Corp\n"
            "Built distributed systems\n"
        )

        mock_registry = MagicMock()
        mock_registry.is_loaded = True
        mock_registry.detect_sections.return_value = [
            {"start_line": 0, "end_line": 2, "heading": "Contact Information", "confidence": 0.91},
            {"start_line": 3, "end_line": 5, "heading": "Education", "confidence": 0.88},
            {"start_line": 6, "end_line": 8, "heading": "Experience", "confidence": 0.90},
        ]

        with patch(_DPIE_CHUNKER_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry
            sections = _infer_sections_from_text(text, fallback_title="Untitled")

        assert len(sections) == 3
        assert sections[0].title == "Contact Information"
        assert sections[1].title == "Education"
        assert sections[2].title == "Experience"
        # Each section should have non-empty text
        for sec in sections:
            assert sec.text.strip()
            assert sec.section_id  # should have a generated id
        mock_registry.detect_sections.assert_called_once()

    def test_infer_sections_falls_back_to_regex(self):
        """When DPIE is not loaded (is_loaded=False), _infer_sections_from_text()
        should fall back to the regex-based heading detection."""
        from src.embedding.chunking.section_chunker import _infer_sections_from_text

        # Text with ALL CAPS headings that the regex detector will find
        text = (
            "EDUCATION\n"
            "BSc Computer Science from MIT\n"
            "Graduated in 2020 with honors\n"
            "\n"
            "EXPERIENCE\n"
            "Software Engineer at Acme Corp\n"
            "Built distributed systems and APIs\n"
        )

        mock_registry = MagicMock()
        mock_registry.is_loaded = False

        with patch(_DPIE_CHUNKER_PATCH) as MockClass:
            MockClass.get_instance.return_value = mock_registry
            sections = _infer_sections_from_text(text, fallback_title="Untitled")

        # DPIE should NOT have been called
        mock_registry.detect_sections.assert_not_called()
        # Regex should have detected the ALL CAPS headings
        assert len(sections) >= 2
        titles = [sec.title for sec in sections]
        assert "EDUCATION" in titles
        assert "EXPERIENCE" in titles


class TestTableExtractionIntegration:

    def test_table_model_has_structured_field(self):
        """Table dataclass should have a 'structured' field for StructuredTable."""
        from src.api.pipeline_models import Table

        t = Table(page=1, text="Item, Qty\nA, 10")
        assert hasattr(t, "structured"), "Table model needs a structured field"
        assert t.structured is None  # Default is None

    def test_chunk_candidate_has_table_meta(self):
        """ChunkCandidate should have a 'table_meta' field for table metadata."""
        from src.api.pipeline_models import ChunkCandidate

        cc = ChunkCandidate(
            text="Item, Qty\nA, 10",
            page=1,
            section_title="Invoice",
            section_id="s1",
            chunk_type="table",
            table_meta={"headers": ["Item", "Qty"], "table_type": "line_items", "row_count": 1, "col_count": 2},
        )
        assert cc.table_meta is not None
        assert cc.table_meta["headers"] == ["Item", "Qty"]
        assert cc.table_meta["table_type"] == "line_items"


class TestOCRDualEngineRetry:
    """Verify per-image dual-engine OCR retry: if primary engine confidence
    is below 70%, the alternate engine is tried and the better result is kept."""

    def _make_extractor(self):
        """Create a DocumentExtractor with mocked doc-intel to avoid side effects."""
        with patch("src.api.dw_document_extractor.DocumentIntelligence"):
            extractor = DocumentExtractor()
        return extractor

    def test_low_confidence_triggers_easyocr_retry(self):
        """pytesseract < 70% should retry with easyocr and pick best."""
        from src.api.dw_document_extractor import DocumentExtractor

        extractor = self._make_extractor()
        mock_image = MagicMock()

        with patch.object(extractor, "_ocr_pytesseract", return_value=("blurry text", 45.0)), \
             patch.object(extractor, "_ocr_easyocr", return_value=("clear text result", 85.0)):
            text, conf = extractor._ocr_image(mock_image)

        assert text == "clear text result"
        assert conf == 85.0

    def test_high_confidence_skips_retry(self):
        """pytesseract >= 70% should not trigger easyocr retry."""
        from src.api.dw_document_extractor import DocumentExtractor

        extractor = self._make_extractor()
        mock_image = MagicMock()

        with patch.object(extractor, "_ocr_pytesseract", return_value=("good text", 82.0)), \
             patch.object(extractor, "_ocr_easyocr") as mock_easy:
            text, conf = extractor._ocr_image(mock_image)

        assert text == "good text"
        assert conf == 82.0
        mock_easy.assert_not_called()


class TestLayoutIntegration:

    def test_pdf_extraction_has_layout_reorder_method(self):
        """PDF extraction should have a method for layout-aware text ordering."""
        with patch("src.api.dw_document_extractor.DocumentIntelligence"):
            extractor = DocumentExtractor()
        assert hasattr(extractor, "_reorder_blocks_by_layout"), \
            "DocumentExtractor needs a layout reordering method"

    def test_layout_reorder_preserves_single_column(self):
        """Single column documents should not be reordered."""
        with patch("src.api.dw_document_extractor.DocumentIntelligence"):
            extractor = DocumentExtractor()

        blocks = [
            {"bbox": [50, 100, 560, 200], "text": "First paragraph", "block_type": "text"},
            {"bbox": [50, 220, 560, 320], "text": "Second paragraph", "block_type": "text"},
        ]

        result = extractor._reorder_blocks_by_layout(blocks, page_width=612)
        texts = [b["text"] for b in result]
        assert texts == ["First paragraph", "Second paragraph"]


class TestEnhancedHeaderFooter:

    def _make_extractor(self):
        with patch("src.api.dw_document_extractor.DocumentIntelligence"):
            extractor = DocumentExtractor()
        return extractor

    def test_strips_running_header_with_page_number(self):
        """Running header like 'Company Inc. | Page N' should be stripped across pages."""
        pages = {
            0: ["Company Inc. | Page 1", "Actual content on page 1", "More content"],
            1: ["Company Inc. | Page 2", "Actual content on page 2", "More content"],
            2: ["Company Inc. | Page 3", "Actual content on page 3", "More content"],
        }

        cleaned, headers, footers = DocumentExtractor._dedupe_page_lines(pages)

        for page_lines in cleaned.values():
            joined = " ".join(page_lines)
            assert "Company Inc. | Page" not in joined
            assert "Actual content" in joined

    def test_strips_copyright_footer(self):
        """Copyright notices appearing on every page should be stripped."""
        pages = {
            0: ["Introduction to the topic", "Some detailed explanation", "Copyright 2024 Acme Corp. All rights reserved."],
            1: ["Background information here", "More detailed explanation", "Copyright 2024 Acme Corp. All rights reserved."],
            2: ["Methodology and approach", "Even more explanation", "Copyright 2024 Acme Corp. All rights reserved."],
        }

        cleaned, headers, footers = DocumentExtractor._dedupe_page_lines(pages)

        for page_lines in cleaned.values():
            joined = " ".join(page_lines)
            assert "Copyright" not in joined
            assert len(page_lines) >= 1  # Content lines should remain

    def test_preserves_unique_headers(self):
        """Section titles that differ per page should NOT be stripped."""
        pages = {
            0: ["Introduction", "Some text here"],
            1: ["Background", "More text here"],
            2: ["Methodology", "Even more text"],
        }

        cleaned, headers, footers = DocumentExtractor._dedupe_page_lines(pages)

        assert "Introduction" in " ".join(cleaned[0])
        assert "Background" in " ".join(cleaned[1])
        assert "Methodology" in " ".join(cleaned[2])


class TestFullIntegrationSmoke:

    def test_all_new_modules_importable(self):
        """All new modules should import without errors."""
        from src.intelligence.table_parser import TableParser, StructuredTable
        from src.intelligence.layout_analyzer import LayoutAnalyzer, LayoutResult
        from src.api.dw_document_extractor import DocumentExtractor, DocumentIntelligence
        from src.embedding.chunking.section_chunker import SectionChunker

        # Verify key classes instantiate
        assert TableParser()
        assert LayoutAnalyzer()
        assert SectionChunker()

    def test_table_parser_end_to_end(self):
        """Full flow: raw rows -> StructuredTable -> classified -> flat text."""
        from src.intelligence.table_parser import TableParser

        raw = [
            ["Item", "Qty", "Price"],
            ["Widget A", "10", "$15.00"],
        ]

        parser = TableParser()
        table = parser.parse(raw, page=1)
        table_type = parser.classify_table_type(table)

        assert table_type == "line_items"
        assert "Item" in table.flat_text
        assert table.data_rows[0]["Item"] == "Widget A"
