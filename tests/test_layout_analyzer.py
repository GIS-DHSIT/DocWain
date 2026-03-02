"""Tests for multi-column layout detection and reading order reconstruction."""
import pytest


class TestMultiColumnDetection:

    def test_detects_two_column_layout(self):
        """Two-column layout detected from bounding boxes."""
        from src.intelligence.layout_analyzer import LayoutAnalyzer

        # Page width = 612pt (letter), two columns
        blocks = [
            {"bbox": [50, 100, 280, 200], "text": "Left column paragraph 1", "block_type": "text"},
            {"bbox": [330, 100, 560, 200], "text": "Right column paragraph 1", "block_type": "text"},
            {"bbox": [50, 220, 280, 320], "text": "Left column paragraph 2", "block_type": "text"},
            {"bbox": [330, 220, 560, 320], "text": "Right column paragraph 2", "block_type": "text"},
        ]

        analyzer = LayoutAnalyzer(page_width=612)
        result = analyzer.analyze(blocks)

        assert result.column_count == 2
        # Reading order: left1, left2, right1, right2 (top-to-bottom per column)
        assert result.ordered_texts[0] == "Left column paragraph 1"
        assert result.ordered_texts[1] == "Left column paragraph 2"
        assert result.ordered_texts[2] == "Right column paragraph 1"
        assert result.ordered_texts[3] == "Right column paragraph 2"

    def test_single_column_unchanged(self):
        """Single column layouts preserve natural order."""
        from src.intelligence.layout_analyzer import LayoutAnalyzer

        blocks = [
            {"bbox": [50, 100, 560, 200], "text": "Paragraph 1", "block_type": "text"},
            {"bbox": [50, 220, 560, 320], "text": "Paragraph 2", "block_type": "text"},
        ]

        analyzer = LayoutAnalyzer(page_width=612)
        result = analyzer.analyze(blocks)

        assert result.column_count == 1
        assert result.ordered_texts == ["Paragraph 1", "Paragraph 2"]

    def test_handles_full_width_header_above_columns(self):
        """Full-width header above multi-column content."""
        from src.intelligence.layout_analyzer import LayoutAnalyzer

        blocks = [
            {"bbox": [50, 50, 560, 90], "text": "Document Title", "block_type": "text"},
            {"bbox": [50, 100, 280, 200], "text": "Left content", "block_type": "text"},
            {"bbox": [330, 100, 560, 200], "text": "Right content", "block_type": "text"},
            {"bbox": [50, 220, 280, 320], "text": "Left content 2", "block_type": "text"},
            {"bbox": [330, 220, 560, 320], "text": "Right content 2", "block_type": "text"},
        ]

        analyzer = LayoutAnalyzer(page_width=612)
        result = analyzer.analyze(blocks)

        assert result.ordered_texts[0] == "Document Title"
        assert "Left content" in result.ordered_texts
        assert "Right content" in result.ordered_texts

    def test_empty_blocks(self):
        """Empty input returns default LayoutResult."""
        from src.intelligence.layout_analyzer import LayoutAnalyzer

        analyzer = LayoutAnalyzer(page_width=612)
        result = analyzer.analyze([])
        assert result.column_count == 1
        assert result.ordered_texts == []
