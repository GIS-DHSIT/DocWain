"""Tests for enhanced temporal/date-range extraction."""
import re
import pytest


class TestTemporalPatterns:
    """Test the new temporal regex patterns directly."""

    def test_full_date_range(self):
        from src.doc_understanding.deep_analyzer import _FULL_DATE_RANGE_RE
        text = "The period from January 1, 2024 to March 31, 2024 showed growth."
        matches = _FULL_DATE_RANGE_RE.findall(text)
        assert len(matches) >= 1

    def test_full_date_range_ordinal(self):
        from src.doc_understanding.deep_analyzer import _FULL_DATE_RANGE_RE
        # Regex expects Month Day, Year format (e.g. "Jan 1st, 2024")
        text = "Jan 1st, 2024 to Mar 31st, 2024"
        matches = _FULL_DATE_RANGE_RE.findall(text)
        assert len(matches) >= 1

    def test_iso_date_range(self):
        from src.doc_understanding.deep_analyzer import _ISO_DATE_RANGE_RE
        text = "Valid from 2024-01-01 to 2024-03-31."
        matches = _ISO_DATE_RANGE_RE.findall(text)
        assert len(matches) == 1

    def test_quarter_reference(self):
        from src.doc_understanding.deep_analyzer import _QUARTER_RE
        for text in ["Q1 2024", "First Quarter 2024", "3rd Quarter of 2023"]:
            matches = _QUARTER_RE.findall(text)
            assert len(matches) >= 1, f"Failed for: {text}"

    def test_fiscal_year(self):
        from src.doc_understanding.deep_analyzer import _FISCAL_YEAR_RE
        for text in ["FY2023-24", "FY 2024", "Fiscal Year 2024"]:
            matches = _FISCAL_YEAR_RE.findall(text)
            assert len(matches) >= 1, f"Failed for: {text}"

    def test_slash_date_range(self):
        from src.doc_understanding.deep_analyzer import _SLASH_DATE_RANGE_RE
        text = "Coverage: 01/01/2024 - 03/31/2024"
        matches = _SLASH_DATE_RANGE_RE.findall(text)
        assert len(matches) == 1

    def test_between_dates(self):
        from src.doc_understanding.deep_analyzer import _BETWEEN_DATES_RE
        text = "Sales between Jan 1, 2024 and Mar 31, 2024 increased."
        matches = _BETWEEN_DATES_RE.findall(text)
        assert len(matches) >= 1

    def test_relative_period(self):
        from src.doc_understanding.deep_analyzer import _RELATIVE_PERIOD_RE
        for text in ["last 6 months", "past 3 years", "previous quarter"]:
            matches = _RELATIVE_PERIOD_RE.findall(text)
            assert len(matches) >= 1, f"Failed for: {text}"

    def test_no_false_positives_on_prose(self):
        from src.doc_understanding.deep_analyzer import _QUARTER_RE, _FISCAL_YEAR_RE
        text = "The company reported quarterly earnings for the period."
        q_matches = _QUARTER_RE.findall(text)
        fy_matches = _FISCAL_YEAR_RE.findall(text)
        assert len(q_matches) == 0
        assert len(fy_matches) == 0


class TestExtractTemporalSpans:
    """Test the full _extract_temporal_spans function."""

    def test_comprehensive_document(self):
        from src.doc_understanding.deep_analyzer import _extract_temporal_spans
        text = """
        Report Period: Q1 2024
        Fiscal Year: FY2023-24
        Coverage: January 1, 2024 to March 31, 2024
        Previous: 2023-01-01 to 2023-03-31
        Trend: Revenue grew over the last 6 months.
        Between Jan 1, 2024 and Mar 31, 2024, sales increased by 15%.
        Date range: 01/01/2024 - 03/31/2024
        """
        # _extract_temporal_spans requires (text, entities, page)
        spans, chrono = _extract_temporal_spans(text, [], page=None)
        # Should find multiple temporal references
        assert len(spans) >= 4

    def test_empty_text(self):
        from src.doc_understanding.deep_analyzer import _extract_temporal_spans
        spans, chrono = _extract_temporal_spans("", [], page=None)
        assert len(spans) == 0

    def test_deduplication(self):
        from src.doc_understanding.deep_analyzer import _extract_temporal_spans
        text = "Q1 2024 results. In Q1 2024, revenue grew. Q1 2024 was strong."
        spans, chrono = _extract_temporal_spans(text, [], page=None)
        # Q1 2024 should appear only once after dedup
        q1_spans = [s for s in spans if "Q1 2024" in str(s.get("raw_text", s.get("description", "")))]
        assert len(q1_spans) <= 2  # Might get 1 from each pattern type
