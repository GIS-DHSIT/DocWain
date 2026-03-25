"""Tests for src.extraction.document_intelligence."""

import json
from unittest.mock import MagicMock

import pytest

from src.extraction.document_intelligence import (
    _parse_json_response,
    build_doc_index_text,
    build_doc_intelligence_text,
    extract_document_intelligence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INTELLIGENCE = {
    "document_type": "contract",
    "parties": [
        {"name": "Acme Corp", "role": "Supplier"},
        {"name": "Globex Inc", "role": "Customer"},
    ],
    "key_dates": [
        {"date": "Jan 2026", "context": "Effective date"},
        {"date": "Dec 2026", "context": "Expiration"},
    ],
    "key_values": [
        {"value": "$500,000", "context": "Total contract value"},
        {"value": "Net 30 days", "context": "Payment terms"},
    ],
    "key_sections": [
        {"title": "Scope of Work", "summary": "Defines deliverables"},
        {"title": "Payment Terms", "summary": "Net 30 payment schedule"},
    ],
    "key_facts": [
        "Supplier warrants all goods for 12 months",
        "Early termination requires 60-day notice",
    ],
    "one_line_summary": "Supply agreement between Acme Corp and Globex Inc for industrial equipment.",
}


# ---------------------------------------------------------------------------
# build_doc_index_text
# ---------------------------------------------------------------------------


class TestBuildDocIndexText:
    def test_compact_oneliner(self):
        result = build_doc_index_text("contract.pdf", SAMPLE_INTELLIGENCE)
        assert "contract.pdf" in result
        assert "contract" in result.lower()
        assert "Acme Corp" in result
        # Should be a single line
        assert "\n" not in result

    def test_minimal_intelligence(self):
        result = build_doc_index_text("file.pdf", {"one_line_summary": "A short doc."})
        assert "file.pdf" in result
        assert "A short doc." in result

    def test_empty_intelligence(self):
        result = build_doc_index_text("empty.pdf", {})
        assert result == "empty.pdf"


# ---------------------------------------------------------------------------
# build_doc_intelligence_text
# ---------------------------------------------------------------------------


class TestBuildDocIntelligenceText:
    def test_narrative_output(self):
        result = build_doc_intelligence_text("contract.pdf", SAMPLE_INTELLIGENCE)
        assert "Document: contract.pdf" in result
        assert "Type: contract" in result
        assert "Acme Corp" in result
        assert "Globex Inc" in result
        assert "Jan 2026" in result
        assert "$500,000" in result
        assert "Scope of Work" in result
        assert "Supplier warrants" in result

    def test_minimal_intelligence(self):
        result = build_doc_intelligence_text("x.pdf", {"one_line_summary": "test"})
        assert "Document: x.pdf" in result
        assert "Summary: test" in result

    def test_length_range(self):
        result = build_doc_intelligence_text("contract.pdf", SAMPLE_INTELLIGENCE)
        # Should produce a substantive multi-line narrative
        assert len(result) > 100


# ---------------------------------------------------------------------------
# extract_document_intelligence — LLM failure
# ---------------------------------------------------------------------------


class TestExtractDocIntelligenceFailure:
    def test_returns_minimal_on_llm_failure(self):
        mock_gw = MagicMock()
        mock_gw.generate_with_metadata.side_effect = RuntimeError("LLM down")

        result = extract_document_intelligence(
            "This is a test document with some content.",
            "test.pdf",
            llm_gateway=mock_gw,
        )
        assert "one_line_summary" in result
        assert "document_type" in result  # Fallback always includes type
        assert "This is a test document" in result["one_line_summary"]


# ---------------------------------------------------------------------------
# extract_document_intelligence — success
# ---------------------------------------------------------------------------


class TestExtractDocIntelligenceSuccess:
    def test_parses_valid_json_response(self):
        mock_gw = MagicMock()
        mock_gw.generate_with_metadata.return_value = (
            json.dumps(SAMPLE_INTELLIGENCE),
            {"tokens": 100},
        )

        result = extract_document_intelligence(
            "Full document text here...",
            "contract.pdf",
            llm_gateway=mock_gw,
        )
        assert result["document_type"] == "contract"
        assert len(result["parties"]) == 2
        assert len(result["key_facts"]) == 2

    def test_parses_markdown_wrapped_json(self):
        wrapped = "```json\n" + json.dumps(SAMPLE_INTELLIGENCE) + "\n```"
        mock_gw = MagicMock()
        mock_gw.generate_with_metadata.return_value = (wrapped, {})

        result = extract_document_intelligence(
            "Some text",
            "doc.pdf",
            llm_gateway=mock_gw,
        )
        assert result["document_type"] == "contract"


# ---------------------------------------------------------------------------
# JSON parsing edge cases
# ---------------------------------------------------------------------------


class TestJsonParsing:
    def test_plain_json(self):
        data = {"document_type": "invoice"}
        assert _parse_json_response(json.dumps(data)) == data

    def test_markdown_code_block(self):
        raw = "```json\n{\"document_type\": \"resume\"}\n```"
        assert _parse_json_response(raw)["document_type"] == "resume"

    def test_markdown_no_language_tag(self):
        raw = "```\n{\"document_type\": \"report\"}\n```"
        assert _parse_json_response(raw)["document_type"] == "report"

    def test_json_with_surrounding_text(self):
        raw = "Here is the result:\n{\"document_type\": \"letter\"}\nDone."
        assert _parse_json_response(raw)["document_type"] == "letter"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            _parse_json_response("not json at all")
