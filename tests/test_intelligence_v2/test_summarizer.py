"""Tests for intelligence_v2 document summarizer."""

import json
from unittest.mock import MagicMock

from src.intelligence_v2.summarizer import AnalysisResult, DocumentSummarizer


VALID_RESPONSE = {
    "document_type": "financial_report",
    "language": "en",
    "summary": "Quarterly earnings report for Q3 2025.",
    "section_summaries": {"Introduction": "Overview of Q3 performance."},
    "entities": [
        {"type": "organization", "value": "Acme Corp", "role": "reporting entity"}
    ],
    "facts": [
        {"claim": "Revenue was $4.2M", "evidence": "See table on page 3"}
    ],
    "relationships": [
        {
            "from": "Acme Corp",
            "relation": "reported_revenue",
            "to": "$4.2M",
            "context": "Q3 2025 earnings",
        }
    ],
    "answerable_topics": ["Acme Corp Q3 2025 revenue"],
}


def _make_llm(return_text, raise_error=False):
    llm = MagicMock()
    if raise_error:
        llm.generate_with_metadata.side_effect = RuntimeError("LLM unavailable")
    else:
        llm.generate_with_metadata.return_value = (return_text, {"usage": {}})
    return llm


def test_summarizer_returns_analysis_result():
    """Valid JSON response yields correct AnalysisResult fields."""
    llm = _make_llm(json.dumps(VALID_RESPONSE))
    summarizer = DocumentSummarizer(llm)
    result = summarizer.analyze("Some text", "report.pdf", "financial")

    assert isinstance(result, AnalysisResult)
    assert result.document_type == "financial_report"
    assert result.language == "en"
    assert result.summary == "Quarterly earnings report for Q3 2025."
    assert "Introduction" in result.section_summaries
    assert len(result.entities) == 1
    assert result.entities[0]["role"] == "reporting entity"
    assert len(result.facts) == 1
    assert len(result.relationships) == 1
    assert "Acme Corp Q3 2025 revenue" in result.answerable_topics


def test_summarizer_handles_llm_returning_markdown_wrapped_json():
    """JSON wrapped in ```json ... ``` markdown blocks is parsed correctly."""
    wrapped = "Here is the analysis:\n```json\n" + json.dumps(VALID_RESPONSE) + "\n```\n"
    llm = _make_llm(wrapped)
    summarizer = DocumentSummarizer(llm)
    result = summarizer.analyze("Some text", "report.pdf", "financial")

    assert isinstance(result, AnalysisResult)
    assert result.document_type == "financial_report"


def test_summarizer_handles_llm_failure():
    """LLM exception returns a fallback result instead of raising."""
    llm = _make_llm("", raise_error=True)
    summarizer = DocumentSummarizer(llm)
    result = summarizer.analyze("Fallback text content", "broken.pdf", "general")

    assert isinstance(result, AnalysisResult)
    assert "Fallback text content" in result.summary


def test_summarizer_handles_malformed_json():
    """Malformed JSON returns a fallback result instead of raising."""
    llm = _make_llm("{not valid json at all!!!")
    summarizer = DocumentSummarizer(llm)
    result = summarizer.analyze("Some document text", "bad.pdf", "general")

    assert isinstance(result, AnalysisResult)
    assert result.summary  # has some fallback summary


def test_summarizer_uses_thinking_mode():
    """Verify think=True is passed to LLM generate call."""
    llm = _make_llm(json.dumps(VALID_RESPONSE))
    summarizer = DocumentSummarizer(llm)
    summarizer.analyze("Text", "doc.pdf", "general")

    llm.generate_with_metadata.assert_called_once()
    _, kwargs = llm.generate_with_metadata.call_args
    assert kwargs.get("think") is True
    assert kwargs.get("temperature") == 0.1
    assert kwargs.get("max_tokens") == 4096


def test_analysis_result_to_dict():
    """AnalysisResult.to_dict() returns a proper dictionary."""
    result = AnalysisResult(
        document_type="report",
        language="en",
        summary="A summary",
        section_summaries={"Intro": "Intro text"},
        entities=[],
        facts=[],
        relationships=[],
        answerable_topics=["topic1"],
    )
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["document_type"] == "report"
    assert d["answerable_topics"] == ["topic1"]
