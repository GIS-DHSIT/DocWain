"""Tests for the DocumentAnalyzer orchestrator."""

import json
from unittest.mock import MagicMock, call

import pytest

from src.intelligence_v2.analyzer import DocumentAnalyzer, _get_text

_LLM_RESPONSE = {
    "document_type": "invoice",
    "language": "en",
    "summary": "Invoice from Acme Corp for IT services.",
    "section_summaries": {"Line Items": "5 items totaling £45,230"},
    "entities": [{"type": "ORGANIZATION", "value": "Acme Corp", "role": "vendor"}],
    "facts": [{"claim": "Total is £45,230", "evidence": "Section: Line Items, Page: 1"}],
    "relationships": [],
    "answerable_topics": ["invoice total", "vendor name"],
}


def _make_mocks():
    llm = MagicMock()
    llm.generate_with_metadata.return_value = (json.dumps(_LLM_RESPONSE), {})
    neo4j = MagicMock()
    neo4j.run_query.return_value = []
    mongodb = MagicMock()
    return llm, neo4j, mongodb


class TestGetText:
    def test_dict_with_full_text(self):
        extracted = {"full_text": "Hello world", "sections": []}
        assert _get_text(extracted) == "Hello world"

    def test_dict_falls_back_to_sections(self):
        extracted = {"full_text": "", "sections": [{"text": "A"}, {"text": "B"}]}
        assert _get_text(extracted) == "A\n\nB"

    def test_object_with_full_text(self):
        obj = MagicMock()
        obj.full_text = "Object text"
        obj.sections = []
        assert _get_text(obj) == "Object text"

    def test_object_falls_back_to_sections(self):
        obj = MagicMock()
        obj.full_text = ""
        sec1, sec2 = MagicMock(), MagicMock()
        sec1.text = "X"
        sec2.text = "Y"
        # Make sections subscriptable but also iterable with .text
        obj.sections = [sec1, sec2]
        result = _get_text(obj)
        assert "X" in result
        assert "Y" in result


class TestDocumentAnalyzer:
    def test_analyzer_writes_intelligence_to_mongodb(self):
        llm, neo4j, mongodb = _make_mocks()
        analyzer = DocumentAnalyzer(llm, neo4j, mongodb)

        analyzer.analyze(
            document_id="doc-1",
            extracted={"full_text": "Invoice from Acme Corp.", "sections": []},
            subscription_id="sub-1",
            profile_id="prof-1",
            filename="invoice.pdf",
            doc_type="invoice",
        )

        mongodb.update_one.assert_called_once()
        args, kwargs = mongodb.update_one.call_args

        # Filter has document_id
        assert args[0] == {"document_id": "doc-1"}

        update = args[1]["$set"]
        assert update["intelligence_ready"] is True
        assert update["intelligence"]["summary"] == "Invoice from Acme Corp for IT services."
        assert "answerable_topics" in update["intelligence"]
        assert update["document_summary"] == "Invoice from Acme Corp for IT services."
        assert "intelligence_completed_at" in update

    def test_analyzer_writes_entities_to_kg(self):
        llm, neo4j, mongodb = _make_mocks()
        analyzer = DocumentAnalyzer(llm, neo4j, mongodb)

        analyzer.analyze(
            document_id="doc-1",
            extracted={"full_text": "Invoice from Acme Corp.", "sections": []},
            subscription_id="sub-1",
            profile_id="prof-1",
        )

        # KG writer calls run_query for Document node + entities at minimum
        assert neo4j.run_query.called

    def test_analyzer_returns_result_dict(self):
        llm, neo4j, mongodb = _make_mocks()
        analyzer = DocumentAnalyzer(llm, neo4j, mongodb)

        result = analyzer.analyze(
            document_id="doc-1",
            extracted={"full_text": "Invoice from Acme Corp.", "sections": []},
            subscription_id="sub-1",
            profile_id="prof-1",
        )

        assert result["document_id"] == "doc-1"
        assert isinstance(result["intelligence"], dict)
        assert result["intelligence"]["summary"] == "Invoice from Acme Corp for IT services."
        assert "kg_stats" in result
        assert "entities" in result["kg_stats"]
        assert "facts" in result["kg_stats"]
        assert "relationships" in result["kg_stats"]
        assert "links" in result["kg_stats"]

    def test_analyzer_handles_neo4j_failure_gracefully(self):
        llm, neo4j, mongodb = _make_mocks()
        neo4j.run_query.side_effect = Exception("Neo4j connection refused")
        analyzer = DocumentAnalyzer(llm, neo4j, mongodb)

        result = analyzer.analyze(
            document_id="doc-1",
            extracted={"full_text": "Invoice from Acme Corp.", "sections": []},
            subscription_id="sub-1",
            profile_id="prof-1",
        )

        # MongoDB should still be written
        mongodb.update_one.assert_called_once()
        # Result should still be returned with summary
        assert result["document_id"] == "doc-1"
        assert result["intelligence"]["summary"] == "Invoice from Acme Corp for IT services."
