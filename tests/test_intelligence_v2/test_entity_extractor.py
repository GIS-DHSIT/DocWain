"""Tests for IntelligenceKGWriter -- KG entity, fact, and relationship writer."""

from unittest.mock import MagicMock, call

import pytest

from src.intelligence_v2.entity_extractor import IntelligenceKGWriter, _entity_id
from src.intelligence_v2.summarizer import AnalysisResult


@pytest.fixture
def store():
    s = MagicMock()
    s.run_query = MagicMock(return_value=[])
    return s


@pytest.fixture
def writer(store):
    return IntelligenceKGWriter(store)


@pytest.fixture
def sample_analysis():
    return AnalysisResult(
        document_type="report",
        language="en",
        summary="A quarterly earnings report.",
        section_summaries={"intro": "Introduction section."},
        entities=[
            {"type": "ORG", "value": "Acme Corp", "role": "subject"},
            {"type": "PERSON", "value": "Jane Doe", "role": "author"},
        ],
        facts=[
            {"claim": "Revenue grew 20%", "evidence": "See table 3"},
            {"claim": "Headcount is 500", "evidence": "HR report"},
        ],
        relationships=[
            {
                "source": {"type": "PERSON", "value": "Jane Doe"},
                "target": {"type": "ORG", "value": "Acme Corp"},
                "relation_type": "WORKS_AT",
                "context": "Author affiliation",
            }
        ],
        answerable_topics=["revenue", "headcount"],
    )


# -------------------------------------------------------------------
# _entity_id helper
# -------------------------------------------------------------------

class TestEntityId:
    def test_basic_format(self):
        eid = _entity_id("sub1", "prof1", "ORG", "Acme Corp")
        assert eid == "sub1::prof1::ORG::acme corp"

    def test_strips_whitespace(self):
        eid = _entity_id("s", "p", "PERSON", "  Jane Doe  ")
        assert eid == "s::p::PERSON::jane doe"


# -------------------------------------------------------------------
# write -- entity nodes
# -------------------------------------------------------------------

class TestWriteEntities:
    def test_write_entities_creates_entity_nodes(self, writer, store, sample_analysis):
        result = writer.write(
            sample_analysis,
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
        )
        # At least one run_query call should have been made
        assert store.run_query.call_count >= 1
        assert result["entities"] == 2

    def test_write_entities_scopes_to_profile(self, writer, store, sample_analysis):
        writer.write(
            sample_analysis,
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
        )
        for c in store.run_query.call_args_list:
            params = c[1] if c[1] else (c[0][1] if len(c[0]) > 1 else {})
            # Every call must carry subscription_id and profile_id
            assert params.get("subscription_id") == "sub-1", (
                f"Missing subscription_id in params: {params}"
            )
            assert params.get("profile_id") == "prof-1", (
                f"Missing profile_id in params: {params}"
            )


# -------------------------------------------------------------------
# write -- relationship edges
# -------------------------------------------------------------------

class TestWriteRelationships:
    def test_write_creates_relationship_edges(self, writer, store, sample_analysis):
        writer.write(
            sample_analysis,
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
        )
        queries = [c[0][0] for c in store.run_query.call_args_list]
        assert any("RELATED_TO" in q for q in queries), (
            "Expected a RELATED_TO edge query"
        )
        assert sample_analysis.relationships[0]["relation_type"] == "WORKS_AT"


# -------------------------------------------------------------------
# write -- empty analysis
# -------------------------------------------------------------------

class TestWriteEmpty:
    def test_write_handles_empty_analysis(self, writer, store):
        empty = AnalysisResult()
        result = writer.write(
            empty,
            document_id="doc-empty",
            subscription_id="sub-1",
            profile_id="prof-1",
        )
        assert result == {"entities": 0, "facts": 0, "relationships": 0}
        # Only the Document MERGE should fire (1 call)
        assert store.run_query.call_count == 1
