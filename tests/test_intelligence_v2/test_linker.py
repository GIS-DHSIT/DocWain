"""Tests for DocumentLinker — intra-profile document linking via shared KG entities."""

from unittest.mock import MagicMock, call

import pytest

from src.intelligence_v2.linker import DocumentLinker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return MagicMock()


@pytest.fixture
def linker(store):
    return DocumentLinker(neo4j_store=store)


DOC_ID = "doc_123"
SUB_ID = "sub_AAA"
PROFILE_ID = "prof_BBB"


# ---------------------------------------------------------------------------
# test_linker_finds_related_docs_by_shared_entities
# ---------------------------------------------------------------------------

def test_linker_finds_related_docs_by_shared_entities(linker, store):
    store.run_query.return_value = [
        {"related_doc_id": "doc_456", "shared_entities": 3, "entity_names": ["Acme Corp", "Widget Inc", "Jane Doe"]},
        {"related_doc_id": "doc_789", "shared_entities": 1, "entity_names": ["Acme Corp"]},
    ]

    results = linker.find_related(DOC_ID, SUB_ID, PROFILE_ID)

    assert len(results) == 2
    # First result has most shared entities
    assert results[0]["document_id"] == "doc_456"
    assert results[0]["shared_entities"] == 3
    assert "Acme Corp" in results[0]["entity_names"]
    # Second result
    assert results[1]["document_id"] == "doc_789"
    assert results[1]["shared_entities"] == 1

    # Verify run_query was called exactly once for find_related
    store.run_query.assert_called_once()


# ---------------------------------------------------------------------------
# test_linker_creates_similarity_edges
# ---------------------------------------------------------------------------

def test_linker_creates_similarity_edges(linker, store):
    # First call = find_related query, second call = MERGE RELATED_DOC edges
    store.run_query.side_effect = [
        [
            {"related_doc_id": "doc_456", "shared_entities": 3, "entity_names": ["Acme Corp"]},
            {"related_doc_id": "doc_789", "shared_entities": 1, "entity_names": ["Widget Inc"]},
        ],
        [],  # return value from the MERGE query
    ]

    created = linker.link(DOC_ID, SUB_ID, PROFILE_ID)

    assert created == 2
    assert store.run_query.call_count == 2

    # The second call should be the MERGE for RELATED_DOC edges
    merge_call = store.run_query.call_args_list[1]
    query_str = merge_call[0][0]
    assert "RELATED_DOC" in query_str
    assert "MERGE" in query_str


# ---------------------------------------------------------------------------
# test_linker_enforces_profile_isolation
# ---------------------------------------------------------------------------

def test_linker_enforces_profile_isolation(linker, store):
    store.run_query.return_value = []

    linker.find_related(DOC_ID, SUB_ID, PROFILE_ID)

    query_str = store.run_query.call_args[0][0]
    params = store.run_query.call_args[0][1]

    # The Cypher query must filter by subscription_id AND profile_id
    assert "subscription_id" in query_str
    assert "profile_id" in query_str

    # The params must carry the actual values
    assert params["subscription_id"] == SUB_ID
    assert params["profile_id"] == PROFILE_ID


# ---------------------------------------------------------------------------
# test_linker_returns_empty_for_no_shared_entities
# ---------------------------------------------------------------------------

def test_linker_returns_empty_for_no_shared_entities(linker, store):
    store.run_query.return_value = []

    results = linker.find_related(DOC_ID, SUB_ID, PROFILE_ID)

    assert results == []


# ---------------------------------------------------------------------------
# test_link_returns_zero_when_no_related_docs
# ---------------------------------------------------------------------------

def test_link_returns_zero_when_no_related_docs(linker, store):
    store.run_query.return_value = []

    created = linker.link(DOC_ID, SUB_ID, PROFILE_ID)

    assert created == 0
    # Only one call (find_related); no MERGE call needed
    store.run_query.assert_called_once()
