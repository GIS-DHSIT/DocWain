"""Tests for Task 7 — IntentAnalyzer answerable_topics + KG hints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agent.intent import IntentAnalyzer, QueryUnderstanding


def _make_llm(task_type="lookup", relevant_documents=None):
    """Return a mock LLM gateway that produces valid JSON."""
    payload = {
        "task_type": task_type,
        "complexity": "simple",
        "resolved_query": "test query",
        "output_format": "prose",
        "relevant_documents": relevant_documents or [],
        "cross_profile": False,
        "entities": [],
    }
    llm = MagicMock()
    llm.generate.return_value = json.dumps(payload)
    return llm


# ------------------------------------------------------------------
# _match_topics
# ------------------------------------------------------------------


def test_match_topics_scores_by_overlap():
    """_match_topics returns docs sorted by word overlap with query."""
    doc_intelligence = [
        {
            "document_id": "doc_aaa",
            "intelligence": {"answerable_topics": ["revenue growth forecast"]},
        },
        {
            "document_id": "doc_bbb",
            "intelligence": {"answerable_topics": ["employee handbook policy"]},
        },
    ]
    matches = IntentAnalyzer._match_topics("What is the revenue growth?", doc_intelligence)
    assert len(matches) == 1
    assert matches[0]["document_id"] == "doc_aaa"
    assert matches[0]["topic_score"] > 0


def test_match_topics_returns_empty_when_no_overlap():
    doc_intelligence = [
        {
            "document_id": "doc_xxx",
            "intelligence": {"answerable_topics": ["unrelated stuff"]},
        },
    ]
    matches = IntentAnalyzer._match_topics("quantum physics breakthroughs", doc_intelligence)
    assert matches == []


def test_match_topics_handles_missing_intelligence():
    """Docs without intelligence or answerable_topics should be skipped."""
    doc_intelligence = [
        {"document_id": "doc_no_intel"},
        {"document_id": "doc_empty", "intelligence": {}},
    ]
    matches = IntentAnalyzer._match_topics("anything", doc_intelligence)
    assert matches == []


# ------------------------------------------------------------------
# analyze() with answerable_topics enrichment
# ------------------------------------------------------------------


def test_intent_analyzer_uses_answerable_topics():
    """IntentAnalyzer should match query against answerable_topics to find relevant docs."""
    llm = _make_llm()
    analyzer = IntentAnalyzer(llm_gateway=llm)

    doc_intelligence = [
        {
            "document_id": "doc_match",
            "intelligence": {"answerable_topics": ["budget allocation"]},
        },
        {
            "document_id": "doc_no_match",
            "intelligence": {"answerable_topics": ["hiring process"]},
        },
    ]

    result = analyzer.analyze(
        query="What is the budget allocation?",
        subscription_id="sub_1",
        profile_id="prof_1",
        doc_intelligence=doc_intelligence,
        conversation_history=None,
    )

    doc_ids = [d["document_id"] for d in result.relevant_documents]
    assert "doc_match" in doc_ids


def test_intent_analyzer_does_not_duplicate_documents():
    """If LLM already returns a doc_id, _match_topics should not add it again."""
    llm = _make_llm(relevant_documents=[{"document_id": "doc_match"}])
    analyzer = IntentAnalyzer(llm_gateway=llm)

    doc_intelligence = [
        {
            "document_id": "doc_match",
            "intelligence": {"answerable_topics": ["budget allocation"]},
        },
    ]

    result = analyzer.analyze(
        query="What is the budget allocation?",
        subscription_id="sub_1",
        profile_id="prof_1",
        doc_intelligence=doc_intelligence,
        conversation_history=None,
    )

    ids = [d["document_id"] for d in result.relevant_documents]
    assert ids.count("doc_match") == 1


# ------------------------------------------------------------------
# analyze() with kg_hints
# ------------------------------------------------------------------


def test_intent_analyzer_accepts_kg_hints():
    """IntentAnalyzer.analyze() should accept optional kg_hints parameter."""
    llm = _make_llm()
    analyzer = IntentAnalyzer(llm_gateway=llm)

    result = analyzer.analyze(
        query="Tell me about project Alpha",
        subscription_id="sub_1",
        profile_id="prof_1",
        doc_intelligence=[],
        conversation_history=None,
        kg_hints={"target_doc_ids": ["doc_789"]},
    )

    doc_ids = [d["document_id"] for d in result.relevant_documents]
    assert "doc_789" in doc_ids


def test_intent_analyzer_kg_hints_default_none():
    """analyze() works fine without kg_hints (backward compat)."""
    llm = _make_llm()
    analyzer = IntentAnalyzer(llm_gateway=llm)

    result = analyzer.analyze(
        query="What is the budget?",
        subscription_id="sub_1",
        profile_id="prof_1",
        doc_intelligence=[],
        conversation_history=None,
    )

    assert isinstance(result, QueryUnderstanding)
