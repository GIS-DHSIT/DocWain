"""Tests for REASON prompt entity/fact enrichment and context_builder aggregation."""

from dataclasses import dataclass

from src.generation.prompts import build_reason_prompt
from src.retrieval.context_builder import build_context
from src.retrieval.retriever import EvidenceChunk


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


def test_reason_prompt_includes_entity_context():
    """Entities and facts should appear in the REASON prompt."""
    doc_context = {
        "summary": "A quarterly report.",
        "entities": [
            {"type": "PERSON", "value": "Alice Smith", "role": "CEO"},
            {"type": "ORG", "value": "Acme Corp"},
            {"type": "DATE", "value": "2025-Q1"},
        ],
        "key_facts": [
            {"claim": "Revenue was $5M", "evidence": "p.12 table 3"},
            {"claim": "Headcount grew 20%"},
        ],
    }

    prompt = build_reason_prompt(
        query="What was the revenue?",
        task_type="extract",
        output_format="prose",
        evidence=[
            {
                "source_index": 1,
                "source_name": "report.pdf",
                "section": "Financials",
                "page": 12,
                "text": "Revenue was $5M in Q1.",
                "score": 0.95,
            }
        ],
        doc_context=doc_context,
    )

    # Entity section present
    assert "## Known Entities" in prompt
    assert "PERSON: Alice Smith (CEO)" in prompt
    assert "ORG: Acme Corp" in prompt
    assert "DATE: 2025-Q1" in prompt

    # Facts section present
    assert "## Pre-Extracted Facts" in prompt
    assert "Revenue was $5M" in prompt
    assert "[p.12 table 3]" in prompt
    assert "Headcount grew 20%" in prompt


def test_reason_prompt_works_without_entity_context():
    """Backward compatible — None doc_context should still work."""
    prompt = build_reason_prompt(
        query="Hello?",
        task_type="lookup",
        output_format="prose",
        evidence=[],
        doc_context=None,
    )

    assert "QUESTION: Hello?" in prompt
    assert "## Known Entities" not in prompt
    assert "## Pre-Extracted Facts" not in prompt


def test_reason_prompt_limits_entities_to_15():
    """At most 15 entities should appear in the prompt."""
    entities = [{"type": "ITEM", "value": f"ent-{i}"} for i in range(30)]
    doc_context = {"entities": entities}

    prompt = build_reason_prompt(
        query="q",
        task_type="lookup",
        output_format="prose",
        evidence=[],
        doc_context=doc_context,
    )

    assert "ent-14" in prompt
    assert "ent-15" not in prompt


def test_reason_prompt_limits_facts_to_10():
    """At most 10 facts should appear in the prompt."""
    facts = [{"claim": f"fact-{i}"} for i in range(20)]
    doc_context = {"key_facts": facts}

    prompt = build_reason_prompt(
        query="q",
        task_type="lookup",
        output_format="prose",
        evidence=[],
        doc_context=doc_context,
    )

    assert "fact-9" in prompt
    assert "fact-10" not in prompt


# ---------------------------------------------------------------------------
# Context builder tests
# ---------------------------------------------------------------------------


def _make_chunk(**overrides) -> EvidenceChunk:
    defaults = dict(
        text="some text",
        source_name="doc.pdf",
        document_id="doc-1",
        profile_id="prof-1",
        section="Intro",
        page_start=1,
        page_end=1,
        score=0.9,
        chunk_id="c-1",
    )
    defaults.update(overrides)
    return EvidenceChunk(**defaults)


def test_context_builder_includes_entities_and_facts():
    """build_context should aggregate entities and key_facts from doc_intelligence."""
    chunks = [_make_chunk(document_id="doc-1")]
    doc_intelligence = {
        "doc-1": {
            "summary": "A report",
            "entities": [
                {"type": "PERSON", "value": "Bob"},
                {"type": "ORG", "value": "Widgets Inc"},
            ],
            "key_facts": [
                {"claim": "Revenue $10M", "evidence": "p.1"},
            ],
        },
    }

    evidence, doc_ctx = build_context(chunks, doc_intelligence)

    assert "entities" in doc_ctx
    assert len(doc_ctx["entities"]) == 2
    assert doc_ctx["entities"][0] == {"type": "PERSON", "value": "Bob"}

    assert "key_facts" in doc_ctx
    assert len(doc_ctx["key_facts"]) == 1
    assert doc_ctx["key_facts"][0]["claim"] == "Revenue $10M"


def test_context_builder_falls_back_to_facts_key():
    """build_context should fall back from 'key_facts' to 'facts'."""
    chunks = [_make_chunk(document_id="doc-1")]
    doc_intelligence = {
        "doc-1": {
            "facts": [
                {"claim": "Profit margin 15%"},
            ],
        },
    }

    _, doc_ctx = build_context(chunks, doc_intelligence)

    assert "key_facts" in doc_ctx
    assert doc_ctx["key_facts"][0]["claim"] == "Profit margin 15%"


def test_context_builder_caps_entities_at_20():
    """Entities should be capped at 20."""
    chunks = [_make_chunk(document_id="doc-1")]
    doc_intelligence = {
        "doc-1": {
            "entities": [{"type": "X", "value": f"e-{i}"} for i in range(30)],
        },
    }

    _, doc_ctx = build_context(chunks, doc_intelligence)

    assert len(doc_ctx["entities"]) == 20


def test_context_builder_caps_facts_at_15():
    """Key facts should be capped at 15."""
    chunks = [_make_chunk(document_id="doc-1")]
    doc_intelligence = {
        "doc-1": {
            "key_facts": [{"claim": f"f-{i}"} for i in range(25)],
        },
    }

    _, doc_ctx = build_context(chunks, doc_intelligence)

    assert len(doc_ctx["key_facts"]) == 15
