"""Tests for retrieval intelligence: query enrichment, section-kind boost, keyword floor, domain inference."""
from __future__ import annotations

import pytest

from src.rag_v3.retrieve import (
    _boost_by_section_kind,
    _enrich_query_for_embedding,
    _infer_domain,
    _infer_query_section_kind,
    _keyword_score,
)
from src.rag_v3.types import Chunk, ChunkSource


def _make_chunk(text: str = "dummy", score: float = 0.5, meta: dict | None = None) -> Chunk:
    return Chunk(
        id="c1",
        text=text,
        score=score,
        source=ChunkSource(document_name="test.pdf", page=1),
        meta=meta or {},
    )


# ── Phase 1: Query enrichment ────────────────────────────────────────────


class TestQueryEnrichment:
    def test_query_enrichment_returns_verbatim(self):
        # After embedding rebuild: queries return as-is for symmetric embedding match
        enriched = _enrich_query_for_embedding("What are the technical skills?")
        assert enriched == "What are the technical skills?"
        assert "technical skills" in enriched

    def test_query_enrichment_no_prefix_for_generic(self):
        query = "How many pages does this document have?"
        enriched = _enrich_query_for_embedding(query)
        assert enriched == query

    def test_infer_query_section_kind_education(self):
        assert _infer_query_section_kind("What is the candidate's education?") == "education"

    def test_infer_query_section_kind_experience(self):
        assert _infer_query_section_kind("Tell me about work history") == "experience"

    def test_infer_query_section_kind_none_for_generic(self):
        assert _infer_query_section_kind("Summarize the document") is None


# ── Phase 2: Section-kind boosting ───────────────────────────────────────


class TestSectionKindBoost:
    def test_section_kind_boost_matching(self):
        c1 = _make_chunk(score=0.5, meta={"section_kind": "skills_technical"})
        c2 = _make_chunk(score=0.6, meta={"section_kind": "experience"})
        c2.id = "c2"
        result = _boost_by_section_kind([c1, c2], "What are the technical skills?")
        # c1 had 0.5 + 0.12 = 0.62, c2 stays 0.6 → c1 should be first
        assert result[0].score == pytest.approx(0.62)
        assert result[0].id == "c1"

    def test_section_kind_boost_no_match(self):
        c1 = _make_chunk(score=0.5, meta={"section_kind": "experience"})
        result = _boost_by_section_kind([c1], "What are the technical skills?")
        assert result[0].score == pytest.approx(0.5)

    def test_section_kind_boost_generic_query(self):
        c1 = _make_chunk(score=0.5, meta={"section_kind": "skills_technical"})
        result = _boost_by_section_kind([c1], "Tell me about this document")
        assert result[0].score == pytest.approx(0.5)


# ── Phase 4: Keyword score floor ─────────────────────────────────────────


class TestKeywordScoreFloor:
    def test_keyword_score_floor_when_match_exists(self):
        # 1 match out of 10 tokens = 0.1, should be floored to 0.2
        tokens = ["zzz1", "zzz2", "zzz3", "zzz4", "zzz5", "zzz6", "zzz7", "zzz8", "zzz9", "python"]
        score = _keyword_score("I know python well", tokens)
        assert score == pytest.approx(0.2)

    def test_keyword_score_no_floor_when_ratio_above(self):
        tokens = ["python", "java"]
        score = _keyword_score("python and java", tokens)
        assert score == pytest.approx(1.0)

    def test_keyword_score_zero_when_no_match(self):
        tokens = ["python", "java"]
        score = _keyword_score("nothing relevant here", tokens)
        assert score == 0.0

    def test_keyword_score_zero_when_empty(self):
        assert _keyword_score("", ["python"]) == 0.0
        assert _keyword_score("python", []) == 0.0


# ── Phase 3: Domain inference ─────────────────────────────────────────────


class TestDomainInference:
    def test_infer_domain_resume(self):
        assert _infer_domain("Show me the candidate's resume") == "hr"

    def test_infer_domain_invoice(self):
        assert _infer_domain("What is the invoice total?") == "invoice"

    def test_infer_domain_generic(self):
        assert _infer_domain("Tell me about the project") == "generic"

    def test_infer_domain_legal(self):
        assert _infer_domain("What does the contract clause say?") == "legal"
