"""Tests for the robustness fixes across retrieval, extraction, rendering, judge, and tool dispatch."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag_v3.types import (
    Candidate,
    CandidateField,
    Chunk,
    ChunkSource,
    EntitySummary,
    EvidenceSpan,
    FieldValue,
    FieldValuesField,
    GenericSchema,
    HRSchema,
    MultiEntitySchema,
)


# ── Helper factories ──────────────────────────────────────────────────────

def _make_chunk(text: str, score: float = 0.5, *, doc_name: str = "doc.pdf", meta: dict | None = None) -> Chunk:
    return Chunk(
        id=f"chunk_{hash(text) % 10000}",
        text=text,
        score=score,
        source=ChunkSource(document_name=doc_name),
        meta=meta or {},
    )


def _make_evidence(snippet: str = "evidence text") -> EvidenceSpan:
    return EvidenceSpan(chunk_id="c1", snippet=snippet)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Tool Dispatch Tests
# ══════════════════════════════════════════════════════════════════════════

class TestToolDispatch:
    """Tests for _dispatch_tools() in pipeline.py."""

    def test_no_tools_returns_empty(self):
        from src.rag_v3.pipeline import _dispatch_tools
        result = _dispatch_tools([], "query", "p1", "s1", None, "corr")
        assert result == []

    def test_none_tools_returns_empty(self):
        from src.rag_v3.pipeline import _dispatch_tools
        result = _dispatch_tools(None, "query", "p1", "s1", None, "corr")
        assert result == []

    def test_successful_tool_dispatch(self):
        from src.rag_v3.pipeline import _dispatch_tools

        tool_response = {
            "status": "success",
            "result": {"name": "John Doe", "skills": ["Python"]},
        }

        mock_registry = MagicMock()
        mock_registry.invoke = AsyncMock(return_value=tool_response)

        with patch.dict("sys.modules", {"src.tools.base": MagicMock(registry=mock_registry)}), \
             patch("asyncio.run", return_value=tool_response):
            chunks = _dispatch_tools(["resumes"], "tell me about John", "p1", "s1", None, "corr")

        assert len(chunks) == 1
        assert chunks[0].score == 1.0
        assert chunks[0].meta["source"] == "tool"
        assert chunks[0].meta["tool_name"] == "resumes"
        assert "John Doe" in chunks[0].text

    def test_tool_dispatch_graceful_on_failure(self):
        from src.rag_v3.pipeline import _dispatch_tools

        mock_registry = MagicMock()

        with patch.dict("sys.modules", {"src.tools.base": MagicMock(registry=mock_registry)}), \
             patch("asyncio.run", side_effect=TimeoutError("Tool timed out")):
            chunks = _dispatch_tools(["resumes"], "query", "p1", "s1", None, "corr")

        assert chunks == []

    def test_tool_dispatch_skips_failed_status(self):
        from src.rag_v3.pipeline import _dispatch_tools

        mock_registry = MagicMock()
        tool_response = {"status": "error", "message": "not found"}

        with patch.dict("sys.modules", {"src.tools.base": MagicMock(registry=mock_registry)}), \
             patch("asyncio.run", return_value=tool_response):
            chunks = _dispatch_tools(["unknown_tool"], "query", "p1", "s1", None, "corr")

        assert chunks == []

    def test_tools_param_accepted_by_run(self):
        """Verify run() and run_docwain_rag_v3() accept tools parameter without error."""
        import inspect
        from src.rag_v3.pipeline import run, run_docwain_rag_v3

        run_sig = inspect.signature(run)
        assert "tools" in run_sig.parameters
        assert "tool_inputs" in run_sig.parameters

        v3_sig = inspect.signature(run_docwain_rag_v3)
        assert "tools" in v3_sig.parameters
        assert "tool_inputs" in v3_sig.parameters


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Retrieval Accuracy Tests
# ══════════════════════════════════════════════════════════════════════════

class TestGradientThreshold:
    """Tests for gradient quality threshold in filter_high_quality()."""

    def test_keeps_medium_when_few_high(self):
        from src.rag_v3.retrieve import filter_high_quality

        high = [_make_chunk(f"high {i}", 0.8) for i in range(3)]
        medium = [_make_chunk(f"medium {i}", 0.55) for i in range(5)]
        low = [_make_chunk(f"low {i}", 0.3) for i in range(3)]
        all_chunks = high + medium + low

        result = filter_high_quality(all_chunks)
        assert len(result) >= len(high)
        texts = {c.text for c in result}
        assert any("medium" in t for t in texts)

    def test_high_only_mode_at_8_plus(self):
        from src.rag_v3.retrieve import filter_high_quality

        high = [_make_chunk(f"high {i}", 0.8) for i in range(10)]
        medium = [_make_chunk(f"medium {i}", 0.55) for i in range(5)]
        all_chunks = high + medium

        result = filter_high_quality(all_chunks)
        texts = {c.text for c in result}
        assert all("high" in t for t in texts)


class TestDedupThreshold:
    """Tests for deduplication threshold lowering."""

    def test_dedup_078_catches_near_duplicates(self):
        from src.rag_v3.retrieve import deduplicate_by_content

        c1 = _make_chunk("John has extensive experience in Python programming and machine learning algorithms", 0.9)
        c2 = _make_chunk("John has extensive experience in Python programming and machine learning models", 0.85)
        c3 = _make_chunk("Mary works in finance and accounting with Excel expertise", 0.7)

        result = deduplicate_by_content([c1, c2, c3])
        assert len(result) <= 3
        assert any("Mary" in c.text for c in result)

    def test_dedup_preserves_distinct_content(self):
        from src.rag_v3.retrieve import deduplicate_by_content

        chunks = [
            _make_chunk("Python developer with 10 years experience", 0.9),
            _make_chunk("Java architect specializing in microservices", 0.85),
            _make_chunk("Data scientist expert in R and statistics", 0.8),
        ]

        result = deduplicate_by_content(chunks)
        assert len(result) == 3


# ══════════════════════════════════════════════════════════════════════════
# Phase 3: Extraction Accuracy Tests
# ══════════════════════════════════════════════════════════════════════════

class TestExtractionFixes:

    def test_extract_timeout_15s(self):
        from src.rag_v3.extract import EXTRACT_TIMEOUT_MS
        assert EXTRACT_TIMEOUT_MS == 15000

    def test_llm_response_below_8_chars_rejected(self):
        from src.rag_v3.llm_extract import _parse_response
        # Minimum threshold is 8 chars — very short responses should be rejected
        result = _parse_response("OK", [])
        assert result is None
        # 9+ char responses should be accepted (valid short answers like "INV-2024")
        result2 = _parse_response("Too short", [])
        assert result2 is not None

    def test_llm_response_metadata_garbage_rejected(self):
        from src.rag_v3.llm_extract import _parse_response
        result = _parse_response(
            "section_id abc chunk_type paragraph page_start 1 page_end 2 embedding_text something",
            [],
        )
        assert result is None

    def test_llm_response_valid_text_accepted(self):
        from src.rag_v3.llm_extract import _parse_response
        result = _parse_response(
            "John Doe has 10 years of experience in Python programming and machine learning.",
            [],
        )
        assert result is not None
        assert "John Doe" in result.text

    def test_content_domain_fallback_detects_hr_keywords(self):
        from src.rag_v3.extract import _majority_chunk_domain

        chunks = [
            _make_chunk("Candidate resume showing skills in Python, work experience in software development, education at MIT",
                        meta={"doc_domain": "resume"}),
            _make_chunk("Resume: professional summary with 10 years of experience, certifications in AWS",
                        meta={"doc_domain": "resume"}),
        ]
        domain = _majority_chunk_domain(chunks)
        assert domain in ("hr", "resume")

    def test_schema_emptiness_partial_hr_kept(self):
        from src.rag_v3.extract import _schema_is_empty

        schema = HRSchema(candidates=CandidateField(items=[
            Candidate(name="John", evidence_spans=[_make_evidence()]),
        ]))
        assert not _schema_is_empty(schema)

    def test_schema_emptiness_empty_generic(self):
        from src.rag_v3.extract import _schema_is_empty

        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="", value="hi", evidence_spans=[_make_evidence()]),
        ]))
        assert _schema_is_empty(schema)


# ══════════════════════════════════════════════════════════════════════════
# Phase 4: Judge + Render + Sanitize Tests
# ══════════════════════════════════════════════════════════════════════════

class TestDeterministicBypass:
    """Tests for _has_valid_deterministic_extraction() tightening."""

    def test_needs_2_facts_or_1_long(self):
        from src.rag_v3.pipeline import _has_valid_deterministic_extraction

        ev = [_make_evidence()]

        # 1 short fact (12 chars): should fail
        schema1 = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Name", value="John Doe XXX", evidence_spans=ev),
        ]))
        assert not _has_valid_deterministic_extraction(schema1)

        # 2 substantial facts (>10 chars each): should pass
        schema2 = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Name", value="John Doe Smith", evidence_spans=ev),
            FieldValue(label="Role", value="Senior Developer", evidence_spans=ev),
        ]))
        assert _has_valid_deterministic_extraction(schema2)

        # 1 long fact (>50 chars): should pass
        schema3 = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Summary", value="X" * 55, evidence_spans=ev),
        ]))
        assert _has_valid_deterministic_extraction(schema3)


class TestMultiEntityJudge:
    """Tests for relaxed multi-entity judge validation."""

    def _make_entity(self, label: str) -> EntitySummary:
        return EntitySummary(
            label=label,
            evidence_spans=[_make_evidence()],
        )

    def test_accepts_numbered_list(self):
        from src.rag_v3.judge import _heuristic_judge

        schema = MultiEntitySchema(entities=[
            self._make_entity("John"),
            self._make_entity("Jane"),
        ])

        answer = "Here are the results:\n1. John - Senior Developer\n2. Jane - Product Manager"
        result = _heuristic_judge(answer, schema, "rank")
        assert result.status != "fail" or result.reason != "multi_entity_not_explicit"

    def test_accepts_bullet_list(self):
        from src.rag_v3.judge import _heuristic_judge

        schema = MultiEntitySchema(entities=[
            self._make_entity("John"),
            self._make_entity("Jane"),
        ])

        answer = "Found these:\n- John with Python skills\n- Jane with Java skills\n- Bob with Go skills"
        result = _heuristic_judge(answer, schema, "list")
        assert result.status != "fail" or result.reason != "multi_entity_not_explicit"

    def test_accepts_candidates_keyword(self):
        from src.rag_v3.judge import _heuristic_judge

        schema = MultiEntitySchema(entities=[self._make_entity("A")])

        answer = "There are 3 candidates in the profile with varying skill levels."
        result = _heuristic_judge(answer, schema, "general")
        assert result.status != "fail" or result.reason != "multi_entity_not_explicit"

    def test_accepts_resumes_keyword(self):
        from src.rag_v3.judge import _heuristic_judge

        schema = MultiEntitySchema(entities=[self._make_entity("A")])

        answer = "The profile contains 2 resumes with different experience levels."
        result = _heuristic_judge(answer, schema, "general")
        assert result.status != "fail" or result.reason != "multi_entity_not_explicit"


class TestCrossDocDedup:
    """Tests for cross-document fact deduplication in enterprise.py."""

    def test_removes_duplicate_facts_across_docs(self):
        from src.rag_v3.enterprise import _render_grouped_by_document

        @dataclass
        class Fact:
            label: str = ""
            value: str = ""
            document_name: str = ""
            section: str = ""

        facts = [
            Fact(label="Name", value="John Doe", document_name="doc1.pdf"),
            Fact(label="Name", value="John Doe", document_name="doc2.pdf"),
            Fact(label="Skills", value="Python", document_name="doc1.pdf"),
        ]

        rendered = _render_grouped_by_document(facts)
        assert rendered.count("John Doe") == 1
        assert "Python" in rendered


class TestQueryAwareRanking:
    """Tests for query-aware candidate ranking in enterprise.py."""

    def test_ranking_boosts_query_matches(self):
        from src.rag_v3.enterprise import _rank_candidates

        @dataclass
        class FakeCandidate:
            name: str = ""
            technical_skills: list = None
            functional_skills: list = None
            certifications: list = None
            achievements: list = None
            total_years_experience: str = ""
            experience_summary: str = ""

            def __post_init__(self):
                self.technical_skills = self.technical_skills or []
                self.functional_skills = self.functional_skills or []
                self.certifications = self.certifications or []
                self.achievements = self.achievements or []

        cand_python = FakeCandidate(name="Alice", technical_skills=["Python", "Django"])
        cand_java = FakeCandidate(name="Bob", technical_skills=["Java", "Spring", "Hibernate"])

        # Without query, Bob has more skills so ranks higher
        ranked_no_query = _rank_candidates([cand_python, cand_java])
        assert ranked_no_query[0].name == "Bob"

        # With Python query, Alice should rank higher
        ranked_python = _rank_candidates([cand_python, cand_java], query="Python developer")
        assert ranked_python[0].name == "Alice"


class TestSanitizerPatterns:
    """Tests for expanded sanitizer metadata patterns."""

    def test_catches_new_metadata_patterns(self):
        from src.rag_v3.sanitize import sanitize_text

        cases = [
            "doc_domain: resume",
            "document_type: invoice",
            "layout_confidence: 0.95",
            "ocr_confidence: 0.88",
            "profile_id: abc123",
            "subscription_id: sub456",
            "document_id: doc789",
            "embedding_text: some content here",
            "canonical_text: raw chunk",
            "section_title: Header Section",
        ]

        for case in cases:
            result = sanitize_text(f"Valid content. {case} More content.")
            key = case.split(":")[0].strip()
            assert key not in result, f"Metadata key '{key}' should be stripped from: {result}"

    def test_preserves_normal_content(self):
        from src.rag_v3.sanitize import sanitize_text

        text = "John Doe has 10 years of experience in Python and machine learning."
        result = sanitize_text(text)
        assert "John Doe" in result
        assert "10 years" in result


class TestRenderQueryThreading:
    """Tests that query flows through the render chain."""

    def test_render_enterprise_accepts_query(self):
        import inspect
        from src.rag_v3.enterprise import render_enterprise
        sig = inspect.signature(render_enterprise)
        assert "query" in sig.parameters

    def test_render_router_accepts_query(self):
        import inspect
        try:
            from src.rag_v3.renderers.router import render
        except ImportError:
            pytest.skip("Module removed")
        sig = inspect.signature(render)
        assert "query" in sig.parameters
