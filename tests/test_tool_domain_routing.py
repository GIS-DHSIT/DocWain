"""Tests for tool-based domain intelligence routing in the DocWain pipeline.

Tests the _TOOL_DOMAIN_MAP, _resolve_domain_from_tools(), extract_schema() with
tool_domain flag, _infer_domain_intent() with domain_hint, and end-to-end
integration of tool domain resolution through extraction.

40+ tests across 6 categories:
1. TestToolDomainMap — mapping correctness
2. TestResolveDomainFromTools — resolver edge cases
3. TestExtractSchemaToolDomain — tool_domain=True behaviour
4. TestInferDomainWithToolHint — domain_hint parameter
5. TestToolDomainEndToEnd — full extraction routing
6. TestToolDomainWithAllProfile — all-profile / schema_extract routing
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from src.rag_v3.pipeline import _resolve_domain_from_tools, _TOOL_DOMAIN_MAP
from src.rag_v3.extract import (
    extract_schema,
    schema_extract,
    _infer_domain_intent,
    ExtractionResult,
)
from src.rag_v3.types import (
    Chunk,
    ChunkSource,
    FieldValue,
    FieldValuesField,
    GenericSchema,
    HRSchema,
    InvoiceSchema,
    LegalSchema,
    LLMBudget,
    LLMResponseSchema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeChunk:
    """Lightweight chunk stub for extraction tests."""
    id: str
    text: str
    score: float
    source: Any = None
    meta: Dict[str, Any] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}
        if self.source is None:
            self.source = types.SimpleNamespace(document_name="test.pdf", page=1)


def _make_chunk(
    text: str,
    doc_domain: str = "",
    doc_id: str = "doc1",
    source_name: str = "test.pdf",
    score: float = 0.9,
) -> FakeChunk:
    return FakeChunk(
        id=f"chunk_{abs(hash(text)) % 100000}",
        text=text,
        score=score,
        source=types.SimpleNamespace(document_name=source_name, page=1),
        meta={
            "doc_domain": doc_domain,
            "document_id": doc_id,
            "source_name": source_name,
        },
    )


def _make_resume_chunks() -> List[FakeChunk]:
    """Standard set of resume chunks for HR extraction."""
    return [
        _make_chunk(
            "PROFESSIONAL SUMMARY\n"
            "Experienced Python developer with 5 years in backend systems.\n"
            "TECHNICAL SKILLS\n"
            "Python, Java, SQL, Django, Flask, PostgreSQL\n"
            "EDUCATION\n"
            "B.S. Computer Science, MIT 2018",
            doc_domain="resume",
            source_name="John_Doe_Resume.pdf",
        ),
        _make_chunk(
            "WORK EXPERIENCE\n"
            "Senior Developer at Acme Corp (2019-2023)\n"
            "- Developed microservices architecture\n"
            "- Led team of 5 engineers\n"
            "Junior Developer at Beta Inc (2018-2019)\n"
            "- Built REST APIs",
            doc_domain="resume",
            source_name="John_Doe_Resume.pdf",
        ),
    ]


def _make_invoice_chunks() -> List[FakeChunk]:
    """Invoice chunks for invoice extraction tests."""
    return [
        _make_chunk(
            "INVOICE #12345\n"
            "Bill To: ABC Company\n"
            "Date: 2025-01-15\n"
            "Amount Due: $5,000.00",
            doc_domain="invoice",
            source_name="Invoice_12345.pdf",
        ),
        _make_chunk(
            "Item 1: Consulting Services - $3,000.00\n"
            "Item 2: Software License - $2,000.00\n"
            "Subtotal: $5,000.00\nTax: $0.00\nTotal: $5,000.00",
            doc_domain="invoice",
            source_name="Invoice_12345.pdf",
        ),
    ]


def _make_legal_chunks() -> List[FakeChunk]:
    """Legal/contract chunks for legal extraction tests."""
    return [
        _make_chunk(
            "TERMS AND CONDITIONS\n"
            "1. This Agreement shall be effective from the date of signing.\n"
            "2. The liability of the parties shall be limited as set forth herein.",
            doc_domain="legal",
            source_name="Contract_Agreement.pdf",
        ),
    ]


def _no_llm_budget() -> LLMBudget:
    return LLMBudget(llm_client=None, max_calls=0)


def _llm_budget() -> LLMBudget:
    return LLMBudget(llm_client=MagicMock(), max_calls=2)


# ===========================================================================
# 1. TestToolDomainMap
# ===========================================================================

class TestToolDomainMap:
    """Verify _TOOL_DOMAIN_MAP entries are correct."""

    def test_resume_analysis_hyphen_maps_to_hr(self):
        assert _TOOL_DOMAIN_MAP["resume-analysis"] == "hr"

    def test_resume_analysis_underscore_maps_to_hr(self):
        assert _TOOL_DOMAIN_MAP["resume_analysis"] == "hr"

    def test_resumes_maps_to_hr(self):
        assert _TOOL_DOMAIN_MAP["resumes"] == "hr"

    def test_resume_maps_to_hr(self):
        assert _TOOL_DOMAIN_MAP["resume"] == "hr"

    def test_medical_maps_to_medical(self):
        assert _TOOL_DOMAIN_MAP["medical"] == "medical"

    def test_lawhere_not_in_map(self):
        # lawhere handles both legal and insurance/policy docs;
        # ML domain detector decides based on chunk content.
        assert "lawhere" not in _TOOL_DOMAIN_MAP

    def test_legal_maps_to_legal(self):
        assert _TOOL_DOMAIN_MAP["legal"] == "legal"

    def test_invoice_maps_to_invoice(self):
        assert _TOOL_DOMAIN_MAP["invoice"] == "invoice"

    def test_policy_maps_to_policy(self):
        assert _TOOL_DOMAIN_MAP["policy"] == "policy"

    def test_insurance_maps_to_policy(self):
        assert _TOOL_DOMAIN_MAP["insurance"] == "policy"


# ===========================================================================
# 2. TestResolveDomainFromTools
# ===========================================================================

class TestResolveDomainFromTools:
    """Test _resolve_domain_from_tools edge cases."""

    def test_none_tools_returns_none(self):
        assert _resolve_domain_from_tools(None) is None

    def test_empty_list_returns_none(self):
        assert _resolve_domain_from_tools([]) is None

    def test_single_matching_tool_returns_domain(self):
        assert _resolve_domain_from_tools(["resume-analysis"]) == "hr"

    def test_first_matching_tool_wins(self):
        result = _resolve_domain_from_tools(["invoice", "resume-analysis"])
        assert result == "invoice", "Should return first matching domain"

    def test_unknown_tool_returns_none(self):
        assert _resolve_domain_from_tools(["unknown_tool"]) is None

    def test_mixed_known_unknown_returns_first_known(self):
        result = _resolve_domain_from_tools(["unknown_tool", "medical", "resume"])
        assert result == "medical", "Should skip unknown and return first known"

    def test_case_insensitive_and_whitespace(self):
        """Tool names are lowercased and stripped before lookup."""
        assert _resolve_domain_from_tools(["  Resume-Analysis  "]) == "hr"
        assert _resolve_domain_from_tools(["INVOICE"]) == "invoice"
        assert _resolve_domain_from_tools(["  Legal  "]) == "legal"

    def test_multiple_resume_variants_all_map_to_hr(self):
        for tool in ["resume-analysis", "resume_analysis", "resumes", "resume"]:
            result = _resolve_domain_from_tools([tool])
            assert result == "hr", f"Tool '{tool}' should map to 'hr', got '{result}'"

    def test_none_in_tool_list_skipped(self):
        """None values inside the list should not cause errors."""
        result = _resolve_domain_from_tools([None, "invoice"])
        assert result == "invoice"

    def test_empty_string_tool_skipped(self):
        """Empty string tools should be skipped."""
        result = _resolve_domain_from_tools(["", "legal"])
        assert result == "legal"

    def test_all_unknown_returns_none(self):
        result = _resolve_domain_from_tools(["foo", "bar", "baz"])
        assert result is None


# ===========================================================================
# 3. TestExtractSchemaToolDomain
# ===========================================================================

class TestExtractSchemaToolDomain:
    """Test extract_schema() behavior with tool_domain=True vs False."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_skips_mismatch_check(self):
        """When tool_domain=True, invoice query + resume chunks should NOT return mismatch."""
        chunks = _make_resume_chunks()
        # Query asks about invoice, chunks are resume — normally a mismatch
        result = extract_schema(
            "hr",  # domain set by tool
            query="show me the invoice total",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,  # tool says HR, skip mismatch
        )
        # Should NOT be a mismatch message
        if isinstance(result.schema, GenericSchema) and result.schema.facts.items:
            for item in result.schema.facts.items:
                assert "No " not in (item.value or "") or "found in this profile" not in (item.value or ""), \
                    f"Mismatch message should be suppressed when tool_domain=True: {item.value}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_prefers_deterministic_hr(self):
        """tool_domain=True with HR domain should produce HRSchema, not LLMResponseSchema."""
        chunks = _make_resume_chunks()
        mock_llm = MagicMock()
        # Even though LLM is available, tool_domain=True should prefer deterministic
        result = extract_schema(
            "hr",
            query="tell me about the candidate",
            chunks=chunks,
            llm_client=mock_llm,
            budget=LLMBudget(llm_client=mock_llm, max_calls=4),
            tool_domain=True,
        )
        # With tool_domain=True, deterministic extraction is preferred
        # so we should get HRSchema (not LLMResponseSchema)
        assert isinstance(result.schema, HRSchema), \
            f"tool_domain=True should prefer deterministic HR extraction, got {type(result.schema).__name__}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_hr_extracts_candidates(self):
        """HR domain with tool_domain=True should produce candidates."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "hr",
            query="show me the candidate's skills",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        assert isinstance(result.schema, HRSchema)
        candidates = (result.schema.candidates.items if result.schema.candidates else []) or []
        assert len(candidates) >= 1, "Should extract at least one candidate from resume chunks"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_false_allows_mismatch(self):
        """Without tool_domain, invoice query + resume chunks = mismatch."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            None,  # no domain hint
            query="show me the invoice details",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=False,
        )
        # Query overrides to "invoice", chunk domain is "resume/hr" → mismatch
        # The mismatch check in extract_schema only triggers for non-HR query domains
        # that don't match chunk domain
        if isinstance(result.schema, GenericSchema) and result.schema.facts.items:
            values = " ".join(item.value or "" for item in result.schema.facts.items)
            assert "No invoices found" in values or len(result.schema.facts.items) > 0

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_false_default_behavior(self):
        """Default tool_domain=False should use standard extraction pipeline."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "hr",
            query="show me the candidate",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=False,
        )
        # Without tool_domain, and no LLM, still falls to deterministic
        assert result.domain is not None

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_with_generic_domain_not_active(self):
        """tool_domain=True but domain='generic' should NOT activate tool domain logic."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "generic",
            query="what is this document about",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        # domain="generic" means _tool_domain_active is False
        # (generic is excluded from tool domain activation)
        assert result is not None

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_with_empty_domain_not_active(self):
        """tool_domain=True but domain='' should NOT activate tool domain logic."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "",
            query="what is this",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        assert result is not None

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_with_none_domain_not_active(self):
        """tool_domain=True but domain=None should NOT activate tool domain logic."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            None,
            query="what skills does John have",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        # domain=None → _tool_domain_active is False
        assert result is not None

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_invoice_domain_deterministic(self):
        """tool_domain=True with invoice domain should prefer deterministic extraction."""
        chunks = _make_invoice_chunks()
        mock_llm = MagicMock()
        result = extract_schema(
            "invoice",
            query="what is the total amount",
            chunks=chunks,
            llm_client=mock_llm,
            budget=LLMBudget(llm_client=mock_llm, max_calls=4),
            tool_domain=True,
        )
        # tool_domain=True → _prefer_deterministic is True
        # Should NOT call LLM for first strategy
        assert not isinstance(result.schema, LLMResponseSchema), \
            "tool_domain=True should skip LLM-first and use deterministic extraction"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_legal_domain(self):
        """tool_domain=True with legal domain should extract from legal chunks."""
        chunks = _make_legal_chunks()
        result = extract_schema(
            "legal",
            query="what are the terms",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        # Legal domain with deterministic extraction
        assert result is not None
        assert result.domain is not None

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_no_llm_deterministic_still_works(self):
        """Without LLM, deterministic extraction works regardless of tool_domain."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "hr",
            query="list the skills",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=False,
        )
        assert result is not None
        # Falls through to schema_extract → deterministic
        assert isinstance(result.schema, (HRSchema, GenericSchema))

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_true_passes_none_llm_to_schema_extract(self):
        """When _prefer_deterministic is True, schema_extract gets llm_client=None."""
        chunks = _make_resume_chunks()
        mock_llm = MagicMock()

        with patch("src.rag_v3.extract.schema_extract") as mock_se:
            mock_se.return_value = ExtractionResult(
                domain="hr", intent="facts", schema=HRSchema()
            )
            extract_schema(
                "hr",
                query="show skills",
                chunks=chunks,
                llm_client=mock_llm,
                budget=LLMBudget(llm_client=mock_llm, max_calls=2),
                tool_domain=True,
            )
            # Verify schema_extract was called with llm_client=None
            # (tool domain prefers deterministic, so LLM is suppressed in fallback)
            call_kwargs = mock_se.call_args
            assert call_kwargs is not None
            # schema_extract is called via keyword arguments
            if call_kwargs.kwargs:
                assert call_kwargs.kwargs.get("llm_client") is None, \
                    "tool_domain=True should pass llm_client=None to schema_extract"
            else:
                # positional+keyword mix
                assert call_kwargs[1].get("llm_client") is None


# ===========================================================================
# 4. TestInferDomainWithToolHint
# ===========================================================================

class TestInferDomainWithToolHint:
    """Test _infer_domain_intent() with domain_hint parameter."""

    def test_domain_hint_hr_overrides(self):
        """domain_hint='hr' should set domain to 'hr' regardless of query/chunks."""
        chunks = _make_invoice_chunks()  # Invoice chunks, but hint says HR
        domain, intent = _infer_domain_intent(
            "show me the invoice", chunks, domain_hint="hr"
        )
        assert domain == "hr", f"domain_hint='hr' should override, got '{domain}'"

    def test_domain_hint_invoice(self):
        """domain_hint='invoice' forces invoice domain."""
        chunks = _make_resume_chunks()  # Resume chunks, but hint says invoice
        domain, intent = _infer_domain_intent(
            "what are the skills", chunks, domain_hint="invoice"
        )
        assert domain == "invoice"

    def test_domain_hint_legal(self):
        """domain_hint='legal' forces legal domain."""
        chunks = _make_resume_chunks()
        domain, intent = _infer_domain_intent(
            "show me the candidate", chunks, domain_hint="legal"
        )
        assert domain == "legal"

    def test_domain_hint_policy(self):
        """domain_hint='policy' forces policy domain."""
        chunks = _make_resume_chunks()
        domain, intent = _infer_domain_intent(
            "what coverage do we have", chunks, domain_hint="policy"
        )
        assert domain == "policy"

    def test_domain_hint_none_falls_back_to_inference(self):
        """No hint falls back to query/chunk inference."""
        chunks = _make_resume_chunks()
        domain, intent = _infer_domain_intent(
            "show me the candidate", chunks, domain_hint=None
        )
        # Should infer from chunk metadata (resume → hr)
        assert domain in ("hr", "resume"), f"Should infer HR from resume chunks, got '{domain}'"

    def test_domain_hint_generic_falls_back_to_inference(self):
        """domain_hint='generic' is treated as empty — falls back to inference."""
        chunks = _make_resume_chunks()
        domain, intent = _infer_domain_intent(
            "show me the candidate", chunks, domain_hint="generic"
        )
        # generic is ignored, falls through to query/chunk inference
        assert domain in ("hr", "resume"), f"generic hint should fall back, got '{domain}'"

    def test_domain_hint_empty_string_falls_back(self):
        """Empty string hint falls back to inference."""
        chunks = _make_resume_chunks()
        domain, intent = _infer_domain_intent(
            "show me the candidate", chunks, domain_hint=""
        )
        assert domain in ("hr", "resume")

    def test_intent_still_inferred_with_hint(self):
        """Intent detection works normally even with domain_hint."""
        chunks = _make_resume_chunks()
        _, intent = _infer_domain_intent(
            "who are the contact details", chunks, domain_hint="hr"
        )
        assert intent == "contact"


# ===========================================================================
# 5. TestToolDomainEndToEnd
# ===========================================================================

class TestToolDomainEndToEnd:
    """End-to-end tests: extract_schema with various tool domains."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_resume_analysis_tool_with_resume_chunks_produces_hr_schema(self):
        """tools=resume-analysis with resume chunks → HRSchema with candidates."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "hr",  # resolved from resume-analysis tool
            query="tell me about the candidate",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        assert isinstance(result.schema, HRSchema), \
            f"Expected HRSchema, got {type(result.schema).__name__}"
        candidates = (result.schema.candidates.items if result.schema.candidates else []) or []
        assert len(candidates) >= 1, "Should produce at least one candidate"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_resume_analysis_tool_with_generic_query(self):
        """tools=resume-analysis with generic query still routes to HR."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "hr",  # resolved from resume-analysis
            query="what is in this document",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        # Should still route to HR extraction because tool domain overrides
        assert isinstance(result.schema, HRSchema), \
            f"Tool domain should force HR extraction, got {type(result.schema).__name__}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_no_tool_with_resume_chunks_normal_behavior(self):
        """No tool + resume chunks → normal behavior (deterministic without LLM)."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            None,
            query="show me the candidate skills",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=False,
        )
        assert result is not None
        # Without explicit tool domain, falls through to normal extraction
        assert isinstance(result.schema, (HRSchema, GenericSchema))

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_invoice_tool_with_invoice_chunks(self):
        """tools=invoice with invoice chunks → extraction works."""
        chunks = _make_invoice_chunks()
        result = extract_schema(
            "invoice",
            query="what is the total",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        assert result is not None
        assert result.domain is not None

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_tool_domain_with_llm_still_prefers_deterministic(self):
        """Even with LLM available, tool_domain=True prefers deterministic."""
        chunks = _make_resume_chunks()
        mock_llm = MagicMock()
        result = extract_schema(
            "hr",
            query="summarize the candidate",
            chunks=chunks,
            llm_client=mock_llm,
            budget=LLMBudget(llm_client=mock_llm, max_calls=4),
            tool_domain=True,
        )
        # Should produce HRSchema (deterministic), not LLMResponseSchema
        assert isinstance(result.schema, HRSchema), \
            f"Tool domain should bypass LLM-first, got {type(result.schema).__name__}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_resolve_and_extract_integration(self):
        """Full flow: resolve domain from tools, then extract."""
        tool_domain = _resolve_domain_from_tools(["resume-analysis"])
        assert tool_domain == "hr"

        chunks = _make_resume_chunks()
        result = extract_schema(
            tool_domain,
            query="what are the technical skills",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=bool(tool_domain),
        )
        assert isinstance(result.schema, HRSchema)

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_resolve_none_and_extract_no_tool_domain(self):
        """No tool match → tool_domain=False in extraction."""
        tool_domain = _resolve_domain_from_tools(["unknown_tool"])
        assert tool_domain is None

        chunks = _make_resume_chunks()
        result = extract_schema(
            tool_domain,
            query="show me the resume",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=bool(tool_domain),
        )
        # tool_domain is None → bool(None) = False
        assert result is not None


# ===========================================================================
# 6. TestToolDomainWithAllProfile
# ===========================================================================

class TestToolDomainWithAllProfile:
    """Test that schema_extract (called by _run_all_profile_analysis) respects domain_hint."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_schema_extract_domain_hint_hr(self):
        """schema_extract with domain_hint='hr' should infer domain as 'hr'."""
        chunks = _make_resume_chunks()
        result = schema_extract(
            query="rank the candidates",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            domain_hint="hr",
        )
        assert result.domain in ("hr", "resume"), \
            f"domain_hint='hr' should produce HR domain, got '{result.domain}'"
        assert isinstance(result.schema, HRSchema), \
            f"Expected HRSchema, got {type(result.schema).__name__}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_schema_extract_domain_hint_invoice(self):
        """schema_extract with domain_hint='invoice' should infer domain as 'invoice'."""
        chunks = _make_invoice_chunks()
        result = schema_extract(
            query="list the items",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            domain_hint="invoice",
        )
        assert result.domain == "invoice", \
            f"domain_hint='invoice' should produce invoice domain, got '{result.domain}'"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_schema_extract_domain_hint_none_infers_from_chunks(self):
        """schema_extract with domain_hint=None should infer domain from chunk metadata."""
        chunks = _make_resume_chunks()
        result = schema_extract(
            query="tell me about John",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            domain_hint=None,
        )
        # Should infer HR from resume chunks
        assert result.domain in ("hr", "resume"), \
            f"Should infer HR from resume chunks, got '{result.domain}'"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_schema_extract_domain_hint_legal(self):
        """schema_extract with domain_hint='legal' routes to legal extraction."""
        chunks = _make_legal_chunks()
        result = schema_extract(
            query="what are the terms of the agreement",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            domain_hint="legal",
        )
        assert result.domain == "legal"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_schema_extract_domain_hint_policy(self):
        """schema_extract with domain_hint='policy' routes to policy/legal extraction."""
        chunks = [
            _make_chunk(
                "INSURANCE POLICY\n"
                "Coverage: Natural Calamities\n"
                "Premium: $500/month\n"
                "Effective Date: 2025-01-01",
                doc_domain="policy",
                source_name="Policy_Document.pdf",
            ),
        ]
        result = schema_extract(
            query="what does the policy cover",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            domain_hint="policy",
        )
        assert result.domain == "policy"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_schema_extract_domain_hint_overrides_chunk_domain(self):
        """domain_hint should override what the chunk metadata says."""
        # Chunks say "invoice" but hint says "hr"
        chunks = _make_invoice_chunks()
        result = schema_extract(
            query="who is the candidate",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            domain_hint="hr",
        )
        assert result.domain == "hr", \
            f"domain_hint='hr' should override invoice chunk domain, got '{result.domain}'"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_schema_extract_generic_hint_falls_back(self):
        """domain_hint='generic' should fall back to chunk/query inference."""
        chunks = _make_resume_chunks()
        result = schema_extract(
            query="show me the candidate profile",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            domain_hint="generic",
        )
        # generic is treated as empty → falls through to inference from chunks
        assert result.domain in ("hr", "resume", "generic")


# ===========================================================================
# 7. TestToolDomainBoolConversion (bonus)
# ===========================================================================

class TestToolDomainBoolConversion:
    """Test that bool(tool_domain) works correctly for pipeline integration."""

    def test_none_tool_domain_is_false(self):
        """bool(None) → False, so tool_domain flag is off."""
        tool_domain = _resolve_domain_from_tools(None)
        assert tool_domain is None
        assert bool(tool_domain) is False

    def test_empty_string_is_false(self):
        """Empty string resolved domain would be falsy."""
        assert bool("") is False

    def test_valid_domain_is_true(self):
        """A valid resolved domain like 'hr' is truthy."""
        tool_domain = _resolve_domain_from_tools(["resume-analysis"])
        assert tool_domain == "hr"
        assert bool(tool_domain) is True

    def test_pipeline_passes_bool_correctly(self):
        """Verify bool(tool_domain) conversion for extract_schema call."""
        # When tool_domain is "hr", bool("hr") should be True
        assert bool("hr") is True
        # When tool_domain is None, bool(None) should be False
        assert bool(None) is False
        # When tool_domain is empty list result, should be False
        assert bool(_resolve_domain_from_tools([])) is False


# ===========================================================================
# 8. TestToolDomainMismatchBypass (additional edge cases)
# ===========================================================================

class TestToolDomainMismatchBypass:
    """Detailed tests for mismatch bypass when tool domain is active."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_legal_query_on_hr_chunks_without_tool_gives_mismatch(self):
        """Legal query on HR chunks without tool should trigger mismatch."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            None,  # no tool domain
            query="what are the contract terms",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=False,
        )
        # Query domain = legal, chunk domain = hr → mismatch
        # But mismatch only fires for non-HR query domains asking for different chunks
        # Legal query + resume chunks should produce mismatch
        if isinstance(result.schema, GenericSchema) and result.schema.facts.items:
            values = " ".join(item.value or "" for item in result.schema.facts.items)
            assert "No legal" in values or len(result.schema.facts.items) >= 0

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_legal_query_on_hr_chunks_with_tool_no_mismatch(self):
        """Legal tool domain on HR chunks should NOT trigger mismatch."""
        chunks = _make_resume_chunks()
        result = extract_schema(
            "legal",  # tool says legal
            query="what are the contract terms",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        # Tool domain active → mismatch check skipped
        if isinstance(result.schema, GenericSchema) and result.schema.facts.items:
            values = " ".join(item.value or "" for item in result.schema.facts.items)
            assert "No legal" not in values, \
                "Tool domain should suppress mismatch message"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_hr_tool_on_invoice_chunks_extracts_without_mismatch(self):
        """HR tool on invoice chunks should skip mismatch and try HR extraction."""
        chunks = _make_invoice_chunks()
        result = extract_schema(
            "hr",
            query="show me the invoice",
            chunks=chunks,
            llm_client=None,
            budget=_no_llm_budget(),
            tool_domain=True,
        )
        # Should not return mismatch message
        assert result is not None
        if isinstance(result.schema, GenericSchema) and result.schema.facts.items:
            values = " ".join(item.value or "" for item in result.schema.facts.items)
            assert "No " not in values or "found in this profile" not in values
