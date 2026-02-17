"""Comprehensive tests for the content generation system.

Tests cover:
- Registry: type lookup, domain listing, detection
- Engine: 6-step pipeline, deterministic fallbacks, error handling
- Prompts: template building, fact injection
- Verifier: grounding checks, hallucination detection
- Tool bridge: gateway integration
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.content_generation.registry import (
    CONTENT_TYPE_REGISTRY,
    DOMAINS,
    ContentType,
    detect_content_type,
    detect_content_type_with_domain,
    get_content_type,
    list_content_types,
    list_domains,
)
from src.content_generation.prompts import ContentPromptBuilder
from src.content_generation.verifier import ContentVerifier, VerificationResult
from src.content_generation.engine import (
    ContentGenerationEngine,
    _build_evidence_text,
    _extract_facts_from_chunks,
    _get_chunk_meta,
    _get_chunk_text,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str = "Sample chunk text",
    *,
    source_name: str = "resume.pdf",
    doc_domain: str = "hr",
    document_id: str = "doc1",
    page: int = 1,
    score: float = 0.8,
) -> SimpleNamespace:
    """Create a mock chunk object."""
    return SimpleNamespace(
        text=text,
        score=score,
        metadata={
            "source_name": source_name,
            "doc_domain": doc_domain,
            "document_id": document_id,
            "page": page,
        },
    )


def _make_dict_chunk(
    text: str = "Sample chunk text",
    *,
    source_name: str = "resume.pdf",
    document_id: str = "doc1",
    score: float = 0.8,
) -> Dict[str, Any]:
    """Create a dict-form chunk."""
    return {
        "canonical_text": text,
        "text": text,
        "score": score,
        "metadata": {
            "source_name": source_name,
            "document_id": document_id,
        },
    }


class FakeLLMClient:
    """Fake LLM client that returns canned responses."""

    def __init__(self, response: str = "Generated content here."):
        self._response = response
        self.calls: List[str] = []

    def generate_with_metadata(self, prompt: str, **kwargs) -> tuple:
        self.calls.append(prompt)
        return (self._response, {})


# ===================================================================
# Test Registry
# ===================================================================


class TestContentTypeRegistry:
    """Tests for content type registration and lookup."""

    def test_registry_not_empty(self):
        assert len(CONTENT_TYPE_REGISTRY) >= 25

    def test_all_types_have_required_fields(self):
        for type_id, ct in CONTENT_TYPE_REGISTRY.items():
            assert ct.id == type_id
            assert ct.domain in DOMAINS
            assert ct.name
            assert ct.description

    def test_get_content_type_exists(self):
        ct = get_content_type("cover_letter")
        assert ct is not None
        assert ct.domain == "hr"
        assert ct.name == "Cover Letter"

    def test_get_content_type_not_found(self):
        assert get_content_type("nonexistent_type_xyz") is None

    def test_list_content_types_all(self):
        types = list_content_types()
        assert len(types) >= 25

    def test_list_content_types_by_domain(self):
        hr_types = list_content_types(domain="hr")
        assert len(hr_types) >= 5
        assert all(ct.domain == "hr" for ct in hr_types)

    def test_list_content_types_empty_domain(self):
        types = list_content_types(domain="nonexistent_domain")
        assert types == []

    def test_list_domains(self):
        domains = list_domains()
        assert len(domains) == len(DOMAINS)
        domain_ids = {d["id"] for d in domains}
        assert "hr" in domain_ids
        assert "invoice" in domain_ids
        assert "general" in domain_ids

    def test_domains_cover_all_types(self):
        """Every content type's domain should exist in DOMAINS."""
        for ct in CONTENT_TYPE_REGISTRY.values():
            assert ct.domain in DOMAINS, f"{ct.id} has unknown domain {ct.domain}"

    def test_hr_domain_types(self):
        hr = list_content_types(domain="hr")
        ids = {ct.id for ct in hr}
        assert "cover_letter" in ids
        assert "professional_summary" in ids
        assert "skills_matrix" in ids
        assert "candidate_comparison" in ids
        assert "interview_prep" in ids

    def test_invoice_domain_types(self):
        inv = list_content_types(domain="invoice")
        ids = {ct.id for ct in inv}
        assert "invoice_summary" in ids
        assert "expense_report" in ids
        assert "payment_reminder" in ids

    def test_cross_doc_types_support_multi_doc(self):
        cross = list_content_types(domain="cross_document")
        for ct in cross:
            assert ct.supports_multi_doc, f"{ct.id} should support multi-doc"
            assert ct.min_chunks >= 2, f"{ct.id} should require min 2 chunks"

    def test_content_type_frozen(self):
        ct = get_content_type("cover_letter")
        with pytest.raises(AttributeError):
            ct.id = "something_else"


# ===================================================================
# Test Detection
# ===================================================================


class TestContentTypeDetection:
    """Tests for natural language → content type detection."""

    def test_detect_cover_letter(self):
        assert detect_content_type("write a cover letter for this candidate") == "cover_letter"

    def test_detect_professional_summary(self):
        assert detect_content_type("generate a professional summary") == "professional_summary"

    def test_detect_skills_matrix(self):
        assert detect_content_type("create a skills matrix") == "skills_matrix"

    def test_detect_candidate_comparison(self):
        assert detect_content_type("compare candidates") == "candidate_comparison"

    def test_detect_interview_prep(self):
        assert detect_content_type("prepare interview questions") == "interview_prep"

    def test_detect_invoice_summary(self):
        assert detect_content_type("summarize the invoices") == "invoice_summary"

    def test_detect_expense_report(self):
        assert detect_content_type("generate an expense report") == "expense_report"

    def test_detect_payment_reminder(self):
        assert detect_content_type("write a payment reminder") == "payment_reminder"

    def test_detect_contract_summary(self):
        assert detect_content_type("summarize the contract") == "contract_summary"

    def test_detect_executive_summary(self):
        assert detect_content_type("write an executive summary") == "executive_summary"

    def test_detect_key_points(self):
        assert detect_content_type("extract the key points") == "key_points"

    def test_detect_faq(self):
        assert detect_content_type("generate FAQ from this document") == "faq_generation"

    def test_detect_action_items(self):
        assert detect_content_type("list the action items") == "action_items"

    def test_detect_talking_points(self):
        assert detect_content_type("create talking points") == "talking_points"

    def test_detect_trend_analysis(self):
        assert detect_content_type("do a trend analysis") == "trend_analysis"

    def test_detect_none_for_unrelated_query(self):
        assert detect_content_type("what is the weather today") is None

    def test_detect_none_for_empty(self):
        assert detect_content_type("") is None

    def test_detect_case_insensitive(self):
        assert detect_content_type("WRITE A COVER LETTER") == "cover_letter"

    def test_detect_with_domain_hr(self):
        ct = detect_content_type_with_domain(
            "generate something for this resume",
            chunk_domain="hr",
        )
        assert ct is not None
        assert ct.domain == "hr"

    def test_detect_with_domain_invoice(self):
        ct = detect_content_type_with_domain(
            "create a report from invoice data",
            chunk_domain="invoice",
        )
        assert ct is not None
        assert ct.domain == "invoice"

    def test_detect_with_domain_no_generate_keyword(self):
        ct = detect_content_type_with_domain(
            "how many pages does this document have",
            chunk_domain="hr",
        )
        assert ct is None

    def test_detect_with_domain_general_fallback(self):
        ct = detect_content_type_with_domain(
            "generate a summary of this",
            chunk_domain=None,
        )
        assert ct is not None
        assert ct.id == "document_summary"


# ===================================================================
# Test Prompts
# ===================================================================


class TestContentPromptBuilder:
    """Tests for prompt building."""

    def setup_method(self):
        self.builder = ContentPromptBuilder()

    def test_build_system_prompt_hr(self):
        ct = get_content_type("cover_letter")
        prompt = self.builder.build_system_prompt(ct)
        assert "HR" in prompt or "professional" in prompt.lower()

    def test_build_system_prompt_invoice(self):
        ct = get_content_type("invoice_summary")
        prompt = self.builder.build_system_prompt(ct)
        assert "financial" in prompt.lower()

    def test_build_system_prompt_fallback(self):
        ct = ContentType(id="test", domain="unknown", name="Test", description="Test")
        prompt = self.builder.build_system_prompt(ct)
        assert "content writer" in prompt.lower()

    def test_build_generation_prompt_includes_task(self):
        ct = get_content_type("cover_letter")
        prompt = self.builder.build_generation_prompt(
            ct, {"skills": ["Python", "Java"]}, "some evidence", "write a cover letter",
        )
        assert "Cover Letter" in prompt
        assert "write a cover letter" in prompt

    def test_build_generation_prompt_includes_facts(self):
        ct = get_content_type("cover_letter")
        prompt = self.builder.build_generation_prompt(
            ct, {"person_name": "Alice", "skills": ["Python"]}, "evidence text",
        )
        assert "Alice" in prompt
        assert "Python" in prompt

    def test_build_generation_prompt_includes_evidence(self):
        ct = get_content_type("document_summary")
        prompt = self.builder.build_generation_prompt(
            ct, {}, "This is the evidence from documents.",
        )
        assert "This is the evidence from documents." in prompt

    def test_build_generation_prompt_truncates_evidence(self):
        ct = get_content_type("document_summary")
        long_evidence = "x" * 10000
        prompt = self.builder.build_generation_prompt(ct, {}, long_evidence)
        assert "[Evidence truncated]" in prompt

    def test_build_generation_prompt_includes_grounding(self):
        ct = get_content_type("document_summary")
        prompt = self.builder.build_generation_prompt(ct, {}, "evidence")
        assert "CRITICAL RULES" in prompt
        assert "hallucinate" in prompt.lower()

    def test_build_generation_prompt_extra_instructions(self):
        ct = get_content_type("document_summary")
        prompt = self.builder.build_generation_prompt(
            ct, {}, "evidence", extra_instructions="Focus on recent experience.",
        )
        assert "Focus on recent experience." in prompt

    def test_build_fact_extraction_prompt(self):
        ct = get_content_type("cover_letter")
        prompt = self.builder.build_fact_extraction_prompt(ct, "sample evidence")
        assert "person_name" in prompt
        assert "skills" in prompt
        assert "JSON" in prompt


# ===================================================================
# Test Verifier
# ===================================================================


class TestContentVerifier:
    """Tests for content grounding verification."""

    def setup_method(self):
        self.verifier = ContentVerifier()
        self.strict_verifier = ContentVerifier(strict=True)

    def test_empty_generated_text(self):
        result = self.verifier.verify("", [_make_chunk()])
        assert not result.grounded
        assert result.score == 0.0

    def test_no_claims_is_grounded(self):
        result = self.verifier.verify(
            "This is a general statement with no specific claims.",
            [_make_chunk("General information about documents.")],
        )
        assert result.grounded
        assert result.total_claims == 0

    def test_verified_numbers(self):
        chunk = _make_chunk("The candidate has 5 years of experience with Python.")
        result = self.verifier.verify(
            "The candidate has 5 years of experience.",
            [chunk],
        )
        assert result.grounded
        assert result.verified_claims > 0

    def test_unverified_number_hallucination(self):
        chunk = _make_chunk("The candidate knows Python and Java.")
        result = self.strict_verifier.verify(
            "The candidate has 15 years of experience with Python.",
            [chunk],
        )
        # Should flag the number 15 as unverified
        assert "15" in result.unverified_claims

    def test_named_entity_verification(self):
        chunk = _make_chunk("John Smith works at Acme Corp.")
        result = self.verifier.verify(
            "John Smith is an employee at Acme Corp.",
            [chunk],
        )
        assert result.grounded

    def test_fabricated_email_warning(self):
        chunk = _make_chunk("Contact us for more information.")
        result = self.verifier.verify(
            "You can reach out at fake@example.com for details.",
            [chunk],
        )
        assert any("email" in w.lower() for w in result.warnings)

    def test_fabricated_phone_warning(self):
        chunk = _make_chunk("The office is located downtown.")
        result = self.verifier.verify(
            "Call us at 555-123-4567 for appointments.",
            [chunk],
        )
        assert any("phone" in w.lower() for w in result.warnings)

    def test_email_in_evidence_no_warning(self):
        chunk = _make_chunk("Contact: real@company.com for support.")
        result = self.verifier.verify(
            "You can email real@company.com for assistance.",
            [chunk],
        )
        email_warnings = [w for w in result.warnings if "email" in w.lower()]
        assert not email_warnings

    def test_strict_vs_lenient_threshold(self):
        chunk = _make_chunk("Alice has skills in Python and data science.")
        text = "Alice has 10 years of experience in Python and data science at Google."
        strict = self.strict_verifier.verify(text, [chunk])
        lenient = self.verifier.verify(text, [chunk])
        # Strict may fail where lenient passes
        assert lenient.score >= strict.score or lenient.score == strict.score

    def test_verification_result_to_dict(self):
        result = VerificationResult(
            grounded=True, score=0.95, total_claims=10, verified_claims=9,
            unverified_claims=["claim1"], warnings=["warn1"],
        )
        d = result.to_dict()
        assert d["grounded"] is True
        assert d["score"] == 0.95
        assert d["total_claims"] == 10
        assert d["unverified_claims"] == ["claim1"]

    def test_dict_chunks_supported(self):
        chunk = _make_dict_chunk("Alice is a Python developer with 5 years experience.")
        result = self.verifier.verify(
            "Alice has 5 years of experience.",
            [chunk],
        )
        assert result.grounded

    def test_facts_used_as_evidence(self):
        result = self.verifier.verify(
            "Alice has skills in Python.",
            [],  # no chunks
            facts={"person_name": "Alice", "skills": ["Python"]},
        )
        assert result.grounded


# ===================================================================
# Test Engine Helpers
# ===================================================================


class TestEngineHelpers:
    """Tests for engine helper functions."""

    def test_get_chunk_text_object(self):
        chunk = _make_chunk("Hello world")
        assert _get_chunk_text(chunk) == "Hello world"

    def test_get_chunk_text_dict(self):
        chunk = {"canonical_text": "From dict", "text": "Fallback"}
        assert _get_chunk_text(chunk) == "From dict"

    def test_get_chunk_text_dict_fallback(self):
        chunk = {"text": "Fallback text"}
        assert _get_chunk_text(chunk) == "Fallback text"

    def test_get_chunk_meta_object(self):
        chunk = _make_chunk("text")
        meta = _get_chunk_meta(chunk)
        assert meta["source_name"] == "resume.pdf"

    def test_get_chunk_meta_dict(self):
        chunk = {"metadata": {"key": "value"}}
        assert _get_chunk_meta(chunk) == {"key": "value"}

    def test_build_evidence_text(self):
        chunks = [
            _make_chunk("First chunk", source_name="doc1.pdf"),
            _make_chunk("Second chunk", source_name="doc2.pdf"),
        ]
        evidence = _build_evidence_text(chunks)
        assert "First chunk" in evidence
        assert "Second chunk" in evidence
        assert "doc1.pdf" in evidence

    def test_build_evidence_text_truncation(self):
        chunks = [_make_chunk("x" * 8000)]
        evidence = _build_evidence_text(chunks, max_chars=100)
        assert "[Evidence truncated]" in evidence

    def test_extract_facts_hr(self):
        ct = get_content_type("cover_letter")
        chunks = [
            _make_chunk(
                "Name: Alice Johnson\n"
                "Skills: Python, Java, Machine Learning\n"
                "Experience: 5 years in software development\n"
                "Certified in AWS Solutions Architect",
                source_name="alice_resume.pdf",
            ),
        ]
        facts = _extract_facts_from_chunks(chunks, ct)
        assert "person_name" in facts
        assert "skills" in facts
        assert len(facts["skills"]) > 0

    def test_extract_facts_invoice(self):
        ct = get_content_type("invoice_summary")
        chunks = [
            _make_chunk(
                "Invoice #12345\n"
                "Date: 01/15/2026\n"
                "Total: $1,500.00\n"
                "Vendor: Acme Supplies Inc.",
            ),
        ]
        facts = _extract_facts_from_chunks(chunks, ct)
        assert "amounts" in facts
        assert "$1,500.00" in facts["amounts"]

    def test_extract_facts_with_filename(self):
        ct = get_content_type("professional_summary")
        chunks = [
            _make_chunk("Some text without names", source_name="John_Smith_Resume.pdf"),
        ]
        facts = _extract_facts_from_chunks(chunks, ct)
        assert "person_name" in facts


# ===================================================================
# Test Engine Pipeline
# ===================================================================


class TestContentGenerationEngine:
    """Tests for the full 6-step engine pipeline."""

    def test_empty_chunks_returns_empty(self):
        engine = ContentGenerationEngine()
        result = engine.generate("write a cover letter", [])
        assert result["context_found"] is False
        assert "Insufficient" in result["response"]

    def test_no_content_type_detected(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk("some text")]
        result = engine.generate("what is the weather", chunks)
        assert "Could not determine" in result["response"]

    def test_deterministic_cover_letter(self):
        engine = ContentGenerationEngine()
        chunks = [
            _make_chunk(
                "Name: Alice Johnson\nSkills: Python, Java, AWS\n"
                "Company: TechCorp\nExperience: 3 years in backend development",
            ),
        ]
        result = engine.generate("write a cover letter", chunks)
        assert result["context_found"] is True
        assert "Dear Hiring Manager" in result["response"]
        assert result["metadata"]["content_type"] == "cover_letter"
        assert result["metadata"]["generation_method"] == "deterministic"

    def test_deterministic_key_points(self):
        engine = ContentGenerationEngine()
        chunks = [
            _make_chunk("This document discusses cloud architecture patterns and best practices."),
        ]
        result = engine.generate("extract key points", chunks)
        assert result["context_found"] is True
        assert "Key Points" in result["response"]
        assert result["metadata"]["content_type"] == "key_points"

    def test_deterministic_summary_fallback(self):
        engine = ContentGenerationEngine()
        chunks = [
            _make_chunk("Patient: John Doe. Diagnosis: Type 2 Diabetes. Medication: Metformin."),
        ]
        result = engine.generate("create a patient summary", chunks)
        assert result["context_found"] is True
        assert result["metadata"]["content_type"] == "patient_summary"

    def test_llm_generation(self):
        llm = FakeLLMClient("Generated professional summary for Alice.")
        engine = ContentGenerationEngine(llm_client=llm)
        chunks = [
            _make_chunk("Alice Johnson has 5 years of experience in Python."),
        ]
        result = engine.generate("write a professional summary", chunks)
        assert "Generated professional summary for Alice" in result["response"]
        assert result["metadata"]["generation_method"] == "llm"
        assert len(llm.calls) == 1

    def test_llm_timeout_falls_back_to_deterministic(self):
        llm = MagicMock()
        llm.generate_with_metadata.side_effect = Exception("Timeout")
        engine = ContentGenerationEngine(llm_client=llm)
        chunks = [
            _make_chunk("Name: Bob Smith\nSkills: JavaScript, React"),
        ]
        result = engine.generate("write a cover letter", chunks)
        assert result["context_found"] is True
        assert "Dear Hiring Manager" in result["response"]
        assert result["metadata"]["generation_method"] == "deterministic"

    def test_explicit_content_type_id(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk("Some resume text with Python skills")]
        result = engine.generate(
            "generate something",
            chunks,
            content_type_id="skills_matrix",
        )
        assert result["metadata"]["content_type"] == "skills_matrix"

    def test_chunk_domain_hint(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk("Invoice data: $5000 from Acme")]
        result = engine.generate(
            "generate a summary",
            chunks,
            chunk_domain="invoice",
        )
        assert result["metadata"]["domain"] == "invoice"

    def test_sources_in_result(self):
        engine = ContentGenerationEngine()
        chunks = [
            _make_chunk("Alice skills", source_name="alice.pdf"),
            _make_chunk("Bob skills", source_name="bob.pdf"),
        ]
        result = engine.generate("write a cover letter", chunks)
        assert len(result["sources"]) >= 1
        source_names = {s["source_name"] for s in result["sources"]}
        assert "alice.pdf" in source_names or "bob.pdf" in source_names

    def test_verification_metadata(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk("Alice has Python skills and 3 years experience.")]
        result = engine.generate("write a cover letter", chunks)
        assert "verification" in result["metadata"]
        v = result["metadata"]["verification"]
        assert "grounded" in v
        assert "score" in v

    def test_dict_chunks_accepted(self):
        engine = ContentGenerationEngine()
        chunks = [
            _make_dict_chunk(
                "Name: Charlie Brown\nSkills: React, Node.js\n5 years experience",
                source_name="charlie_cv.pdf",
            ),
        ]
        result = engine.generate("write a cover letter", chunks)
        assert result["context_found"] is True

    def test_multi_doc_requirement_enforced(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk("single doc")]
        result = engine.generate(
            "some query",
            chunks,
            content_type_id="comparison_report",
        )
        assert "requires at least" in result["response"]

    def test_llm_override_per_call(self):
        default_llm = FakeLLMClient("default response")
        override_llm = FakeLLMClient("override response")
        engine = ContentGenerationEngine(llm_client=default_llm)
        chunks = [_make_chunk("Alice skills")]
        result = engine.generate(
            "write a professional summary",
            chunks,
            llm_client=override_llm,
        )
        assert "override response" in result["response"]
        assert len(override_llm.calls) == 1
        assert len(default_llm.calls) == 0

    def test_top_chunks_selection(self):
        chunks = [
            _make_chunk("low score", score=0.1),
            _make_chunk("high score with Python skills", score=0.9),
            _make_chunk("medium score", score=0.5),
        ]
        selected = ContentGenerationEngine._select_top_chunks(chunks)
        assert selected[0].score == 0.9


# ===================================================================
# Test Tool Bridge
# ===================================================================


class TestToolBridge:
    """Tests for gateway tool bridge integration."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_content_generate_registered(self):
        import src.content_generation.tool_bridge  # noqa: F401 — trigger registration
        from src.tools.base import registry
        assert "content_generate" in registry._registry

    def test_content_types_registered(self):
        import src.content_generation.tool_bridge  # noqa: F401 — trigger registration
        from src.tools.base import registry
        assert "content_types" in registry._registry

    def test_content_types_handler(self):
        from src.content_generation.tool_bridge import content_types_handler
        result = self._run(content_types_handler({"input": {}}))
        assert result["result"]["total"] >= 25
        assert len(result["result"]["content_types"]) >= 25
        assert len(result["result"]["domains"]) == len(DOMAINS)

    def test_content_types_handler_filter_domain(self):
        from src.content_generation.tool_bridge import content_types_handler
        result = self._run(content_types_handler({"input": {"domain": "hr"}}))
        assert result["result"]["total"] >= 5
        for ct in result["result"]["content_types"]:
            assert ct["domain"] == "hr"

    def test_content_generate_handler_no_query(self):
        from src.content_generation.tool_bridge import content_generate_handler
        result = self._run(content_generate_handler({"input": {}}))
        assert result["grounded"] is False
        assert "error" in result["result"]

    def test_content_generate_handler_with_chunks(self):
        from src.content_generation.tool_bridge import content_generate_handler
        chunks = [
            {
                "text": "Alice Johnson\nSkills: Python, Java\n3 years experience at TechCorp",
                "score": 0.8,
                "metadata": {"source_name": "alice.pdf", "document_id": "doc1"},
            },
        ]
        result = self._run(content_generate_handler({
            "input": {
                "query": "write a cover letter",
                "chunks": chunks,
            },
        }))
        assert result["context_found"] is True
        assert result["result"]["content_type"] == "cover_letter"

    def test_content_generate_handler_explicit_type(self):
        from src.content_generation.tool_bridge import content_generate_handler
        chunks = [
            {
                "text": "Python, Java, AWS, Docker, Kubernetes",
                "score": 0.9,
                "metadata": {"source_name": "resume.pdf", "document_id": "doc1"},
            },
        ]
        result = self._run(content_generate_handler({
            "input": {
                "query": "organize these skills",
                "content_type": "skills_matrix",
                "chunks": chunks,
            },
        }))
        assert result["result"]["content_type"] == "skills_matrix"


# ===================================================================
# Test Edge Cases
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_chunk_text(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk("")]
        result = engine.generate("write a cover letter", chunks)
        # Should still attempt generation with whatever is available
        assert "response" in result

    def test_none_llm_client(self):
        engine = ContentGenerationEngine(llm_client=None)
        chunks = [_make_chunk("Skills: Python, Java")]
        result = engine.generate("write a cover letter", chunks)
        assert result["metadata"]["generation_method"] == "deterministic"

    def test_large_number_of_chunks(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk(f"Chunk {i} text content") for i in range(50)]
        result = engine.generate("write a cover letter", chunks)
        # Should not crash, should select top chunks
        assert "response" in result

    def test_special_characters_in_query(self):
        engine = ContentGenerationEngine()
        chunks = [_make_chunk("Alice's resume with Python & Java")]
        result = engine.generate(
            "write a cover letter for Alice's application (urgent!)",
            chunks,
        )
        assert "response" in result

    def test_mixed_chunk_types(self):
        engine = ContentGenerationEngine()
        chunks = [
            _make_chunk("Object chunk text"),
            _make_dict_chunk("Dict chunk text"),
        ]
        result = engine.generate("write a cover letter", chunks)
        assert "response" in result

    def test_verifier_with_no_evidence(self):
        verifier = ContentVerifier()
        result = verifier.verify("Some generated text.", [])
        # With no evidence, claims can't be verified but no crash
        assert isinstance(result, VerificationResult)
