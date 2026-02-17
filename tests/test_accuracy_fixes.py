"""Tests for accuracy fixes: LLM timeout, HR domain skip removal, domain classification,
name extraction, and deterministic extraction improvements."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ── LLM Extract Timeout + Context Reduction ─────────────────────────


class TestLLMExtractTimeout:
    def test_timeout_is_60_seconds(self):
        from src.rag_v3.llm_extract import LLM_EXTRACT_TIMEOUT_S
        assert LLM_EXTRACT_TIMEOUT_S == 60.0

    def test_max_context_chars_reduced(self):
        from src.rag_v3.llm_extract import LLM_MAX_CONTEXT_CHARS
        assert LLM_MAX_CONTEXT_CHARS == 6144

    def test_max_chunks_limit_exists(self):
        from src.rag_v3.llm_extract import LLM_MAX_CHUNKS
        assert LLM_MAX_CHUNKS == 8

    def test_top_chunks_selected_by_score(self):
        """Verify only top-scored chunks are sent to LLM."""
        from src.rag_v3.llm_extract import llm_extract_and_respond
        from src.rag_v3.types import LLMBudget

        # Create 20 mock chunks with varying scores
        chunks = []
        for i in range(20):
            c = MagicMock()
            c.score = float(i) / 20
            c.text = f"Chunk {i} content about software engineering"
            c.id = f"chunk_{i}"
            c.meta = {"source_name": "doc.pdf"}
            c.source = None
            chunks.append(c)

        client = MagicMock()
        budget = LLMBudget(llm_client=client, max_calls=2)
        # Return a valid answer
        client.generate_with_metadata.return_value = (
            "This is a detailed answer about software engineering with enough text to pass validation.",
            {},
        )

        result = llm_extract_and_respond(
            query="What are the skills?",
            chunks=chunks,
            llm_client=client,
            budget=budget,
        )

        # Should have been called once
        assert client.generate_with_metadata.call_count == 1
        # The prompt should NOT contain all 20 chunks
        prompt = client.generate_with_metadata.call_args[0][0]
        # Should only include top chunks (by score)
        assert "Chunk 0" not in prompt  # Low-score chunk excluded


class TestHRDomainSkipRemoved:
    """LLM-first extraction should work for ALL domains including HR."""

    def test_llm_result_returned_for_hr_domain(self):
        from src.rag_v3.extract import extract_schema
        from src.rag_v3.types import LLMBudget, LLMResponseSchema

        chunks = []
        for i in range(3):
            c = MagicMock()
            c.text = f"John Doe is an experienced software engineer with Python expertise."
            c.id = f"chunk_{i}"
            c.score = 0.9
            c.meta = {"document_id": "doc1", "source_name": "resume.pdf", "doc_domain": "resume"}
            c.source = None
            chunks.append(c)

        client = MagicMock()
        budget = LLMBudget(llm_client=client, max_calls=2)
        client.generate_with_metadata.return_value = (
            "John Doe is an experienced software engineer with strong Python skills and 10 years of experience in web development.",
            {},
        )

        result = extract_schema(
            "resume",
            query="Tell me about this candidate",
            chunks=chunks,
            llm_client=client,
            budget=budget,
        )

        # Should return LLM result, NOT fall through to deterministic
        assert isinstance(result.schema, LLMResponseSchema)

    def test_llm_result_returned_for_generic_domain(self):
        from src.rag_v3.extract import extract_schema
        from src.rag_v3.types import LLMBudget, LLMResponseSchema

        chunks = []
        for i in range(3):
            c = MagicMock()
            c.text = f"The quarterly revenue was $1.2M with 15% growth."
            c.id = f"chunk_{i}"
            c.score = 0.8
            c.meta = {"document_id": "doc1", "source_name": "report.pdf"}
            c.source = None
            chunks.append(c)

        client = MagicMock()
        budget = LLMBudget(llm_client=client, max_calls=2)
        client.generate_with_metadata.return_value = (
            "The quarterly revenue was $1.2M, representing a 15% growth from the previous quarter.",
            {},
        )

        result = extract_schema(
            "generic",
            query="What was the revenue?",
            chunks=chunks,
            llm_client=client,
            budget=budget,
        )

        assert isinstance(result.schema, LLMResponseSchema)


# ── Domain Classification Fixes ─────────────────────────────────────


class TestDomainClassificationFixes:
    def test_purchase_order_no_false_positive_on_resume(self):
        """Resume text should NOT be classified as purchase_order."""
        from src.intelligence.domain_classifier import classify_domain

        resume_text = (
            "John Doe - Software Engineer\n"
            "Skills: Python, Java, Docker\n"
            "Experience: 10 years\n"
            "Education: B.Tech Computer Science\n"
            "Managed vendor relationships and procurement processes.\n"
            "Responsible for order tracking and item management."
        )
        result = classify_domain(resume_text, metadata={"source_name": "John_Resume.pdf"})
        assert result.domain == "resume"

    def test_short_keywords_require_word_boundaries(self):
        """Short keywords like 'cv' should not match inside words."""
        from src.intelligence.domain_classifier import _score_keywords

        text = "The service provided excellent customer value."
        scores = _score_keywords(text)
        # "cv" is inside "service" — should NOT match
        assert scores.get("resume", 0) == 0

    def test_cv_as_word_matches(self):
        """'cv' as a standalone word should match resume domain."""
        from src.intelligence.domain_classifier import _score_keywords

        text = "Please find my cv attached."
        scores = _score_keywords(text)
        assert scores["resume"] > 0

    def test_resume_keywords_expanded(self):
        from src.intelligence.domain_classifier import DOMAIN_KEYWORDS
        assert "technical skills" in DOMAIN_KEYWORDS["resume"]
        assert "professional experience" in DOMAIN_KEYWORDS["resume"]

    def test_purchase_order_keywords_no_po(self):
        """Short 'po' keyword should be removed (too many false positives)."""
        from src.intelligence.domain_classifier import DOMAIN_KEYWORDS
        assert "po" not in DOMAIN_KEYWORDS["purchase_order"]
        assert "item" not in DOMAIN_KEYWORDS["purchase_order"]
        assert "vendor" not in DOMAIN_KEYWORDS["purchase_order"]

    def test_schema_normalizer_no_reclassify_generic(self):
        """When doc_domain is 'generic' (intentionally classified), don't reclassify per chunk."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "source_name": "report.pdf",
            "canonical_text": "The vendor management and purchase order process was improved by 30%.",
            "section_title": "Results",
            "chunk_id": "c1",
            "doc_domain": "generic",  # Explicitly set as generic
        }
        payload = build_qdrant_payload(raw)
        # Should keep "generic" — not reclassify as purchase_order
        assert payload["doc_domain"] == "generic"


# ── Name Extraction Fixes ───────────────────────────────────────────


class TestNameExtractionFixes:
    def test_name_from_filename_strips_update(self):
        from src.rag_v3.extract import _name_from_filename

        result = _name_from_filename("Dhayal CV Update1.pdf")
        assert result is not None
        assert "Dhayal" in result
        # "Update1" should be stripped by the regex
        assert "update" not in result.lower()

    def test_name_from_filename_preserves_dev_name(self):
        from src.rag_v3.extract import _name_from_filename

        result = _name_from_filename("Dev_Resume_IP.pdf")
        # "Dev" is a valid name (e.g., Dev Patel); "IP" is noise
        # After stripping "resume" and "ip", "Dev" remains as a valid name
        assert result is not None
        assert "Dev" in result

    def test_name_from_filename_normal_name(self):
        from src.rag_v3.extract import _name_from_filename

        result = _name_from_filename("Bharath Kumar_Resume.pdf")
        assert result is not None
        assert "Bharath" in result
        assert "Kumar" in result

    def test_name_from_filename_gaurav(self):
        from src.rag_v3.extract import _name_from_filename

        result = _name_from_filename("Gaurav Fegade_SAP EWM.pdf")
        assert result is not None
        assert "Gaurav" in result or "Fegade" in result

    def test_last_resort_name_from_filename_in_extract_hr(self):
        """When all name extraction fails, use cleaned filename as last resort."""
        from src.rag_v3.extract import _extract_hr

        # Create chunks with no name patterns in text, but with source_name
        chunks = []
        for i in range(2):
            c = MagicMock()
            c.text = "SAP EWM configuration and warehouse management processes."
            c.id = f"chunk_{i}"
            c.meta = {
                "document_id": "doc1",
                "source_name": "Dhayal CV Update1.pdf",
                "section_kind": "experience",
            }
            c.source = None
            chunks.append(c)

        result = _extract_hr(chunks)
        candidates = result.candidates.items or []
        assert len(candidates) >= 1
        # Should have extracted name from filename
        cand = candidates[0]
        assert cand.name is not None
        assert "Dhayal" in cand.name


# ── Deterministic Extraction Improvements ────────────────────────────


class TestDeterministicExtractionImprovements:
    def test_education_extraction_btech(self):
        from src.rag_v3.extract import _extract_education_from_text

        text = "B.Tech in Computer Science from XYZ University, 2015"
        result = _extract_education_from_text(text)
        assert len(result) > 0
        assert any("B.Tech" in e or "btech" in e.lower() for e in result)

    def test_education_extraction_bcom(self):
        from src.rag_v3.extract import _extract_education_from_text

        text = "B.Com from St. Xavier's College, Mumbai. Graduated in 2012."
        result = _extract_education_from_text(text)
        assert len(result) > 0

    def test_education_extraction_university_line(self):
        from src.rag_v3.extract import _extract_education_from_text

        text = "University of Mumbai\nBachelor of Engineering in IT\n2010-2014"
        result = _extract_education_from_text(text)
        assert len(result) > 0

    def test_education_extraction_line_by_line(self):
        from src.rag_v3.extract import _extract_education_from_text

        text = (
            "SAP Skills:\nMM, SD, WM, EWM\n\n"
            "Education:\nB.Tech in Mechanical Engineering from Anna University, 2012\n"
            "MBA in Operations from IIM, 2016"
        )
        result = _extract_education_from_text(text)
        assert len(result) >= 1
        assert any("B.Tech" in e for e in result)

    def test_certification_extraction_itil(self):
        from src.rag_v3.extract import _extract_certifications_from_text

        text = "Certifications: ITIL Foundation, Six Sigma Green Belt"
        result = _extract_certifications_from_text(text)
        assert len(result) > 0
        assert any("ITIL" in c for c in result)

    def test_certification_extraction_line_by_line(self):
        from src.rag_v3.extract import _extract_certifications_from_text

        text = (
            "Certification Achieved:\n"
            "AWS Certified Solutions Architect\n"
            "Google Cloud Professional Data Engineer"
        )
        result = _extract_certifications_from_text(text)
        assert len(result) > 0

    def test_experience_summary_multi_line(self):
        """Experience summary should aggregate multiple short lines."""
        from src.rag_v3.extract import _extract_hr

        chunks = []
        c = MagicMock()
        c.text = (
            "Experienced SAP consultant\n"
            "with 8 years of hands-on experience\n"
            "in warehouse management and logistics"
        )
        c.id = "chunk_0"
        c.meta = {
            "document_id": "doc1",
            "source_name": "Consultant_Resume.pdf",
            "section_kind": "section_text",
        }
        c.source = None
        chunks.append(c)

        result = _extract_hr(chunks)
        candidates = result.candidates.items or []
        assert len(candidates) >= 1
        cand = candidates[0]
        # Should have aggregated experience summary from multiple lines
        if cand.experience_summary:
            assert len(cand.experience_summary) > 30


# ── LLM Generation Prompt Quality ───────────────────────────────────


class TestLLMPromptQuality:
    def test_evidence_truncated_to_max_context(self):
        from src.rag_v3.llm_extract import build_generation_prompt, LLM_MAX_CONTEXT_CHARS

        # Create very long evidence
        long_evidence = "A" * 20000
        prompt = build_generation_prompt(
            query="test query",
            evidence_text=long_evidence,
            intent="factual",
        )
        # Evidence should be truncated
        assert len(prompt) < LLM_MAX_CONTEXT_CHARS + 2000

    def test_chunk_text_truncated_in_evidence(self):
        from src.rag_v3.llm_extract import _build_grouped_evidence

        # Create chunk with very long text
        chunk = MagicMock()
        chunk.text = "X" * 2000
        chunk.id = "c1"
        chunk.meta = {"source_name": "doc.pdf"}
        chunk.source = None

        evidence = _build_grouped_evidence([chunk])
        # Chunk text should be truncated to 800 chars
        assert len(evidence) <= 900  # 800 + some headers


# ── Intent Classification ────────────────────────────────────────────


class TestIntentClassificationAccuracy:
    def test_resume_query_is_factual_or_summary(self):
        from src.rag_v3.llm_extract import classify_query_intent

        result = classify_query_intent("Tell me about this candidate's experience")
        assert result in ("factual", "summary")

    def test_ranking_query(self):
        from src.rag_v3.llm_extract import classify_query_intent

        result = classify_query_intent("Rank all candidates by experience")
        assert result == "ranking"

    def test_comparison_query(self):
        from src.rag_v3.llm_extract import classify_query_intent

        result = classify_query_intent("Compare the skills of John and Jane")
        assert result == "comparison"

    def test_all_candidates_is_cross_document(self):
        from src.rag_v3.llm_extract import classify_query_intent

        result = classify_query_intent("What are the skills of all candidates?")
        assert result == "cross_document"


# ── Section Kind Re-classification on Domain Change ─────────────────


class TestQueryDomainOverride:
    """Tests that query-based domain detection overrides chunk majority vote."""

    def test_resume_in_query_forces_hr(self):
        from src.rag_v3.extract import _query_domain_override
        assert _query_domain_override("List the skills mentioned in Abinaya's resume") == "hr"

    def test_cv_in_query_forces_hr(self):
        from src.rag_v3.extract import _query_domain_override
        assert _query_domain_override("What is the education in John's CV?") == "hr"

    def test_candidate_in_query_forces_hr(self):
        from src.rag_v3.extract import _query_domain_override
        assert _query_domain_override("Tell me about the candidate") == "hr"

    def test_invoice_in_query_forces_invoice(self):
        from src.rag_v3.extract import _query_domain_override
        assert _query_domain_override("What is the total invoice amount?") == "invoice"

    def test_skills_plus_education_forces_hr(self):
        from src.rag_v3.extract import _query_domain_override
        assert _query_domain_override("What are the skills and education?") == "hr"

    def test_generic_query_returns_none(self):
        from src.rag_v3.extract import _query_domain_override
        assert _query_domain_override("Tell me about the document") is None

    def test_domain_override_in_infer_domain_intent(self):
        """_infer_domain_intent should use query override even when chunks say invoice."""
        from src.rag_v3.extract import _infer_domain_intent

        # Create mock chunks with invoice domain
        chunk = MagicMock()
        chunk.meta = {"doc_domain": "invoice"}
        chunk.text = "Invoice total: $5000"
        chunks = [chunk] * 5

        domain, intent = _infer_domain_intent("List skills in Abinaya's resume", chunks)
        assert domain == "hr", f"Domain should be 'hr' for resume query, got '{domain}'"


class TestSectionKindReclassification:
    """Tests that invoice section kinds are re-classified when document domain is resume."""

    def test_invoice_kinds_reclassified_for_resume(self):
        """When doc_domain=resume and section_kind is an invoice kind, re-classify."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "source_name": "John_Resume.pdf",
            "doc_domain": "resume",
            "section_kind": "financial_summary",
            "canonical_text": "Certified SAP SCM Consultant with strong implementation experience in SAP S/4 HANA. Skills include Python, Java, SQL, Docker, Kubernetes.",
        }
        payload = build_qdrant_payload(raw)
        # Section kind should NOT remain as "financial_summary" for a resume
        assert payload["section_kind"] != "financial_summary"

    def test_invoice_kinds_preserved_for_invoice(self):
        """Invoice section kinds should be preserved when doc_domain is invoice."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "source_name": "INV12345.pdf",
            "doc_domain": "invoice",
            "section_kind": "financial_summary",
            "canonical_text": "Total: $5,000.00. Tax: $500.00. Grand Total: $5,500.00.",
        }
        payload = build_qdrant_payload(raw)
        assert payload["section_kind"] == "financial_summary"

    def test_all_invoice_kinds_reclassified(self):
        """All invoice-specific section kinds should be caught."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        invoice_kinds = ["financial_summary", "line_items", "invoice_metadata", "parties_addresses", "terms_conditions"]
        for kind in invoice_kinds:
            raw = {
                "subscription_id": "sub1",
                "profile_id": "prof1",
                "document_id": "doc1",
                "source_name": "Resume.pdf",
                "doc_domain": "resume",
                "section_kind": kind,
                "canonical_text": "Experience in Python and Java development. Skills include Docker, Kubernetes, AWS.",
            }
            payload = build_qdrant_payload(raw)
            assert payload["section_kind"] != kind, f"section_kind {kind} should be re-classified for resume"


class TestHRExtractionSectionKindInference:
    """Tests that _extract_hr re-infers section_kind for invoice-tagged chunks."""

    def _make_chunk(self, text, section_kind="section_text", doc_id="doc1", source_name="Resume.pdf"):
        chunk = MagicMock()
        chunk.text = text
        chunk.id = f"chunk_{hash(text) % 10000}"
        chunk.score = 0.8
        chunk.meta = {
            "section_kind": section_kind,
            "document_id": doc_id,
            "source_name": source_name,
            "section_title": "",
        }
        return chunk

    def test_certifications_found_despite_invoice_section_kind(self):
        """Certifications should be extracted even when section_kind is 'line_items'."""
        from src.rag_v3.extract import _extract_hr

        chunks = [
            self._make_chunk(
                "SAP S/4 HANA – Sales\nSAP S/4 HANA – Production Planning & Manufacturing\n"
                "SAP Analytics Cloud (SAC)\nCAPM – Certified Associate in Project Management\n"
                "Business Analysis – Fundamental",
                section_kind="line_items",
            ),
            self._make_chunk(
                "K V MADHU AADITHYA\nSAP SCM Consultant\nContact: +91 9488213034\nMadhuaadithya25@gmail.com",
                section_kind="invoice_metadata",
            ),
        ]
        result = _extract_hr(chunks)
        candidates = result.candidates.items if result.candidates else []
        assert len(candidates) >= 1
        cand = candidates[0]
        # Should find at least some certifications
        assert cand.certifications, "Certifications should be extracted from line_items section_kind"

    def test_skills_found_despite_invoice_section_kind(self):
        """Technical skills should be extracted even when section_kind is 'financial_summary'."""
        from src.rag_v3.extract import _extract_hr

        chunks = [
            self._make_chunk(
                "Technical Skills:\nPython, Java, SQL, Docker, Kubernetes, AWS, React, Node.js",
                section_kind="financial_summary",
            ),
        ]
        result = _extract_hr(chunks)
        candidates = result.candidates.items if result.candidates else []
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.technical_skills, "Skills should be found despite wrong section_kind"


class TestCertificationPatterns:
    """Tests for enhanced certification extraction patterns."""

    def test_capm_found(self):
        from src.rag_v3.extract import _extract_certifications_from_text

        text = "CAPM – Certified Associate in Project Management"
        result = _extract_certifications_from_text(text)
        assert len(result) >= 1
        assert any("CAPM" in c for c in result)

    def test_sap_modules_found(self):
        from src.rag_v3.extract import _extract_certifications_from_text

        text = "SAP S/4 HANA – Sales\nSAP S/4 HANA – Production Planning\nSAP Analytics Cloud (SAC)"
        result = _extract_certifications_from_text(text)
        assert len(result) >= 2

    def test_business_analysis_found(self):
        from src.rag_v3.extract import _extract_certifications_from_text

        text = "Business Analysis – Fundamental"
        result = _extract_certifications_from_text(text)
        assert len(result) >= 1
        assert any("Business Analysis" in c for c in result)


class TestRenderingNoMissingReason:
    """Tests that rendering omits optional fields instead of showing 'Not explicitly mentioned'."""

    def _cand(self, **kwargs):
        """Helper to create Candidate with required evidence_spans."""
        from src.rag_v3.types import Candidate
        kwargs.setdefault("evidence_spans", [])
        return Candidate(**kwargs)

    def test_single_candidate_omits_missing_fields(self):
        from src.rag_v3.enterprise import _render_hr
        from src.rag_v3.types import HRSchema, CandidateField, MISSING_REASON

        cand = self._cand(
            name="Aadithya",
            technical_skills=["SAP", "SQL"],
            experience_summary="SAP SCM Consultant",
            certifications=["CAPM"],
            education=["MBA from Academy of Management"],
            # These are intentionally empty/None
            functional_skills=None,
            achievements=None,
            total_years_experience=None,
            linkedins=None,
        )
        schema = HRSchema(candidates=CandidateField(items=[cand], missing_reason=None))
        rendered = _render_hr(schema, intent="summary")
        assert MISSING_REASON not in rendered

    def test_multi_candidate_omits_missing_fields(self):
        from src.rag_v3.enterprise import _render_hr
        from src.rag_v3.types import HRSchema, CandidateField, MISSING_REASON

        cand1 = self._cand(name="Alice", technical_skills=["Python"])
        cand2 = self._cand(name="Bob", certifications=["PMP"])
        schema = HRSchema(candidates=CandidateField(items=[cand1, cand2], missing_reason=None))
        rendered = _render_hr(schema, intent="summary")
        assert MISSING_REASON not in rendered

    def test_contact_rendering_omits_missing_linkedin(self):
        from src.rag_v3.enterprise import _render_hr
        from src.rag_v3.types import HRSchema, CandidateField, MISSING_REASON

        cand = self._cand(
            name="Abinaya",
            emails=["abinaya@gmail.com"],
            phones=["9843720090"],
            # LinkedIn intentionally absent
        )
        schema = HRSchema(candidates=CandidateField(items=[cand], missing_reason=None))
        rendered = _render_hr(schema, intent="contact")
        assert MISSING_REASON not in rendered
        assert "LinkedIn" not in rendered  # Should be omitted entirely

    def test_rank_rendering_omits_missing_fields(self):
        from src.rag_v3.enterprise import _render_hr
        from src.rag_v3.types import HRSchema, CandidateField, MISSING_REASON

        cand1 = self._cand(name="Alice", technical_skills=["Python", "Java"])
        cand2 = self._cand(name="Bob", technical_skills=["Go"])
        schema = HRSchema(candidates=CandidateField(items=[cand1, cand2], missing_reason=None))
        rendered = _render_hr(schema, intent="rank")
        assert MISSING_REASON not in rendered
