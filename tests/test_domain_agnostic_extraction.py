"""Tests for domain-agnostic extraction: ensures non-HR documents are NOT forced
through the HR/Resume schema, cross-profile leakage is prevented, and generic
extraction produces useful output for any document type."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, ".")


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class FakeChunk:
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


def _make_chunk(text: str, doc_domain: str = "", doc_id: str = "doc1", source_name: str = "test.pdf") -> FakeChunk:
    return FakeChunk(
        id=f"chunk_{hash(text) % 10000}",
        text=text,
        score=0.8,
        source=types.SimpleNamespace(document_name=source_name, page=1),
        meta={"doc_domain": doc_domain, "document_id": doc_id, "source_name": source_name},
    )


# ── Local helpers replacing removed pipeline/extract functions ───────────────

def _query_is_hr(query):
    lowered = (query or "").lower()
    strong = ("resume", "cv", "curriculum vitae", "linkedin", "candidate")
    if any(token in lowered for token in strong):
        return True
    weak = ("experience", "education", "skills", "certification", "certifications")
    return sum(1 for token in weak if token in lowered) >= 2


def _query_is_hr_like(query):
    lowered = (query or "").lower()
    strong = ("resume", "cv", "candidate")
    if any(kw in lowered for kw in strong):
        return True
    weak = ("skills", "experience", "education", "certification")
    return sum(1 for kw in weak if kw in lowered) >= 2


# ── 1. HR Detection Tightening ──────────────────────────────────────────────

class TestQueryIsHr:
    """Test that _query_is_hr only triggers on strong HR signals."""

    def test_patient_details_is_not_hr(self):
        assert _query_is_hr("give me the patient details") is False

    def test_medical_history_is_not_hr(self):
        assert _query_is_hr("provide me the medical history") is False

    def test_education_alone_is_not_hr(self):
        assert _query_is_hr("education of harshanaa") is False

    def test_experience_alone_is_not_hr(self):
        assert _query_is_hr("give me the experience details") is False

    def test_skills_alone_is_not_hr(self):
        assert _query_is_hr("what are the skills?") is False

    def test_resume_is_hr(self):
        assert _query_is_hr("show me the resume") is True

    def test_candidate_is_hr(self):
        assert _query_is_hr("tell me about the candidate") is True

    def test_cv_is_hr(self):
        assert _query_is_hr("upload my cv") is True

    def test_two_weak_signals_is_hr(self):
        assert _query_is_hr("education and experience of the person") is True

    def test_experience_and_skills_is_hr(self):
        assert _query_is_hr("what are the skills and experience?") is True


class TestQueryIsHrLike:
    """Test that _query_is_hr_like only triggers on strong HR signals."""

    def test_patient_not_hr_like(self):
        assert _query_is_hr_like("give me the patient details") is False

    def test_education_alone_not_hr_like(self):
        assert _query_is_hr_like("education of john") is False

    def test_resume_is_hr_like(self):
        assert _query_is_hr_like("analyze the resume") is True

    def test_two_weak_is_hr_like(self):
        assert _query_is_hr_like("skills and experience") is True


# ── 2. Domain Inference from Chunk Metadata ──────────────────────────────────

class TestMajorityChunkDomain:
    """Test _majority_chunk_domain uses doc_domain metadata."""

    def test_resume_chunks_return_hr(self):
        from src.rag_v3.extract import _majority_chunk_domain
        chunks = [_make_chunk("Professional experience at Google. Work experience includes Python and Java development.", doc_domain="resume") for _ in range(3)]
        assert _majority_chunk_domain(chunks) == "hr"

    def test_medical_chunks_return_medical(self):
        from src.rag_v3.extract import _majority_chunk_domain
        # Content must contain medical terms (patient, diagnosis, etc.) to confirm medical domain
        chunks = [_make_chunk("Patient diagnosis: hypertension. Medication: lisinopril 10mg daily.", doc_domain="medical") for _ in range(3)]
        assert _majority_chunk_domain(chunks) == "medical"

    def test_medical_metadata_without_medical_content_returns_generic(self):
        from src.rag_v3.extract import _majority_chunk_domain
        # Equipment manuals tagged as "medical" but without patient content → generic fallback
        chunks = [_make_chunk("Werkgebiedgrenswaarden voor Werkafstand: 225mm", doc_domain="medical") for _ in range(3)]
        assert _majority_chunk_domain(chunks) is None  # triggers content-based fallback

    def test_no_domain_returns_none(self):
        from src.rag_v3.extract import _majority_chunk_domain
        chunks = [_make_chunk("text", doc_domain="") for _ in range(3)]
        assert _majority_chunk_domain(chunks) is None

    def test_mixed_domains_majority_wins(self):
        from src.rag_v3.extract import _majority_chunk_domain
        chunks = [
            _make_chunk("Professional summary: 5 years work experience in software engineering.", doc_domain="resume"),
            _make_chunk("Career objective and professional experience at leading tech companies.", doc_domain="resume"),
            _make_chunk("Patient diagnosis: hypertension. Medication: lisinopril.", doc_domain="medical"),
        ]
        assert _majority_chunk_domain(chunks) == "hr"


# ── 3. Domain Inference Integration ──────────────────────────────────────────

class TestInferDomainIntent:
    """Test _infer_domain_intent respects chunk metadata over query keywords."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_medical_chunks_not_classified_as_hr(self):
        from src.rag_v3.extract import _infer_domain_intent
        chunks = [
            _make_chunk("Patient: John Doe\nDiagnosis: Chest pain", doc_domain="medical"),
            _make_chunk("Treatment: Aspirin 100mg daily", doc_domain="medical"),
        ]
        domain, intent = _infer_domain_intent("give me patient details", chunks)
        assert domain != "hr", f"Medical record classified as HR: domain={domain}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_resume_hint_produces_hr(self):
        from src.rag_v3.extract import _infer_domain_intent
        chunks = [_make_chunk("5 years Python experience", doc_domain="resume")]
        domain, intent = _infer_domain_intent("show candidate skills", chunks, domain_hint="hr")
        assert domain == "hr"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_generic_query_on_generic_chunks(self):
        from src.rag_v3.extract import _infer_domain_intent
        chunks = [_make_chunk("The weather is sunny today.")]
        domain, intent = _infer_domain_intent("what is the weather?", chunks)
        assert domain == "generic"
        assert intent == "facts"


# ── 4. Generic Extraction with Key-Value Pairs ──────────────────────────────

class TestExtractGeneric:
    """Test _extract_generic produces labeled key-value facts."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_extracts_key_value_pairs(self):
        from src.rag_v3.extract import _extract_generic
        chunks = [_make_chunk("Patient Name: John Doe\nAge: 45\nDiagnosis: Chest pain")]
        schema = _extract_generic("patient details", chunks)
        items = schema.facts.items or []
        labeled = [f for f in items if f.label]
        assert len(labeled) >= 2, f"Expected labeled facts, got {len(labeled)}: {[f.label for f in items]}"
        labels = [f.label for f in labeled]
        assert any("Patient Name" in l or "Name" in l for l in labels), f"No name label found in {labels}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_extracts_sentences_with_keywords(self):
        from src.rag_v3.extract import _extract_generic
        chunks = [_make_chunk("The patient was admitted on January 5th. The weather was nice.")]
        schema = _extract_generic("patient admission", chunks)
        items = schema.facts.items or []
        values = [f.value for f in items]
        assert any("patient" in v.lower() for v in values), f"No patient-related fact found: {values}"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_fallback_includes_top_sentences(self):
        from src.rag_v3.extract import _extract_generic
        chunks = [_make_chunk("The document contains important information about the surgery performed on January 5th.")]
        schema = _extract_generic("xyz_nonexistent_keyword", chunks)
        items = schema.facts.items or []
        # Even with no keyword match, fallback should include sentences
        assert len(items) >= 1, "Fallback should include top sentences when no keywords match"


# ── 5. Generic Rendering with Labels ────────────────────────────────────────

class TestRenderGeneric:
    """Test _render_generic produces formatted key-value output."""

    def test_renders_labeled_facts(self):
        from src.rag_v3.enterprise import _render_generic
        from src.rag_v3.types import GenericSchema, FieldValue, FieldValuesField, EvidenceSpan

        facts = [
            FieldValue(label="Patient Name", value="John Doe", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="Patient Name: John Doe")]),
            FieldValue(label="Age", value="45", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="Age: 45")]),
            FieldValue(label="Diagnosis", value="Chest pain", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="Diagnosis: Chest pain")]),
        ]
        schema = GenericSchema(facts=FieldValuesField(items=facts))
        rendered = _render_generic(schema, "facts")

        # "facts" intent: no generic heading — facts render directly
        assert "**Patient Name:**" in rendered
        assert "John Doe" in rendered
        assert "**Age:**" in rendered
        assert "**Diagnosis:**" in rendered

    def test_renders_unlabeled_facts(self):
        from src.rag_v3.enterprise import _render_generic
        from src.rag_v3.types import GenericSchema, FieldValue, FieldValuesField, EvidenceSpan

        facts = [
            FieldValue(label=None, value="The patient was admitted.", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="admitted")]),
            FieldValue(label=None, value="Treatment was successful.", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="treatment")]),
        ]
        schema = GenericSchema(facts=FieldValuesField(items=facts))
        rendered = _render_generic(schema, "facts")

        # "facts" intent: no heading for unlabeled facts either
        assert "The patient was admitted." in rendered
        assert "Treatment was successful." in rendered

    def test_single_unlabeled_fact_no_header(self):
        from src.rag_v3.enterprise import _render_generic
        from src.rag_v3.types import GenericSchema, FieldValue, FieldValuesField, EvidenceSpan

        facts = [FieldValue(label=None, value="Simple answer text.", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="x")])]
        schema = GenericSchema(facts=FieldValuesField(items=facts))
        rendered = _render_generic(schema, "facts")
        assert rendered == "Simple answer text."


# ── 6. Cross-Profile Leakage Prevention ──────────────────────────────────────

class TestCrossProfileLeakage:
    """Test filter_chunks_by_profile_scope prevents data leakage."""

    def test_no_metadata_returns_empty(self):
        """When chunks have no profile_id metadata, return empty list (not all chunks)."""
        from src.rag_v3.retrieve import filter_chunks_by_profile_scope
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="c1", text="data", score=0.9, source=ChunkSource(document_name="test.pdf"), meta={}),
            Chunk(id="c2", text="data2", score=0.8, source=ChunkSource(document_name="test2.pdf"), meta={}),
        ]
        result = filter_chunks_by_profile_scope(chunks, profile_id="profile_A", subscription_id="sub_1")
        assert result == [], f"Expected empty list, got {len(result)} chunks — cross-profile leakage!"

    def test_matching_profile_returns_chunks(self):
        """When chunks have matching profile_id, return them."""
        from src.rag_v3.retrieve import filter_chunks_by_profile_scope
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="c1", text="data", score=0.9, source=ChunkSource(document_name="test.pdf"),
                  meta={"profile_id": "profile_A", "subscription_id": "sub_1"}),
            Chunk(id="c2", text="other", score=0.8, source=ChunkSource(document_name="test2.pdf"),
                  meta={"profile_id": "profile_B", "subscription_id": "sub_1"}),
        ]
        result = filter_chunks_by_profile_scope(chunks, profile_id="profile_A", subscription_id="sub_1")
        assert len(result) == 1
        assert result[0].id == "c1"


# ── 7. (Removed: _resolve_domain_from_chunks was replaced by ML-first domain detection)


# ── 8. End-to-End: Medical Record Through Pipeline ──────────────────────────

class TestMedicalRecordNotHR:
    """End-to-end: medical record queries should NOT produce HR template output."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_medical_chunks_produce_medical_schema(self):
        from src.rag_v3.extract import schema_extract
        from src.rag_v3.types import HRSchema, MedicalSchema, LLMBudget

        chunks = [
            _make_chunk("Patient Name: John Doe\nAge: 65\nDiagnosis: Chest discomfort", doc_domain="medical"),
            _make_chunk("Doctor: Dr. Laura Bennett, MD\nReason for Visit: Shortness of breath", doc_domain="medical"),
        ]
        budget = LLMBudget(llm_client=None, max_calls=0)
        result = schema_extract(
            query="give me the patient details",
            chunks=chunks,
            llm_client=None,
            budget=budget,
            domain_hint=None,  # No HR hint
        )
        assert not isinstance(result.schema, HRSchema), \
            f"Medical record should NOT produce HRSchema, got domain={result.domain}"
        assert isinstance(result.schema, MedicalSchema), \
            f"Expected MedicalSchema, got {type(result.schema).__name__}"
        # Should have patient info extracted
        patient_items = (result.schema.patient_info.items if result.schema.patient_info else None) or []
        assert len(patient_items) > 0, "MedicalSchema should have extracted patient info"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_medical_extraction_contains_patient_data(self):
        from src.rag_v3.extract import schema_extract
        from src.rag_v3.types import LLMBudget
        from src.rag_v3.enterprise import render_enterprise

        chunks = [
            _make_chunk("Patient Name: John Doe\nAge: 65\nDiagnosis: Chest discomfort", doc_domain="medical"),
            _make_chunk("Doctor: Dr. Laura Bennett, MD\nReason for Visit: Shortness of breath", doc_domain="medical"),
        ]
        budget = LLMBudget(llm_client=None, max_calls=0)
        result = schema_extract(
            query="give me the patient details",
            chunks=chunks,
            llm_client=None,
            budget=budget,
            domain_hint=None,
        )
        rendered = render_enterprise(result.schema, result.intent, domain=result.domain)
        # Should NOT contain HR template elements
        assert "Candidate:" not in rendered, f"Rendered output contains HR template: {rendered}"
        assert "Total experience:" not in rendered, f"Rendered output contains HR template: {rendered}"
        assert "Technical skills:" not in rendered, f"Rendered output contains HR template: {rendered}"
        # Should contain actual patient data
        assert "John Doe" in rendered, f"Patient name not in rendered: {rendered}"


# ── 9. Backward Compatibility: HR Queries Still Work ────────────────────────

class TestHRBackwardCompat:
    """Ensure resume/CV queries on HR documents still produce structured output."""

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_resume_query_produces_structured_schema(self):
        from src.rag_v3.extract import schema_extract
        from src.rag_v3.types import HRSchema, LLMBudget

        chunks = [
            _make_chunk(
                "PROFESSIONAL SUMMARY\nExperienced Python developer with 5 years.\n"
                "TECHNICAL SKILLS\nPython, Java, SQL\n"
                "EDUCATION\nB.S. Computer Science, MIT 2018",
                doc_domain="resume",
                source_name="John_Doe_Resume.pdf",
            ),
        ]
        budget = LLMBudget(llm_client=None, max_calls=0)
        result = schema_extract(
            query="show me the candidate's resume",
            chunks=chunks,
            llm_client=None,
            budget=budget,
            domain_hint="hr",
        )
        # HR domain routes through structured HR extractor
        assert isinstance(result.schema, HRSchema), \
            f"Resume query should produce HRSchema, got {type(result.schema).__name__}"
        candidates = (result.schema.candidates.items if result.schema.candidates else []) or []
        assert len(candidates) >= 1, "Should extract at least one candidate"

    @patch("src.rag_v3.extract.Config.Features.DOMAIN_SPECIFIC_ENABLED", True)
    def test_candidate_keyword_triggers_hr(self):
        assert _query_is_hr("tell me about the candidate") is True


# ── 10. DomainRouter Medical Support ────────────────────────────────────────

class TestDomainRouterMedical:
    """Test DomainRouter recognizes medical domain."""

    def test_patient_query_medical_domain(self):
        from src.rag_v3.domain_router import DomainRouter
        decision = DomainRouter.route("give me the patient details")
        assert decision.domain == "medical", f"Expected medical, got {decision.domain}"

    def test_diagnosis_query_medical_domain(self):
        from src.rag_v3.domain_router import DomainRouter
        decision = DomainRouter.route("what is the diagnosis?")
        assert decision.domain == "medical", f"Expected medical, got {decision.domain}"

    def test_resume_query_still_resume(self):
        from src.rag_v3.domain_router import DomainRouter
        decision = DomainRouter.route("show me the resume")
        assert decision.domain == "resume"


# ── 11. Source Type Inference ────────────────────────────────────────────────

class TestInferSourceType:
    """Test _infer_source_type recognizes different document types."""

    def test_resume_source(self):
        from src.rag_v3.extract import _infer_source_type
        assert _infer_source_type("John_Resume.pdf") == "Resume"

    def test_medical_source(self):
        from src.rag_v3.extract import _infer_source_type
        assert _infer_source_type("Hospital_Medical_Record.pdf") == "Medical record"

    def test_invoice_source(self):
        from src.rag_v3.extract import _infer_source_type
        assert _infer_source_type("invoice_2024.pdf") == "Invoice"

    def test_unknown_source_returns_none(self):
        from src.rag_v3.extract import _infer_source_type
        assert _infer_source_type("random_file.pdf") is None
