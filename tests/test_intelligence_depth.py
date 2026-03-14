"""Tests for intelligence depth overhaul (bug fixes + intelligence improvements).

Covers:
  A1: Tool name leak fix
  A2: Query echo/acknowledgement disabled by default
  A3: Evidence coverage asymmetric containment
  A4: LLM context limit scaling for multi-doc
  A5: Invoice metadata fields
  B1: Analytical persona enhancement
  B2: Chain-of-thought reasoning preamble
  B3: Enhanced evidence chain (topic groups, contradictions, stats)
  B4: Domain insights in enterprise rendering
  B5: Deterministic cross-document synthesis
  B6: LLM synthesis post-processing
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# A1: Tool name leak fix
# ---------------------------------------------------------------------------

class TestToolNameLeak:
    """Verify that tool:tool_name no longer leaks into ChunkSource."""

    def test_chunk_source_no_tool_prefix(self):
        """ChunkSource constructed from tool results should use doc name, not tool:name."""
        from src.rag_v3.types import ChunkSource
        # Simulating what _dispatch_tools now does:
        result = {"document_name": "JohnDoe_Resume.pdf", "rendered": "some content"}
        _tool_doc_name = result.get("document_name", "") or result.get("doc_name", "") or ""
        source = ChunkSource(document_name=_tool_doc_name)
        assert "tool:" not in source.document_name
        assert source.document_name == "JohnDoe_Resume.pdf"

    def test_chunk_source_empty_fallback(self):
        """When result has no doc name, should fall back to empty string."""
        result = {"rendered": "some content"}
        _tool_doc_name = result.get("document_name", "") or result.get("doc_name", "") or ""
        assert _tool_doc_name == ""

    def test_chunk_source_doc_name_field(self):
        """Should also check doc_name field."""
        result = {"doc_name": "Invoice_2025.pdf", "rendered": "data"}
        _tool_doc_name = result.get("document_name", "") or result.get("doc_name", "") or ""
        assert _tool_doc_name == "Invoice_2025.pdf"


# ---------------------------------------------------------------------------
# A2: Query echo/acknowledgement disabled
# ---------------------------------------------------------------------------

class TestAcknowledgementDisabled:
    """Verify that _build_answer defaults to include_acknowledgement=False."""

    def test_default_acknowledgement_false(self):
        """The default parameter should be False."""
        import inspect
        from src.rag_v3.pipeline import _build_answer

        sig = inspect.signature(_build_answer)
        default = sig.parameters["include_acknowledgement"].default
        assert default is False, f"Expected False, got {default}"


# ---------------------------------------------------------------------------
# A3: Evidence coverage asymmetric containment
# ---------------------------------------------------------------------------

class TestEvidenceCoverage:
    """Verify asymmetric containment replaces Jaccard for evidence coverage."""

    def test_asymmetric_containment_function_exists(self):
        from src.intelligence.confidence_scorer import _asymmetric_containment
        assert callable(_asymmetric_containment)

    def test_asymmetric_containment_basic(self):
        from src.intelligence.confidence_scorer import _asymmetric_containment
        response_tokens = {"python", "java", "skills"}
        evidence_tokens = {"python", "java", "skills", "experience", "years", "education", "projects"}
        # All response tokens are in evidence → 1.0
        score = _asymmetric_containment(response_tokens, evidence_tokens)
        assert score == 1.0

    def test_asymmetric_not_jaccard(self):
        """Asymmetric should give higher score than Jaccard when evidence >> response."""
        from src.intelligence.confidence_scorer import _asymmetric_containment, _jaccard
        resp = {"python", "java"}
        evidence = {"python", "java", "go", "rust", "typescript", "react", "django", "flask"}
        asym = _asymmetric_containment(resp, evidence)
        jacc = _jaccard(resp, evidence)
        assert asym > jacc, f"Asymmetric {asym} should be > Jaccard {jacc}"
        assert asym == 1.0  # Both response tokens found in evidence

    def test_evidence_coverage_no_longer_zero(self):
        """Realistic scenario: response words mostly found in evidence chunks."""
        from src.intelligence.confidence_scorer import score_evidence_coverage
        response = "The candidate has 5 years of Python and Java experience with AWS certifications."
        chunks = [
            "Python experience: 5 years. Certified AWS Solutions Architect.",
            "Java programming skills. Experience with microservices and cloud.",
        ]
        score, reason = score_evidence_coverage(response, chunks)
        assert score > 0.0, f"Coverage should be > 0 but got {score}: {reason}"

    def test_empty_inputs(self):
        from src.intelligence.confidence_scorer import _asymmetric_containment
        assert _asymmetric_containment(set(), {"a", "b"}) == 0.0
        assert _asymmetric_containment(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# A4: LLM context limit scaling
# ---------------------------------------------------------------------------

class TestLLMContextScaling:
    """Verify multi-doc queries get expanded context limits."""

    def test_single_doc_limits(self):
        from src.rag_v3.llm_extract import _effective_max_chunks, _effective_context_chars
        assert _effective_max_chunks(1) == 10
        assert _effective_context_chars(1) == 8192

    def test_multi_doc_limits(self):
        from src.rag_v3.llm_extract import _effective_max_chunks, _effective_context_chars
        assert _effective_max_chunks(2) == 20
        assert _effective_context_chars(3) == 16384

    def test_build_grouped_evidence_respects_limit(self):
        from src.rag_v3.llm_extract import _build_grouped_evidence
        # Default limit
        result = _build_grouped_evidence([])
        assert result == ""  # No chunks → empty

    def test_build_grouped_evidence_custom_limit(self):
        from src.rag_v3.llm_extract import _build_grouped_evidence
        # Should accept max_context_chars parameter
        result = _build_grouped_evidence([], max_context_chars=12288)
        assert result == ""


# ---------------------------------------------------------------------------
# A5: Invoice metadata fields
# ---------------------------------------------------------------------------

class TestInvoiceMetadata:
    """Verify InvoiceSchema has invoice_metadata field."""

    def test_invoice_schema_has_metadata(self):
        from src.rag_v3.types import InvoiceSchema
        schema = InvoiceSchema()
        assert hasattr(schema, "invoice_metadata")
        assert schema.invoice_metadata is not None

    def test_invoice_metadata_field_type(self):
        from src.rag_v3.types import InvoiceSchema, FieldValuesField, FieldValue, EvidenceSpan
        schema = InvoiceSchema(
            invoice_metadata=FieldValuesField(items=[
                FieldValue(label="Invoice Number", value="INV-2025-001", evidence_spans=[
                    EvidenceSpan(chunk_id="c1", snippet="Invoice Number: INV-2025-001"),
                ]),
            ])
        )
        assert len(schema.invoice_metadata.items) == 1
        assert schema.invoice_metadata.items[0].value == "INV-2025-001"

    def test_schema_is_empty_checks_metadata(self):
        from src.rag_v3.extract import _schema_is_empty
        from src.rag_v3.types import InvoiceSchema, FieldValuesField, FieldValue, EvidenceSpan
        # Empty schema
        empty = InvoiceSchema()
        assert _schema_is_empty(empty) is True

        # Schema with only invoice_metadata
        with_meta = InvoiceSchema(
            invoice_metadata=FieldValuesField(items=[
                FieldValue(label="Invoice No", value="12345", evidence_spans=[
                    EvidenceSpan(chunk_id="c1", snippet="Invoice No: 12345"),
                ]),
            ])
        )
        assert _schema_is_empty(with_meta) is False

    def test_invoice_names_includes_metadata(self):
        from src.rag_v3.line_classifier import INVOICE_NAMES
        assert "invoice_metadata" in INVOICE_NAMES


# ---------------------------------------------------------------------------
# B1: Analytical persona enhancement
# ---------------------------------------------------------------------------

class TestAnalyticalPersona:
    """Verify analytical intelligence section added to persona."""

    def test_persona_contains_analytical_section(self):
        from src.prompting.persona import _DOCWAIN_PERSONA
        assert "ANALYTICAL INTELLIGENCE" in _DOCWAIN_PERSONA

    def test_persona_contains_statistical_guidance(self):
        from src.prompting.persona import _DOCWAIN_PERSONA
        assert "statistical summaries" in _DOCWAIN_PERSONA

    def test_persona_contains_pattern_guidance(self):
        from src.prompting.persona import _DOCWAIN_PERSONA
        assert "patterns, outliers" in _DOCWAIN_PERSONA

    def test_persona_contains_synthesis_guidance(self):
        from src.prompting.persona import _DOCWAIN_PERSONA
        assert "synthesis statement" in _DOCWAIN_PERSONA


# ---------------------------------------------------------------------------
# B2: Chain-of-thought reasoning preamble
# ---------------------------------------------------------------------------

class TestReasoningPreamble:
    """Verify reasoning preamble in generation prompts."""

    def test_preamble_constant_exists(self):
        from src.rag_v3.llm_extract import _REASONING_PREAMBLE
        assert "ANALYTICAL APPROACH" in _REASONING_PREAMBLE
        assert "patterns" in _REASONING_PREAMBLE.lower()
        assert "statistics" in _REASONING_PREAMBLE.lower()

    def test_complex_intents_defined(self):
        from src.rag_v3.llm_extract import _COMPLEX_INTENTS
        assert "comparison" in _COMPLEX_INTENTS
        assert "ranking" in _COMPLEX_INTENTS
        assert "analytics" in _COMPLEX_INTENTS

    def test_preamble_in_complex_prompt(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="Compare all candidates",
            evidence_text="Evidence here",
            intent="comparison",
            num_documents=3,
        )
        assert "ANALYTICAL APPROACH" in prompt

    def test_no_preamble_for_factual(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="What is the email?",
            evidence_text="Evidence here",
            intent="factual",
            num_documents=1,
        )
        assert "ANALYTICAL APPROACH" not in prompt

    def test_preamble_for_multi_doc_even_factual(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="What are the emails?",
            evidence_text="Evidence here",
            intent="factual",
            num_documents=3,
        )
        assert "ANALYTICAL APPROACH" in prompt


# ---------------------------------------------------------------------------
# B3: Enhanced evidence chain
# ---------------------------------------------------------------------------

class TestEnhancedEvidenceChain:
    """Verify topic grouping, contradictions, and numeric stats."""

    def test_evidence_chain_has_new_fields(self):
        from src.rag_v3.evidence_chain import EvidenceChain
        chain = EvidenceChain(query="test")
        assert hasattr(chain, "topic_groups")
        assert hasattr(chain, "contradictions")
        assert hasattr(chain, "numeric_stats")

    def test_topic_grouping(self):
        from src.rag_v3.evidence_chain import _group_facts_by_topic, EvidenceFact
        facts = [
            EvidenceFact(text="Python programming skills are strong", source="doc1", chunk_id="c1", relevance=0.8),
            EvidenceFact(text="Python development experience 5 years", source="doc2", chunk_id="c2", relevance=0.7),
            EvidenceFact(text="Python machine learning projects", source="doc3", chunk_id="c3", relevance=0.6),
            EvidenceFact(text="Java enterprise applications", source="doc4", chunk_id="c4", relevance=0.5),
        ]
        groups = _group_facts_by_topic(facts)
        # Should have at least one group with "python"
        assert len(groups) >= 1
        topic_names = [g.topic for g in groups]
        assert "python" in topic_names

    def test_numeric_stats(self):
        from src.rag_v3.evidence_chain import _compute_numeric_stats, EvidenceFact
        facts = [
            EvidenceFact(text="Total Amount: 5000.00", source="inv1", chunk_id="c1", relevance=0.9),
            EvidenceFact(text="Total Amount: 3000.00", source="inv2", chunk_id="c2", relevance=0.8),
            EvidenceFact(text="Total Amount: 7000.00", source="inv3", chunk_id="c3", relevance=0.7),
        ]
        stats = _compute_numeric_stats(facts)
        assert len(stats) >= 1
        total_stat = stats[0]
        assert total_stat.count == 3
        assert total_stat.total == 15000.0
        assert total_stat.average == 5000.0

    def test_contradiction_detection(self):
        from src.rag_v3.evidence_chain import _detect_contradictions, EvidenceFact
        facts = [
            EvidenceFact(text="Total Amount: 5000", source="doc1", chunk_id="c1", relevance=0.9),
            EvidenceFact(text="Total Amount: 3000", source="doc2", chunk_id="c2", relevance=0.8),
        ]
        contradictions = _detect_contradictions(facts)
        assert len(contradictions) >= 1
        assert contradictions[0].label == "total amount"

    def test_render_with_stats(self):
        from src.rag_v3.evidence_chain import build_evidence_chain
        # Build a mock chunk
        chunk = types.SimpleNamespace(
            text="Total Amount: 5000.00\nVendor: Acme Corp",
            id="c1",
            score=0.9,
            meta={"source_name": "invoice1.pdf"},
            source=types.SimpleNamespace(document_name="invoice1.pdf"),
        )
        chain = build_evidence_chain("What is the total?", [chunk])
        rendered = chain.render_for_prompt()
        assert "EVIDENCE FOUND" in rendered


# ---------------------------------------------------------------------------
# B4: Domain insights in enterprise rendering
# ---------------------------------------------------------------------------

class TestDomainInsights:
    """Verify insight functions for HR, invoice, and generic domains."""

    def test_hr_insights_with_multiple_candidates(self):
        from src.rag_v3.enterprise import _compute_hr_insights
        from src.rag_v3.types import Candidate, EvidenceSpan

        candidates = [
            Candidate(
                name="Alice", total_years_experience="5 years",
                technical_skills=["Python", "Java", "AWS"],
                functional_skills=["Agile"],
                evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="s")],
            ),
            Candidate(
                name="Bob", total_years_experience="8 years",
                technical_skills=["Python", "Go", "Docker"],
                functional_skills=["Scrum"],
                evidence_spans=[EvidenceSpan(chunk_id="c2", snippet="s")],
            ),
        ]
        insights = _compute_hr_insights(candidates)
        assert "2 candidates" in insights
        assert "5" in insights  # min years
        assert "8" in insights  # max years
        assert "python" in insights.lower()  # shared skill

    def test_hr_insights_single_candidate(self):
        from src.rag_v3.enterprise import _compute_hr_insights
        from src.rag_v3.types import Candidate, EvidenceSpan

        candidates = [
            Candidate(name="Alice", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="s")]),
        ]
        assert _compute_hr_insights(candidates) == ""

    def test_invoice_insights(self):
        from src.rag_v3.enterprise import _compute_invoice_insights
        from src.rag_v3.types import InvoiceSchema, FieldValuesField, FieldValue, EvidenceSpan

        schema = InvoiceSchema(
            totals=FieldValuesField(items=[
                FieldValue(label="Total", value="$5,000.00", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="s")]),
            ]),
            invoice_metadata=FieldValuesField(items=[
                FieldValue(label="Invoice Number", value="INV-001", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="s")]),
            ]),
        )
        insights = _compute_invoice_insights(schema)
        assert "Invoice Number" in insights
        assert "line item" in insights.lower() or "total" in insights.lower()

    def test_generic_insights(self):
        from src.rag_v3.enterprise import _compute_generic_insights
        from src.rag_v3.types import FieldValue, EvidenceSpan

        facts = [
            FieldValue(label="Name", value="Alice", document_name="doc1.pdf", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="s")]),
            FieldValue(label="Name", value="Bob", document_name="doc2.pdf", evidence_spans=[EvidenceSpan(chunk_id="c2", snippet="s")]),
            FieldValue(label="Role", value="Engineer", document_name="doc1.pdf", evidence_spans=[EvidenceSpan(chunk_id="c3", snippet="s")]),
        ]
        insights = _compute_generic_insights(facts)
        assert "2 documents" in insights


# ---------------------------------------------------------------------------
# B5: Deterministic cross-document synthesis
# ---------------------------------------------------------------------------

class TestCrossDocSynthesis:
    """Verify _synthesize_cross_document for different domains."""

    def test_hr_synthesis(self):
        from src.rag_v3.pipeline import _synthesize_cross_document
        from src.rag_v3.types import HRSchema, CandidateField, Candidate, EvidenceSpan

        schema = HRSchema(candidates=CandidateField(items=[
            Candidate(
                name="Alice", total_years_experience="5 years",
                technical_skills=["Python", "Java"],
                evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="s")],
            ),
            Candidate(
                name="Bob", total_years_experience="8 years",
                technical_skills=["Python", "Go"],
                evidence_spans=[EvidenceSpan(chunk_id="c2", snippet="s")],
            ),
        ]))

        result = _synthesize_cross_document(
            schemas=[schema],
            doc_contexts=[MagicMock(), MagicMock()],
            query="Compare candidates",
            domain="hr",
        )
        assert "candidates" in result.lower() or "experience" in result.lower()

    def test_single_doc_returns_empty(self):
        from src.rag_v3.pipeline import _synthesize_cross_document
        result = _synthesize_cross_document(
            schemas=[MagicMock()],
            doc_contexts=[MagicMock()],
            query="Test",
            domain="generic",
        )
        assert result == ""

    def test_generic_synthesis(self):
        from src.rag_v3.pipeline import _synthesize_cross_document
        from src.rag_v3.types import GenericSchema, FieldValuesField, FieldValue, EvidenceSpan

        schemas = [
            GenericSchema(facts=FieldValuesField(items=[
                FieldValue(label="Topic", value="ML", evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="s")]),
            ])),
            GenericSchema(facts=FieldValuesField(items=[
                FieldValue(label="Topic", value="AI", evidence_spans=[EvidenceSpan(chunk_id="c2", snippet="s")]),
            ])),
        ]
        result = _synthesize_cross_document(
            schemas=schemas,
            doc_contexts=[MagicMock(), MagicMock()],
            query="Compare documents",
            domain="generic",
        )
        # Might be empty if no repeated themes, but shouldn't crash
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# B6: LLM synthesis config
# ---------------------------------------------------------------------------

class TestLLMSynthesisConfig:
    """Verify Config.Synthesis exists and has correct defaults."""

    def test_synthesis_config_exists(self):
        from src.api.config import Config
        assert hasattr(Config, "Synthesis")

    def test_synthesis_defaults(self):
        from src.api.config import Config
        assert Config.Synthesis.ENABLED is True
        assert Config.Synthesis.TIMEOUT <= 20.0
        assert Config.Synthesis.MIN_DOCUMENTS == 2

    def test_llm_synthesize_function_exists(self):
        from src.rag_v3.pipeline import _llm_synthesize
        assert callable(_llm_synthesize)

    def test_llm_synthesize_skips_single_doc(self):
        """Should return None for single document queries."""
        from src.rag_v3.pipeline import _llm_synthesize
        from src.rag_v3.types import LLMBudget
        result = _llm_synthesize(
            rendered_text="Some text",
            query="Test query",
            domain="generic",
            intent="factual",
            num_documents=1,
            llm_client=MagicMock(),
            budget=LLMBudget(llm_client=MagicMock(), max_calls=1),
            correlation_id="test",
        )
        assert result is None  # MIN_DOCUMENTS=2, so single doc skipped


# ---------------------------------------------------------------------------
# Integration: Invoice metadata rendering
# ---------------------------------------------------------------------------

class TestInvoiceMetadataRendering:
    """Verify invoice_metadata is rendered in enterprise output."""

    def test_render_invoice_with_metadata(self):
        from src.rag_v3.enterprise import render_enterprise
        from src.rag_v3.types import (
            InvoiceSchema, FieldValuesField, FieldValue, EvidenceSpan,
            InvoiceItemsField,
        )

        schema = InvoiceSchema(
            invoice_metadata=FieldValuesField(items=[
                FieldValue(label="Invoice Number", value="INV-2025-001",
                           evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="Invoice Number: INV-2025-001")]),
                FieldValue(label="Invoice Date", value="January 15, 2025",
                           evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="Invoice Date: January 15, 2025")]),
            ]),
            totals=FieldValuesField(items=[
                FieldValue(label="Total", value="$5,000", evidence_spans=[
                    EvidenceSpan(chunk_id="c1", snippet="Total: $5,000"),
                ]),
            ]),
        )
        rendered = render_enterprise(schema, intent="facts", domain="invoice")
        assert "INV-2025-001" in rendered
        assert "January 15, 2025" in rendered


# ---------------------------------------------------------------------------
# Integration: Full evidence chain with new features
# ---------------------------------------------------------------------------

class TestEvidenceChainIntegration:
    """Integration test for build_evidence_chain with all enhancements."""

    def test_full_chain_build(self):
        from src.rag_v3.evidence_chain import build_evidence_chain

        chunks = [
            types.SimpleNamespace(
                text="Salary: 85000\nExperience: 5 years Python development",
                id="c1", score=0.9,
                meta={"source_name": "resume_alice.pdf"},
                source=types.SimpleNamespace(document_name="resume_alice.pdf"),
            ),
            types.SimpleNamespace(
                text="Salary: 92000\nExperience: 8 years Java and Python",
                id="c2", score=0.85,
                meta={"source_name": "resume_bob.pdf"},
                source=types.SimpleNamespace(document_name="resume_bob.pdf"),
            ),
            types.SimpleNamespace(
                text="Salary: 78000\nExperience: 3 years JavaScript",
                id="c3", score=0.8,
                meta={"source_name": "resume_carol.pdf"},
                source=types.SimpleNamespace(document_name="resume_carol.pdf"),
            ),
        ]

        chain = build_evidence_chain("What are the salary ranges?", chunks)
        assert chain.num_documents == 3
        assert len(chain.supporting_facts) >= 1

        rendered = chain.render_for_prompt()
        assert "EVIDENCE FOUND" in rendered
        assert len(rendered) > 50
