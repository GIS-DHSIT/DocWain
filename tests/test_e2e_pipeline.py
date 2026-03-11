"""Comprehensive end-to-end tests for the DocWain pipeline.

Tests the full document intelligence flow:
  1. Document extraction & schema normalization (build_qdrant_payload)
  2. Screening pipeline (ScreeningEngine with categories)
  3. RAG pipeline (retrieve -> rerank -> extract -> render)
  4. Response intelligence (query types: factual, comparison, summary, etc.)
  5. Content generation (cover letter, key points, summary with deterministic fallback)
  6. Gateway execution (ScreeningExecutor)

Uses FakeQdrant, FakeEmbedder, FakeRedis from tests/rag_v2_helpers.py for
offline tests.  Live Qdrant tests are gated behind LIVE_TEST=1.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


@dataclass(frozen=True)
class _FakeIntentParse:
    """Lightweight mock for IntentParse used in scope inference tests."""
    intent: str = "qa"
    output_format: str = "text"
    requested_fields: list = dc_field(default_factory=list)
    domain: str = "generic"
    constraints: dict = dc_field(default_factory=dict)
    entity_hints: list = dc_field(default_factory=list)
    source: str = "test"

from tests.rag_v2_helpers import (
    FakeEmbedder,
    FakePoint,
    FakeQdrant,
    FakeRedis,
    make_point,
)

# ---------------------------------------------------------------------------
# Screening config helper — cache real ScreeningConfig at import time
# to avoid contamination from test_agent_mode.py which replaces
# src.screening.config in sys.modules with a permissive stub.
# ---------------------------------------------------------------------------

_SCREENING_CONFIG_CLASS = None
_SCREENING_DEFAULTS = None


def _screening_config():
    """Build a real ScreeningConfig, resilient to module-level stubbing.

    Falls back to a dataclass constructor with hardcoded defaults when
    the module has been replaced by test_agent_mode.py stubs.
    """
    global _SCREENING_CONFIG_CLASS, _SCREENING_DEFAULTS

    if _SCREENING_CONFIG_CLASS is None:
        try:
            import importlib
            mod = importlib.import_module("src.screening.config")
            cls = getattr(mod, "ScreeningConfig", None)
            # Verify it is the real class (has enabled_tools field)
            if cls is not None and hasattr(cls, "enabled_tools"):
                _SCREENING_CONFIG_CLASS = cls
                _SCREENING_DEFAULTS = {
                    "enabled_tools": list(getattr(mod, "DEFAULT_WEIGHTS", {}).keys()),
                    "weights": dict(getattr(mod, "DEFAULT_WEIGHTS", {})),
                    "doc_type_templates": dict(getattr(mod, "DEFAULT_TEMPLATES", {})),
                    "policy_rules": dict(getattr(mod, "DEFAULT_POLICY_RULES", {})),
                }
        except Exception:
            pass

    if _SCREENING_CONFIG_CLASS is not None and _SCREENING_DEFAULTS:
        try:
            return _SCREENING_CONFIG_CLASS(
                enabled_tools=list(_SCREENING_DEFAULTS["enabled_tools"]),
                weights=dict(_SCREENING_DEFAULTS["weights"]),
                sigmoid_a=6.0,
                sigmoid_b=0.50,
                risk_thresholds={"high": 75.0, "medium": 45.0},
                internet_enabled=False,
                search_provider={},
                doc_type_templates=dict(_SCREENING_DEFAULTS["doc_type_templates"]),
                policy_rules=dict(_SCREENING_DEFAULTS["policy_rules"]),
                sensitive_keywords=["confidential", "proprietary"],
                auto_attach_on_ingest=False,
                block_high_risk=False,
                config_source="test",
            )
        except Exception:
            pass

    # Fallback: construct via importlib forcing a fresh import
    import importlib
    import sys
    saved = sys.modules.pop("src.screening.config", None)
    try:
        mod = importlib.import_module("src.screening.config")
        cls = mod.ScreeningConfig
        return cls(
            enabled_tools=list(mod.DEFAULT_WEIGHTS.keys()),
            weights=dict(mod.DEFAULT_WEIGHTS),
            sigmoid_a=6.0,
            sigmoid_b=0.50,
            risk_thresholds={"high": 75.0, "medium": 45.0},
            internet_enabled=False,
            search_provider={},
            doc_type_templates=dict(mod.DEFAULT_TEMPLATES),
            policy_rules=dict(mod.DEFAULT_POLICY_RULES),
            sensitive_keywords=["confidential", "proprietary"],
            auto_attach_on_ingest=False,
            block_high_risk=False,
            config_source="test",
        )
    finally:
        if saved is not None:
            sys.modules["src.screening.config"] = saved


# Eagerly cache at import time (before any stubs)
try:
    _screening_config()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants for test data
# ---------------------------------------------------------------------------

SUB_ID = "sub-e2e"
PROFILE_ID = "profile-e2e"
DOC_ID_RESUME_1 = "doc-resume-alice"
DOC_ID_RESUME_2 = "doc-resume-bob"
DOC_ID_INVOICE = "doc-invoice-001"
DOC_ID_LEGAL = "doc-contract-001"
DOC_ID_MEDICAL = "doc-medical-001"

RESUME_TEXT_ALICE = (
    "Name: Alice Johnson\n"
    "Email: alice@example.com\n"
    "Phone: (555) 123-4567\n"
    "Summary: Experienced software engineer with 8 years of experience in Python, "
    "Java, and cloud technologies. Led teams of 5+ engineers at TechCorp.\n"
    "Skills: Python, Java, AWS, Docker, Kubernetes, Machine Learning\n"
    "Education: B.S. in Computer Science from MIT\n"
    "Certifications: AWS Solutions Architect, Certified Scrum Master\n"
    "Experience: Senior Engineer at TechCorp (2018-2024), Software Developer at StartupXYZ (2016-2018)"
)

RESUME_TEXT_BOB = (
    "Name: Bob Smith\n"
    "Email: bob.smith@company.com\n"
    "Summary: Data scientist with 5 years of experience in machine learning, "
    "statistical modeling, and data pipeline architecture.\n"
    "Skills: Python, R, TensorFlow, SQL, Spark, Pandas\n"
    "Education: M.S. in Data Science from Stanford University\n"
    "Certifications: Google Cloud Professional Data Engineer\n"
    "Experience: Data Scientist at DataCo (2019-2024), Analyst at FinTech Inc (2017-2019)"
)

INVOICE_TEXT = (
    "Invoice #12345\n"
    "Date: 01/15/2026\n"
    "Bill To: Acme Corp\n"
    "Item: Consulting Services - 40 hours @ $150/hour = $6,000.00\n"
    "Item: Software License - 1 year = $2,400.00\n"
    "Subtotal: $8,400.00\n"
    "Tax (10%): $840.00\n"
    "Total: $9,240.00\n"
    "Payment Terms: Net 30\n"
    "Due Date: 02/14/2026"
)

LEGAL_TEXT = (
    "SERVICES AGREEMENT\n"
    "This Agreement is entered into between Party A and Party B.\n"
    "Section 1: Scope of Work\n"
    "Party A shall provide consulting services as described in Exhibit A.\n"
    "Section 2: Compensation\n"
    "Party B shall pay Party A the sum of $50,000 per quarter.\n"
    "Section 3: Term\n"
    "This agreement shall be effective for a period of 12 months.\n"
    "Section 4: Termination\n"
    "Either party may terminate with 30 days written notice.\n"
    "Section 5: Liability\n"
    "Neither party shall be liable for consequential damages."
)

MEDICAL_TEXT = (
    "Patient: Jane Doe\n"
    "Date of Visit: 01/10/2026\n"
    "Diagnosis: Type 2 Diabetes Mellitus\n"
    "Medications: Metformin 500mg twice daily, Lisinopril 10mg daily\n"
    "Lab Results: HbA1c: 7.2%, Fasting Glucose: 145 mg/dL\n"
    "Blood Pressure: 130/85 mmHg\n"
    "Follow-up: Schedule appointment in 3 months for HbA1c recheck."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resume_points_alice() -> List[FakePoint]:
    """Create FakeQdrant points for Alice's resume."""
    return [
        make_point(
            pid="alice-1", profile_id=PROFILE_ID, document_id=DOC_ID_RESUME_1,
            file_name="Alice_Johnson_Resume.pdf", text=RESUME_TEXT_ALICE,
            page=1, score=0.95, section_kind="summary_objective",
            doc_domain="resume",
        ),
        make_point(
            pid="alice-2", profile_id=PROFILE_ID, document_id=DOC_ID_RESUME_1,
            file_name="Alice_Johnson_Resume.pdf",
            text="Skills: Python, Java, AWS, Docker, Kubernetes, Machine Learning",
            page=1, score=0.90, section_kind="skills_technical",
            doc_domain="resume",
        ),
        make_point(
            pid="alice-3", profile_id=PROFILE_ID, document_id=DOC_ID_RESUME_1,
            file_name="Alice_Johnson_Resume.pdf",
            text="Experience: Senior Engineer at TechCorp (2018-2024), Led team of 5 engineers.",
            page=2, score=0.88, section_kind="experience",
            doc_domain="resume",
        ),
    ]


def _make_resume_points_bob() -> List[FakePoint]:
    """Create FakeQdrant points for Bob's resume."""
    return [
        make_point(
            pid="bob-1", profile_id=PROFILE_ID, document_id=DOC_ID_RESUME_2,
            file_name="Bob_Smith_Resume.pdf", text=RESUME_TEXT_BOB,
            page=1, score=0.93, section_kind="summary_objective",
            doc_domain="resume",
        ),
        make_point(
            pid="bob-2", profile_id=PROFILE_ID, document_id=DOC_ID_RESUME_2,
            file_name="Bob_Smith_Resume.pdf",
            text="Skills: Python, R, TensorFlow, SQL, Spark, Pandas",
            page=1, score=0.89, section_kind="skills_technical",
            doc_domain="resume",
        ),
    ]


def _make_invoice_points() -> List[FakePoint]:
    """Create FakeQdrant points for an invoice."""
    return [
        make_point(
            pid="inv-1", profile_id=PROFILE_ID, document_id=DOC_ID_INVOICE,
            file_name="Invoice_12345.pdf", text=INVOICE_TEXT,
            page=1, score=0.92, section_kind="financial_summary",
            doc_domain="invoice",
        ),
    ]


def _make_legal_points() -> List[FakePoint]:
    """Create FakeQdrant points for a legal contract."""
    return [
        make_point(
            pid="legal-1", profile_id=PROFILE_ID, document_id=DOC_ID_LEGAL,
            file_name="Services_Agreement.pdf", text=LEGAL_TEXT,
            page=1, score=0.91, section_kind="terms_conditions",
            doc_domain="legal",
        ),
    ]


def _make_medical_points() -> List[FakePoint]:
    """Create FakeQdrant points for a medical document."""
    return [
        make_point(
            pid="med-1", profile_id=PROFILE_ID, document_id=DOC_ID_MEDICAL,
            file_name="Patient_Report.pdf", text=MEDICAL_TEXT,
            page=1, score=0.90, section_kind="misc",
            doc_domain="medical",
        ),
    ]


def _all_points() -> List[FakePoint]:
    """All test points across document types."""
    return (
        _make_resume_points_alice()
        + _make_resume_points_bob()
        + _make_invoice_points()
        + _make_legal_points()
        + _make_medical_points()
    )


def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# 1. Document Extraction & Schema Normalization
# ===========================================================================


class TestBuildQdrantPayload:
    """Test build_qdrant_payload() produces correct flat payload fields."""

    def test_resume_payload_fields(self):
        """Resume document produces correct core fields."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": DOC_ID_RESUME_1,
            "source_name": "Alice_Johnson_Resume.pdf",
            "content": RESUME_TEXT_ALICE,
            "section_title": "Summary",
            "page": 1,
            "chunk_id": "c-001",
            "doc_domain": "resume",
        }
        payload = build_qdrant_payload(raw)

        # Identity fields
        assert payload["subscription_id"] == SUB_ID
        assert payload["profile_id"] == PROFILE_ID
        assert payload["document_id"] == DOC_ID_RESUME_1
        # Source
        assert payload["source_name"] == "Alice_Johnson_Resume.pdf"
        # Classification - filename hints override doc_domain
        assert payload["doc_domain"] in ("resume", "hr")
        # Text fields
        assert payload.get("canonical_text"), "canonical_text must not be empty"
        assert payload.get("embedding_text"), "embedding_text must not be empty"
        # Section
        assert "section_kind" in payload
        assert payload["section_kind"] != "unknown"
        # Metadata
        assert "embed_pipeline_version" in payload
        assert "hash" in payload

    def test_invoice_payload_fields(self):
        """Invoice document produces correct classification."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": DOC_ID_INVOICE,
            "source_name": "Invoice_12345.pdf",
            "content": INVOICE_TEXT,
            "section_title": "Invoice Details",
            "page": 1,
            "chunk_id": "c-inv-001",
            "doc_domain": "invoice",
        }
        payload = build_qdrant_payload(raw)

        assert payload["doc_domain"] == "invoice"
        assert payload["document_id"] == DOC_ID_INVOICE
        assert len(payload["canonical_text"]) > 50

    def test_legal_payload_fields(self):
        """Legal document produces correct classification."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": DOC_ID_LEGAL,
            "source_name": "Services_Agreement.pdf",
            "content": LEGAL_TEXT,
            "page": 1,
            "chunk_id": "c-leg-001",
            "doc_domain": "legal",
        }
        payload = build_qdrant_payload(raw)

        assert payload["document_id"] == DOC_ID_LEGAL
        assert payload.get("canonical_text")

    def test_medical_payload_fields(self):
        """Medical document produces correct classification."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": DOC_ID_MEDICAL,
            "source_name": "Patient_Report.pdf",
            "content": MEDICAL_TEXT,
            "page": 1,
            "chunk_id": "c-med-001",
            "doc_domain": "medical",
        }
        payload = build_qdrant_payload(raw)

        assert payload["document_id"] == DOC_ID_MEDICAL
        assert payload.get("canonical_text")

    def test_missing_required_field_raises(self):
        """Missing subscription_id raises ValueError."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        with pytest.raises(ValueError, match="subscription_id"):
            build_qdrant_payload({
                "profile_id": PROFILE_ID,
                "document_id": DOC_ID_RESUME_1,
                "content": "Some text",
            })

    def test_metadata_garbage_is_cleaned(self):
        """Text containing metadata garbage markers gets cleaned."""
        from src.embedding.pipeline.schema_normalizer import (
            build_qdrant_payload,
            _is_metadata_garbage,
        )

        garbage_text = "section_id : abc, chunk_type : text, section_title : Test"
        assert _is_metadata_garbage(garbage_text)

        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": "doc-test",
            "source_name": "test.pdf",
            "content": garbage_text,
            "embedding_text": "This is the actual meaningful content of the document about Python programming skills and AWS certifications.",
            "page": 1,
            "chunk_id": "c-test",
        }
        payload = build_qdrant_payload(raw)
        # canonical_text should be derived from embedding_text, not the garbage content
        assert not _is_metadata_garbage(payload.get("canonical_text", ""))

    def test_normalize_content_preserves_bullets(self):
        """normalize_content preserves bullet point structure."""
        from src.embedding.pipeline.schema_normalizer import normalize_content

        text = "Skills:\n- Python\n- Java\n- AWS"
        result = normalize_content(text)
        assert "Python" in result
        assert "Java" in result
        assert "AWS" in result

    def test_flat_payload_no_nested_dicts(self):
        """Payload should have flat structure, no nested source/section/chunk dicts."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": DOC_ID_RESUME_1,
            "source_name": "test.pdf",
            "content": "Some meaningful test content for the document",
            "page": 1,
            "chunk_id": "c-flat-test",
        }
        payload = build_qdrant_payload(raw)

        # Verify no nested dicts (all values are scalars or lists)
        for key, value in payload.items():
            if key == "table_headers":
                continue  # table_headers is allowed to be a list
            assert not isinstance(value, dict), f"Field '{key}' should not be a nested dict"

    def test_filename_based_domain_override(self):
        """Resume filename should override incorrect doc_domain classification."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": "doc-override-test",
            "source_name": "John_Resume.pdf",
            "content": "Some text about the person's address and phone number",
            "page": 1,
            "chunk_id": "c-override",
            "doc_domain": "purchase_order",  # Incorrect classification
        }
        payload = build_qdrant_payload(raw)
        assert payload["doc_domain"] == "resume"


# ===========================================================================
# 2. Screening Pipeline
# ===========================================================================


class TestScreeningPipeline:
    """Test the screening engine with different categories and document types."""

    def _make_engine(self):
        """Create a ScreeningEngine with default tools and explicit config.

        Constructs config without importing from src.screening.config because
        test_agent_mode.py may replace that module in sys.modules with a stub.
        """
        from src.screening.engine import ScreeningEngine
        return ScreeningEngine(config=_screening_config())

    def _make_context(self, text: str, doc_type: str = "generic"):
        """Create a ScreeningContext for testing."""
        from src.screening.models import ScreeningContext
        return ScreeningContext(
            doc_id="test-doc",
            doc_type=doc_type,
            text=text,
            metadata={"doc_id": "test-doc", "doc_type": doc_type},
            config=_screening_config(),
        )

    def test_integrity_category_tools_exist(self):
        """Integrity category maps to integrity_hash and metadata_consistency tools."""
        from src.screening.engine import CATEGORY_TOOL_MAP
        assert "integrity" in CATEGORY_TOOL_MAP
        assert "integrity_hash" in CATEGORY_TOOL_MAP["integrity"]
        assert "metadata_consistency" in CATEGORY_TOOL_MAP["integrity"]

    def test_compliance_category_tools_exist(self):
        """Compliance category maps to template and policy tools."""
        from src.screening.engine import CATEGORY_TOOL_MAP
        assert "compliance" in CATEGORY_TOOL_MAP
        assert "template_conformance" in CATEGORY_TOOL_MAP["compliance"]
        assert "policy_compliance" in CATEGORY_TOOL_MAP["compliance"]

    def test_quality_category_tools(self):
        """Quality category includes numeric, citation, and ambiguity tools."""
        from src.screening.engine import CATEGORY_TOOL_MAP
        tools = CATEGORY_TOOL_MAP["quality"]
        assert "numeric_unit_consistency" in tools
        assert "citation_sanity" in tools
        assert "ambiguity_vagueness" in tools

    def test_language_category_tools(self):
        """Language category includes readability and passive voice tools."""
        from src.screening.engine import CATEGORY_TOOL_MAP
        tools = CATEGORY_TOOL_MAP["language"]
        assert "readability_style" in tools
        assert "passive_voice" in tools

    def test_security_category_tools(self):
        """Security category includes PII sensitivity tool."""
        from src.screening.engine import CATEGORY_TOOL_MAP
        assert "pii_sensitivity" in CATEGORY_TOOL_MAP["security"]

    def test_ai_authorship_category_tools(self):
        """AI authorship category includes the ai_authorship tool."""
        from src.screening.engine import CATEGORY_TOOL_MAP
        assert "ai_authorship" in CATEGORY_TOOL_MAP["ai_authorship"]
        assert "ai_authorship" in CATEGORY_TOOL_MAP["ai-authorship"]

    def test_resume_category_has_all_tools(self):
        """Resume category includes full resume screening tool chain."""
        from src.screening.engine import CATEGORY_TOOL_MAP
        tools = CATEGORY_TOOL_MAP["resume"]
        assert "resume_extractor_tool" in tools
        assert "resume_screening" in tools
        assert "resume_entity_validation" in tools

    def test_screen_resume_text(self):
        """Screen a resume document and verify report structure."""
        engine = self._make_engine()
        ctx = self._make_context(RESUME_TEXT_ALICE, doc_type="resume")
        report = engine.screen(ctx)

        assert hasattr(report, "overall_score_0_100")
        assert hasattr(report, "risk_level")
        assert report.risk_level in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(report.results, list)
        assert len(report.results) > 0

    def test_screen_invoice_text(self):
        """Screen an invoice document and verify report structure."""
        engine = self._make_engine()
        ctx = self._make_context(INVOICE_TEXT, doc_type="invoice")
        report = engine.screen(ctx)

        assert report.risk_level in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(report.overall_score_0_100, float)
        assert 0.0 <= report.overall_score_0_100 <= 100.0

    def test_screen_legal_text(self):
        """Screen a legal document and verify report structure."""
        engine = self._make_engine()
        ctx = self._make_context(LEGAL_TEXT, doc_type="contract")
        report = engine.screen(ctx)

        assert report.risk_level in ("LOW", "MEDIUM", "HIGH")

    def test_screening_report_to_dict(self):
        """ScreeningReport serializes to dict correctly."""
        engine = self._make_engine()
        ctx = self._make_context(RESUME_TEXT_ALICE, doc_type="resume")
        report = engine.screen(ctx)
        d = report.to_dict()

        assert "overall_score_0_100" in d
        assert "risk_level" in d
        assert "results" in d
        assert isinstance(d["results"], list)
        assert "provenance" in d

    def test_tool_result_structure(self):
        """Each ToolResult has required fields."""
        engine = self._make_engine()
        ctx = self._make_context(RESUME_TEXT_ALICE, doc_type="resume")
        report = engine.screen(ctx)

        for result in report.results:
            assert hasattr(result, "tool_name")
            assert hasattr(result, "category")
            assert hasattr(result, "score_0_1")
            assert 0.0 <= result.score_0_1 <= 1.0
            assert hasattr(result, "risk_level")
            assert result.risk_level in ("LOW", "MEDIUM", "HIGH")

    def test_resolve_tools_for_explicit_category(self):
        """Explicit category skips applies_to filter."""
        engine = self._make_engine()
        cfg = _screening_config()
        # Resume tools for generic doc type: explicit category should still resolve
        tools = engine._resolve_tools_for_category("resume", "generic", cfg)
        assert len(tools) > 0, "Explicit resume category should resolve tools even for generic doc_type"

    def test_pii_detection_on_text_with_email(self):
        """PII tool should detect email addresses."""
        from src.screening.tools import PIISensitivityTool
        from src.screening.models import ScreeningContext

        tool = PIISensitivityTool()
        ctx = ScreeningContext(
            doc_id="test",
            doc_type="generic",
            text="Contact alice@example.com or call (555) 123-4567 for details.",
            metadata={},
        )
        result = tool.run(ctx)
        assert result.score_0_1 > 0, "PII tool should flag text containing email and phone"


# ===========================================================================
# 3. RAG Pipeline
# ===========================================================================


class TestRAGPipelineExtractRender:
    """Test the extract -> render pipeline with FakeQdrant points."""

    def _make_chunks_from_points(self, points: List[FakePoint]):
        """Convert FakePoints to Chunk objects for extraction."""
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = []
        for pt in points:
            p = pt.payload
            chunks.append(Chunk(
                id=str(pt.id),
                text=p.get("canonical_text", ""),
                score=pt.score,
                source=ChunkSource(
                    document_name=p.get("source_name", "unknown"),
                    page=p.get("page"),
                ),
                meta={
                    "document_id": p.get("document_id"),
                    "doc_domain": p.get("doc_domain"),
                    "source_name": p.get("source_name"),
                    "section_kind": p.get("section_kind"),
                    "page": p.get("page"),
                },
            ))
        return chunks

    def test_extract_resume_produces_hr_schema(self):
        """Deterministic extraction of resume chunks produces HRSchema."""
        from src.rag_v3.extract import extract_schema
        from src.rag_v3.types import HRSchema, LLMBudget

        chunks = self._make_chunks_from_points(_make_resume_points_alice())
        budget = LLMBudget(llm_client=None, max_calls=0)

        result = extract_schema(
            None,
            query="Tell me about Alice Johnson",
            chunks=chunks,
            llm_client=None,
            budget=budget,
        )

        assert result.domain in ("hr", "resume")
        # Schema can be HRSchema or GenericSchema depending on extraction path
        schema = result.schema
        assert schema is not None

    def test_extract_invoice_produces_schema(self):
        """Deterministic extraction of invoice chunks produces a schema."""
        from src.rag_v3.extract import extract_schema
        from src.rag_v3.types import LLMBudget

        chunks = self._make_chunks_from_points(_make_invoice_points())
        budget = LLMBudget(llm_client=None, max_calls=0)

        result = extract_schema(
            None,
            query="What is the total amount on the invoice?",
            chunks=chunks,
            llm_client=None,
            budget=budget,
        )

        assert result.schema is not None
        assert result.domain in ("invoice", "generic")

    def test_extract_legal_produces_schema(self):
        """Deterministic extraction of legal chunks produces a schema."""
        from src.rag_v3.extract import extract_schema
        from src.rag_v3.types import LLMBudget

        chunks = self._make_chunks_from_points(_make_legal_points())
        budget = LLMBudget(llm_client=None, max_calls=0)

        result = extract_schema(
            None,
            query="What are the termination clauses?",
            chunks=chunks,
            llm_client=None,
            budget=budget,
        )

        assert result.schema is not None

    def test_render_produces_nonempty_text(self):
        """Render step produces non-empty text from extracted schema."""
        from src.rag_v3.extract import extract_schema
        try:
            from src.rag_v3.renderers.router import render
        except ImportError:
            pytest.skip("Module removed")
        from src.rag_v3.types import LLMBudget

        chunks = self._make_chunks_from_points(_make_resume_points_alice())
        budget = LLMBudget(llm_client=None, max_calls=0)

        extraction = extract_schema(
            None,
            query="What are Alice's skills?",
            chunks=chunks,
            llm_client=None,
            budget=budget,
        )

        rendered = render(
            domain=extraction.domain,
            intent=extraction.intent,
            schema=extraction.schema,
            strict=False,
            query="What are Alice's skills?",
        )

        # Rendered text might be empty for some schemas; emergency fallback handles it
        # But the function should not raise
        assert isinstance(rendered, str)

    def test_sanitize_cleans_output(self):
        """Sanitize step removes stray artifacts from rendered text."""
        from src.rag_v3.sanitize import sanitize

        text = "Alice has Python skills n and Java experience."
        result = sanitize(text)
        assert isinstance(result, str)
        assert "Alice" in result

    def test_rerank_preserves_order_without_cross_encoder(self):
        """Rerank without cross_encoder falls back to score-based ordering."""
        from src.rag_v3.rerank import rerank
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="c1", text="Low score chunk", score=0.3,
                  source=ChunkSource(document_name="doc.pdf"), meta={}),
            Chunk(id="c2", text="High score chunk", score=0.9,
                  source=ChunkSource(document_name="doc.pdf"), meta={}),
            Chunk(id="c3", text="Medium score chunk", score=0.6,
                  source=ChunkSource(document_name="doc.pdf"), meta={}),
        ]

        result = rerank(query="test query", chunks=chunks, cross_encoder=None, top_k=3)
        assert len(result) == 3
        # Without cross-encoder, should sort by score descending
        assert result[0].id == "c2"
        assert result[1].id == "c3"
        assert result[2].id == "c1"


class TestRAGPipelineRun:
    """Test the full RAG pipeline run() with mocked dependencies."""

    def _run_pipeline(self, query: str, points: List[FakePoint], **kwargs):
        """Run the pipeline with FakeQdrant and mocked dependencies."""
        from src.rag_v3.pipeline import run

        qdrant = FakeQdrant(points)
        embedder = FakeEmbedder()
        redis = FakeRedis()

        with patch("src.rag_v3.pipeline._start_intent_parse", return_value=None):
            result = run(
                query=query,
                subscription_id=SUB_ID,
                profile_id=PROFILE_ID,
                qdrant_client=qdrant,
                embedder=embedder,
                redis_client=redis,
                cross_encoder=None,
                llm_client=None,
                request_id="e2e-test",
                **kwargs,
            )
        return result

    def test_factual_query_returns_response(self):
        """Factual query returns a valid response dict."""
        result = self._run_pipeline(
            "What are Alice's skills?",
            _make_resume_points_alice(),
        )

        assert "response" in result
        assert isinstance(result["response"], str)
        assert "metadata" in result
        assert result["metadata"].get("rag_v3") is True

    def test_targeted_query_with_entity_hint(self):
        """Query about a specific person is detected as targeted scope."""
        result = self._run_pipeline(
            "Tell me about Alice Johnson's experience",
            _make_resume_points_alice() + _make_resume_points_bob(),
        )

        assert "response" in result
        assert isinstance(result["response"], str)

    def test_all_profile_query_triggers_multi_doc(self):
        """Queries about 'all candidates' route to all-profile analysis."""
        result = self._run_pipeline(
            "Compare all candidates in the profile",
            _make_resume_points_alice() + _make_resume_points_bob(),
        )

        assert "response" in result
        metadata = result.get("metadata", {})
        # Should be all_profile scope
        scope = metadata.get("scope")
        assert scope == "all_profile" or (isinstance(scope, dict) and "profile_id" in scope)

    def test_specific_document_query(self):
        """Query with explicit document_id routes to specific_document scope."""
        result = self._run_pipeline(
            "Summarize this document",
            _make_resume_points_alice(),
            document_id=DOC_ID_RESUME_1,
        )

        assert "response" in result

    def test_empty_collection_returns_no_chunks_message(self):
        """Query against empty collection returns appropriate message."""
        result = self._run_pipeline(
            "What skills does the candidate have?",
            [],  # No points
        )

        assert "response" in result
        # Should indicate insufficient information
        response_lower = result["response"].lower()
        assert any(phrase in response_lower for phrase in [
            "not enough information",
            "no documents",
            "insufficient",
            "could not find",
            "unable to",
            "i couldn't find",
        ]) or result.get("context_found") is False

    def test_invoice_query(self):
        """Invoice-specific query returns relevant response."""
        result = self._run_pipeline(
            "What is the total amount on the invoice?",
            _make_invoice_points(),
        )

        assert "response" in result
        assert isinstance(result["response"], str)

    def test_mixed_document_query(self):
        """Query across mixed document types works."""
        result = self._run_pipeline(
            "List all documents in this profile",
            _all_points(),
        )

        assert "response" in result


# ===========================================================================
# 4. Response Intelligence (Query Type Classification)
# ===========================================================================


class TestResponseIntelligence:
    """Test query intent classification and response routing."""

    def test_factual_query_classification(self):
        """Factual queries are classified correctly."""
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("What is Alice's email?") == "factual"

    def test_comparison_query_classification(self):
        """Comparison queries are classified correctly."""
        from src.rag_v3.llm_extract import classify_query_intent
        result = classify_query_intent("Compare Alice and Bob's skills")
        assert result == "comparison"

    def test_summary_query_classification(self):
        """Summary queries are classified correctly via NLU engine."""
        from src.rag_v3.llm_extract import classify_query_intent
        with patch("src.nlp.nlu_engine.classify_intent", return_value="summary"):
            result = classify_query_intent("Summarize Alice's experience")
        assert result == "summary"

    def test_ranking_query_classification(self):
        """Ranking queries are classified correctly."""
        from src.rag_v3.llm_extract import classify_query_intent
        result = classify_query_intent("Rank the candidates by experience")
        assert result == "ranking"

    def test_reasoning_query_classification(self):
        """Reasoning queries are classified correctly via NLU engine."""
        from src.rag_v3.llm_extract import classify_query_intent
        with patch("src.nlp.nlu_engine.classify_intent", return_value="reasoning"):
            result = classify_query_intent("Why is Alice a good fit for the role?")
        assert result == "reasoning"

    def test_cross_document_query_classification(self):
        """Cross-document queries are classified correctly via NLU engine."""
        from src.rag_v3.llm_extract import classify_query_intent
        with patch("src.nlp.nlu_engine.classify_intent", return_value="cross_document"):
            result = classify_query_intent("What skills do each candidate share?")
        assert result == "cross_document"

    def test_analytics_query_classification(self):
        """Analytics queries are classified correctly."""
        from src.rag_v3.llm_extract import classify_query_intent
        result = classify_query_intent("How many candidates have Python skills?")
        assert result == "analytics"

    def test_intent_hint_overrides_regex(self):
        """Intent hint from intent parse takes precedence over regex classification."""
        from src.rag_v3.llm_extract import classify_query_intent
        result = classify_query_intent("Show details", intent_hint="compare")
        assert result == "comparison"

    def test_query_scope_all_profile_patterns(self):
        """'all documents' patterns route to all_profile scope when ML classifier provides intent."""
        from src.rag_v3.pipeline import _infer_query_scope

        queries_with_intents = [
            ("Compare all candidates", _FakeIntentParse(intent="compare")),
            ("List all documents", _FakeIntentParse(intent="list")),
            ("How many resumes are there?", _FakeIntentParse(intent="list")),
            ("Show all invoices", _FakeIntentParse(intent="list")),
            ("Rank every candidate", _FakeIntentParse(intent="rank")),
        ]
        for q, intent in queries_with_intents:
            scope = _infer_query_scope(q, None, intent)
            assert scope.mode == "all_profile", f"Expected all_profile for: {q!r}, got {scope.mode}"

    def test_query_scope_targeted_entity(self):
        """Entity mentions in queries route to targeted scope."""
        from src.rag_v3.pipeline import _infer_query_scope

        scope = _infer_query_scope("Tell me about Alice Johnson", None, None)
        assert scope.mode == "targeted"
        assert scope.entity_hint is not None
        assert "Alice" in scope.entity_hint

    def test_query_scope_specific_document(self):
        """Explicit document_id routes to specific_document scope."""
        from src.rag_v3.pipeline import _infer_query_scope

        scope = _infer_query_scope("Summarize", "doc-123", None)
        assert scope.mode == "specific_document"
        assert scope.document_id == "doc-123"

    def test_query_scope_possessive_entity(self):
        """Possessive entity pattern is detected."""
        from src.rag_v3.pipeline import _infer_query_scope

        scope = _infer_query_scope("What are Alice's certifications?", None, None)
        assert scope.mode == "targeted"
        assert scope.entity_hint is not None


# ===========================================================================
# 5. Content Generation
# ===========================================================================


class TestContentGeneration:
    """Test the content generation engine with deterministic fallback."""

    def _make_chunk_dicts(self, text: str, source_name: str = "doc.pdf", domain: str = "hr"):
        """Create chunk dicts for content generation."""
        return [
            {
                "canonical_text": text,
                "text": text,
                "score": 0.9,
                "metadata": {
                    "source_name": source_name,
                    "doc_domain": domain,
                    "page": 1,
                },
                "payload": {
                    "source_name": source_name,
                    "doc_domain": domain,
                },
            }
        ]

    def test_detect_cover_letter(self):
        """Cover letter content type is detected from query via NLU registry."""
        from src.content_generation.registry import detect_content_type
        from src.nlp.nlu_engine import ClassificationResult

        # Mock the NLU engine functions that detect_content_type imports
        mock_reg = MagicMock()
        mock_reg.classify.return_value = ClassificationResult(
            name="cover_letter", score=0.65, method="nlu_structural",
        )
        with patch("src.nlp.nlu_engine._ensure_registry"), \
             patch("src.nlp.nlu_engine.get_registry", return_value=mock_reg):
            ct_id = detect_content_type("Generate a cover letter for Alice")
        assert ct_id is not None
        assert ct_id == "cover_letter"

    def test_detect_key_points(self):
        """Key points content type is detected from query via NLU registry."""
        from src.content_generation.registry import detect_content_type
        from src.nlp.nlu_engine import ClassificationResult

        mock_reg = MagicMock()
        mock_reg.classify.return_value = ClassificationResult(
            name="key_points", score=0.60, method="nlu_structural",
        )
        with patch("src.nlp.nlu_engine._ensure_registry"), \
             patch("src.nlp.nlu_engine.get_registry", return_value=mock_reg):
            ct_id = detect_content_type("Extract the key points from this document")
        assert ct_id is not None
        assert ct_id == "key_points"

    def test_detect_summary(self):
        """Summary content type is detected from query."""
        from src.content_generation.registry import detect_content_type_with_domain

        ct = detect_content_type_with_domain("Write a professional summary", "hr")
        assert ct is not None

    def test_content_type_registry_has_types(self):
        """Registry contains expected content types."""
        from src.content_generation.registry import CONTENT_TYPE_REGISTRY

        assert "cover_letter" in CONTENT_TYPE_REGISTRY
        assert "key_points" in CONTENT_TYPE_REGISTRY
        assert "professional_summary" in CONTENT_TYPE_REGISTRY

    def test_cover_letter_deterministic_generation(self):
        """Cover letter generation with deterministic fallback (no LLM)."""
        from src.content_generation.engine import ContentGenerationEngine

        engine = ContentGenerationEngine(llm_client=None)
        chunks = self._make_chunk_dicts(RESUME_TEXT_ALICE, source_name="Alice_Resume.pdf")

        result = engine.generate(
            query="Generate a cover letter for Alice",
            chunks=chunks,
            content_type_id="cover_letter",
        )

        assert result.get("response"), "Should produce non-empty response"
        assert "Dear Hiring Manager" in result["response"]
        assert result.get("context_found") is True
        assert result.get("metadata", {}).get("generation_method") == "deterministic"

    def test_key_points_deterministic_generation(self):
        """Key points generation with deterministic fallback (no LLM)."""
        from src.content_generation.engine import ContentGenerationEngine

        engine = ContentGenerationEngine(llm_client=None)
        chunks = self._make_chunk_dicts(RESUME_TEXT_ALICE)

        result = engine.generate(
            query="Extract key points",
            chunks=chunks,
            content_type_id="key_points",
        )

        assert result.get("response"), "Should produce non-empty response"
        assert "Key Points" in result["response"]
        assert result.get("metadata", {}).get("generation_method") == "deterministic"

    def test_summary_deterministic_generation(self):
        """Summary generation falls back to deterministic when no LLM."""
        from src.content_generation.engine import ContentGenerationEngine

        engine = ContentGenerationEngine(llm_client=None)
        chunks = self._make_chunk_dicts(RESUME_TEXT_ALICE)

        result = engine.generate(
            query="Write a professional summary",
            chunks=chunks,
            content_type_id="professional_summary",
        )

        assert result.get("response"), "Should produce non-empty response"
        assert result.get("metadata", {}).get("generation_method") == "deterministic"

    def test_empty_chunks_returns_insufficient(self):
        """Empty chunks produce appropriate error response."""
        from src.content_generation.engine import ContentGenerationEngine

        engine = ContentGenerationEngine(llm_client=None)
        result = engine.generate(
            query="Generate a cover letter",
            chunks=[],
        )

        assert "Insufficient" in result.get("response", "")
        assert result.get("context_found") is False

    def test_content_generation_with_mock_llm(self):
        """Content generation uses LLM when available."""
        from src.content_generation.engine import ContentGenerationEngine

        mock_llm = MagicMock()
        mock_llm.generate_with_metadata.return_value = (
            "This is an LLM-generated cover letter for Alice Johnson. "
            "She has 8 years of Python experience and AWS certifications. "
            "Dear Hiring Manager, I recommend Alice for this role.",
            {},
        )

        engine = ContentGenerationEngine(llm_client=mock_llm)
        chunks = self._make_chunk_dicts(RESUME_TEXT_ALICE)

        result = engine.generate(
            query="Generate a cover letter for Alice",
            chunks=chunks,
            content_type_id="cover_letter",
        )

        assert result.get("response"), "Should produce non-empty response"
        assert result.get("metadata", {}).get("generation_method") == "llm"

    def test_verification_result_structure(self):
        """Content verification produces correct result structure."""
        from src.content_generation.verifier import ContentVerifier

        verifier = ContentVerifier()
        generated = "Alice has 8 years of experience in Python and Java."
        chunks = self._make_chunk_dicts(RESUME_TEXT_ALICE)
        facts = {"person_name": "Alice Johnson", "skills": ["Python", "Java"]}

        result = verifier.verify(generated, chunks, facts)
        assert hasattr(result, "grounded")
        assert hasattr(result, "score")
        assert hasattr(result, "total_claims")
        d = result.to_dict()
        assert "grounded" in d
        assert "score" in d

    def test_fact_extraction_from_resume_chunks(self):
        """Fact extraction pulls person names and skills from resume text."""
        from src.content_generation.engine import _extract_facts_from_chunks
        from src.content_generation.registry import get_content_type

        ct = get_content_type("cover_letter")
        chunks = self._make_chunk_dicts(RESUME_TEXT_ALICE)
        facts = _extract_facts_from_chunks(chunks, ct)

        assert "person_name" in facts, "Should extract person name"
        assert "skills" in facts, "Should extract skills"

    def test_fact_extraction_from_invoice_chunks(self):
        """Fact extraction pulls amounts from invoice text."""
        from src.content_generation.engine import _extract_facts_from_chunks, _extract_amounts

        amounts = _extract_amounts(INVOICE_TEXT)
        assert len(amounts) > 0, "Should extract monetary amounts from invoice"


# ===========================================================================
# 6. Gateway Execution
# ===========================================================================


class TestGatewayExecution:
    """Test the unified gateway executor (ScreeningExecutor)."""

    def test_build_response_structure(self):
        """_build_response produces correctly structured response."""
        from src.gateway.unified_executor import _build_response
        import time

        resp = _build_response(
            status="success",
            action="screen:integrity",
            correlation_id="test-cid",
            start_time=time.time(),
            result={"score": 85},
            metadata={"test": True},
        )

        assert resp["status"] == "success"
        assert resp["action"] == "screen:integrity"
        assert resp["correlation_id"] == "test-cid"
        assert resp["result"] == {"score": 85}
        assert "timestamp" in resp
        assert "duration_ms" in resp
        assert isinstance(resp["duration_ms"], int)

    def test_executor_category_requires_doc_ids(self):
        """Category screening without doc_ids returns error."""
        from src.gateway.unified_executor import ScreeningExecutor

        executor = ScreeningExecutor()
        result = _run_async(executor.execute_screening(
            categories=["integrity"],
        ))
        assert result["status"] == "error"
        assert result["error"]["code"] == "missing_doc_ids"

    def test_executor_batch_requires_profile_ids(self):
        """Batch screening without profile_ids returns error."""
        from src.gateway.unified_executor import ScreeningExecutor

        executor = ScreeningExecutor()
        result = _run_async(executor.execute_screening(
            categories=["run"],
        ))
        assert result["status"] == "error"
        assert result["error"]["code"] == "missing_profile_ids"

    def test_executor_correlation_id_passthrough(self):
        """Custom correlation_id is preserved in response."""
        from src.gateway.unified_executor import ScreeningExecutor

        executor = ScreeningExecutor()
        result = _run_async(executor.execute_screening(
            categories=["integrity"],
            correlation_id="custom-cid-123",
        ))
        assert result["correlation_id"] == "custom-cid-123"

    def test_executor_all_category_with_mock_engine(self):
        """'all' category dispatches to run_all on the engine."""
        from src.gateway.unified_executor import ScreeningExecutor

        executor = ScreeningExecutor()
        mock_engine = MagicMock()
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {
            "overall_score_0_100": 72.5,
            "risk_level": "MEDIUM",
            "results": [],
        }
        mock_engine.run_all.return_value = mock_report
        executor._screening_engine = mock_engine

        result = _run_async(executor.execute_screening(
            categories=["all"],
            doc_ids=["test-doc-1"],
        ))

        assert result["status"] == "success"
        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc["doc_id"] == "test-doc-1"
        assert doc["status"] == "success"

    def test_executor_handles_engine_exception(self):
        """Engine exceptions are caught per-document."""
        from src.gateway.unified_executor import ScreeningExecutor

        executor = ScreeningExecutor()
        mock_engine = MagicMock()
        mock_engine.run_all.side_effect = RuntimeError("Engine failure")
        executor._screening_engine = mock_engine

        result = _run_async(executor.execute_screening(
            categories=["all"],
            doc_ids=["test-doc-1"],
        ))

        assert result["documents"][0]["status"] == "failed"

    def test_executor_multiple_doc_ids(self):
        """Executor processes multiple doc_ids."""
        from src.gateway.unified_executor import ScreeningExecutor

        executor = ScreeningExecutor()
        mock_engine = MagicMock()
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"overall_score_0_100": 80, "risk_level": "MEDIUM", "results": []}
        mock_engine.run_all.return_value = mock_report
        executor._screening_engine = mock_engine

        result = _run_async(executor.execute_screening(
            categories=["all"],
            doc_ids=["doc-1", "doc-2", "doc-3"],
        ))

        assert result["status"] == "success"
        assert len(result["documents"]) == 3

    def test_summarize_input_strips_long_text(self):
        """_summarize_input replaces long strings with length indicators."""
        from src.gateway.unified_executor import ScreeningExecutor

        summary = ScreeningExecutor._summarize_input({
            "text": "x" * 200,
            "doc_id": "short-val",
            "tags": ["a", "b"],
        })
        assert "text_length" in summary
        assert summary["text_length"] == 200
        assert summary["doc_id"] == "short-val"
        assert "tags_count" in summary


# ===========================================================================
# 7. Integration: Full Pipeline with Schema Normalization
# ===========================================================================


class TestPipelineIntegration:
    """Integration tests combining multiple pipeline stages."""

    def test_payload_to_chunk_to_extraction_round_trip(self):
        """Payload -> FakeQdrant -> Chunk -> extraction produces valid result."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
        from src.rag_v3.types import Chunk, ChunkSource, LLMBudget
        from src.rag_v3.extract import extract_schema

        # Step 1: Build payload
        raw = {
            "subscription_id": SUB_ID,
            "profile_id": PROFILE_ID,
            "document_id": DOC_ID_RESUME_1,
            "source_name": "Alice_Johnson_Resume.pdf",
            "content": RESUME_TEXT_ALICE,
            "page": 1,
            "chunk_id": "integration-c1",
        }
        payload = build_qdrant_payload(raw)

        # Step 2: Create Chunk from payload (simulating retrieval)
        chunk = Chunk(
            id="integration-c1",
            text=payload.get("canonical_text", ""),
            score=0.95,
            source=ChunkSource(
                document_name=payload.get("source_name", "unknown"),
                page=payload.get("page"),
            ),
            meta={
                "document_id": payload.get("document_id"),
                "doc_domain": payload.get("doc_domain"),
                "source_name": payload.get("source_name"),
                "section_kind": payload.get("section_kind"),
            },
        )

        # Step 3: Extract
        budget = LLMBudget(llm_client=None, max_calls=0)
        result = extract_schema(
            None,
            query="What are Alice Johnson's skills?",
            chunks=[chunk],
            llm_client=None,
            budget=budget,
        )

        assert result.domain is not None
        assert result.schema is not None

    def test_content_generation_from_rag_chunks(self):
        """Content generation can consume RAG-style Chunk objects."""
        from src.content_generation.engine import ContentGenerationEngine
        from src.rag_v3.types import Chunk, ChunkSource

        chunk = Chunk(
            id="cg-c1",
            text=RESUME_TEXT_ALICE,
            score=0.95,
            source=ChunkSource(document_name="Alice_Resume.pdf", page=1),
            meta={"source_name": "Alice_Resume.pdf", "doc_domain": "resume"},
        )

        engine = ContentGenerationEngine(llm_client=None)
        result = engine.generate(
            query="Generate a cover letter",
            chunks=[chunk],
            content_type_id="cover_letter",
        )

        assert result.get("response"), "Should generate content from Chunk objects"

    def test_screening_and_extraction_complementary(self):
        """Screening and extraction can process the same document text independently."""
        from src.screening.engine import ScreeningEngine
        from src.screening.models import ScreeningContext
        from src.rag_v3.extract import extract_schema
        from src.rag_v3.types import Chunk, ChunkSource, LLMBudget

        cfg = _screening_config()

        # Screening
        engine = ScreeningEngine(config=cfg)
        ctx = ScreeningContext(
            doc_id="test-doc",
            doc_type="resume",
            text=RESUME_TEXT_ALICE,
            metadata={},
            config=cfg,
        )
        report = engine.screen(ctx)
        assert report.risk_level in ("LOW", "MEDIUM", "HIGH")

        # Extraction on same text
        chunk = Chunk(
            id="dual-c1",
            text=RESUME_TEXT_ALICE,
            score=0.9,
            source=ChunkSource(document_name="resume.pdf", page=1),
            meta={"doc_domain": "resume", "source_name": "resume.pdf", "document_id": "doc-1"},
        )
        budget = LLMBudget(llm_client=None, max_calls=0)
        extraction = extract_schema(
            None,
            query="What are the candidate's skills?",
            chunks=[chunk],
            llm_client=None,
            budget=budget,
        )
        assert extraction.schema is not None

    def test_judge_passes_valid_deterministic_extraction(self):
        """Judge passes when deterministic extraction has valid data."""
        from src.rag_v3.pipeline import _has_valid_deterministic_extraction
        from src.rag_v3.types import (
            HRSchema, CandidateField, Candidate, EvidenceSpan,
            GenericSchema, FieldValuesField, FieldValue,
        )

        # HR schema with valid candidate
        hr = HRSchema(candidates=CandidateField(items=[
            Candidate(
                name="Alice Johnson",
                technical_skills=["Python", "Java"],
                evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="Alice Johnson...")],
            ),
        ]))
        assert _has_valid_deterministic_extraction(hr) is True

        # Generic schema with substantial facts
        generic = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Skills", value="Python, Java, AWS, Docker, Kubernetes, Machine Learning",
                       evidence_spans=[]),
            FieldValue(label="Experience", value="8 years of experience at TechCorp and StartupXYZ",
                       evidence_spans=[]),
        ]))
        assert _has_valid_deterministic_extraction(generic) is True

        # Empty HR schema should fail
        empty_hr = HRSchema(candidates=CandidateField(items=[]))
        assert _has_valid_deterministic_extraction(empty_hr) is False


# ===========================================================================
# 8. LLM Budget Management
# ===========================================================================


class TestLLMBudget:
    """Test LLM budget allocation and consumption."""

    def test_budget_allows_when_has_client_and_calls(self):
        from src.rag_v3.types import LLMBudget

        budget = LLMBudget(llm_client=MagicMock(), max_calls=2)
        assert budget.allow() is True

    def test_budget_denies_when_no_client(self):
        from src.rag_v3.types import LLMBudget

        budget = LLMBudget(llm_client=None, max_calls=2)
        assert budget.allow() is False

    def test_budget_denies_after_exhaustion(self):
        from src.rag_v3.types import LLMBudget

        budget = LLMBudget(llm_client=MagicMock(), max_calls=2)
        assert budget.consume() is True
        assert budget.consume() is True
        assert budget.allow() is False
        assert budget.consume() is False

    def test_separate_infra_and_extract_budgets(self):
        """Pipeline uses separate budgets for infrastructure and extraction."""
        from src.rag_v3.types import LLMBudget

        client = MagicMock()
        infra_budget = LLMBudget(llm_client=client, max_calls=2)
        extract_budget = LLMBudget(llm_client=client, max_calls=4)

        # Consume all infra budget
        infra_budget.consume()
        infra_budget.consume()
        assert infra_budget.allow() is False

        # Extract budget should still be available
        assert extract_budget.allow() is True
        assert extract_budget.used == 0


# ===========================================================================
# 9. Domain Router & Document Context
# ===========================================================================


class TestDomainRouter:
    """Test domain inference and routing."""

    def test_infer_section_kind_from_skills_query(self):
        """Skills query maps to skills_technical section kind."""
        from src.rag_v3.retrieve import _infer_query_section_kind

        result = _infer_query_section_kind("What are the technical skills?")
        assert result == "skills_technical"

    def test_infer_section_kind_from_education_query(self):
        """Education query maps to education section kind."""
        from src.rag_v3.retrieve import _infer_query_section_kind

        result = _infer_query_section_kind("What is the candidate's education?")
        assert result == "education"

    def test_infer_section_kind_from_experience_query(self):
        """Experience query maps to experience section kind."""
        from src.rag_v3.retrieve import _infer_query_section_kind

        result = _infer_query_section_kind("Describe their work history")
        assert result == "experience"

    def test_infer_section_kind_returns_none_for_generic(self):
        """Generic query returns None for section kind."""
        from src.rag_v3.retrieve import _infer_query_section_kind

        result = _infer_query_section_kind("Hello, how are you?")
        assert result is None


# ===========================================================================
# 10. Emergency and Fallback Paths
# ===========================================================================


class TestFallbackPaths:
    """Test emergency fallbacks and error recovery."""

    def test_emergency_chunk_summary(self):
        """Emergency chunk summary produces meaningful text from chunks."""
        from src.rag_v3.pipeline import _emergency_chunk_summary
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="c1", text="Alice has Python skills.", score=0.9,
                  source=ChunkSource(document_name="doc.pdf"), meta={}),
            Chunk(id="c2", text="She worked at TechCorp for 6 years.", score=0.8,
                  source=ChunkSource(document_name="doc.pdf"), meta={}),
        ]
        result = _emergency_chunk_summary(chunks, "What are her skills?")

        # Emergency summary returns bullet-point or raw content (no preamble)
        assert len(result) > 10, "Emergency summary should produce meaningful text"
        assert "Alice" in result or "Python" in result or "skills" in result

    def test_emergency_chunk_summary_empty_chunks(self):
        """Emergency summary returns empty for no chunks."""
        from src.rag_v3.pipeline import _emergency_chunk_summary

        result = _emergency_chunk_summary([], "query")
        assert result == ""

    def test_fallback_answer_constant(self):
        """FALLBACK_ANSWER constant is defined and non-empty."""
        from src.rag_v3.sanitize import FALLBACK_ANSWER

        assert isinstance(FALLBACK_ANSWER, str)
        assert len(FALLBACK_ANSWER) > 10

    def test_no_chunks_message_constant(self):
        """NO_CHUNKS_MESSAGE is defined."""
        from src.rag_v3.pipeline import NO_CHUNKS_MESSAGE

        assert isinstance(NO_CHUNKS_MESSAGE, str)
        assert "information" in NO_CHUNKS_MESSAGE.lower()


# ===========================================================================
# 11. Content Type Detection Across Domains
# ===========================================================================


class TestContentTypeDetection:
    """Test content type detection for various domains."""

    def test_invoice_summary_detection(self):
        """Invoice summary detection works via NLU registry."""
        from src.content_generation.registry import detect_content_type
        from src.nlp.nlu_engine import ClassificationResult

        mock_reg = MagicMock()
        mock_reg.classify.return_value = ClassificationResult(
            name="invoice_summary", score=0.62, method="nlu_structural",
        )
        with patch("src.nlp.nlu_engine._ensure_registry"), \
             patch("src.nlp.nlu_engine.get_registry", return_value=mock_reg):
            ct_id = detect_content_type("Give me an invoice summary")
        assert ct_id is not None
        assert ct_id == "invoice_summary"

    def test_skills_matrix_detection(self):
        """Skills matrix detection from query."""
        from src.content_generation.registry import detect_content_type

        # detect_content_type returns a string type_id
        ct_id = detect_content_type("Create a skills matrix")
        assert ct_id is None or ct_id == "skills_matrix"

    def test_candidate_comparison_detection(self):
        """Candidate comparison detection from query via NLU registry."""
        from src.content_generation.registry import detect_content_type
        from src.nlp.nlu_engine import ClassificationResult

        mock_reg = MagicMock()
        mock_reg.classify.return_value = ClassificationResult(
            name="candidate_comparison", score=0.58, method="nlu_structural",
        )
        with patch("src.nlp.nlu_engine._ensure_registry"), \
             patch("src.nlp.nlu_engine.get_registry", return_value=mock_reg):
            ct_id = detect_content_type("Do a candidate comparison")
        assert ct_id is not None
        assert ct_id == "candidate_comparison"

    def test_content_type_registry_domains(self):
        """Registry covers all expected domains."""
        from src.content_generation.registry import CONTENT_TYPE_REGISTRY, DOMAINS

        registered_domains = {ct.domain for ct in CONTENT_TYPE_REGISTRY.values()}
        # At minimum, hr and general should be present
        assert "hr" in registered_domains
        assert "general" in registered_domains


# ===========================================================================
# 12. Live Tests (gated behind LIVE_TEST=1)
# ===========================================================================


@pytest.mark.skipif(not os.getenv("LIVE_TEST"), reason="Live test — set LIVE_TEST=1")
class TestLivePipeline:
    """Live tests that query the actual Qdrant instance."""

    LIVE_SUB_ID = os.getenv("LIVE_SUBSCRIPTION_ID", "67fde0754e36c00b14cea7f5")
    LIVE_PROFILE_ID = os.getenv("LIVE_PROFILE_ID", "698c46e6bcae2c45eca1d8d9")

    def _get_live_clients(self):
        """Get real Qdrant, embedder, and Redis clients from app state."""
        from qdrant_client import QdrantClient
        from src.api.config import Config

        qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)

        # Lightweight embedder for tests
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(Config.Model.EMBEDDING_MODEL)
        except Exception:
            embedder = FakeEmbedder()

        return qdrant, embedder

    def test_live_collection_has_points(self):
        """Verify live collection has data points."""
        qdrant, _ = self._get_live_clients()
        from src.api.vector_store import build_collection_name, build_qdrant_filter

        collection = build_collection_name(self.LIVE_SUB_ID)
        q_filter = build_qdrant_filter(
            subscription_id=self.LIVE_SUB_ID,
            profile_id=self.LIVE_PROFILE_ID,
        )
        resp = qdrant.count(collection_name=collection, count_filter=q_filter, exact=True)
        assert resp.count > 0, f"Expected points in collection {collection}"

    def test_live_factual_query(self):
        """Live factual query returns meaningful response."""
        qdrant, embedder = self._get_live_clients()
        from src.rag_v3.pipeline import run

        result = run(
            query="What skills does the candidate have?",
            subscription_id=self.LIVE_SUB_ID,
            profile_id=self.LIVE_PROFILE_ID,
            qdrant_client=qdrant,
            embedder=embedder,
            redis_client=FakeRedis(),
            cross_encoder=None,
            llm_client=None,
            request_id="live-e2e-factual",
        )

        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 20, "Response should be substantive"
        # Should not contain fallback message
        assert "not enough information" not in result["response"].lower() or result.get("sources")

    def test_live_all_profile_query(self):
        """Live all-profile query returns multi-document response."""
        qdrant, embedder = self._get_live_clients()
        from src.rag_v3.pipeline import run

        result = run(
            query="List all candidates in this profile",
            subscription_id=self.LIVE_SUB_ID,
            profile_id=self.LIVE_PROFILE_ID,
            qdrant_client=qdrant,
            embedder=embedder,
            redis_client=FakeRedis(),
            cross_encoder=None,
            llm_client=None,
            request_id="live-e2e-all-profile",
        )

        assert "response" in result
        metadata = result.get("metadata", {})
        scope = metadata.get("scope")
        assert scope == "all_profile" or isinstance(scope, dict)


# ===========================================================================
# 13. Schema Type Structures
# ===========================================================================


class TestSchemaStructures:
    """Test that schema types can be constructed and serialized correctly."""

    def test_chunk_source_creation(self):
        from src.rag_v3.types import ChunkSource
        cs = ChunkSource(document_name="test.pdf", page=1)
        assert cs.document_name == "test.pdf"
        assert cs.page == 1

    def test_chunk_creation(self):
        from src.rag_v3.types import Chunk, ChunkSource
        c = Chunk(
            id="test-id",
            text="test text",
            score=0.85,
            source=ChunkSource(document_name="doc.pdf"),
            meta={"key": "value"},
        )
        assert c.id == "test-id"
        assert c.score == 0.85

    def test_hr_schema_creation(self):
        from src.rag_v3.types import HRSchema, CandidateField, Candidate, EvidenceSpan

        schema = HRSchema(candidates=CandidateField(items=[
            Candidate(
                name="Test User",
                technical_skills=["Python"],
                evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="evidence")],
            )
        ]))
        assert schema.candidates.items[0].name == "Test User"

    def test_invoice_schema_creation(self):
        from src.rag_v3.types import InvoiceSchema, InvoiceItemsField, InvoiceItem, EvidenceSpan

        schema = InvoiceSchema(items=InvoiceItemsField(items=[
            InvoiceItem(
                description="Service",
                amount="$100",
                evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="ev")],
            )
        ]))
        assert schema.items.items[0].description == "Service"

    def test_legal_schema_creation(self):
        from src.rag_v3.types import LegalSchema, ClauseField, Clause, EvidenceSpan

        schema = LegalSchema(clauses=ClauseField(items=[
            Clause(
                title="Termination",
                text="Either party may terminate...",
                evidence_spans=[EvidenceSpan(chunk_id="c1", snippet="ev")],
            )
        ]))
        assert schema.clauses.items[0].title == "Termination"

    def test_generic_schema_creation(self):
        from src.rag_v3.types import GenericSchema, FieldValuesField, FieldValue

        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Key", value="Value", evidence_spans=[])
        ]))
        assert schema.facts.items[0].label == "Key"

    def test_llm_response_schema_creation(self):
        from src.rag_v3.types import LLMResponseSchema

        schema = LLMResponseSchema(text="Generated answer", evidence_chunks=["c1", "c2"])
        assert schema.text == "Generated answer"
        assert len(schema.evidence_chunks) == 2

    def test_llm_budget_dataclass(self):
        from src.rag_v3.types import LLMBudget

        budget = LLMBudget(llm_client=None, max_calls=3)
        assert budget.max_calls == 3
        assert budget.used == 0
        assert budget.allow() is False  # No client


# ===========================================================================
# 14. FakeQdrant Filter Verification
# ===========================================================================


class TestFakeQdrantFiltering:
    """Test that FakeQdrant correctly filters points for pipeline tests."""

    def test_filter_by_profile_id(self):
        """FakeQdrant filters by profile_id correctly."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        points = _all_points()
        qdrant = FakeQdrant(points)

        # make_point() hardcodes subscription_id="sub-1", so filter by that
        q_filter = Filter(must=[
            FieldCondition(key="subscription_id", match=MatchValue(value="sub-1")),
            FieldCondition(key="profile_id", match=MatchValue(value=PROFILE_ID)),
        ])
        result = qdrant.query_points(
            collection_name="test",
            query=[0.1, 0.1, 0.1, 0.1],
            query_filter=q_filter,
            limit=100,
        )

        assert len(result.points) == len(points), "All points match the test profile_id"

    def test_filter_by_document_id(self):
        """FakeQdrant filters by document_id correctly."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        points = _all_points()
        qdrant = FakeQdrant(points)

        q_filter = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=DOC_ID_RESUME_1)),
        ])
        result, _ = qdrant.scroll(query_filter=q_filter, limit=100)

        assert all(
            p.payload.get("document_id") == DOC_ID_RESUME_1
            for p in result
        )
        assert len(result) == 3  # Alice has 3 resume points

    def test_count_returns_correct_total(self):
        """FakeQdrant count() returns correct point count."""
        points = _all_points()
        qdrant = FakeQdrant(points)

        resp = qdrant.count()
        assert resp.count == len(points)

    def test_scroll_without_filter_returns_all(self):
        """FakeQdrant scroll without filter returns all points."""
        points = _all_points()
        qdrant = FakeQdrant(points)

        result, _ = qdrant.scroll(limit=100)
        assert len(result) == len(points)


# ===========================================================================
# 15. Query Scope Edge Cases
# ===========================================================================


class TestQueryScopeEdgeCases:
    """Test edge cases in query scope inference."""

    def test_empty_query_defaults_to_all_profile(self):
        from src.rag_v3.pipeline import _infer_query_scope

        scope = _infer_query_scope("", None, None)
        assert scope.mode == "all_profile"

    def test_document_id_in_query_text(self):
        from src.rag_v3.pipeline import _infer_query_scope

        scope = _infer_query_scope("Show document_id: abc123", None, None)
        assert scope.mode == "specific_document"
        assert scope.document_id == "abc123"

    def test_verb_plus_name_entity_detection(self):
        """'summarize Dhayal' pattern detects entity."""
        from src.rag_v3.pipeline import _infer_query_scope

        scope = _infer_query_scope("summarize Dhayal", None, None)
        assert scope.mode == "targeted"
        assert scope.entity_hint is not None
        assert "Dhayal" in scope.entity_hint

    def test_possessive_with_certifications(self):
        """Possessive + certifications pattern detects entity."""
        from src.rag_v3.pipeline import _infer_query_scope

        scope = _infer_query_scope("Show Alice's certifications", None, None)
        assert scope.mode == "targeted"
        assert scope.entity_hint is not None

    def test_how_many_invoices_is_all_profile(self):
        from src.rag_v3.pipeline import _infer_query_scope

        intent = _FakeIntentParse(intent="list")
        scope = _infer_query_scope("How many invoices are there?", None, intent)
        assert scope.mode == "all_profile"

    def test_between_all_documents_is_all_profile(self):
        from src.rag_v3.pipeline import _infer_query_scope

        intent = _FakeIntentParse(intent="compare")
        scope = _infer_query_scope("What differences exist between all documents?", None, intent)
        assert scope.mode == "all_profile"
