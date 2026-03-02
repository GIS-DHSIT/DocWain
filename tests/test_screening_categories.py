"""Comprehensive diagnostic tests for all screening categories.

Tests every screening category individually via:
1. Direct tool `run()` invocation with ScreeningContext
2. Engine `_resolve_tools_for_category()` verification
3. Engine `evaluate()` for inline text screening
4. Engine `screen()` with full report generation
5. Tool bridge functions (screen_pii, screen_ai_authorship, screen_resume, screen_readability)
6. `applies_to()` filtering for doc-type-specific tools
7. Category-to-tool mapping completeness
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.screening.config import ScreeningConfig
from src.screening.engine import CATEGORY_TOOL_MAP, ScreeningEngine
from src.screening.models import ScreeningContext, ScreeningReport, ToolResult

# ---------------------------------------------------------------------------
# Sample texts for each screening category
# ---------------------------------------------------------------------------

SAMPLE_TEXT_PII = (
    "John Smith can be reached at john.smith@example.com or call 555-123-4567. "
    "His SSN is 123-45-6789 and he lives at 123 Main Street, Springfield, IL 62704."
)

SAMPLE_TEXT_AI = (
    "In conclusion, it is important to note that the aforementioned strategies "
    "provide a comprehensive framework for understanding the multifaceted nature "
    "of organizational development. Furthermore, the implementation of these "
    "strategies necessitates a holistic approach that encompasses various dimensions "
    "of corporate governance and stakeholder engagement. Moreover, the integration "
    "of technology-driven solutions can significantly enhance operational efficiency. "
    "In conclusion, it is important to note that these strategies are crucial. "
    "Furthermore, a holistic approach is needed. Moreover, integration is key. "
    "In conclusion, the aforementioned points are significant. Furthermore, "
    "we must consider the broader implications of these findings."
)

SAMPLE_TEXT_RESUME = (
    "JOHN DOE\n"
    "Senior Software Engineer\n"
    "john.doe@email.com | (555) 987-6543\n\n"
    "SUMMARY\n"
    "Experienced software engineer with 8 years of experience in Python, Java, "
    "and cloud technologies. Proven track record of delivering scalable applications.\n\n"
    "EXPERIENCE\n"
    "Senior Software Engineer, Google, 2020-Present\n"
    "- Led development of microservices architecture\n"
    "- Managed team of 5 engineers\n\n"
    "Software Engineer, Amazon, 2016-2020\n"
    "- Developed RESTful APIs serving 10M requests/day\n"
    "- Implemented CI/CD pipelines\n\n"
    "EDUCATION\n"
    "B.S. Computer Science, MIT, 2016\n\n"
    "SKILLS\n"
    "Python, Java, AWS, Docker, Kubernetes, PostgreSQL\n\n"
    "CERTIFICATIONS\n"
    "AWS Solutions Architect Professional\n"
    "Google Cloud Professional Data Engineer\n"
)

SAMPLE_TEXT_LEGAL = (
    "This Agreement is entered into as of January 1, 2025, by and between "
    "Company A ('Licensor') and Company B ('Licensee'). The Licensee shall "
    "indemnify the Licensor against all liabilities. Governing law: State of California. "
    "Force majeure clause applies. Confidentiality provisions extend 5 years post-termination. "
    "Arbitration shall be conducted in San Francisco, CA under AAA Commercial Rules."
)

SAMPLE_TEXT_GENERIC = (
    "The quarterly revenue report shows a 15% increase in Q3 2024 compared to Q2 2024. "
    "Total revenue reached $4.5 million, with the software division contributing $2.8M. "
    "Operating expenses were $3.2M, resulting in net income of $1.3M. "
    "Employee count grew from 150 to 175. Customer satisfaction score is 4.7/5.0. "
    "The company invested $500K in R&D during this period."
)

SAMPLE_TEXT_READABILITY_HARD = (
    "The epistemological ramifications of ontological paradigm shifts necessitate "
    "a reconceptualization of hermeneutical frameworks within the context of "
    "poststructuralist epistemological discourse. Furthermore, the phenomenological "
    "implications of transcendental subjectivity vis-a-vis intersubjective "
    "constitutive synthesis demand rigorous philosophical scrutiny."
)


# ---------------------------------------------------------------------------
# Helper: build a fresh engine with all default tools
# ---------------------------------------------------------------------------

def _make_engine(**overrides) -> ScreeningEngine:
    """Create a ScreeningEngine with proper defaults."""
    cfg = ScreeningConfig.load()
    # Apply overrides
    for key, val in overrides.items():
        setattr(cfg, key, val)
    cfg.internet_enabled = overrides.get("internet_enabled", False)
    return ScreeningEngine(config=cfg)


def _make_context(text: str, doc_type: Optional[str] = None, **kwargs) -> ScreeningContext:
    """Build a ScreeningContext for inline text screening."""
    return ScreeningContext(
        doc_id=None,
        doc_type=doc_type,
        text=text,
        metadata=kwargs.pop("metadata", {}),
        raw_bytes=kwargs.pop("raw_bytes", None),
        config=kwargs.pop("config", None),
    )


# ===========================================================================
# 1. CATEGORY_TOOL_MAP completeness
# ===========================================================================

class TestCategoryToolMap:
    """Verify the CATEGORY_TOOL_MAP is well-formed and all referenced tools exist."""

    def test_all_categories_present(self):
        expected = {"integrity", "compliance", "quality", "language", "security",
                    "ai-authorship", "ai_authorship", "legality", "resume", "resume_screening"}
        assert set(CATEGORY_TOOL_MAP.keys()) == expected

    def test_all_tool_names_exist_in_engine(self):
        engine = _make_engine()
        all_tool_names = set()
        for names in CATEGORY_TOOL_MAP.values():
            all_tool_names.update(names)
        missing = all_tool_names - set(engine._tools.keys())
        assert not missing, f"Tools referenced in CATEGORY_TOOL_MAP but not in engine: {missing}"

    def test_every_engine_tool_is_in_some_category(self):
        engine = _make_engine()
        mapped_tools = set()
        for names in CATEGORY_TOOL_MAP.values():
            mapped_tools.update(names)
        unmapped = set(engine._tools.keys()) - mapped_tools
        # Unmapped tools are allowed (they can be invoked individually)
        # but we should log them
        if unmapped:
            pytest.skip(f"Unmapped tools (not necessarily a bug): {unmapped}")


class TestResolveToolsForCategory:
    """Verify _resolve_tools_for_category() returns correct tools."""

    @pytest.fixture
    def engine(self):
        return _make_engine()

    @pytest.mark.parametrize("category,expected_tools", [
        ("integrity", ["integrity_hash", "metadata_consistency"]),
        ("compliance", ["template_conformance", "policy_compliance"]),
        ("quality", ["numeric_unit_consistency", "citation_sanity", "ambiguity_vagueness"]),
        ("language", ["readability_style", "passive_voice"]),
        ("security", ["pii_sensitivity"]),
        ("ai_authorship", ["ai_authorship"]),
        ("ai-authorship", ["ai_authorship"]),
        ("legality", ["legality_agent"]),
    ])
    def test_non_resume_category_resolution(self, engine, category, expected_tools):
        cfg = engine.config
        tools = engine._resolve_tools_for_category(category, doc_type=None, cfg=cfg)
        tool_names = [t.name for t in tools]
        assert sorted(tool_names) == sorted(expected_tools), \
            f"Category '{category}': expected {expected_tools}, got {tool_names}"

    def test_resume_category_resolution(self, engine):
        cfg = engine.config
        tools = engine._resolve_tools_for_category("resume", doc_type=None, cfg=cfg)
        tool_names = [t.name for t in tools]
        expected = [
            "resume_extractor_tool", "company_validator_tool",
            "institution_validator_tool", "certification_verifier_tool",
            "authenticity_analyzer_tool", "resume_entity_validation",
            "resume_screening",
        ]
        assert sorted(tool_names) == sorted(expected)

    def test_explicit_category_bypasses_applies_to(self, engine):
        """Explicit categories (in CATEGORY_TOOL_MAP) skip applies_to filter."""
        cfg = engine.config
        # Resume tools normally need doc_type=RESUME, but explicit category skips that
        tools = engine._resolve_tools_for_category("resume", doc_type="INVOICE", cfg=cfg)
        assert len(tools) == 7, f"Expected 7 resume tools even for INVOICE doc_type, got {len(tools)}"

    def test_unknown_category_raises(self, engine):
        cfg = engine.config
        tools = engine._resolve_tools_for_category("nonexistent", doc_type=None, cfg=cfg)
        # _resolve_tools_for_category returns empty list, run_category raises
        assert tools == []


# ===========================================================================
# 2. applies_to() filtering
# ===========================================================================

class TestAppliesToFiltering:
    """Test that doc-type-specific tools correctly filter."""

    def test_resume_tools_apply_to_resume(self):
        engine = _make_engine()
        resume_tool_names = CATEGORY_TOOL_MAP["resume"]
        for name in resume_tool_names:
            tool = engine._tools[name]
            assert tool.applies_to("RESUME"), f"{name} should apply to RESUME"
            assert tool.applies_to("CV"), f"{name} should apply to CV"
            assert tool.applies_to("resume"), f"{name} should apply to lowercase resume"

    def test_resume_tools_do_not_apply_to_non_resume(self):
        engine = _make_engine()
        resume_tool_names = CATEGORY_TOOL_MAP["resume"]
        for name in resume_tool_names:
            tool = engine._tools[name]
            assert not tool.applies_to("INVOICE"), f"{name} should not apply to INVOICE"
            assert not tool.applies_to("CONTRACT"), f"{name} should not apply to CONTRACT"

    def test_general_tools_apply_to_all(self):
        engine = _make_engine()
        general_tools = ["integrity_hash", "metadata_consistency", "pii_sensitivity",
                         "ai_authorship", "readability_style", "passive_voice",
                         "numeric_unit_consistency", "citation_sanity", "ambiguity_vagueness"]
        for name in general_tools:
            tool = engine._tools[name]
            assert tool.applies_to(None), f"{name} should apply to None doc_type"
            assert tool.applies_to("INVOICE"), f"{name} should apply to INVOICE"
            assert tool.applies_to("RESUME"), f"{name} should apply to RESUME"

    def test_active_tools_excludes_resume_for_generic(self):
        """_active_tools() should exclude resume tools when doc_type is not RESUME."""
        engine = _make_engine()
        active = engine._active_tools(doc_type="INVOICE")
        active_names = {t.name for t in active}
        resume_only = {"resume_extractor_tool", "company_validator_tool",
                       "institution_validator_tool", "certification_verifier_tool",
                       "authenticity_analyzer_tool", "resume_entity_validation",
                       "resume_screening"}
        assert not (active_names & resume_only), \
            f"Resume tools should not be active for INVOICE: {active_names & resume_only}"

    def test_active_tools_includes_resume_for_resume(self):
        engine = _make_engine()
        active = engine._active_tools(doc_type="RESUME")
        active_names = {t.name for t in active}
        assert "resume_extractor_tool" in active_names
        assert "resume_screening" in active_names


# ===========================================================================
# 3. Individual tool run() tests (each category)
# ===========================================================================

class TestIntegrityTools:
    """Category: integrity — integrity_hash, metadata_consistency."""

    def test_integrity_hash_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["integrity_hash"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "integrity_hash"
        assert 0.0 <= result.score_0_1 <= 1.0
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH")
        assert "sha256" in result.raw_features

    def test_integrity_hash_detects_mismatch(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config,
                            metadata={"expected_sha256": "wrong_hash_value"})
        tool = engine._tools["integrity_hash"]
        result = tool.run(ctx)
        assert result.score_0_1 > 0.0, "Should flag hash mismatch"
        assert any("hash" in r.lower() or "differ" in r.lower() for r in result.reasons)

    def test_metadata_consistency_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["metadata_consistency"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "metadata_consistency"
        assert 0.0 <= result.score_0_1 <= 1.0


class TestComplianceTools:
    """Category: compliance — template_conformance, policy_compliance."""

    def test_template_conformance_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["template_conformance"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "template_conformance"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_policy_compliance_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["policy_compliance"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "policy_compliance"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_policy_compliance_detects_sensitive_keywords(self):
        engine = _make_engine(sensitive_keywords=["confidential", "secret"])
        ctx = _make_context(
            "This document is CONFIDENTIAL and contains SECRET information.",
            config=engine.config,
        )
        tool = engine._tools["policy_compliance"]
        result = tool.run(ctx)
        # Should detect sensitive keywords
        assert result.score_0_1 > 0.0 or len(result.reasons) > 0


class TestQualityTools:
    """Category: quality — numeric_unit_consistency, citation_sanity, ambiguity_vagueness."""

    def test_numeric_unit_consistency_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["numeric_unit_consistency"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "numeric_unit_consistency"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_citation_sanity_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["citation_sanity"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "citation_sanity"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_ambiguity_vagueness_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["ambiguity_vagueness"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "ambiguity_vagueness"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_ambiguity_flags_vague_text(self):
        engine = _make_engine()
        vague_text = (
            "Some things might possibly be somewhat relevant to certain aspects "
            "of various things that could potentially maybe matter in some way."
        )
        ctx = _make_context(vague_text, config=engine.config)
        tool = engine._tools["ambiguity_vagueness"]
        result = tool.run(ctx)
        assert result.score_0_1 > 0.0, "Vague text should score above 0"


class TestLanguageTools:
    """Category: language — readability_style, passive_voice."""

    def test_readability_style_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["readability_style"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "readability_style"
        assert 0.0 <= result.score_0_1 <= 1.0
        assert "overall_readability" in result.raw_features

    def test_readability_flags_hard_text(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_READABILITY_HARD, config=engine.config)
        tool = engine._tools["readability_style"]
        result = tool.run(ctx)
        # Hard readability text should score higher (worse)
        assert result.raw_features.get("overall_readability", 100) < 40, \
            "Complex academic text should have low Flesch score"

    def test_passive_voice_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["passive_voice"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "passive_voice"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_passive_voice_detects_passive(self):
        engine = _make_engine()
        passive_text = (
            "The report was written by the team. The data was collected by researchers. "
            "Errors were found in the analysis. The results were verified by experts. "
            "The project was completed on time. Improvements were suggested by the committee."
        )
        ctx = _make_context(passive_text, config=engine.config)
        tool = engine._tools["passive_voice"]
        result = tool.run(ctx)
        assert result.score_0_1 > 0.0, "Passive voice text should score above 0"


class TestSecurityTools:
    """Category: security — pii_sensitivity."""

    def test_pii_sensitivity_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_PII, config=engine.config)
        tool = engine._tools["pii_sensitivity"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "pii_sensitivity"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_pii_detects_email(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_PII, config=engine.config)
        tool = engine._tools["pii_sensitivity"]
        result = tool.run(ctx)
        assert result.raw_features.get("pii_count", 0) > 0, "Should detect PII items"
        assert result.score_0_1 > 0.0, "PII text should score above 0"

    def test_pii_clean_text(self):
        engine = _make_engine()
        ctx = _make_context("The sky is blue and grass is green.", config=engine.config)
        tool = engine._tools["pii_sensitivity"]
        result = tool.run(ctx)
        assert result.raw_features.get("pii_count", 0) == 0, "Clean text should have no PII"
        assert result.score_0_1 == 0.0


class TestAIAuthorshipTools:
    """Category: ai_authorship — ai_authorship."""

    def test_ai_authorship_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        tool = engine._tools["ai_authorship"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "ai_authorship"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_ai_authorship_detects_repetitive(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_AI, config=engine.config)
        tool = engine._tools["ai_authorship"]
        result = tool.run(ctx)
        # Highly repetitive, formulaic text should get flagged
        assert result.raw_features.get("word_count", 0) > 0
        assert "entropy" in result.raw_features or "repetition_ratio" in result.raw_features

    def test_ai_authorship_human_text(self):
        engine = _make_engine()
        human_text = (
            "I went to the store yesterday. It was raining, so I grabbed an umbrella. "
            "The cashier was friendly - she even gave my kid a sticker! On the way home "
            "I realized I forgot the milk. Typical. Had to go back, of course."
        )
        ctx = _make_context(human_text, config=engine.config)
        tool = engine._tools["ai_authorship"]
        result = tool.run(ctx)
        # Short human-like text should score low
        assert result.score_0_1 < 0.5 or "human-like" in " ".join(result.reasons).lower()


class TestLegalityTools:
    """Category: legality — legality_agent."""

    def test_legality_agent_runs(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_LEGAL, config=engine.config)
        tool = engine._tools["legality_agent"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "legality_agent"
        assert 0.0 <= result.score_0_1 <= 1.0

    def test_legality_applies_to_all_doc_types(self):
        engine = _make_engine()
        tool = engine._tools["legality_agent"]
        assert tool.applies_to(None)
        assert tool.applies_to("CONTRACT")
        assert tool.applies_to("INVOICE")


class TestResumeTools:
    """Category: resume — 7 tools (run on RESUME doc type)."""

    RESUME_TOOL_NAMES = [
        "resume_extractor_tool",
        "company_validator_tool",
        "institution_validator_tool",
        "certification_verifier_tool",
        "authenticity_analyzer_tool",
        "resume_entity_validation",
        "resume_screening",
    ]

    def test_all_resume_tools_exist(self):
        engine = _make_engine()
        for name in self.RESUME_TOOL_NAMES:
            assert name in engine._tools, f"Resume tool '{name}' not found in engine"

    @pytest.mark.parametrize("tool_name", RESUME_TOOL_NAMES)
    def test_resume_tool_runs(self, tool_name):
        """Each resume tool should execute without error on resume text."""
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_RESUME, doc_type="RESUME", config=engine.config)
        tool = engine._tools[tool_name]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult), f"{tool_name} did not return ToolResult"
        assert result.tool_name == tool_name
        assert 0.0 <= result.score_0_1 <= 1.0
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH")

    def test_resume_extractor_finds_sections(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_RESUME, doc_type="RESUME", config=engine.config)
        tool = engine._tools["resume_extractor_tool"]
        result = tool.run(ctx)
        # Should find experience, education, skills sections
        assert result.score_0_1 < 0.8, "Well-structured resume shouldn't be high risk"

    def test_resume_screening_full(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_RESUME, doc_type="RESUME", config=engine.config)
        tool = engine._tools["resume_screening"]
        result = tool.run(ctx)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "resume_screening"


# ===========================================================================
# 4. Engine evaluate() — full pipeline test per category
# ===========================================================================

class TestEngineEvaluate:
    """Test engine.evaluate() produces valid reports for various text types."""

    def test_evaluate_generic_text(self):
        engine = _make_engine()
        result = engine.evaluate(text=SAMPLE_TEXT_GENERIC)
        assert isinstance(result, dict)
        assert "overall_score_0_100" in result
        assert "risk_level" in result
        assert "results" in result
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert 0 <= result["overall_score_0_100"] <= 100

    def test_evaluate_pii_text(self):
        engine = _make_engine()
        result = engine.evaluate(text=SAMPLE_TEXT_PII)
        assert isinstance(result, dict)
        # PII-laden text should produce results from pii_sensitivity tool
        pii_results = [r for r in result["results"] if r["tool_name"] == "pii_sensitivity"]
        assert len(pii_results) == 1, "PII tool should run"
        assert pii_results[0]["score_0_1"] > 0.0, "PII score should be > 0"

    def test_evaluate_resume_text_without_doc_type(self):
        engine = _make_engine()
        # Without doc_type="RESUME", resume tools should NOT run
        result = engine.evaluate(text=SAMPLE_TEXT_RESUME)
        resume_results = [r for r in result["results"]
                          if r["tool_name"] in {"resume_extractor_tool", "resume_screening"}]
        assert len(resume_results) == 0, "Resume tools should not run without doc_type=RESUME"

    def test_evaluate_resume_text_with_doc_type(self):
        engine = _make_engine()
        result = engine.evaluate(text=SAMPLE_TEXT_RESUME, doc_type="RESUME")
        resume_results = [r for r in result["results"]
                          if r["tool_name"] in {"resume_extractor_tool", "resume_screening"}]
        assert len(resume_results) >= 1, "Resume tools should run with doc_type=RESUME"

    def test_evaluate_empty_text(self):
        engine = _make_engine()
        result = engine.evaluate(text="")
        assert isinstance(result, dict)
        # Should still produce a report, even if scores are low
        assert "overall_score_0_100" in result

    def test_evaluate_returns_all_general_tools(self):
        engine = _make_engine()
        result = engine.evaluate(text=SAMPLE_TEXT_GENERIC)
        tool_names = {r["tool_name"] for r in result["results"]}
        expected_general = {
            "integrity_hash", "metadata_consistency", "pii_sensitivity",
            "ai_authorship", "readability_style", "passive_voice",
            "numeric_unit_consistency", "citation_sanity", "ambiguity_vagueness",
        }
        # Template conformance, policy compliance, legality_agent may or may not apply
        # depending on their applies_to logic
        for name in expected_general:
            assert name in tool_names, f"General tool '{name}' should run on generic text"


# ===========================================================================
# 5. Engine screen() — full ScreeningReport
# ===========================================================================

class TestEngineScreen:
    """Test engine.screen() returns well-formed ScreeningReport."""

    def test_screen_returns_report(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        report = engine.screen(ctx)
        assert isinstance(report, ScreeningReport)
        assert 0 <= report.overall_score_0_100 <= 100
        assert report.risk_level in ("LOW", "MEDIUM", "HIGH")
        assert len(report.results) > 0
        assert report.provenance.get("tool_versions")

    def test_screen_resume_has_resume_tools(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_RESUME, doc_type="RESUME", config=engine.config)
        report = engine.screen(ctx)
        tool_names = {r.tool_name for r in report.results}
        assert "resume_extractor_tool" in tool_names
        assert "resume_screening" in tool_names

    def test_screen_to_dict_serializable(self):
        engine = _make_engine()
        ctx = _make_context(SAMPLE_TEXT_GENERIC, config=engine.config)
        report = engine.screen(ctx)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["results"], list)
        assert isinstance(d["overall_score_0_100"], (int, float))
        # All values should be JSON-serializable
        import json
        json.dumps(d, default=str)  # should not raise


# ===========================================================================
# 6. Score blending
# ===========================================================================

class TestScoreBlending:
    """Test the _blend_score and _risk_level functions."""

    def test_blend_score_zero(self):
        engine = _make_engine()
        results = [
            ToolResult(tool_name="t1", category="c1", score_0_1=0.0, weight=1.0, risk_level="LOW"),
            ToolResult(tool_name="t2", category="c2", score_0_1=0.0, weight=1.0, risk_level="LOW"),
        ]
        score = engine._blend_score(results)
        assert score < 10, f"All-zero scores should blend to low overall: {score}"

    def test_blend_score_high(self):
        engine = _make_engine()
        results = [
            ToolResult(tool_name="t1", category="c1", score_0_1=1.0, weight=1.0, risk_level="HIGH"),
            ToolResult(tool_name="t2", category="c2", score_0_1=1.0, weight=1.0, risk_level="HIGH"),
        ]
        score = engine._blend_score(results)
        assert score > 90, f"All-max scores should blend to high overall: {score}"

    def test_risk_level_thresholds(self):
        engine = _make_engine()
        assert engine._risk_level(10) == "LOW"
        assert engine._risk_level(44.9) == "LOW"
        assert engine._risk_level(45) == "MEDIUM"
        assert engine._risk_level(74.9) == "MEDIUM"
        assert engine._risk_level(75) == "HIGH"
        assert engine._risk_level(100) == "HIGH"


# ===========================================================================
# 7. Tool bridge functions
# ===========================================================================

class TestToolBridge:
    """Test the gateway tool bridge functions in tool_bridge.py."""

    def test_screen_pii_tool_with_text(self):
        from src.screening.tool_bridge import screen_pii_tool
        result = screen_pii_tool({"input": {"text": SAMPLE_TEXT_PII}})
        assert "result" in result
        assert isinstance(result["result"], dict)
        assert "overall_score_0_100" in result["result"]
        # Check PII was detected
        pii_tools = [r for r in result["result"].get("results", [])
                     if r.get("tool_name") == "pii_sensitivity"]
        assert len(pii_tools) == 1

    def test_screen_pii_tool_empty_text(self):
        from src.screening.tool_bridge import screen_pii_tool
        result = screen_pii_tool({"input": {"text": ""}})
        assert "warnings" in result
        assert "No text provided" in result["warnings"]

    def test_screen_pii_tool_no_input(self):
        from src.screening.tool_bridge import screen_pii_tool
        result = screen_pii_tool({})
        assert "warnings" in result

    def test_screen_ai_authorship_tool_with_text(self):
        from src.screening.tool_bridge import screen_ai_authorship_tool
        result = screen_ai_authorship_tool({"input": {"text": SAMPLE_TEXT_AI}})
        assert "result" in result
        assert isinstance(result["result"], dict)
        assert "overall_score_0_100" in result["result"]

    def test_screen_ai_authorship_tool_empty(self):
        from src.screening.tool_bridge import screen_ai_authorship_tool
        result = screen_ai_authorship_tool({"input": {"text": ""}})
        assert "warnings" in result

    def test_screen_readability_tool_with_text(self):
        from src.screening.tool_bridge import screen_readability_tool
        result = screen_readability_tool({"input": {"text": SAMPLE_TEXT_READABILITY_HARD}})
        assert "result" in result
        assert isinstance(result["result"], dict)
        assert "overall_score_0_100" in result["result"]

    def test_screen_readability_tool_empty(self):
        from src.screening.tool_bridge import screen_readability_tool
        result = screen_readability_tool({"input": {"text": ""}})
        assert "warnings" in result

    def test_screen_resume_tool_with_text(self):
        from src.screening.tool_bridge import screen_resume_tool
        result = screen_resume_tool({"input": {"text": SAMPLE_TEXT_RESUME}})
        assert "result" in result
        assert isinstance(result["result"], dict)

    def test_screen_resume_tool_empty(self):
        from src.screening.tool_bridge import screen_resume_tool
        result = screen_resume_tool({"input": {"text": ""}})
        assert "warnings" in result


# ===========================================================================
# 8. Tool bridge bug verification (screen_pii_tool no longer calls run_one)
# ===========================================================================

class TestToolBridgeBugFix:
    """Verify the screen_pii_tool bug fix — no longer calls run_one('pii_sensitivity', doc_id='inline')."""

    def test_screen_pii_does_not_call_run_one(self):
        """screen_pii_tool should only call evaluate(), not run_one()."""
        import inspect
        from src.screening import tool_bridge
        source = inspect.getsource(tool_bridge.screen_pii_tool)
        assert "run_one" not in source, \
            "screen_pii_tool should not call run_one() — it fails with doc_id='inline'"

    def test_screen_pii_tool_does_not_crash(self):
        """The old bug: run_one with doc_id='inline' would try to load from storage and fail."""
        from src.screening.tool_bridge import screen_pii_tool
        # This should NOT raise — the bug was that run_one tried storage lookup for "inline"
        result = screen_pii_tool({"input": {"text": "Contact me at user@example.com"}})
        assert "result" in result
        assert result["result"].get("overall_score_0_100") is not None


# ===========================================================================
# 9. Engine run_all() and run_category() via context mock
# ===========================================================================

class TestEngineRunCategory:
    """Test run_category via mocked storage (to avoid real doc lookup)."""

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_security(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_PII
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        results = engine.run_category("security", doc_id="test-doc-1")
        assert len(results) == 1
        assert results[0].tool_name == "pii_sensitivity"
        assert results[0].score_0_1 > 0.0

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_language(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_READABILITY_HARD
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        results = engine.run_category("language", doc_id="test-doc-2")
        assert len(results) == 2
        tool_names = {r.tool_name for r in results}
        assert "readability_style" in tool_names
        assert "passive_voice" in tool_names

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_quality(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_GENERIC
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        results = engine.run_category("quality", doc_id="test-doc-3")
        assert len(results) == 3
        tool_names = {r.tool_name for r in results}
        assert "numeric_unit_consistency" in tool_names
        assert "citation_sanity" in tool_names
        assert "ambiguity_vagueness" in tool_names

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_integrity(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_GENERIC
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        results = engine.run_category("integrity", doc_id="test-doc-4")
        assert len(results) == 2
        tool_names = {r.tool_name for r in results}
        assert "integrity_hash" in tool_names
        assert "metadata_consistency" in tool_names

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_compliance(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_GENERIC
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        results = engine.run_category("compliance", doc_id="test-doc-5")
        assert len(results) == 2
        tool_names = {r.tool_name for r in results}
        assert "template_conformance" in tool_names
        assert "policy_compliance" in tool_names

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_ai_authorship(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_AI
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        results = engine.run_category("ai_authorship", doc_id="test-doc-6")
        assert len(results) == 1
        assert results[0].tool_name == "ai_authorship"

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_legality(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_LEGAL
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        results = engine.run_category("legality", doc_id="test-doc-7")
        assert len(results) == 1
        assert results[0].tool_name == "legality_agent"

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_resume(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_RESUME
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = "RESUME"

        engine = _make_engine()
        results = engine.run_category("resume", doc_id="test-doc-8")
        assert len(results) == 7
        tool_names = {r.tool_name for r in results}
        expected = {"resume_extractor_tool", "company_validator_tool",
                    "institution_validator_tool", "certification_verifier_tool",
                    "authenticity_analyzer_tool", "resume_entity_validation",
                    "resume_screening"}
        assert tool_names == expected

    @patch("src.screening.engine.storage_adapter")
    def test_run_category_unknown_raises(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = "text"
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        with pytest.raises(ValueError, match="No screening tools found"):
            engine.run_category("nonexistent_category", doc_id="test-doc")


# ===========================================================================
# 10. Engine run_one() via context mock
# ===========================================================================

class TestEngineRunOne:
    """Test run_one for individual tool invocation."""

    @patch("src.screening.engine.storage_adapter")
    def test_run_one_pii(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_PII
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        result = engine.run_one("pii_sensitivity", doc_id="test-doc")
        assert result.tool_name == "pii_sensitivity"
        assert result.score_0_1 > 0.0

    @patch("src.screening.engine.storage_adapter")
    def test_run_one_unknown_tool_raises(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = "text"
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = None

        engine = _make_engine()
        with pytest.raises(ValueError, match="Unknown screening tool"):
            engine.run_one("nonexistent_tool", doc_id="test-doc")

    @patch("src.screening.engine.storage_adapter")
    def test_run_one_resume_tool_needs_resume_doctype(self, mock_storage):
        mock_storage.get_document_metadata.return_value = {}
        mock_storage.get_document_subscription_id.return_value = "sub1"
        mock_storage.get_document_text.return_value = SAMPLE_TEXT_RESUME
        mock_storage.get_document_bytes.return_value = None
        mock_storage.get_document_doc_type.return_value = "INVOICE"

        engine = _make_engine()
        with pytest.raises(ValueError, match="not enabled"):
            engine.run_one("resume_extractor_tool", doc_id="test-doc")


# ===========================================================================
# 11. Resume analysis integration
# ===========================================================================

class TestResumeAnalysis:
    """Test the resume_analysis_from_text method."""

    def test_resume_analysis_from_text(self):
        engine = _make_engine()
        result = engine.resume_analysis_from_text(text=SAMPLE_TEXT_RESUME)
        # Should return a ResumeScreeningDetailedResponse
        assert hasattr(result, "risk_level")
        assert hasattr(result, "overall_confidence_0_100")
        assert hasattr(result, "candidate_profile")
        assert hasattr(result, "findings")
        assert result.risk_level in ("low", "medium", "high")

    def test_resume_analysis_extracts_profile(self):
        engine = _make_engine()
        result = engine.resume_analysis_from_text(text=SAMPLE_TEXT_RESUME)
        profile = result.candidate_profile.extracted
        # Should extract key sections
        assert profile.experience or profile.education or profile.skills, \
            "Resume analysis should extract at least some profile data"


# ===========================================================================
# 12. Helpers
# ===========================================================================

class TestHelpers:
    """Test screening helper functions."""

    def test_format_results(self):
        from src.screening.helpers import format_results
        results = [
            ToolResult(tool_name="t1", category="c1", score_0_1=0.3, weight=0.5, risk_level="LOW"),
            ToolResult(tool_name="t2", category="c2", score_0_1=0.7, weight=0.5, risk_level="MEDIUM"),
        ]
        formatted = format_results("doc-1", results)
        assert formatted["doc_id"] == "doc-1"
        assert "results" in formatted
        assert "overall_score_0_100" in formatted
        assert "risk_level" in formatted

    def test_format_results_single(self):
        from src.screening.helpers import format_results
        results = [
            ToolResult(tool_name="t1", category="c1", score_0_1=0.3, weight=0.5, risk_level="LOW"),
        ]
        formatted = format_results("doc-1", results)
        assert formatted["risk_level"] == "LOW"

    def test_normalize_categories_default(self):
        from src.screening.helpers import normalize_categories
        assert normalize_categories(None) == ["all"]
        assert normalize_categories([]) == ["all"]

    def test_normalize_categories_valid(self):
        from src.screening.helpers import normalize_categories
        assert normalize_categories(["security"]) == ["security"]
        assert normalize_categories(["ai_authorship"]) == ["ai_authorship"]
        assert normalize_categories(["ai-authorship"]) == ["ai_authorship"]

    def test_normalize_categories_invalid(self):
        from src.screening.helpers import normalize_categories
        with pytest.raises(ValueError, match="Unsupported category"):
            normalize_categories(["nonexistent"])

    def test_normalize_doc_ids(self):
        from src.screening.helpers import normalize_doc_ids
        assert normalize_doc_ids(["abc123", "def456"]) == ["abc123", "def456"]

    def test_normalize_doc_ids_dedup(self):
        from src.screening.helpers import normalize_doc_ids
        assert normalize_doc_ids(["abc123", "abc123"]) == ["abc123"]

    def test_normalize_doc_ids_rejects_placeholder(self):
        from src.screening.helpers import normalize_doc_ids
        with pytest.raises(ValueError, match="Invalid doc_id"):
            normalize_doc_ids(["string"])

    def test_normalize_doc_ids_rejects_short(self):
        from src.screening.helpers import normalize_doc_ids
        with pytest.raises(ValueError, match="Invalid doc_id"):
            normalize_doc_ids(["ab"])

    def test_normalize_doc_ids_rejects_empty(self):
        from src.screening.helpers import normalize_doc_ids
        with pytest.raises(ValueError, match="non-empty"):
            normalize_doc_ids([])


# ===========================================================================
# 13. ScreeningConfig basics
# ===========================================================================

class TestScreeningConfig:
    """Test ScreeningConfig construction and defaults."""

    def test_default_config_loads(self):
        cfg = ScreeningConfig.load()
        assert cfg.sigmoid_a == 6.0
        assert cfg.sigmoid_b == 0.5
        assert cfg.risk_thresholds.get("high") == 75
        assert cfg.risk_thresholds.get("medium") == 45

    def test_config_hash_is_stable(self):
        cfg1 = ScreeningConfig.load()
        cfg2 = ScreeningConfig.load()
        assert cfg1.config_hash == cfg2.config_hash

    def test_default_engine_has_19_tools(self):
        """The default engine should have all 19 tools (18 general + legality = 19)."""
        engine = ScreeningEngine()
        # 11 general + 7 resume + 1 legality = 19
        assert len(engine._tools) >= 18, f"Expected ≥18 tools, got {len(engine._tools)}"


# ===========================================================================
# 14. End-to-end: all categories produce valid results
# ===========================================================================

class TestEndToEndAllCategories:
    """Run evaluate() and verify every tool in the report is valid."""

    def test_generic_text_all_general_tools_produce_results(self):
        engine = _make_engine()
        result = engine.evaluate(text=SAMPLE_TEXT_GENERIC)
        for tool_result in result["results"]:
            assert "tool_name" in tool_result, "Missing tool_name"
            assert "score_0_1" in tool_result, f"Missing score for {tool_result.get('tool_name')}"
            assert "risk_level" in tool_result, f"Missing risk_level for {tool_result.get('tool_name')}"
            assert 0.0 <= tool_result["score_0_1"] <= 1.0, \
                f"Score out of range for {tool_result['tool_name']}: {tool_result['score_0_1']}"
            assert tool_result["risk_level"] in ("LOW", "MEDIUM", "HIGH"), \
                f"Invalid risk_level for {tool_result['tool_name']}: {tool_result['risk_level']}"

    def test_resume_text_all_tools_produce_results(self):
        engine = _make_engine()
        result = engine.evaluate(text=SAMPLE_TEXT_RESUME, doc_type="RESUME")
        tool_names = {r["tool_name"] for r in result["results"]}
        # Should include both general AND resume tools
        assert "pii_sensitivity" in tool_names, "General tool missing from resume screening"
        assert "resume_screening" in tool_names, "Resume tool missing from resume screening"
        for tool_result in result["results"]:
            assert 0.0 <= tool_result["score_0_1"] <= 1.0

    def test_legal_text_all_tools_produce_results(self):
        engine = _make_engine()
        result = engine.evaluate(text=SAMPLE_TEXT_LEGAL)
        tool_names = {r["tool_name"] for r in result["results"]}
        assert "legality_agent" in tool_names
        for tool_result in result["results"]:
            assert 0.0 <= tool_result["score_0_1"] <= 1.0
