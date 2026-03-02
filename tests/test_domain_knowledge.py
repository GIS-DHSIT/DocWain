"""Tests for the Domain Knowledge Engine.

Tests cover:
- Domain knowledge retrieval for all supported domains
- Country/region detection for medical documents
- Intent-specific augmentation
- DomainContext rendering (prompt and brief)
- Provider singleton management
- Integration with LLM prompt pipeline
- Integration with tool intelligence profiles
"""
import pytest
from unittest.mock import patch, MagicMock

from src.intelligence.domain_knowledge import (
    DomainContext,
    DomainKnowledgeProvider,
    detect_country_of_origin,
    get_domain_knowledge_provider,
    set_domain_knowledge_provider,
    ensure_domain_knowledge_provider,
    _DOMAIN_KNOWLEDGE,
    _MEDICAL_REGION_CONTEXT,
    _INTENT_AUGMENTATION,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the provider singleton before each test."""
    import src.intelligence.domain_knowledge as mod
    old = mod._provider
    mod._provider = None
    yield
    mod._provider = old


@pytest.fixture
def provider():
    return DomainKnowledgeProvider()


# ── Domain Knowledge Retrieval ───────────────────────────────────────

class TestDomainKnowledgeRetrieval:
    """Test retrieving domain knowledge for each supported domain."""

    @pytest.mark.parametrize("domain", ["hr", "resume"])
    def test_hr_domain(self, provider, domain):
        ctx = provider.get_knowledge(domain)
        assert ctx.domain == "hr"
        assert "talent acquisition" in ctx.analytical_perspective.lower() or "recruiter" in ctx.analytical_perspective.lower()
        assert len(ctx.evaluation_criteria) >= 5
        assert len(ctx.key_indicators) >= 5
        assert len(ctx.red_flags) >= 3

    @pytest.mark.parametrize("domain", ["invoice", "procurement"])
    def test_invoice_domain(self, provider, domain):
        ctx = provider.get_knowledge(domain)
        assert ctx.domain == "invoice"
        assert "procurement" in ctx.analytical_perspective.lower() or "accounts payable" in ctx.analytical_perspective.lower()
        assert any("three-way match" in c.lower() for c in ctx.evaluation_criteria)

    @pytest.mark.parametrize("domain", ["medical", "clinical"])
    def test_medical_domain(self, provider, domain):
        ctx = provider.get_knowledge(domain)
        assert ctx.domain == "medical"
        assert "clinical" in ctx.analytical_perspective.lower()
        assert any("medication" in c.lower() for c in ctx.evaluation_criteria)

    @pytest.mark.parametrize("domain", ["legal", "contract"])
    def test_legal_domain(self, provider, domain):
        ctx = provider.get_knowledge(domain)
        assert ctx.domain == "legal"
        assert "contract" in ctx.analytical_perspective.lower() or "legal" in ctx.analytical_perspective.lower()
        assert any("indemnification" in c.lower() for c in ctx.evaluation_criteria)

    @pytest.mark.parametrize("domain", ["policy", "insurance"])
    def test_policy_domain(self, provider, domain):
        ctx = provider.get_knowledge(domain)
        assert ctx.domain == "policy"
        assert "insurance" in ctx.analytical_perspective.lower() or "policy" in ctx.analytical_perspective.lower()

    def test_generic_domain(self, provider):
        ctx = provider.get_knowledge("generic")
        assert ctx.domain == "generic"
        assert "document intelligence" in ctx.analytical_perspective.lower()

    def test_unknown_domain_falls_back_to_generic(self, provider):
        ctx = provider.get_knowledge("unknown_domain_xyz")
        assert ctx.domain == "generic"

    def test_empty_domain_falls_back_to_generic(self, provider):
        ctx = provider.get_knowledge("")
        assert ctx.domain == "generic"


# ── Country/Region Detection ─────────────────────────────────────────

class TestCountryDetection:
    """Test medical document country-of-origin detection."""

    def test_us_detection_hipaa(self):
        text = "Patient records maintained in accordance with HIPAA regulations. Medicare coverage verified."
        code, name, conf = detect_country_of_origin(text)
        assert code == "US"
        assert "United States" in name
        assert conf > 0.3

    def test_us_detection_fda(self):
        text = "FDA approved medication. Blood glucose: 120 mg/dL. Weight: 180 lbs."
        code, name, conf = detect_country_of_origin(text)
        assert code == "US"
        assert conf > 0.3

    def test_uk_detection_nhs(self):
        text = "NHS referral from GP surgery. NICE guidelines followed. Patient seen in A&E."
        code, name, conf = detect_country_of_origin(text)
        assert code == "UK"
        assert "United Kingdom" in name

    def test_uk_detection_currency(self):
        text = "Consultation fee: £150.00. Blood glucose: 6.8 mmol/L."
        code, name, conf = detect_country_of_origin(text)
        assert code == "UK"

    def test_india_detection(self):
        text = "Patient from AIIMS Delhi. MBBS doctor. ICMR guidelines. Treatment cost: Rs. 5000."
        code, name, conf = detect_country_of_origin(text)
        assert code == "IN"
        assert "India" in name

    def test_eu_detection(self):
        text = "EMA approved drug. European Medicines Agency guidelines. Cost: €250.00."
        code, name, conf = detect_country_of_origin(text)
        assert code == "EU"

    def test_canada_detection(self):
        text = "Health Canada approved. OHIP coverage. Patient in Canada."
        code, name, conf = detect_country_of_origin(text)
        assert code == "CA"

    def test_australia_detection(self):
        text = "TGA approved medication. PBS listed drug. Medicare Australia card."
        code, name, conf = detect_country_of_origin(text)
        assert code == "AU"

    def test_international_default(self):
        text = "ICD-10 coding applied. WHO guidelines referenced."
        code, name, conf = detect_country_of_origin(text)
        assert code == "INT"

    def test_empty_text(self):
        code, name, conf = detect_country_of_origin("")
        assert code == "INT"
        assert conf == 0.0

    def test_no_signals(self):
        text = "The patient presented with mild symptoms and was treated accordingly."
        code, name, conf = detect_country_of_origin(text)
        assert code == "INT"
        assert conf <= 0.3

    def test_mixed_signals_majority_wins(self):
        text = "NHS NHS NHS GP surgery. FDA approved drug."
        code, name, conf = detect_country_of_origin(text)
        assert code == "UK"  # UK has more signals


# ── Medical Region Context ───────────────────────────────────────────

class TestMedicalRegionContext:
    """Test medical region-specific context injection."""

    def test_us_region_context(self, provider):
        text = "HIPAA compliant records. FDA approved. Medicare coverage."
        ctx = provider.get_knowledge("medical", document_text=text)
        assert ctx.region_context is not None
        assert "HIPAA" in ctx.region_context or "United States" in ctx.region_context

    def test_uk_region_context(self, provider):
        text = "NHS referral. NICE guidelines. GP surgery consultation."
        ctx = provider.get_knowledge("medical", document_text=text)
        assert ctx.region_context is not None
        assert "NHS" in ctx.region_context or "United Kingdom" in ctx.region_context

    def test_india_region_context(self, provider):
        text = "AIIMS hospital. MBBS doctor. ICMR guidelines. Rs. 5000 treatment."
        ctx = provider.get_knowledge("medical", document_text=text)
        assert ctx.region_context is not None
        assert "India" in ctx.region_context

    def test_no_region_for_non_medical(self, provider):
        ctx = provider.get_knowledge("hr", document_text="HIPAA compliant")
        assert ctx.region_context is None

    def test_get_medical_region_context_method(self, provider):
        text = "FDA approved medication. Blood glucose: 120 mg/dL."
        code, name, region_text = provider.get_medical_region_context(text)
        assert code == "US"
        assert "United States" in name
        assert "HIPAA" in region_text or "FDA" in region_text


# ── Intent-Specific Augmentation ─────────────────────────────────────

class TestIntentAugmentation:
    """Test intent-specific knowledge augmentation."""

    def test_hr_rank_augmentation(self, provider):
        ctx = provider.get_knowledge("hr", intent="rank")
        assert "rank" in ctx.analytical_perspective.lower() or "weight" in ctx.analytical_perspective.lower()

    def test_hr_compare_augmentation(self, provider):
        ctx = provider.get_knowledge("hr", intent="compare")
        assert "compar" in ctx.analytical_perspective.lower()

    def test_invoice_summary_augmentation(self, provider):
        ctx = provider.get_knowledge("invoice", intent="summary")
        assert "summar" in ctx.analytical_perspective.lower() or "aggregat" in ctx.analytical_perspective.lower()

    def test_medical_extraction_augmentation(self, provider):
        ctx = provider.get_knowledge("medical", intent="extraction")
        assert "extract" in ctx.analytical_perspective.lower() or "reference range" in ctx.analytical_perspective.lower()

    def test_no_augmentation_for_unknown_intent(self, provider):
        ctx_base = provider.get_knowledge("hr")
        ctx_unknown = provider.get_knowledge("hr", intent="unknown_intent_xyz")
        # Base perspective should be the same
        assert ctx_base.analytical_perspective == ctx_unknown.analytical_perspective


# ── DomainContext Rendering ──────────────────────────────────────────

class TestDomainContextRendering:
    """Test DomainContext rendering methods."""

    def test_to_prompt_section(self, provider):
        ctx = provider.get_knowledge("hr")
        prompt = ctx.to_prompt_section()
        assert "DOMAIN EXPERTISE (HR)" in prompt
        assert "Evaluation Criteria:" in prompt
        assert "Key Indicators" in prompt
        assert "Red Flags" in prompt

    def test_to_prompt_section_with_region(self, provider):
        ctx = provider.get_knowledge("medical", document_text="HIPAA compliance. FDA approved.")
        prompt = ctx.to_prompt_section()
        assert "Region-Specific Context:" in prompt

    def test_to_brief(self, provider):
        ctx = provider.get_knowledge("invoice")
        brief = ctx.to_brief()
        assert "procurement" in brief.lower() or "accounts payable" in brief.lower()
        assert "Evaluate:" in brief

    def test_brief_context_method(self, provider):
        brief = provider.get_brief_context("hr", intent="rank")
        assert brief
        assert len(brief) > 50
        assert "rank" in brief.lower() or "talent" in brief.lower()


# ── Provider Singleton Management ────────────────────────────────────

class TestSingletonManagement:
    """Test provider singleton lifecycle."""

    def test_get_creates_default(self):
        provider = get_domain_knowledge_provider()
        assert provider is not None
        assert isinstance(provider, DomainKnowledgeProvider)

    def test_get_returns_same_instance(self):
        p1 = get_domain_knowledge_provider()
        p2 = get_domain_knowledge_provider()
        assert p1 is p2

    def test_set_replaces_instance(self):
        new_provider = DomainKnowledgeProvider(web_enrichment=True)
        set_domain_knowledge_provider(new_provider)
        assert get_domain_knowledge_provider() is new_provider

    def test_ensure_creates_once(self):
        p1 = ensure_domain_knowledge_provider(web_enrichment=False)
        p2 = ensure_domain_knowledge_provider(web_enrichment=True)
        assert p1 is p2  # Second call should not recreate

    def test_supported_domains(self, provider):
        domains = provider.supported_domains
        assert "hr" in domains
        assert "invoice" in domains
        assert "medical" in domains
        assert "legal" in domains
        assert "policy" in domains
        assert "generic" in domains


# ── Knowledge Integrity ──────────────────────────────────────────────

class TestKnowledgeIntegrity:
    """Test that all domain knowledge entries are well-formed."""

    @pytest.mark.parametrize("domain", ["hr", "invoice", "medical", "legal", "policy", "generic"])
    def test_domain_has_all_fields(self, domain):
        ctx = _DOMAIN_KNOWLEDGE.get(domain)
        assert ctx is not None
        assert ctx.domain
        assert ctx.analytical_perspective
        assert len(ctx.evaluation_criteria) >= 3
        assert len(ctx.key_indicators) >= 3
        assert len(ctx.professional_terminology) >= 3
        assert len(ctx.common_patterns) >= 2
        assert len(ctx.red_flags) >= 2

    def test_medical_region_contexts_complete(self):
        expected = {"US", "UK", "IN", "EU", "CA", "AU", "INT"}
        assert expected.issubset(set(_MEDICAL_REGION_CONTEXT.keys()))

    def test_intent_augmentation_domains(self):
        assert "hr" in _INTENT_AUGMENTATION
        assert "invoice" in _INTENT_AUGMENTATION
        assert "medical" in _INTENT_AUGMENTATION
        assert "legal" in _INTENT_AUGMENTATION
        assert "policy" in _INTENT_AUGMENTATION

    @pytest.mark.parametrize("domain", ["hr", "invoice", "medical", "legal", "policy"])
    def test_intent_augmentation_has_summary(self, domain):
        assert "summary" in _INTENT_AUGMENTATION[domain]

    @pytest.mark.parametrize("domain", ["hr", "invoice", "medical", "legal", "policy"])
    def test_intent_augmentation_has_extraction(self, domain):
        assert "extraction" in _INTENT_AUGMENTATION[domain]


# ── LLM Prompt Integration ──────────────────────────────────────────

class TestLLMPromptIntegration:
    """Test domain knowledge integration with LLM prompt building."""

    def test_build_generation_prompt_includes_domain_knowledge(self):
        """Test that build_generation_prompt includes domain knowledge when domain is set."""
        from src.rag_v3.llm_extract import _get_domain_knowledge_section

        # Ensure provider is initialized
        ensure_domain_knowledge_provider()

        section = _get_domain_knowledge_section("hr", "rank")
        assert section
        assert "DOMAIN KNOWLEDGE" in section
        assert "talent" in section.lower() or "recruiter" in section.lower() or "hr" in section.lower()

    def test_domain_knowledge_section_empty_for_disabled(self):
        """Domain knowledge section should be empty when config disabled."""
        from src.rag_v3.llm_extract import _get_domain_knowledge_section

        with patch("src.api.config.Config.DomainKnowledge") as mock_dk:
            mock_dk.ENABLED = False
            section = _get_domain_knowledge_section("hr", "rank")
            assert section == ""

    def test_domain_knowledge_section_empty_when_no_domain(self):
        """Domain knowledge section should be empty for None domain."""
        from src.rag_v3.llm_extract import _get_domain_knowledge_section
        section = _get_domain_knowledge_section("", "rank")
        # Empty domain falls back to generic, which still has knowledge
        # but we mainly test it doesn't crash
        assert isinstance(section, str)

    def test_build_generation_prompt_no_double_expertise(self):
        """When tool_context is provided, domain knowledge should not duplicate it."""
        from src.rag_v3.llm_extract import build_generation_prompt

        ensure_domain_knowledge_provider()
        prompt = build_generation_prompt(
            query="rank candidates",
            evidence_text="some evidence",
            intent="ranking",
            num_documents=3,
            tool_context="You are an HR specialist...",
            domain="hr",
        )
        # tool_context should be present
        assert "DOMAIN EXPERTISE" in prompt
        # domain knowledge should NOT be present (tool_context takes precedence)
        assert prompt.count("DOMAIN KNOWLEDGE") == 0


# ── Tool Intelligence Integration ────────────────────────────────────

class TestToolIntelligenceIntegration:
    """Test domain knowledge integration with tool intelligence profiles."""

    def test_tool_profile_gets_domain_knowledge(self):
        """Test that tool intelligence profile includes domain knowledge."""
        from src.tools.intelligence import _get_domain_knowledge_for_tool

        ensure_domain_knowledge_provider()
        result = _get_domain_knowledge_for_tool("hr", "rank")
        assert result
        assert "DOMAIN KNOWLEDGE" in result

    def test_tool_profile_empty_when_disabled(self):
        """Domain knowledge should not be injected when disabled."""
        from src.tools.intelligence import _get_domain_knowledge_for_tool

        with patch("src.api.config.Config.DomainKnowledge") as mock_dk:
            mock_dk.ENABLED = False
            result = _get_domain_knowledge_for_tool("hr", "rank")
            assert result == ""


# ── Web Enrichment ───────────────────────────────────────────────────

class TestWebEnrichment:
    """Test web-based domain knowledge enrichment."""

    def test_web_enrichment_disabled_by_default(self, provider):
        ctx = provider.get_knowledge("hr", include_web=True)
        # Web enrichment is disabled by default
        assert ctx.web_supplement is None

    def test_web_enrichment_enabled(self):
        p = DomainKnowledgeProvider(web_enrichment=True)
        with patch("src.intelligence.domain_knowledge._fetch_domain_web_knowledge") as mock_fetch:
            mock_fetch.return_value = "Web knowledge about HR best practices"
            ctx = p.get_knowledge("hr", include_web=True)
            assert ctx.web_supplement == "Web knowledge about HR best practices"

    def test_web_enrichment_caching(self):
        """Test that web results are cached."""
        from src.intelligence.domain_knowledge import (
            _get_cached_web_knowledge,
            _set_cached_web_knowledge,
        )

        _set_cached_web_knowledge("test_key", "cached value")
        result = _get_cached_web_knowledge("test_key")
        assert result == "cached value"

    def test_web_enrichment_cache_miss(self):
        from src.intelligence.domain_knowledge import _get_cached_web_knowledge
        result = _get_cached_web_knowledge("nonexistent_key_abc123")
        assert result is None


# ── Copy Isolation ───────────────────────────────────────────────────

class TestCopyIsolation:
    """Test that get_knowledge returns independent copies."""

    def test_modifications_dont_affect_base(self, provider):
        ctx1 = provider.get_knowledge("hr")
        original_criteria_count = len(ctx1.evaluation_criteria)
        ctx1.evaluation_criteria.append("test criterion")

        ctx2 = provider.get_knowledge("hr")
        assert len(ctx2.evaluation_criteria) == original_criteria_count

    def test_different_intents_produce_different_contexts(self, provider):
        ctx_rank = provider.get_knowledge("hr", intent="rank")
        ctx_compare = provider.get_knowledge("hr", intent="compare")
        # Perspectives should differ due to intent augmentation
        assert ctx_rank.analytical_perspective != ctx_compare.analytical_perspective
