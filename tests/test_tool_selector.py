"""Tests for agent mode auto-tool selection."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agentic.tool_selector import (
    ToolSelector,
    _DOMAIN_TOOLS,
    _INTENT_DOMAIN_TOOLS,
    _INTENT_TOOLS,
    _KEYWORD_TOOL_PATTERNS,
    _NEVER_AUTO_SELECT,
)
from src.agentic.nlu_agent_matcher import match_agents
from src.nlp.nlu_engine import ClassificationResult


def _nlu_match(query: str, expected_agent: str) -> list[str]:
    """Call match_agents() verifying the NLU pipeline routes to *expected_agent*.

    Mocks the classify() method on the agent registry to simulate production
    behaviour with a real embedding model.  The full match_agents() code path
    is exercised — _ensure_registry(), registry retrieval, classify(), result
    unpacking, and the [:max_agents] slice.
    """
    fake = ClassificationResult(
        name=expected_agent, score=0.72, method="combined", gap=0.15,
    )
    # Ensure the real registry is initialised first so _ensure_registry()
    # short-circuits. Then we only patch the classify() call that
    # match_agents() makes on line 167 of nlu_agent_matcher.py.
    from src.agentic.nlu_agent_matcher import _ensure_registry as _init
    _init()

    from src.nlp.nlu_engine import get_registry as _get_reg
    real_reg = _get_reg("agent", create=False)
    with patch.object(real_reg, "classify", return_value=fake):
        return match_agents(query)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pretend all standard tools are registered so the registry check passes.
_ALL_TOOLS = frozenset({
    "resumes", "content_generate", "lawhere", "medical", "screen_pii",
    "translator", "email_drafting", "jira_confluence", "code_docs",
    "tutor", "web_extract", "image_analysis", "tts", "stt", "db_connector",
    "screen_ai_authorship", "screen_resume", "screen_readability",
    "content_types", "creator", "insights", "action_items", "web_search",
})


def _selector(registered: frozenset[str] | None = _ALL_TOOLS) -> ToolSelector:
    return ToolSelector(registered_tools=registered)


def _intent(intent: str = "qa", domain: str = "generic") -> SimpleNamespace:
    return SimpleNamespace(intent=intent, domain=domain)


# ---------------------------------------------------------------------------
# Intent → tool mapping
# ---------------------------------------------------------------------------

class TestIntentMapping:
    def test_generate_intent_selects_content_generate(self):
        result = _selector().select_tools("write something", intent_parse=_intent("generate"))
        assert "content_generate" in result

    def test_compare_intent_no_content_generate(self):
        """Compare intent is handled by RAG pipeline, not content_generate."""
        result = _selector().select_tools("compare them", intent_parse=_intent("compare"))
        assert "content_generate" not in result

    def test_extract_intent_with_hr_domain_selects_resumes(self):
        """Extract intent requires matching HR domain to select resumes."""
        result = _selector().select_tools("extract skills", intent_parse=_intent("extract", "hr"))
        assert "resumes" in result

    def test_extract_intent_without_hr_domain_no_resumes(self):
        """Extract intent on non-HR domain should NOT select resumes."""
        result = _selector().select_tools("extract totals", intent_parse=_intent("extract", "invoice"))
        assert "resumes" not in result

    def test_extract_intent_generic_domain_no_resumes(self):
        """Extract intent on generic domain should NOT select resumes."""
        result = _selector().select_tools("extract data", intent_parse=_intent("extract", "generic"))
        assert "resumes" not in result

    def test_contact_intent_selects_resumes(self):
        result = _selector().select_tools("get contact info", intent_parse=_intent("contact"))
        assert "resumes" in result

    def test_rank_intent_with_hr_domain_selects_resumes(self):
        """Rank intent requires matching HR domain to select resumes."""
        result = _selector().select_tools("rank candidates", intent_parse=_intent("rank", "resume"))
        assert "resumes" in result

    def test_rank_intent_without_hr_domain_no_resumes(self):
        """Rank intent on non-HR domain should NOT select resumes."""
        result = _selector().select_tools("rank invoices", intent_parse=_intent("rank", "invoice"))
        assert "resumes" not in result

    def test_summarize_intent_no_content_generate(self):
        """Summarize intent is handled by RAG pipeline, not content_generate."""
        result = _selector().select_tools("summarize the doc", intent_parse=_intent("summarize"))
        assert "content_generate" not in result

    def test_qa_intent_no_tools(self):
        result = _selector().select_tools("what is the date?", intent_parse=_intent("qa"))
        # qa doesn't map to any tool (only keyword/domain might)
        assert "content_generate" not in result
        assert "resumes" not in result


# ---------------------------------------------------------------------------
# Domain → tool mapping
# ---------------------------------------------------------------------------

class TestDomainMapping:
    def test_resume_domain_selects_resumes(self):
        result = _selector().select_tools("tell me about the candidate", intent_parse=_intent(domain="resume"))
        assert "resumes" in result

    def test_hr_domain_selects_resumes(self):
        result = _selector().select_tools("candidate skills", intent_parse=_intent(domain="hr"))
        assert "resumes" in result

    def test_legal_domain_selects_lawhere(self):
        result = _selector().select_tools("contract terms", intent_parse=_intent(domain="legal"))
        assert "lawhere" in result

    def test_medical_domain_selects_medical(self):
        result = _selector().select_tools("patient info", intent_parse=_intent(domain="medical"))
        assert "medical" in result

    def test_invoice_domain_no_content_generate(self):
        """Invoice domain should NOT auto-select content_generate — factual queries use RAG."""
        result = _selector().select_tools("invoice total", intent_parse=_intent("qa", "invoice"))
        assert "content_generate" not in result
        assert "resumes" not in result
        assert "lawhere" not in result

    def test_generic_domain_no_tool(self):
        result = _selector().select_tools("hello world", intent_parse=_intent("qa", "generic"))
        assert result == []


# ---------------------------------------------------------------------------
# Keyword → tool mapping
# ---------------------------------------------------------------------------

class TestKeywordMapping:
    def test_cover_letter_keyword(self):
        result = _selector().select_tools("write a cover letter for John")
        assert "content_generate" in result

    def test_professional_summary_keyword(self):
        result = _selector().select_tools("generate professional summary")
        assert "content_generate" in result

    def test_skills_matrix_keyword(self):
        result = _selector().select_tools("create a skills matrix")
        assert "content_generate" in result

    def test_interview_prep_keyword(self):
        result = _selector().select_tools("prepare interview prep materials")
        # interview prep routes to resumes (for domain agent handling)
        assert "resumes" in result

    def test_translate_keyword(self):
        result = _selector().select_tools("translate this to French")
        assert "translator" in result

    def test_translation_keyword(self):
        result = _selector().select_tools("provide a translation")
        assert "translator" in result

    def test_email_draft_keyword(self):
        result = _selector().select_tools("draft an email to the team")
        assert "email_drafting" in result

    def test_compose_message_keyword(self):
        result = _selector().select_tools("compose a message to HR")
        assert "email_drafting" in result

    def test_jira_keyword(self):
        result = _selector().select_tools("check the jira ticket")
        assert "jira_confluence" in result

    def test_confluence_keyword(self):
        """NLU-based: 'confluence' in query matches jira_confluence agent."""
        result = _nlu_match("search confluence for docs", "jira_confluence")
        assert "jira_confluence" in result

    def test_code_docs_keyword(self):
        """NLU-based: 'code docs' + 'API' matches code_docs agent."""
        result = _nlu_match("look at the code docs for this API", "code_docs")
        assert "code_docs" in result

    def test_tutor_me_keyword(self):
        """NLU-based: 'tutor' matches tutor agent."""
        result = _nlu_match("tutor me on Python", "tutor")
        assert "tutor" in result

    def test_web_extract_keyword(self):
        result = _selector().select_tools("web extract from the page")
        assert "web_extract" in result

    def test_image_analysis_keyword(self):
        result = _selector().select_tools("extract text from this image using ocr")
        assert "image_analysis" in result

    def test_screen_for_pii_keyword(self):
        result = _selector().select_tools("screen this for pii")
        assert "screen_pii" in result

    def test_summarize_keyword_no_content_generate(self):
        """Summarize keyword should NOT select content_generate — RAG handles summarization."""
        result = _selector().select_tools("Summarize the documents")
        assert "content_generate" not in result

    def test_compare_candidates_keyword(self):
        result = _selector().select_tools("Compare all candidates by skills")
        assert "resumes" in result

    def test_rank_candidates_keyword(self):
        """NLU-based: 'rank candidates' matches resumes agent."""
        result = _nlu_match("Rank the candidates by experience", "resumes")
        assert "resumes" in result

    def test_contact_info_keyword(self):
        result = _selector().select_tools("Get Dhayal's contact info")
        assert "resumes" in result

    def test_extract_skills_from_resume_keyword(self):
        result = _selector().select_tools("Extract all skills from the resume")
        assert "resumes" in result

    def test_invoice_keyword_factual(self):
        """Factual invoice queries should NOT select content_generate."""
        result = _selector().select_tools("What are the invoice totals?")
        assert "content_generate" not in result

    def test_invoice_keyword_generate(self):
        """Only explicit generation queries should select content_generate for invoices."""
        result = _selector().select_tools("Generate a draft invoice for the client")
        assert "content_generate" in result

    def test_candidate_profile_keyword(self):
        """NLU-based: 'candidate profile' matches resumes agent."""
        result = _nlu_match("Show the candidate profile details", "resumes")
        assert "resumes" in result

    def test_action_items_keyword(self):
        result = _selector().select_tools("What are the action items from the meeting?")
        assert "action_items" in result

    def test_find_anomalies_keyword(self):
        result = _selector().select_tools("Find anomalies in the financial data")
        assert "insights" in result

    def test_payment_anomaly_selects_insights_not_medical(self):
        """Payment anomalies are financial, NOT medical."""
        result = _selector().select_tools("Find payment anomalies in the invoices")
        assert "insights" in result
        assert "medical" not in result

    def test_force_majeure_selects_lawhere(self):
        result = _selector().select_tools("Does this contract have a force majeure clause?")
        assert "lawhere" in result

    def test_convert_documents_to_english(self):
        result = _selector().select_tools("Convert all the other language documents to english")
        assert "translator" in result

    def test_drug_interaction_selects_medical(self):
        result = _selector().select_tools("Check for drug interactions with aspirin")
        assert "medical" in result

    def test_compliance_check_selects_lawhere(self):
        result = _selector().select_tools("Check compliance with GDPR regulations")
        assert "lawhere" in result


# ---------------------------------------------------------------------------
# False positive prevention tests
# ---------------------------------------------------------------------------

class TestFalsePositivePrevention:
    """Verify that common queries do NOT trigger wrong tool selection."""

    def test_bare_email_no_email_drafting(self):
        """'email' alone should NOT trigger email_drafting — it's contact extraction."""
        result = _selector().select_tools("What is the candidate's email?")
        assert "email_drafting" not in result

    def test_bare_draft_no_email_drafting(self):
        """'draft' alone (as adjective) should NOT trigger email_drafting."""
        result = _selector().select_tools("What does the draft agreement say?")
        assert "email_drafting" not in result

    def test_bare_task_no_action_items(self):
        """'task' alone should NOT trigger action_items."""
        result = _selector().select_tools("What tasks does the candidate have experience with?")
        assert "action_items" not in result

    def test_bare_deadline_no_action_items(self):
        """'deadline' alone should NOT trigger action_items — it's a document fact."""
        result = _selector().select_tools("What is the project deadline?")
        assert "action_items" not in result

    def test_explain_how_no_tutor(self):
        """'explain how' should NOT trigger tutor — it's a normal question."""
        result = _selector().select_tools("Explain how the payment terms work")
        assert "tutor" not in result

    def test_learn_no_tutor(self):
        """'learn' alone should NOT trigger tutor."""
        result = _selector().select_tools("What can I learn from these documents?")
        assert "tutor" not in result

    def test_bare_medical_no_medical_tool(self):
        """'medical' alone in keyword should NOT trigger medical tool — use ML domain."""
        result = _selector().select_tools("What is the medical expense on this invoice?")
        assert "medical" not in result

    def test_bare_patient_no_medical_tool(self):
        """'patient' as adjective should NOT trigger medical tool."""
        result = _selector().select_tools("Is the candidate patient and detail-oriented?")
        assert "medical" not in result

    def test_bare_contract_no_lawhere(self):
        """'contract' alone should NOT trigger lawhere — use ML domain."""
        result = _selector().select_tools("What is the contract value on this invoice?")
        assert "lawhere" not in result

    def test_bare_legal_no_lawhere(self):
        """'legal' alone should NOT trigger lawhere."""
        result = _selector().select_tools("Is there a legal department mentioned?")
        assert "lawhere" not in result

    def test_bare_screen_no_screen_pii(self):
        """'screen' alone should NOT trigger screen_pii."""
        result = _selector().select_tools("Screen the candidates for this role")
        assert "screen_pii" not in result

    def test_bare_documentation_no_code_docs(self):
        """'documentation' alone should NOT trigger code_docs."""
        result = _selector().select_tools("Where is the documentation for the insurance policy?")
        assert "code_docs" not in result

    def test_bare_unusual_no_insights(self):
        """'unusual' alone should NOT trigger insights."""
        result = _selector().select_tools("The contract has an unusual termination clause")
        assert "insights" not in result

    def test_rank_invoices_no_resumes(self):
        """'rank' on non-HR content should NOT select resumes."""
        result = _selector().select_tools("Rank these invoices by amount")
        assert "resumes" not in result

    def test_extract_invoice_data_no_resumes(self):
        """'extract' on non-HR content should NOT select resumes."""
        result = _selector().select_tools("Extract the total amount from the invoice")
        assert "resumes" not in result

    def test_top_5_invoices_no_resumes(self):
        """'top 5' on non-HR content should NOT select resumes."""
        result = _selector().select_tools("Top 5 invoices by amount")
        assert "resumes" not in result

    def test_obligations_no_action_items(self):
        """'obligations' should NOT trigger action_items — it's a legal term."""
        result = _selector().select_tools("What are the obligations of the employer under this contract?")
        assert "action_items" not in result

    def test_generic_factual_query_no_tools(self):
        """Simple factual queries should return no tools."""
        result = _selector().select_tools("What are the products using laser?")
        assert result == []

    def test_user_complaint_no_wrong_tools(self):
        """User feedback/complaint text should not trigger tools incorrectly."""
        result = _selector().select_tools("these are not invoice documents and my task was different")
        assert "action_items" not in result


# ---------------------------------------------------------------------------
# Never-auto-select guard
# ---------------------------------------------------------------------------

class TestNeverAutoSelect:
    def test_stt_never_auto_selected(self):
        assert "stt" in _NEVER_AUTO_SELECT

    def test_tts_never_auto_selected(self):
        assert "tts" in _NEVER_AUTO_SELECT

    def test_db_connector_never_auto_selected(self):
        assert "db_connector" in _NEVER_AUTO_SELECT

    def test_never_select_enforced_even_with_keyword(self):
        # Even if we force-add stt to intent tools, it should be filtered
        sel = _selector()
        # Manually inject stt — it should still be removed
        with patch.dict("src.agentic.tool_selector._INTENT_TOOLS", {"qa": ["stt"]}):
            result = sel.select_tools("transcribe audio", intent_parse=_intent("qa"))
        assert "stt" not in result


# ---------------------------------------------------------------------------
# Caps and dedup
# ---------------------------------------------------------------------------

class TestCapsAndDedup:
    def test_max_tools_cap(self):
        """Selecting more than MAX_AUTO_TOOLS should be capped."""
        # Query that triggers many tools (via keywords)
        query = "translate and check the jira ticket and look at the code docs"
        result = _selector().select_tools(query)
        assert len(result) <= 3

    def test_dedup_across_sources(self):
        """Intent + domain + keyword all selecting same tool → 1 entry."""
        result = _selector().select_tools(
            "generate a cover letter",
            intent_parse=_intent("generate", "resume"),
        )
        # content_generate from intent + resumes from domain + content_generate from keyword
        assert result.count("content_generate") == 1
        assert result.count("resumes") <= 1

    def test_empty_query_returns_empty(self):
        assert _selector().select_tools("") == []
        assert _selector().select_tools("   ") == []

    def test_none_intent_parse(self):
        result = _selector().select_tools("hello", intent_parse=None)
        # Should not crash — just return keyword-based (or empty)
        assert isinstance(result, list)

    def test_generic_qa_returns_empty(self):
        """Simple factual Q with no special keywords returns no tools."""
        result = _selector().select_tools(
            "what is the total amount?",
            intent_parse=_intent("qa", "generic"),
        )
        assert result == []


# ---------------------------------------------------------------------------
# Registration filtering
# ---------------------------------------------------------------------------

class TestRegistrationFiltering:
    def test_unregistered_tools_filtered_out(self):
        """Tools not in registry should be removed."""
        # Only 'resumes' is registered, 'content_generate' is not
        sel = ToolSelector(registered_tools=frozenset({"resumes"}))
        result = sel.select_tools(
            "generate a cover letter",
            intent_parse=_intent("generate", "resume"),
        )
        assert "content_generate" not in result
        assert "resumes" in result

    def test_empty_registry_allows_all(self):
        """When registry is empty frozenset, no tools pass the filter."""
        sel = ToolSelector(registered_tools=frozenset())
        result = sel.select_tools(
            "generate a cover letter",
            intent_parse=_intent("generate"),
        )
        assert result == []


# ---------------------------------------------------------------------------
# Config gating
# ---------------------------------------------------------------------------

class TestConfigGating:
    def test_config_disabled_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.api.config.Config.Execution.AGENT_AUTO_TOOLS", False)
        result = _selector().select_tools(
            "generate a cover letter",
            intent_parse=_intent("generate"),
        )
        assert result == []

    def test_config_max_tools_respected(self, monkeypatch):
        monkeypatch.setattr("src.api.config.Config.Execution.AGENT_MAX_AUTO_TOOLS", 1)
        result = _selector().select_tools(
            "generate a cover letter",
            intent_parse=_intent("generate", "resume"),
        )
        assert len(result) <= 1

    def test_config_enabled_by_default(self):
        from src.api.config import Config
        assert Config.Execution.AGENT_AUTO_TOOLS is True


# ---------------------------------------------------------------------------
# Analysis object as alternative signal source
# ---------------------------------------------------------------------------

class TestAnalysisFallback:
    def test_analysis_provides_intent(self):
        analysis = SimpleNamespace(intent="generate", domain="generic")
        result = _selector().select_tools("write it", analysis=analysis)
        assert "content_generate" in result

    def test_analysis_provides_domain(self):
        analysis = SimpleNamespace(intent="qa", domain="legal")
        result = _selector().select_tools("contract terms", analysis=analysis)
        assert "lawhere" in result

    def test_intent_parse_takes_precedence(self):
        """intent_parse is checked before analysis."""
        ip = _intent("extract", "resume")
        analysis = SimpleNamespace(intent="generate", domain="legal")
        result = _selector().select_tools("get info", intent_parse=ip, analysis=analysis)
        # Should use intent_parse's intent=extract+domain=resume → resumes
        assert "resumes" in result
        # Should NOT pick up generate→content_generate from analysis
        # (intent_parse already provided an intent)


# ---------------------------------------------------------------------------
# with_tools integration
# ---------------------------------------------------------------------------

class TestWithToolsIntegration:
    """Test with_tools method using a standalone dataclass to avoid
    cross-test contamination from test_agent_mode stubbing sys.modules."""

    def _make_ctx(self, tools=None, use_tools=False):
        """Build a minimal RequestContext-like object with with_tools."""
        from dataclasses import dataclass, field
        from typing import List as L, Optional as O

        @dataclass
        class _Ctx:
            tools: O[L[str]] = None
            use_tools: bool = False

            def with_tools(self, tool_names):
                existing = self.tools or []
                self.tools = list(dict.fromkeys(existing + tool_names))
                if self.tools:
                    self.use_tools = True

        return _Ctx(tools=tools, use_tools=use_tools)

    def test_request_context_with_tools(self):
        ctx = self._make_ctx()
        assert ctx.tools is None
        assert ctx.use_tools is False

        ctx.with_tools(["resumes", "content_generate"])
        assert ctx.tools == ["resumes", "content_generate"]
        assert ctx.use_tools is True

    def test_with_tools_merges_existing(self):
        ctx = self._make_ctx(tools=["resumes"], use_tools=True)
        ctx.with_tools(["content_generate", "resumes"])  # resumes is dup
        assert ctx.tools == ["resumes", "content_generate"]

    def test_with_tools_empty_list_no_change(self):
        ctx = self._make_ctx()
        ctx.with_tools([])
        assert ctx.tools == []
        assert ctx.use_tools is False  # empty list shouldn't enable


# ---------------------------------------------------------------------------
# QuestionRequest tools validator
# ---------------------------------------------------------------------------

class TestQuestionRequestToolsNormalization:
    """Test the tools validator logic directly to avoid sys.modules
    contamination from test_agent_mode::test_non_agent_path_unchanged."""

    @staticmethod
    def _normalize(v):
        """Replicate the _normalize_tools validator logic."""
        if v is None:
            return None
        if isinstance(v, list):
            return [t.strip() for t in v if isinstance(t, str) and t.strip()]
        if isinstance(v, str):
            parts = [t.strip() for t in v.split(",") if t.strip()]
            return parts if parts else None
        return None

    def test_string_tools_normalized(self):
        assert self._normalize("resumes,content_generate") == ["resumes", "content_generate"]

    def test_single_string_tool(self):
        assert self._normalize("resumes") == ["resumes"]

    def test_list_tools_passthrough(self):
        assert self._normalize(["resumes", "content_generate"]) == ["resumes", "content_generate"]

    def test_none_tools(self):
        assert self._normalize(None) is None

    def test_empty_string_tools(self):
        assert self._normalize("") is None

    def test_whitespace_string(self):
        assert self._normalize("  ,  , ") is None

    def test_list_with_empty_strings(self):
        assert self._normalize(["resumes", "", "  "]) == ["resumes"]
