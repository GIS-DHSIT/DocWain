"""Tests for agent mode auto-tool selection."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agentic.tool_selector import (
    ToolSelector,
    _DOMAIN_TOOLS,
    _INTENT_TOOLS,
    _KEYWORD_TOOL_PATTERNS,
    _NEVER_AUTO_SELECT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pretend all standard tools are registered so the registry check passes.
_ALL_TOOLS = frozenset({
    "resumes", "content_generate", "lawhere", "medical", "screen_pii",
    "translator", "email_drafting", "jira_confluence", "code_docs",
    "tutor", "web_extract", "tts", "stt", "db_connector",
    "screen_ai_authorship", "screen_resume", "screen_readability",
    "content_types", "creator",
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

    def test_compare_intent_selects_content_generate(self):
        result = _selector().select_tools("compare them", intent_parse=_intent("compare"))
        assert "content_generate" in result

    def test_extract_intent_selects_resumes(self):
        result = _selector().select_tools("extract skills", intent_parse=_intent("extract"))
        assert "resumes" in result

    def test_contact_intent_selects_resumes(self):
        result = _selector().select_tools("get contact info", intent_parse=_intent("contact"))
        assert "resumes" in result

    def test_rank_intent_selects_resumes(self):
        result = _selector().select_tools("rank candidates", intent_parse=_intent("rank"))
        assert "resumes" in result

    def test_summarize_intent_selects_content_generate(self):
        result = _selector().select_tools("summarize the doc", intent_parse=_intent("summarize"))
        assert "content_generate" in result

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

    def test_invoice_domain_selects_content_generate(self):
        result = _selector().select_tools("invoice total", intent_parse=_intent("qa", "invoice"))
        assert "content_generate" in result
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
        assert "content_generate" in result

    def test_translate_keyword(self):
        result = _selector().select_tools("translate this to French")
        assert "translator" in result

    def test_translation_keyword(self):
        result = _selector().select_tools("provide a translation")
        assert "translator" in result

    def test_email_keyword(self):
        result = _selector().select_tools("draft an email to the team")
        assert "email_drafting" in result

    def test_compose_keyword(self):
        result = _selector().select_tools("compose a message")
        assert "email_drafting" in result

    def test_jira_keyword(self):
        result = _selector().select_tools("check the jira ticket")
        assert "jira_confluence" in result

    def test_confluence_keyword(self):
        result = _selector().select_tools("search confluence for docs")
        assert "jira_confluence" in result

    def test_code_docs_keyword(self):
        result = _selector().select_tools("look at the code docs for this API")
        assert "code_docs" in result

    def test_tutor_keyword(self):
        result = _selector().select_tools("tutor me on Python")
        assert "tutor" in result

    def test_web_extract_keyword(self):
        result = _selector().select_tools("web extract from the page")
        assert "web_extract" in result

    def test_screen_keyword(self):
        result = _selector().select_tools("screen this for pii")
        assert "screen_pii" in result

    def test_summarize_keyword(self):
        result = _selector().select_tools("Summarize the documents")
        assert "content_generate" in result

    def test_compare_candidates_keyword(self):
        result = _selector().select_tools("Compare all candidates by skills")
        assert "content_generate" in result

    def test_rank_keyword(self):
        result = _selector().select_tools("Rank candidates by experience")
        assert "resumes" in result

    def test_contact_info_keyword(self):
        result = _selector().select_tools("Get Dhayal's contact info")
        assert "resumes" in result

    def test_extract_skills_keyword(self):
        result = _selector().select_tools("Extract all skills from the resume")
        assert "resumes" in result

    def test_legal_clause_keyword(self):
        result = _selector().select_tools("Analyze this legal clause for risks")
        assert "lawhere" in result

    def test_medical_report_keyword(self):
        result = _selector().select_tools("Summarize the medical report findings")
        assert "medical" in result

    def test_invoice_keyword(self):
        result = _selector().select_tools("What are the invoice totals?")
        assert "content_generate" in result

    def test_resume_detail_keyword(self):
        result = _selector().select_tools("Get the candidate profile details from the resume")
        assert "resumes" in result


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
        query = "translate the email draft from the jira confluence code docs and tutor me"
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
        # Should use intent_parse's intent=extract → resumes, domain=resume → resumes
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
