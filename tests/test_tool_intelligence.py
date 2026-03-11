"""Tests for Tool Intelligence Layer.

Tests the ToolProfile dataclass, 14 tool profiles, enhancement engine,
tool context builder, listing function, pipeline integration, and the
GET /api/tools endpoint.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.tools.intelligence import (
    TOOL_PROFILES,
    ToolProfile,
    _LEGAL_DISCLAIMER,
    _MEDICAL_DISCLAIMER,
    _apply_post_processing,
    _build_evidence_from_chunks,
    build_tool_context_for_llm,
    build_tool_enhanced_prompt,
    enhance_tool_result,
    get_tool_profile,
    list_tools_with_capabilities,
)

# ── Helpers ──────────────────────────────────────────────────────────

_VALID_DOMAINS = {"hr", "medical", "legal", "general", "generic"}
_VALID_INTENTS = {
    "factual", "contact", "rank", "compare", "summary",
    "extraction", "cross_document", "reasoning", "generation",
    "analytics", "analysis", "extract",
}


class _FakeChunk:
    """Minimal chunk stand-in for tests."""

    def __init__(self, text: str, source_name: str = "doc.pdf"):
        self.text = text
        self.source = type("S", (), {"document_name": source_name})()
        self.score = 0.9
        self.meta = {}


class _FakeLLM:
    """LLM stub that echoes the prompt prefix and returns a canned response."""

    def __init__(self, response: str = "Enhanced answer from LLM."):
        self._response = response
        self.last_prompt: Optional[str] = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


class _TimeoutLLM:
    """LLM stub that always times out."""

    def generate(self, prompt: str) -> str:
        import time
        time.sleep(30)
        return "never"


class _ErrorLLM:
    """LLM stub that always raises."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM backend unavailable")


class _EmptyLLM:
    """LLM stub that returns empty string."""

    def generate(self, prompt: str) -> str:
        return ""


# ═══════════════════════════════════════════════════════════════════════
# TestToolProfileRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestToolProfileRegistry:
    """Verify all 14 tool profiles are correctly defined."""

    def test_profile_count(self):
        assert len(TOOL_PROFILES) == 17

    def test_all_expected_tools_present(self):
        expected = {
            "resumes", "medical", "lawhere", "creator", "email_drafting",
            "tutor", "image_analysis", "translator", "code_docs",
            "web_extract", "jira_confluence", "db_connector", "stt", "tts",
            "insights", "action_items", "web_search",
        }
        assert set(TOOL_PROFILES.keys()) == expected

    def test_each_profile_is_tool_profile_instance(self):
        for name, profile in TOOL_PROFILES.items():
            assert isinstance(profile, ToolProfile), f"{name} is not a ToolProfile"

    @pytest.mark.parametrize("name", list(TOOL_PROFILES.keys()))
    def test_profile_has_nonempty_system_prompt(self, name):
        assert len(TOOL_PROFILES[name].system_prompt) > 20

    @pytest.mark.parametrize("name", list(TOOL_PROFILES.keys()))
    def test_profile_has_nonempty_description(self, name):
        assert len(TOOL_PROFILES[name].description) > 10

    @pytest.mark.parametrize("name", list(TOOL_PROFILES.keys()))
    def test_profile_domain_is_valid(self, name):
        assert TOOL_PROFILES[name].domain in _VALID_DOMAINS

    @pytest.mark.parametrize("name", list(TOOL_PROFILES.keys()))
    def test_profile_capabilities_nonempty(self, name):
        assert len(TOOL_PROFILES[name].capabilities) >= 1

    @pytest.mark.parametrize("name", list(TOOL_PROFILES.keys()))
    def test_profile_name_matches_key(self, name):
        assert TOOL_PROFILES[name].name == name

    @pytest.mark.parametrize("name", list(TOOL_PROFILES.keys()))
    def test_profile_has_display_name(self, name):
        assert len(TOOL_PROFILES[name].display_name) > 2

    @pytest.mark.parametrize("name", list(TOOL_PROFILES.keys()))
    def test_profile_supported_intents_valid(self, name):
        for intent in TOOL_PROFILES[name].supported_intents:
            assert intent in _VALID_INTENTS, f"{name}: unknown intent {intent}"

    def test_medical_has_medical_disclaimer(self):
        assert "medical_disclaimer" in TOOL_PROFILES["medical"].post_processing

    def test_lawhere_has_legal_disclaimer(self):
        assert "legal_disclaimer" in TOOL_PROFILES["lawhere"].post_processing

    def test_external_tools_flagged(self):
        for name in ("stt", "tts", "db_connector", "jira_confluence"):
            assert TOOL_PROFILES[name].requires_external is True

    def test_non_external_tools_not_flagged(self):
        for name in ("resumes", "medical", "lawhere", "creator"):
            assert TOOL_PROFILES[name].requires_external is False

    def test_output_format_values(self):
        valid_formats = {"structured", "narrative", "tabular"}
        for name, profile in TOOL_PROFILES.items():
            assert profile.output_format in valid_formats, f"{name}: bad format {profile.output_format}"


# ═══════════════════════════════════════════════════════════════════════
# TestGetToolProfile
# ═══════════════════════════════════════════════════════════════════════

class TestGetToolProfile:
    """Test get_tool_profile() lookup."""

    def test_known_tool_returns_profile(self):
        profile = get_tool_profile("resumes")
        assert profile is not None
        assert profile.name == "resumes"
        assert profile.domain == "hr"

    def test_unknown_tool_returns_none(self):
        assert get_tool_profile("nonexistent_tool") is None

    def test_empty_string_returns_none(self):
        assert get_tool_profile("") is None

    def test_case_sensitive(self):
        assert get_tool_profile("Resumes") is None
        assert get_tool_profile("RESUMES") is None

    def test_all_profiles_accessible(self):
        for name in TOOL_PROFILES:
            assert get_tool_profile(name) is not None


# ═══════════════════════════════════════════════════════════════════════
# TestBuildToolEnhancedPrompt
# ═══════════════════════════════════════════════════════════════════════

class TestBuildToolEnhancedPrompt:
    """Test build_tool_enhanced_prompt() output structure."""

    def _profile(self):
        return TOOL_PROFILES["resumes"]

    def test_prompt_contains_system_prompt(self):
        prompt = build_tool_enhanced_prompt(
            self._profile(), "what are the skills?", "some evidence",
            {"skills": ["Python"]}, "factual",
        )
        assert "senior HR analyst" in prompt

    def test_prompt_contains_reasoning_instruction(self):
        prompt = build_tool_enhanced_prompt(
            self._profile(), "what are the skills?", "some evidence",
            {"skills": ["Python"]}, "factual",
        )
        assert "REASONING APPROACH" in prompt
        assert "career trajectory" in prompt

    def test_prompt_contains_query(self):
        prompt = build_tool_enhanced_prompt(
            self._profile(), "what are the skills of Alice?", "evidence",
            {}, "factual",
        )
        assert "what are the skills of Alice?" in prompt

    def test_prompt_contains_raw_result_json(self):
        prompt = build_tool_enhanced_prompt(
            self._profile(), "q", "ev",
            {"skills": ["Python", "Java"]}, "factual",
        )
        assert "PRE-EXTRACTED DATA" in prompt
        assert "Python" in prompt

    def test_raw_result_capped_at_2000_chars(self):
        big_result = {"data": "x" * 5000}
        prompt = build_tool_enhanced_prompt(
            self._profile(), "q", "ev", big_result, "factual",
        )
        # The raw JSON section should be truncated
        pre_idx = prompt.index("PRE-EXTRACTED DATA")
        ev_idx = prompt.index("DOCUMENT EVIDENCE")
        raw_section = prompt[pre_idx:ev_idx]
        # Section includes the raw JSON capped at ~2000 chars + header
        assert len(raw_section) < 2200

    def test_evidence_capped_at_max_evidence_chars(self):
        profile = self._profile()
        big_evidence = "a" * (profile.max_evidence_chars + 5000)
        prompt = build_tool_enhanced_prompt(
            profile, "q", big_evidence, {}, "factual",
        )
        ev_idx = prompt.index("DOCUMENT EVIDENCE")
        remaining = prompt[ev_idx:]
        # Evidence should be capped
        assert len(remaining) < profile.max_evidence_chars + 500

    def test_intent_matched_rendering_hint_included(self):
        prompt = build_tool_enhanced_prompt(
            self._profile(), "q", "ev", {}, "rank",
        )
        assert "OUTPUT INSTRUCTIONS" in prompt
        assert "Score candidates" in prompt

    def test_fallback_to_factual_hint(self):
        prompt = build_tool_enhanced_prompt(
            self._profile(), "q", "ev", {}, "unknown_intent",
        )
        assert "OUTPUT INSTRUCTIONS" in prompt
        assert "comprehensive professional profile" in prompt


# ═══════════════════════════════════════════════════════════════════════
# TestBuildEvidenceFromChunks
# ═══════════════════════════════════════════════════════════════════════

class TestBuildEvidenceFromChunks:
    """Test _build_evidence_from_chunks() helper."""

    def test_empty_chunks(self):
        assert _build_evidence_from_chunks([]) == ""
        assert _build_evidence_from_chunks(None) == ""

    def test_single_chunk(self):
        chunks = [_FakeChunk("Hello world", "resume.pdf")]
        ev = _build_evidence_from_chunks(chunks)
        assert "[resume.pdf]" in ev
        assert "Hello world" in ev

    def test_multiple_chunks_separated(self):
        chunks = [_FakeChunk("First", "a.pdf"), _FakeChunk("Second", "b.pdf")]
        ev = _build_evidence_from_chunks(chunks)
        assert "[a.pdf]" in ev
        assert "[b.pdf]" in ev
        assert "\n\n" in ev

    def test_chunk_text_capped(self):
        long_text = "x" * 2000
        chunks = [_FakeChunk(long_text, "doc.pdf")]
        ev = _build_evidence_from_chunks(chunks)
        # Each chunk capped at _MAX_EVIDENCE_PER_CHUNK (600)
        assert len(ev) < 700

    def test_max_chunks_limit(self):
        chunks = [_FakeChunk(f"text_{i}", f"doc_{i}.pdf") for i in range(20)]
        ev = _build_evidence_from_chunks(chunks)
        # Should only include _MAX_EVIDENCE_CHUNKS (4)
        assert "doc_3.pdf" in ev
        assert "doc_4.pdf" not in ev


# ═══════════════════════════════════════════════════════════════════════
# TestEnhanceToolResult
# ═══════════════════════════════════════════════════════════════════════

class TestEnhanceToolResult:
    """Test enhance_tool_result() end-to-end."""

    def test_with_mock_llm_returns_enhanced_dict(self):
        llm = _FakeLLM("The candidate has 5 years of Python experience.")
        result = enhance_tool_result(
            tool_name="resumes",
            raw_result={"skills": ["Python"]},
            query="what skills does Alice have?",
            chunks=[_FakeChunk("Alice knows Python and Java", "resume.pdf")],
            llm_client=llm,
        )
        assert result is not None
        assert result["intelligence"] == "tool_enhanced"
        assert result["tool"] == "resumes"
        assert result["domain"] == "hr"
        assert "enhanced_response" in result
        assert "raw_result" in result
        assert len(result["enhanced_response"]) > 10

    def test_without_llm_returns_none(self):
        result = enhance_tool_result(
            tool_name="resumes",
            raw_result={"skills": ["Python"]},
            query="skills",
            chunks=[],
            llm_client=None,
        )
        assert result is None

    def test_unknown_tool_returns_none(self):
        result = enhance_tool_result(
            tool_name="totally_fake_tool",
            raw_result={},
            query="query",
            chunks=[],
            llm_client=_FakeLLM(),
        )
        assert result is None

    def test_medical_adds_disclaimer(self):
        llm = _FakeLLM("Patient has hypertension controlled by medication.")
        result = enhance_tool_result(
            tool_name="medical",
            raw_result={"diagnoses": ["hypertension"]},
            query="what are the diagnoses?",
            chunks=[_FakeChunk("Diagnosis: hypertension", "medical.pdf")],
            llm_client=llm,
        )
        assert result is not None
        assert "informational purposes" in result["enhanced_response"]
        assert "medical advice" in result["enhanced_response"]

    def test_legal_adds_disclaimer(self):
        llm = _FakeLLM("The contract obligates Party A to deliver goods.")
        result = enhance_tool_result(
            tool_name="lawhere",
            raw_result={"parties": ["Party A", "Party B"]},
            query="what are the obligations?",
            chunks=[_FakeChunk("Party A shall deliver", "contract.pdf")],
            llm_client=llm,
        )
        assert result is not None
        assert "legal advice" in result["enhanced_response"]

    def test_llm_error_returns_none(self):
        result = enhance_tool_result(
            tool_name="resumes",
            raw_result={"skills": ["Python"]},
            query="skills",
            chunks=[],
            llm_client=_ErrorLLM(),
        )
        assert result is None

    def test_empty_llm_response_returns_none(self):
        result = enhance_tool_result(
            tool_name="resumes",
            raw_result={},
            query="skills",
            chunks=[],
            llm_client=_EmptyLLM(),
        )
        assert result is None

    def test_empty_chunks_still_works(self):
        llm = _FakeLLM("Answer from pre-extracted data only.")
        result = enhance_tool_result(
            tool_name="resumes",
            raw_result={"name": "Alice"},
            query="who is this?",
            chunks=[],
            llm_client=llm,
        )
        assert result is not None
        assert result["enhanced_response"] == "Answer from pre-extracted data only."

    def test_none_chunks_still_works(self):
        llm = _FakeLLM("Answer from raw data.")
        result = enhance_tool_result(
            tool_name="creator",
            raw_result={"content": "draft"},
            query="create content",
            chunks=None,
            llm_client=llm,
        )
        assert result is not None

    def test_intent_hint_passed_through(self):
        llm = _FakeLLM("Ranked list of candidates.")
        result = enhance_tool_result(
            tool_name="resumes",
            raw_result={},
            query="rank candidates",
            chunks=[],
            llm_client=llm,
            intent_hint="rank",
        )
        assert result is not None
        # Verify LLM received a prompt with rank rendering hint
        assert "Score candidates" in llm.last_prompt

    def test_correlation_id_passed(self):
        llm = _FakeLLM("Enhanced answer with enough text for validation.")
        result = enhance_tool_result(
            tool_name="resumes",
            raw_result={},
            query="q",
            chunks=[],
            llm_client=llm,
            correlation_id="test-corr-123",
        )
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# TestApplyPostProcessing
# ═══════════════════════════════════════════════════════════════════════

class TestApplyPostProcessing:
    """Test _apply_post_processing() helper."""

    def test_medical_disclaimer_appended(self):
        text = "Patient findings here."
        result = _apply_post_processing(text, ["medical_disclaimer"])
        assert result.startswith("Patient findings here.")
        assert _MEDICAL_DISCLAIMER in result

    def test_legal_disclaimer_appended(self):
        text = "Contract analysis."
        result = _apply_post_processing(text, ["legal_disclaimer"])
        assert _LEGAL_DISCLAIMER in result

    def test_no_post_processing(self):
        text = "Just text."
        result = _apply_post_processing(text, [])
        assert result == "Just text."

    def test_unknown_post_processing_ignored(self):
        text = "Text."
        result = _apply_post_processing(text, ["unknown_pp"])
        assert result == "Text."


# ═══════════════════════════════════════════════════════════════════════
# TestBuildToolContextForLLM
# ═══════════════════════════════════════════════════════════════════════

class TestBuildToolContextForLLM:
    """Test build_tool_context_for_llm() for LLM-first injection."""

    def test_known_tool_returns_context(self):
        ctx = build_tool_context_for_llm(["resumes"], "factual")
        assert ctx is not None
        assert "Resumes Agent" in ctx
        assert "senior HR analyst" in ctx

    def test_extraction_focus_included(self):
        ctx = build_tool_context_for_llm(["medical"], "factual")
        assert ctx is not None
        assert "Extraction focus:" in ctx
        assert "diagnoses" in ctx.lower()

    def test_rendering_hint_for_intent(self):
        ctx = build_tool_context_for_llm(["resumes"], "rank")
        assert ctx is not None
        assert "Output guidance:" in ctx
        assert "Score candidates" in ctx

    def test_empty_list_returns_none(self):
        assert build_tool_context_for_llm([], "factual") is None

    def test_unknown_tool_returns_none(self):
        assert build_tool_context_for_llm(["fake_tool"], "factual") is None

    def test_multiple_tools_combined(self):
        ctx = build_tool_context_for_llm(["resumes", "medical"], "factual")
        assert ctx is not None
        assert "Resumes Agent" in ctx
        assert "Doc on Doc Agent" in ctx

    def test_mixed_known_unknown(self):
        ctx = build_tool_context_for_llm(["resumes", "fake"], "factual")
        assert ctx is not None
        assert "Resumes Agent" in ctx


# ═══════════════════════════════════════════════════════════════════════
# TestListToolsWithCapabilities
# ═══════════════════════════════════════════════════════════════════════

class TestListToolsWithCapabilities:
    """Test list_tools_with_capabilities() discovery payload."""

    def test_returns_dict_with_tools_and_total(self):
        result = list_tools_with_capabilities()
        assert "tools" in result
        assert "total" in result
        assert isinstance(result["tools"], list)
        assert result["total"] == 17

    def test_each_tool_has_required_fields(self):
        result = list_tools_with_capabilities()
        required = {
            "name", "display_name", "description", "domain",
            "capabilities", "supported_intents", "requires_external",
            "output_format", "intelligence_enabled", "registered",
        }
        for tool in result["tools"]:
            missing = required - set(tool.keys())
            assert not missing, f"Tool {tool.get('name')} missing: {missing}"

    def test_intelligence_enabled_for_all(self):
        result = list_tools_with_capabilities()
        for tool in result["tools"]:
            assert tool["intelligence_enabled"] is True, f"{tool['name']} not intelligence_enabled"

    def test_domain_values_valid(self):
        result = list_tools_with_capabilities()
        for tool in result["tools"]:
            assert tool["domain"] in _VALID_DOMAINS

    def test_capabilities_are_lists(self):
        result = list_tools_with_capabilities()
        for tool in result["tools"]:
            assert isinstance(tool["capabilities"], list)
            assert len(tool["capabilities"]) >= 1


# ═══════════════════════════════════════════════════════════════════════
# TestPipelineDispatchIntegration
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineDispatchIntegration:
    """Test _dispatch_tools() integration with tool intelligence."""

    def _make_chunks(self, n: int = 3) -> list:
        from src.rag_v3.types import Chunk, ChunkSource
        return [
            Chunk(
                id=f"c{i}",
                text=f"Evidence text {i} about Python and Java skills.",
                score=0.9 - i * 0.1,
                source=ChunkSource(document_name=f"doc{i}.pdf"),
            )
            for i in range(n)
        ]

    @patch("src.tools.base.registry")
    def test_dispatch_with_llm_produces_enhanced_chunks(self, mock_registry):
        """When LLM is available, tool results are enhanced."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        # Mock the registry to return a successful tool result
        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "success", "result": {"skills": ["Python"]}}

        mock_registry.invoke = fake_invoke

        llm = _FakeLLM("Enhanced: The candidate excels in Python.")
        chunks = self._make_chunks()

        result = _dispatch_tools(
            tool_names=["resumes"],
            query="what skills does Alice have?",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            chunks=chunks,
            llm_client=llm,
            intent_hint="factual",
        )

        assert len(result) == 1
        assert result[0].meta.get("source") == "tool_enhanced"
        assert result[0].meta.get("tool_name") == "resumes"
        assert result[0].meta.get("domain") == "hr"
        assert "Enhanced" in result[0].text

    @patch("src.tools.base.registry")
    def test_dispatch_without_llm_produces_raw_chunks(self, mock_registry):
        """Without LLM, tool results fall back to raw serialization."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "success", "result": {"skills": ["Python"]}}

        mock_registry.invoke = fake_invoke

        result = _dispatch_tools(
            tool_names=["resumes"],
            query="skills",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            chunks=None,
            llm_client=None,  # No LLM
            intent_hint=None,
        )

        assert len(result) == 1
        assert result[0].meta.get("source") == "tool"  # raw, not enhanced
        assert "Python" in result[0].text

    @patch("src.tools.base.registry")
    def test_dispatch_enhancement_failure_falls_back(self, mock_registry):
        """When enhancement fails, raw result is used."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "success", "result": {"data": "raw"}}

        mock_registry.invoke = fake_invoke

        result = _dispatch_tools(
            tool_names=["resumes"],
            query="skills",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            chunks=None,
            llm_client=_ErrorLLM(),  # Will fail
            intent_hint=None,
        )

        assert len(result) == 1
        assert result[0].meta.get("source") == "tool"  # raw fallback

    @patch("src.tools.base.registry")
    def test_dispatch_unknown_tool_no_enhancement(self, mock_registry):
        """Tools without profiles get raw serialization."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "success", "result": {"value": "hello"}}

        mock_registry.invoke = fake_invoke

        llm = _FakeLLM("Should not be used.")
        result = _dispatch_tools(
            tool_names=["screen_pii"],  # No intelligence profile
            query="scan for pii",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            chunks=None,
            llm_client=llm,
            intent_hint=None,
        )

        assert len(result) == 1
        assert result[0].meta.get("source") == "tool"

    @patch("src.tools.base.registry")
    def test_dispatch_tool_failure_skipped(self, mock_registry):
        """Failed tool dispatch returns empty list."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "error", "result": None}

        mock_registry.invoke = fake_invoke

        result = _dispatch_tools(
            tool_names=["resumes"],
            query="skills",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            chunks=None,
            llm_client=_FakeLLM(),
            intent_hint=None,
        )

        assert len(result) == 0

    @patch("src.tools.base.registry")
    def test_dispatch_backward_compat_no_new_params(self, mock_registry):
        """Old callers without llm_client/intent_hint still work (defaults to None)."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "success", "result": {"value": "data"}}

        mock_registry.invoke = fake_invoke

        result = _dispatch_tools(
            tool_names=["creator"],
            query="generate content",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
        )

        # Without llm_client it falls back to raw
        assert len(result) == 1
        assert result[0].meta.get("source") == "tool"

    @patch("src.tools.base.registry")
    def test_dispatch_multiple_tools(self, mock_registry):
        """Multiple tools each get enhancement attempt."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        call_count = 0

        async def fake_invoke(name, payload, correlation_id=None):
            nonlocal call_count
            call_count += 1
            return {"status": "success", "result": {"tool": name}}

        mock_registry.invoke = fake_invoke

        llm = _FakeLLM("Enhanced output for this tool.")
        result = _dispatch_tools(
            tool_names=["resumes", "medical"],
            query="analyze",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            chunks=None,
            llm_client=llm,
            intent_hint="factual",
        )

        assert call_count == 2
        assert len(result) == 2

    @patch("src.tools.base.registry")
    def test_enhanced_chunk_score_is_one(self, mock_registry):
        """Enhanced chunks get score=1.0 for priority."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "success", "result": {"skills": ["Python"]}}

        mock_registry.invoke = fake_invoke

        llm = _FakeLLM("Enhanced answer.")
        result = _dispatch_tools(
            tool_names=["resumes"],
            query="skills",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            llm_client=llm,
        )

        assert result[0].score == 1.0

    @patch("src.tools.base.registry")
    def test_enhanced_chunk_text_capped_at_2000(self, mock_registry):
        """Enhanced response text is capped at 2000 chars in the chunk."""
        import asyncio
        from src.rag_v3.pipeline import _dispatch_tools

        async def fake_invoke(name, payload, correlation_id=None):
            return {"status": "success", "result": {"data": "x"}}

        mock_registry.invoke = fake_invoke

        llm = _FakeLLM("y" * 5000)  # Very long response
        result = _dispatch_tools(
            tool_names=["resumes"],
            query="skills",
            profile_id="p1",
            subscription_id="s1",
            tool_inputs=None,
            correlation_id="test-123",
            llm_client=llm,
        )

        assert len(result[0].text) <= 2000


# ═══════════════════════════════════════════════════════════════════════
# TestLLMExtractToolContext
# ═══════════════════════════════════════════════════════════════════════

class TestLLMExtractToolContext:
    """Test tool_context injection into LLM extraction prompts."""

    def test_build_generation_prompt_with_tool_context(self):
        from src.rag_v3.llm_extract import build_generation_prompt

        prompt = build_generation_prompt(
            query="summarize the resume",
            evidence_text="Alice has 5 years of Python experience.",
            intent="summary",
            tool_context="[Resume Analysis]\nYou are a senior HR analyst.",
        )
        assert "DOMAIN EXPERTISE:" in prompt
        assert "senior HR analyst" in prompt

    def test_build_generation_prompt_without_tool_context(self):
        from src.rag_v3.llm_extract import build_generation_prompt

        prompt = build_generation_prompt(
            query="summarize",
            evidence_text="evidence",
            intent="factual",
            tool_context=None,
        )
        assert "DOMAIN EXPERTISE:" not in prompt

    def test_build_generation_prompt_empty_tool_context(self):
        from src.rag_v3.llm_extract import build_generation_prompt

        prompt = build_generation_prompt(
            query="summarize",
            evidence_text="evidence",
            intent="factual",
            tool_context="",
        )
        # Empty string is falsy, no section added
        assert "DOMAIN EXPERTISE:" not in prompt


# ═══════════════════════════════════════════════════════════════════════
# TestGetApiToolsEndpoint
# ═══════════════════════════════════════════════════════════════════════

class TestGetApiToolsEndpoint:
    """Test the GET /api/tools endpoint."""

    def test_endpoint_returns_200(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/tools")
        assert resp.status_code == 200

    def test_endpoint_returns_tools_list(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/tools")
        data = resp.json()
        assert "tools" in data
        assert "total" in data
        assert isinstance(data["tools"], list)
        assert data["total"] >= 14

    def test_endpoint_tool_entries_have_fields(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/tools")
        data = resp.json()
        for tool in data["tools"]:
            assert "name" in tool
            assert "display_name" in tool
            assert "domain" in tool
            assert "capabilities" in tool

    def test_endpoint_includes_resumes_tool(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/tools")
        names = [t["name"] for t in resp.json()["tools"]]
        assert "resumes" in names
        assert "medical" in names
        assert "lawhere" in names

    def test_endpoint_intelligence_enabled_flag(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/tools")
        for tool in resp.json()["tools"]:
            assert "intelligence_enabled" in tool
            assert tool["intelligence_enabled"] is True


# ═══════════════════════════════════════════════════════════════════════
# TestHealthEndpoint
# ═══════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Test the GET /api/admin/tools/status health check."""

    # health_router is mounted under api_router (prefix=/api), so full path
    # is /api + /admin/tools/status = /api/admin/tools/status
    _URL = "/api/admin/tools/status"

    def test_tools_status_returns_200(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(self._URL)
        assert resp.status_code == 200

    def test_tools_status_has_expected_fields(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        data = client.get(self._URL).json()
        assert "intelligence_profiles" in data
        assert "tools" in data
        assert data["intelligence_profiles"] == 17

    def test_tools_status_tool_entries(self):
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        data = client.get(self._URL).json()
        for tool in data["tools"]:
            assert "name" in tool
            assert "domain" in tool
            assert "intelligence" in tool
