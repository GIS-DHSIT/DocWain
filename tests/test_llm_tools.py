"""Tests for src/tools/llm_tools.py — shared LLM infrastructure for tools.

Tests domain options, gateway access, generation, structured output,
IQ scoring, and prompt building.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from src.tools.llm_tools import (
    ToolIQScore,
    _DOMAIN_OPTIONS,
    _SOURCE_CONFIDENCE,
    _is_field_present,
    _parse_json_response,
    build_extraction_prompt,
    build_generation_prompt,
    get_tool_llm_client,
    score_tool_response,
    tool_generate,
    tool_generate_structured,
)


# ── Helpers ─────────────────────────────────────────────────────────

class _FakeLLM:
    """LLM stub that returns a configurable response."""

    def __init__(self, response: str = "Generated text.", metadata: Optional[Dict] = None):
        self._response = response
        self._metadata = metadata or {}
        self.last_prompt: Optional[str] = None
        self.last_options: Optional[Dict] = None

    def generate_with_metadata(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        self.last_prompt = prompt
        self.last_options = kwargs.get("options", {})
        return self._response, self._metadata

    def in_cooldown(self) -> bool:
        return False


class _CooldownLLM(_FakeLLM):
    def in_cooldown(self) -> bool:
        return True


class _SlowLLM:
    """LLM that blocks forever (for timeout tests)."""
    def generate_with_metadata(self, prompt: str, **kwargs):
        import time
        time.sleep(100)

    def in_cooldown(self) -> bool:
        return False


class _ErrorLLM:
    """LLM that always raises."""
    def generate_with_metadata(self, prompt: str, **kwargs):
        raise RuntimeError("LLM exploded")

    def in_cooldown(self) -> bool:
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDomainOptions:
    """All 6 domain dicts exist with required keys."""

    def test_all_domains_present(self):
        assert set(_DOMAIN_OPTIONS.keys()) == {"hr", "medical", "legal", "general", "creative", "translation"}

    @pytest.mark.parametrize("domain", list(_DOMAIN_OPTIONS.keys()))
    def test_required_keys(self, domain):
        opts = _DOMAIN_OPTIONS[domain]
        for key in ("temperature", "num_ctx", "num_predict", "top_p", "top_k"):
            assert key in opts, f"{domain} missing {key}"

    def test_medical_lowest_temperature(self):
        assert _DOMAIN_OPTIONS["medical"]["temperature"] <= _DOMAIN_OPTIONS["hr"]["temperature"]

    def test_creative_highest_temperature(self):
        assert _DOMAIN_OPTIONS["creative"]["temperature"] >= _DOMAIN_OPTIONS["general"]["temperature"]

    def test_legal_larger_context(self):
        assert _DOMAIN_OPTIONS["legal"]["num_ctx"] >= _DOMAIN_OPTIONS["general"]["num_ctx"]

    def test_temperature_values_in_range(self):
        for domain, opts in _DOMAIN_OPTIONS.items():
            assert 0.0 <= opts["temperature"] <= 1.0, f"{domain} temperature out of range"


class TestGetToolLLMClient:
    """Singleton access, cooldown, disabled, unavailable."""

    @patch("src.llm.gateway.get_llm_gateway")
    def test_returns_gateway(self, mock_gw):
        client = _FakeLLM()
        mock_gw.return_value = client
        assert get_tool_llm_client() is client

    @patch("src.llm.gateway.get_llm_gateway")
    def test_returns_none_when_cooldown(self, mock_gw):
        mock_gw.return_value = _CooldownLLM()
        assert get_tool_llm_client() is None

    @patch("src.llm.gateway.get_llm_gateway", return_value=None)
    def test_returns_none_when_no_gateway(self, mock_gw):
        assert get_tool_llm_client() is None

    @patch("src.llm.gateway.get_llm_gateway", side_effect=ImportError("no module"))
    def test_returns_none_on_import_error(self, mock_gw):
        assert get_tool_llm_client() is None

    @patch("src.llm.gateway.get_llm_gateway")
    def test_returns_none_when_disabled(self, mock_gw, monkeypatch):
        from src.api.config import Config
        monkeypatch.setattr(Config.Tools, "LLM_ENABLED", False)
        mock_gw.return_value = _FakeLLM()
        assert get_tool_llm_client() is None


class TestToolGenerate:
    """Generation: success, timeout, error→None, domain routing."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_success(self, mock_client):
        mock_client.return_value = _FakeLLM("Hello world from LLM.")
        result = tool_generate("test prompt", domain="general", timeout=5.0)
        assert result == "Hello world from LLM."

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_no_client_returns_none(self, mock_client):
        assert tool_generate("test") is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_timeout_returns_none(self, mock_client):
        mock_client.return_value = _SlowLLM()
        result = tool_generate("test", timeout=0.1)
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_error_returns_none(self, mock_client):
        mock_client.return_value = _ErrorLLM()
        result = tool_generate("test", timeout=5.0)
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_uses_domain_options(self, mock_client):
        fake = _FakeLLM("response text here.")
        mock_client.return_value = fake
        tool_generate("prompt", domain="medical", timeout=5.0)
        assert fake.last_options is not None
        assert fake.last_options.get("temperature") == 0.05

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_hr_domain_options(self, mock_client):
        fake = _FakeLLM("response text here.")
        mock_client.return_value = fake
        tool_generate("prompt", domain="hr", timeout=5.0)
        assert fake.last_options.get("temperature") == 0.1

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_creative_domain_options(self, mock_client):
        fake = _FakeLLM("response text here.")
        mock_client.return_value = fake
        tool_generate("prompt", domain="creative", timeout=5.0)
        assert fake.last_options.get("temperature") == 0.7

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_unknown_domain_uses_general(self, mock_client):
        fake = _FakeLLM("response text here.")
        mock_client.return_value = fake
        tool_generate("prompt", domain="nonexistent", timeout=5.0)
        assert fake.last_options.get("temperature") == 0.3

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_short_response_returns_none(self, mock_client):
        mock_client.return_value = _FakeLLM("hi")
        result = tool_generate("test", timeout=5.0)
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_empty_response_returns_none(self, mock_client):
        mock_client.return_value = _FakeLLM("")
        result = tool_generate("test", timeout=5.0)
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_whitespace_response_returns_none(self, mock_client):
        mock_client.return_value = _FakeLLM("   \n  ")
        result = tool_generate("test", timeout=5.0)
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_strips_whitespace(self, mock_client):
        mock_client.return_value = _FakeLLM("  clean output  ")
        result = tool_generate("test", timeout=5.0)
        assert result == "clean output"


class TestToolGenerateStructured:
    """JSON parse, fence stripping, malformed→None."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_parses_json(self, mock_client):
        mock_client.return_value = _FakeLLM('{"name": "Alice", "age": 30}')
        result = tool_generate_structured("extract", domain="hr", timeout=5.0)
        assert result == {"name": "Alice", "age": 30}

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_strips_markdown_fences(self, mock_client):
        mock_client.return_value = _FakeLLM('```json\n{"key": "value"}\n```')
        result = tool_generate_structured("extract", timeout=5.0)
        assert result == {"key": "value"}

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_extracts_embedded_json(self, mock_client):
        mock_client.return_value = _FakeLLM('Here is the result: {"data": 42} end.')
        result = tool_generate_structured("extract", timeout=5.0)
        assert result == {"data": 42}

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_malformed_returns_none(self, mock_client):
        mock_client.return_value = _FakeLLM("This is not JSON at all")
        result = tool_generate_structured("extract", timeout=5.0)
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_no_client_returns_none(self, mock_client):
        result = tool_generate_structured("extract")
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_array_response_returns_none(self, mock_client):
        mock_client.return_value = _FakeLLM('[1, 2, 3, 4, 5]')
        result = tool_generate_structured("extract", timeout=5.0)
        assert result is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_appends_json_instruction(self, mock_client):
        fake = _FakeLLM('{"ok": true, "val": "x"}')
        mock_client.return_value = fake
        tool_generate_structured("my prompt", timeout=5.0)
        assert "valid JSON" in fake.last_prompt

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_nested_json(self, mock_client):
        nested = {"outer": {"inner": [1, 2, 3]}, "key": "val"}
        mock_client.return_value = _FakeLLM(json.dumps(nested))
        result = tool_generate_structured("extract", timeout=5.0)
        assert result == nested

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_empty_object(self, mock_client):
        # "{}" is only 2 chars, which is under tool_generate's 5-char minimum.
        # Wrap in enough text so tool_generate accepts it.
        mock_client.return_value = _FakeLLM('Here: {}')
        result = tool_generate_structured("extract", timeout=5.0)
        assert result == {}

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_partial_json_braces(self, mock_client):
        mock_client.return_value = _FakeLLM('Some text {badly formed json')
        result = tool_generate_structured("extract", timeout=5.0)
        assert result is None


class TestParseJsonResponse:
    """Direct tests for _parse_json_response."""

    def test_direct_json(self):
        assert _parse_json_response('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_embedded_json(self):
        assert _parse_json_response('prefix {"a": 1} suffix') == {"a": 1}

    def test_garbage_returns_none(self):
        assert _parse_json_response("not json") is None

    def test_empty_returns_none(self):
        assert _parse_json_response("") is None


class TestToolIQScore:
    """Dataclass and as_dict."""

    def test_defaults(self):
        iq = ToolIQScore()
        assert iq.overall == 0.0
        assert iq.source == "unknown"

    def test_as_dict(self):
        iq = ToolIQScore(overall=0.85, completeness=0.9, confidence=0.78, source="llm", domain="hr")
        d = iq.as_dict()
        assert d["overall"] == 0.85
        assert d["source"] == "llm"
        assert d["domain"] == "hr"
        assert isinstance(d["field_coverage"], dict)

    def test_as_dict_rounds(self):
        iq = ToolIQScore(overall=0.12345, completeness=0.67891, confidence=0.11111)
        d = iq.as_dict()
        assert d["overall"] == 0.123
        assert d["completeness"] == 0.679

    def test_custom_field_coverage(self):
        iq = ToolIQScore(field_coverage={"name": True, "email": False})
        assert iq.field_coverage["name"] is True
        assert iq.field_coverage["email"] is False

    def test_score_ranges(self):
        iq = ToolIQScore(overall=0.5, completeness=0.5, confidence=0.5)
        assert 0.0 <= iq.overall <= 1.0
        assert 0.0 <= iq.completeness <= 1.0
        assert 0.0 <= iq.confidence <= 1.0

    def test_all_source_bases_exist(self):
        for source in ("llm", "regex", "template", "hybrid", "argos"):
            assert source in _SOURCE_CONFIDENCE

    def test_llm_highest_confidence(self):
        assert _SOURCE_CONFIDENCE["llm"] > _SOURCE_CONFIDENCE["regex"]
        assert _SOURCE_CONFIDENCE["llm"] > _SOURCE_CONFIDENCE["template"]


class TestScoreToolResponse:
    """Completeness calc, source bases, domain adjustments."""

    def test_full_completeness(self):
        result = {"name": "Alice", "skills": ["Python"], "age": 30}
        iq = score_tool_response(result, "hr", ["name", "skills", "age"], "llm")
        assert iq.completeness == 1.0

    def test_partial_completeness(self):
        result = {"name": "Alice", "skills": [], "age": 30}
        iq = score_tool_response(result, "hr", ["name", "skills", "age"], "llm")
        assert 0.6 < iq.completeness < 0.7  # 2/3

    def test_zero_completeness(self):
        result = {}
        iq = score_tool_response(result, "hr", ["name", "skills"], "llm")
        assert iq.completeness == 0.0

    def test_empty_expected_fields(self):
        result = {"name": "Alice"}
        iq = score_tool_response(result, "hr", [], "llm")
        assert iq.completeness == 0.0

    def test_llm_source_base(self):
        result = {"name": "Alice"}
        iq = score_tool_response(result, "hr", ["name"], "llm")
        assert iq.confidence > 0.7

    def test_regex_source_base(self):
        result = {"name": "Alice"}
        iq = score_tool_response(result, "hr", ["name"], "regex")
        assert iq.confidence < _SOURCE_CONFIDENCE["llm"]

    def test_template_source_base(self):
        result = {"content": "text"}
        iq = score_tool_response(result, "general", ["content"], "template")
        assert iq.confidence < _SOURCE_CONFIDENCE["regex"]

    def test_overall_formula(self):
        result = {"a": "x", "b": "y"}
        iq = score_tool_response(result, "general", ["a", "b"], "llm")
        expected = 0.6 * iq.completeness + 0.4 * iq.confidence
        assert abs(iq.overall - expected) < 0.001

    def test_field_coverage_tracking(self):
        result = {"name": "Alice", "email": "", "phone": None}
        iq = score_tool_response(result, "hr", ["name", "email", "phone"], "llm")
        assert iq.field_coverage["name"] is True
        assert iq.field_coverage["email"] is False
        assert iq.field_coverage["phone"] is False

    def test_unknown_source(self):
        result = {"x": "y"}
        iq = score_tool_response(result, "general", ["x"], "magic")
        assert iq.confidence > 0  # Falls back to 0.40


class TestIsFieldPresent:
    """Value presence check."""

    def test_none_not_present(self):
        assert _is_field_present(None) is False

    def test_empty_string_not_present(self):
        assert _is_field_present("") is False
        assert _is_field_present("  ") is False

    def test_empty_list_not_present(self):
        assert _is_field_present([]) is False

    def test_empty_dict_not_present(self):
        assert _is_field_present({}) is False

    def test_string_present(self):
        assert _is_field_present("Alice") is True

    def test_list_present(self):
        assert _is_field_present(["Python"]) is True

    def test_dict_present(self):
        assert _is_field_present({"key": "val"}) is True

    def test_number_present(self):
        assert _is_field_present(42) is True
        assert _is_field_present(0) is True


class TestBuildExtractionPrompt:
    """Uses ToolProfile, includes schema, caps text."""

    def test_includes_schema(self):
        prompt = build_extraction_prompt("resumes", "some text", "query", '{"name": ""}')
        assert '{"name": ""}' in prompt

    def test_includes_text(self):
        prompt = build_extraction_prompt("resumes", "resume content here", "query", "{}")
        assert "resume content here" in prompt

    def test_includes_query(self):
        prompt = build_extraction_prompt("resumes", "text", "find skills", "{}")
        assert "find skills" in prompt

    def test_caps_text_length(self):
        long_text = "x" * 10000
        prompt = build_extraction_prompt("resumes", long_text, "query", "{}")
        assert len(prompt) < 10000  # Should be capped

    def test_includes_system_prompt(self):
        prompt = build_extraction_prompt("resumes", "text", "query", "{}")
        # Should include the resume ToolProfile system prompt
        assert "HR" in prompt or "resume" in prompt.lower() or "analyst" in prompt.lower()


class TestBuildGenerationPrompt:
    """Generation prompt for creator/email tools."""

    def test_includes_instructions(self):
        prompt = build_generation_prompt("creator", "Write a summary", "reference text")
        assert "Write a summary" in prompt

    def test_includes_reference(self):
        prompt = build_generation_prompt("creator", "instructions", "my reference material")
        assert "my reference material" in prompt

    def test_caps_reference(self):
        long_ref = "y" * 10000
        prompt = build_generation_prompt("creator", "instr", long_ref)
        assert len(prompt) < 10000

    def test_includes_system_prompt_for_email(self):
        prompt = build_generation_prompt("email_drafting", "Draft email", "context")
        assert "communication" in prompt.lower() or "email" in prompt.lower()

    def test_unknown_tool_still_works(self):
        prompt = build_generation_prompt("nonexistent", "do something", "ref")
        assert "do something" in prompt
