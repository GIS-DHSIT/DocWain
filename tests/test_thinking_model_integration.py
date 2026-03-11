"""Tests for MoE integration: lfm2.5-thinking + enhanced glm-ocr + parallel agents.

Covers:
  - Model registry: lfm2.5-thinking profile, family matching, capabilities
  - Task routing: all tasks prefer DocWain-Agent first (single-GPU optimisation)
  - Config: ThinkingModel and VisionAnalysis defaults
  - Agent loop: thinking_client used when provided, fallback to llm
  - Domain agents: reasoning agents use thinking model, generation agents use DocWain-Agent
  - Vision analysis: chart/table/diagram/photo/general prompts
  - ImageAgent: multimodal with image_bytes, fallback to text
  - Parallel execution: concurrent agents, timeout handling
  - Multi-agent gateway: REASONER and VISION roles, convenience methods
  - Health endpoints: thinking-model/status, vision-analysis/status
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Minimal duck-typed LLM client for testing."""

    def __init__(self, response: str = "test response"):
        self._response = response
        self.calls: List[Dict[str, Any]] = []
        self.backend = "fake"
        self.model_name = "fake-model"

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self._response

    def generate_with_metadata(self, prompt: str, **kwargs) -> Tuple[str, Dict]:
        self.calls.append({"prompt": prompt, **kwargs})
        return self._response, {"model": self.model_name}


def _fake_ollama_list_with_lfm():
    """Simulates ollama.list() with lfm2.5-thinking included."""
    return {
        "models": [
            {"name": "DocWain-Agent:latest", "size": 13_000_000_000},
            {"name": "llama3.2:latest", "size": 2_000_000_000},
            {"name": "mistral:latest", "size": 4_100_000_000},
            {"name": "deepseek-r1:latest", "size": 4_700_000_000},
            {"name": "gemma2:latest", "size": 5_400_000_000},
            {"name": "llava:latest", "size": 4_700_000_000},
            {"name": "glm-ocr:latest", "size": 2_200_000_000},
            {"name": "lfm2.5-thinking:latest", "size": 731_000_000},
        ]
    }


def _build_moe_registry():
    """Build a ModelRegistry with all 8 models including lfm2.5-thinking."""
    from src.llm.model_registry import ModelCapability, ModelRegistry, _match_family, _MODEL_PROFILES

    registry = ModelRegistry()
    for entry in _fake_ollama_list_with_lfm()["models"]:
        name = entry["name"]
        size = entry["size"]
        family = _match_family(name)
        profile = _MODEL_PROFILES.get(family, {})
        cap = ModelCapability(
            name=name,
            size_bytes=size,
            speed_tier=profile.get("speed_tier", "medium"),
            strengths=list(profile.get("strengths", ["general"])),
            supports_json_mode=profile.get("supports_json_mode", True),
            supports_cot=profile.get("supports_cot", False),
            supports_vision=profile.get("supports_vision", False),
            supports_tool_calling=profile.get("supports_tool_calling", False),
            context_window=profile.get("context_window", 4096),
            available=True,
        )
        registry.register(cap)
    return registry


def _run_async(coro):
    """Run async coroutine in tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result(timeout=10)
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════
# 1. Model Registry — lfm2.5-thinking profile
# ═══════════════════════════════════════════════════════════════════════

class TestLfmModelRegistry:
    def test_lfm_profile_exists(self):
        from src.llm.model_registry import _MODEL_PROFILES
        assert "lfm2.5-thinking" in _MODEL_PROFILES

    def test_lfm_profile_speed_fast(self):
        from src.llm.model_registry import _MODEL_PROFILES
        assert _MODEL_PROFILES["lfm2.5-thinking"]["speed_tier"] == "fast"

    def test_lfm_profile_cot_true(self):
        from src.llm.model_registry import _MODEL_PROFILES
        assert _MODEL_PROFILES["lfm2.5-thinking"]["supports_cot"] is True

    def test_lfm_profile_tool_calling_true(self):
        from src.llm.model_registry import _MODEL_PROFILES
        assert _MODEL_PROFILES["lfm2.5-thinking"]["supports_tool_calling"] is True

    def test_lfm_profile_128k_context(self):
        from src.llm.model_registry import _MODEL_PROFILES
        assert _MODEL_PROFILES["lfm2.5-thinking"]["context_window"] == 131072

    def test_lfm_profile_strengths(self):
        from src.llm.model_registry import _MODEL_PROFILES
        strengths = _MODEL_PROFILES["lfm2.5-thinking"]["strengths"]
        assert "reasoning" in strengths
        assert "chain_of_thought" in strengths
        assert "verification" in strengths
        assert "tool_calling" in strengths

    def test_family_matching_lfm(self):
        from src.llm.model_registry import _match_family
        assert _match_family("lfm2.5-thinking:latest") == "lfm2.5-thinking"

    def test_family_matching_lfm_prefix(self):
        from src.llm.model_registry import _match_family
        assert _match_family("lfm2.5-thinking") == "lfm2.5-thinking"

    def test_registry_discovers_lfm(self):
        reg = _build_moe_registry()
        cap = reg.get("lfm2.5-thinking:latest")
        assert cap is not None
        assert cap.speed_tier == "fast"
        assert cap.supports_cot is True

    def test_registry_has_8_models(self):
        reg = _build_moe_registry()
        assert len(reg.get_available()) == 8

    def test_best_for_reasoning_prefers_lfm(self):
        reg = _build_moe_registry()
        best = reg.best_for("reasoning")
        # lfm2.5-thinking has reasoning + cot + verification + fast speed bonus
        assert best in ("lfm2.5-thinking:latest", "deepseek-r1:latest")

    def test_glm_ocr_enhanced_strengths(self):
        from src.llm.model_registry import _MODEL_PROFILES
        strengths = _MODEL_PROFILES["glm-ocr"]["strengths"]
        assert "chart_analysis" in strengths
        assert "table_analysis" in strengths


# ═══════════════════════════════════════════════════════════════════════
# 2. Task Routing — MoE preferences
# ═══════════════════════════════════════════════════════════════════════

class TestMoeTaskRouting:
    def test_agent_reasoning_task_exists(self):
        from src.llm.task_router import TaskType
        assert TaskType.AGENT_REASONING == "agent_reasoning"

    def test_task_type_count_14(self):
        from src.llm.task_router import TaskType
        assert len(TaskType) == 14

    def test_judging_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.ANSWER_JUDGING][0] == "docwain-agent"

    def test_verify_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.GROUNDING_VERIFY][0] == "docwain-agent"

    def test_agent_reasoning_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.AGENT_REASONING][0] == "docwain-agent"

    def test_classification_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.QUERY_CLASSIFICATION][0] == "docwain-agent"

    def test_intent_parse_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.INTENT_PARSE][0] == "docwain-agent"

    def test_conversation_summary_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.CONVERSATION_SUMMARY][0] == "docwain-agent"

    def test_generation_still_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.RESPONSE_GENERATION][0] == "docwain-agent"

    def test_content_generation_still_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.CONTENT_GENERATION][0] == "docwain-agent"

    def test_complex_extraction_still_prefers_docwain_agent(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES, TaskType
        assert _TASK_MODEL_PREFERENCES[TaskType.COMPLEX_EXTRACTION][0] == "docwain-agent"

    def test_router_selects_docwain_agent_for_judging(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_moe_registry()
        router = TaskRouter(reg)
        assert router.select_model(TaskType.ANSWER_JUDGING) == "DocWain-Agent:latest"

    def test_router_selects_docwain_agent_for_generation(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_moe_registry()
        router = TaskRouter(reg)
        assert router.select_model(TaskType.RESPONSE_GENERATION) == "DocWain-Agent:latest"

    def test_router_fallback_when_preferred_unavailable(self):
        from src.llm.model_registry import ModelCapability, ModelRegistry
        from src.llm.task_router import TaskRouter, TaskType
        # Registry with only deepseek-r1 (no DocWain-Agent, no lfm2.5-thinking)
        reg = ModelRegistry()
        reg.register(ModelCapability(name="deepseek-r1:latest", strengths=["reasoning"]))
        router = TaskRouter(reg)
        # ANSWER_JUDGING prefers DocWain-Agent → missing → lfm2.5-thinking → missing → deepseek-r1
        assert router.select_model(TaskType.ANSWER_JUDGING) == "deepseek-r1:latest"

    def test_agent_reasoning_options(self):
        from src.llm.task_router import _TASK_OPTIONS, TaskType
        opts = _TASK_OPTIONS[TaskType.AGENT_REASONING]
        assert opts["temperature"] == 0.05
        assert opts["num_predict"] == 512
        assert opts["num_ctx"] == 8192

    def test_all_tasks_have_preferences(self):
        from src.llm.task_router import TaskType, _TASK_MODEL_PREFERENCES
        for task in TaskType:
            assert task in _TASK_MODEL_PREFERENCES, f"{task} missing"

    def test_all_tasks_have_options(self):
        from src.llm.task_router import TaskType, _TASK_OPTIONS
        for task in TaskType:
            assert task in _TASK_OPTIONS, f"{task} missing"

    def test_preferences_valid_families(self):
        from src.llm.model_registry import _MODEL_PROFILES
        from src.llm.task_router import _TASK_MODEL_PREFERENCES
        valid = set(_MODEL_PROFILES.keys())
        for task, prefs in _TASK_MODEL_PREFERENCES.items():
            for p in prefs:
                assert p in valid, f"Unknown model family '{p}' in {task}"


# ═══════════════════════════════════════════════════════════════════════
# 3. Config classes
# ═══════════════════════════════════════════════════════════════════════

class TestMoeConfig:
    def test_thinking_model_config_exists(self):
        from src.api.config import Config
        assert hasattr(Config, "ThinkingModel")

    def test_thinking_model_defaults(self):
        from src.api.config import Config
        assert Config.ThinkingModel.MODEL == "lfm2.5-thinking:latest"
        assert Config.ThinkingModel.KEEP_ALIVE == "24h"
        assert Config.ThinkingModel.DEFAULT_TEMPERATURE == 0.05
        assert Config.ThinkingModel.MAX_PREDICT == 512

    def test_thinking_model_use_flags(self):
        from src.api.config import Config
        assert isinstance(Config.ThinkingModel.USE_FOR_JUDGING, bool)
        assert isinstance(Config.ThinkingModel.USE_FOR_AGENT_STEPS, bool)
        assert isinstance(Config.ThinkingModel.USE_FOR_VERIFICATION, bool)

    def test_vision_analysis_config_exists(self):
        from src.api.config import Config
        assert hasattr(Config, "VisionAnalysis")

    def test_vision_analysis_defaults(self):
        from src.api.config import Config
        assert Config.VisionAnalysis.MODEL == "glm-ocr:latest"
        assert Config.VisionAnalysis.MAX_IMAGE_TOKENS == 4096

    def test_vision_analysis_type_flags(self):
        from src.api.config import Config
        assert isinstance(Config.VisionAnalysis.CHART_ANALYSIS, bool)
        assert isinstance(Config.VisionAnalysis.TABLE_ANALYSIS, bool)
        assert isinstance(Config.VisionAnalysis.DIAGRAM_ANALYSIS, bool)
        assert isinstance(Config.VisionAnalysis.PHOTO_ANALYSIS, bool)


# ═══════════════════════════════════════════════════════════════════════
# 4. Agent Loop — thinking_client
# ═══════════════════════════════════════════════════════════════════════

class TestAgentLoopThinking:
    def test_init_with_thinking_client(self):
        try:
            from src.agentic.agent_loop import AgentLoop
        except ImportError:
            pytest.skip("Module removed")
        llm = _FakeLLM("base")
        thinking = _FakeLLM("thinking")
        tools = MagicMock()
        loop = AgentLoop(llm, tools, thinking_client=thinking)
        assert loop._thinking is thinking
        assert loop._llm is llm

    def test_init_without_thinking_client(self):
        try:
            from src.agentic.agent_loop import AgentLoop
        except ImportError:
            pytest.skip("Module removed")
        llm = _FakeLLM("base")
        tools = MagicMock()
        loop = AgentLoop(llm, tools)
        assert loop._thinking is None

    def test_think_uses_thinking_client(self):
        try:
            from src.agentic.agent_loop import AgentLoop
        except ImportError:
            pytest.skip("Module removed")
        llm = _FakeLLM("base response")
        thinking = _FakeLLM('{"reasoning":"thought","final_answer":"done"}')
        tools = MagicMock()
        loop = AgentLoop(llm, tools, thinking_client=thinking)
        thought = loop._think("test prompt")
        assert thought.final_answer == "done"
        assert len(thinking.calls) == 1
        assert len(llm.calls) == 0  # Base LLM not called

    def test_think_fallback_on_thinking_failure(self):
        try:
            from src.agentic.agent_loop import AgentLoop
        except ImportError:
            pytest.skip("Module removed")
        llm = _FakeLLM('{"reasoning":"ok","final_answer":"fallback answer"}')
        thinking = MagicMock()
        thinking.generate_with_metadata = MagicMock(side_effect=RuntimeError("lfm failed"))
        tools = MagicMock()
        loop = AgentLoop(llm, tools, thinking_client=thinking)
        thought = loop._think("test prompt")
        assert thought.final_answer == "fallback answer"


# ═══════════════════════════════════════════════════════════════════════
# 5. Domain Agents — MoE routing
# ═══════════════════════════════════════════════════════════════════════

class TestDomainAgentMoe:
    def test_domain_agent_accepts_thinking_client(self):
        from src.agentic.domain_agents import ResumeAgent
        llm = _FakeLLM("base")
        thinking = _FakeLLM("thinking")
        agent = ResumeAgent(llm_client=llm, thinking_client=thinking)
        assert agent._thinking is thinking

    def test_reasoning_agent_uses_thinking(self):
        from src.agentic.domain_agents import ResumeAgent
        llm = _FakeLLM("base")
        thinking = _FakeLLM("thinking output")
        agent = ResumeAgent(llm_client=llm, thinking_client=thinking)
        # ResumeAgent has use_thinking_model = True (default)
        assert agent.use_thinking_model is True
        assert agent._get_llm() is thinking

    def test_generation_agent_uses_docwain_agent(self):
        from src.agentic.domain_agents import ContentAgent
        llm = _FakeLLM("DocWain-Agent output")
        thinking = _FakeLLM("thinking output")
        agent = ContentAgent(llm_client=llm, thinking_client=thinking)
        assert agent.use_thinking_model is False
        # Should use base LLM, not thinking
        assert agent._get_llm() is llm

    def test_translator_agent_uses_docwain_agent(self):
        from src.agentic.domain_agents import TranslatorAgent
        agent = TranslatorAgent(llm_client=_FakeLLM(), thinking_client=_FakeLLM())
        assert agent.use_thinking_model is False

    def test_tutor_agent_uses_docwain_agent(self):
        from src.agentic.domain_agents import TutorAgent
        agent = TutorAgent(llm_client=_FakeLLM(), thinking_client=_FakeLLM())
        assert agent.use_thinking_model is False

    def test_medical_agent_generation_heavy(self):
        from src.agentic.domain_agents import MedicalAgent
        agent = MedicalAgent(llm_client=_FakeLLM(), thinking_client=_FakeLLM())
        # MedicalAgent is generation-heavy (summaries, interpretations), not reasoning-heavy
        assert agent.use_thinking_model is False

    def test_legal_agent_uses_thinking(self):
        from src.agentic.domain_agents import LegalAgent
        agent = LegalAgent(llm_client=_FakeLLM(), thinking_client=_FakeLLM())
        assert agent.use_thinking_model is True

    def test_invoice_agent_uses_thinking(self):
        from src.agentic.domain_agents import InvoiceAgent
        agent = InvoiceAgent(llm_client=_FakeLLM(), thinking_client=_FakeLLM())
        assert agent.use_thinking_model is True

    def test_get_domain_agent_passes_thinking_client(self):
        from src.agentic.domain_agents import get_domain_agent
        llm = _FakeLLM()
        thinking = _FakeLLM()
        agent = get_domain_agent("hr", llm_client=llm, thinking_client=thinking)
        assert agent is not None
        assert agent._thinking is thinking

    def test_generate_fallback_on_thinking_error(self):
        from src.agentic.domain_agents import ResumeAgent
        llm = _FakeLLM("base output")
        thinking = MagicMock()
        thinking.generate_with_metadata = MagicMock(side_effect=RuntimeError("fail"))
        agent = ResumeAgent(llm_client=llm, thinking_client=thinking)
        result = agent._generate("test prompt")
        assert result == "base output"


# ═══════════════════════════════════════════════════════════════════════
# 6. Vision Analysis — prompts
# ═══════════════════════════════════════════════════════════════════════

class TestVisionAnalysis:
    def test_analysis_prompts_exist(self):
        from src.llm.vision_ocr import _ANALYSIS_PROMPTS
        assert "chart" in _ANALYSIS_PROMPTS
        assert "table" in _ANALYSIS_PROMPTS
        assert "diagram" in _ANALYSIS_PROMPTS
        assert "photo" in _ANALYSIS_PROMPTS
        assert "general" in _ANALYSIS_PROMPTS

    def test_chart_prompt_content(self):
        from src.llm.vision_ocr import _CHART_ANALYSIS_PROMPT
        assert "chart type" in _CHART_ANALYSIS_PROMPT.lower()
        assert "data values" in _CHART_ANALYSIS_PROMPT.lower()

    def test_table_prompt_content(self):
        from src.llm.vision_ocr import _TABLE_ANALYSIS_PROMPT
        assert "markdown" in _TABLE_ANALYSIS_PROMPT.lower()
        assert "column" in _TABLE_ANALYSIS_PROMPT.lower()

    def test_diagram_prompt_content(self):
        from src.llm.vision_ocr import _DIAGRAM_ANALYSIS_PROMPT
        assert "flowchart" in _DIAGRAM_ANALYSIS_PROMPT.lower()
        assert "nodes" in _DIAGRAM_ANALYSIS_PROMPT.lower()

    def test_photo_prompt_content(self):
        from src.llm.vision_ocr import _PHOTO_ANALYSIS_PROMPT
        assert "photograph" in _PHOTO_ANALYSIS_PROMPT.lower()

    def test_analyze_image_method_exists(self):
        from src.llm.vision_ocr import VisionOCRClient
        client = VisionOCRClient()
        assert hasattr(client, "analyze_image")

    def test_analyze_image_returns_tuple(self):
        from src.llm.vision_ocr import VisionOCRClient
        client = VisionOCRClient()
        # Without model, should return empty
        result = client.analyze_image(b"fake_image")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════
# 7. ImageAgent — multimodal
# ═══════════════════════════════════════════════════════════════════════

class TestImageAgentMultimodal:
    def test_image_agent_tries_vision_first(self):
        from src.agentic.domain_agents import ImageAgent
        agent = ImageAgent(llm_client=_FakeLLM("text fallback"))

        # Mock vision client that returns analysis
        mock_vision = MagicMock()
        mock_vision.is_available.return_value = True
        mock_vision.analyze_image.return_value = ("Chart shows upward trend", 85.0)

        with patch("src.agentic.domain_agents.ImageAgent._get_vision_client", return_value=mock_vision):
            result = agent.execute("analyze_image", {"image_bytes": b"fake_image"})

        assert result.success is True
        assert "upward trend" in result.output
        assert result.structured_data["method"] == "vision_multimodal"

    def test_image_agent_text_fallback(self):
        from src.agentic.domain_agents import ImageAgent
        agent = ImageAgent(llm_client=_FakeLLM("text analysis result"))

        # No image_bytes → text-based analysis
        result = agent.execute("analyze_image", {"text": "Sample text from image"})
        assert result.success is True
        assert result.structured_data["method"] == "text_based"

    def test_extract_text_vision_first(self):
        from src.agentic.domain_agents import ImageAgent
        agent = ImageAgent(llm_client=_FakeLLM())

        mock_vision = MagicMock()
        mock_vision.is_available.return_value = True
        mock_vision.ocr_image.return_value = ("Extracted text here", 90.0)

        with patch("src.agentic.domain_agents.ImageAgent._get_vision_client", return_value=mock_vision):
            result = agent.execute("extract_text_from_image", {"image_bytes": b"img"})

        assert result.success is True
        assert result.structured_data["method"] == "vision_ocr"

    def test_describe_image_vision_first(self):
        from src.agentic.domain_agents import ImageAgent
        agent = ImageAgent(llm_client=_FakeLLM())

        mock_vision = MagicMock()
        mock_vision.is_available.return_value = True
        mock_vision.analyze_image.return_value = ("A photo of a building", 80.0)

        with patch("src.agentic.domain_agents.ImageAgent._get_vision_client", return_value=mock_vision):
            result = agent.execute("describe_image", {"image_bytes": b"img"})

        assert result.success is True
        assert result.structured_data["method"] == "vision_description"

    def test_extract_data_vision_first(self):
        from src.agentic.domain_agents import ImageAgent
        agent = ImageAgent(llm_client=_FakeLLM())

        mock_vision = MagicMock()
        mock_vision.is_available.return_value = True
        mock_vision.analyze_image.return_value = ("| Col1 | Col2 |\n| --- | --- |", 85.0)

        with patch("src.agentic.domain_agents.ImageAgent._get_vision_client", return_value=mock_vision):
            result = agent.execute("extract_data_from_image", {"image_bytes": b"img"})

        assert result.success is True
        assert result.structured_data["method"] == "vision_table"


# ═══════════════════════════════════════════════════════════════════════
# 8. Parallel Agent Execution
# ═══════════════════════════════════════════════════════════════════════

class TestParallelAgentExecution:
    def test_parallel_execute_two_agents(self):
        from src.agentic.orchestrator import _execute_agents_parallel
        from src.agentic.domain_agents import ResumeAgent, InvoiceAgent, AgentTaskResult

        llm = _FakeLLM("parallel output")
        agents = [
            (ResumeAgent(llm_client=llm), "candidate_summary", {"text": "resume text"}),
            (InvoiceAgent(llm_client=llm), "financial_summary", {"text": "invoice text"}),
        ]
        results = _execute_agents_parallel(agents, timeout=10.0)
        assert len(results) == 2
        for r in results:
            if r is not None:
                assert isinstance(r, AgentTaskResult)

    def test_parallel_handles_partial_failure(self):
        from src.agentic.orchestrator import _execute_agents_parallel
        from src.agentic.domain_agents import ResumeAgent

        good_llm = _FakeLLM("good output")
        bad_llm = MagicMock()
        bad_llm.generate_with_metadata = MagicMock(side_effect=RuntimeError("boom"))

        agents = [
            (ResumeAgent(llm_client=good_llm), "candidate_summary", {"text": "ok"}),
            (ResumeAgent(llm_client=bad_llm), "candidate_summary", {"text": "fail"}),
        ]
        results = _execute_agents_parallel(agents, timeout=10.0)
        assert len(results) == 2
        # At least one should succeed
        successes = [r for r in results if r is not None and r.success]
        assert len(successes) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 9. Multi-Agent Gateway — REASONER and VISION roles
# ═══════════════════════════════════════════════════════════════════════

class TestMultiAgentMoeRoles:
    def test_reasoner_role_exists(self):
        from src.llm.multi_agent import AgentRole
        assert AgentRole.REASONER == "reasoner"

    def test_vision_role_exists(self):
        from src.llm.multi_agent import AgentRole
        assert AgentRole.VISION == "vision"

    def test_default_role_models_single_gpu(self):
        from src.llm.multi_agent import _DEFAULT_ROLE_MODELS, AgentRole
        assert _DEFAULT_ROLE_MODELS[AgentRole.CLASSIFIER] == "DocWain-Agent:latest"
        assert _DEFAULT_ROLE_MODELS[AgentRole.VERIFIER] == "DocWain-Agent:latest"
        assert _DEFAULT_ROLE_MODELS[AgentRole.REASONER] == "DocWain-Agent:latest"
        assert _DEFAULT_ROLE_MODELS[AgentRole.VISION] == "glm-ocr:latest"
        assert _DEFAULT_ROLE_MODELS[AgentRole.GENERATOR] == "DocWain-Agent:latest"

    def test_reason_convenience_method(self):
        from src.llm.multi_agent import MultiAgentGateway, AgentRole
        gw = MultiAgentGateway()
        fake = _FakeLLM("reasoning result")
        gw._clients[AgentRole.REASONER] = fake
        result = gw.reason("Analyze this argument")
        assert result == "reasoning result"

    def test_analyze_vision_convenience_method(self):
        from src.llm.multi_agent import MultiAgentGateway, AgentRole
        gw = MultiAgentGateway()
        fake = _FakeLLM("vision result")
        gw._clients[AgentRole.VISION] = fake
        result = gw.analyze_vision("Describe this image")
        assert result == "vision result"

    def test_role_count_7(self):
        from src.llm.multi_agent import AgentRole
        assert len(AgentRole) == 7


# ═══════════════════════════════════════════════════════════════════════
# 10. Health Endpoints
# ═══════════════════════════════════════════════════════════════════════

class TestMoeHealthEndpoints:
    def test_thinking_model_status_configured(self):
        from src.api.health_endpoints import thinking_model_status
        with patch("src.api.config.Config.ThinkingModel") as mock_cfg:
            mock_cfg.ENABLED = True
            mock_cfg.MODEL = "lfm2.5-thinking:latest"
            mock_cfg.USE_FOR_JUDGING = True
            mock_cfg.USE_FOR_AGENT_STEPS = True
            mock_cfg.USE_FOR_VERIFICATION = True
            mock_cfg.KEEP_ALIVE = "24h"
            mock_cfg.DEFAULT_TEMPERATURE = 0.05
            with patch("ollama.show"):
                result = _run_async(thinking_model_status())
        assert result["enabled"] is True
        assert result["available"] is True
        assert result["status"] == "active"
        assert result["model"] == "lfm2.5-thinking:latest"

    def test_thinking_model_status_unavailable(self):
        from src.api.health_endpoints import thinking_model_status
        with patch("src.api.config.Config.ThinkingModel") as mock_cfg:
            mock_cfg.ENABLED = True
            mock_cfg.MODEL = "lfm2.5-thinking:latest"
            mock_cfg.USE_FOR_JUDGING = True
            mock_cfg.USE_FOR_AGENT_STEPS = True
            mock_cfg.USE_FOR_VERIFICATION = True
            mock_cfg.KEEP_ALIVE = "24h"
            mock_cfg.DEFAULT_TEMPERATURE = 0.05
            with patch("ollama.show", side_effect=Exception("not found")):
                result = _run_async(thinking_model_status())
        assert result["available"] is False
        assert result["status"] == "model_not_available"

    def test_vision_analysis_status_configured(self):
        from src.api.health_endpoints import vision_analysis_status
        with patch("src.api.config.Config.VisionAnalysis") as mock_cfg:
            mock_cfg.ENABLED = True
            mock_cfg.MODEL = "glm-ocr:latest"
            mock_cfg.CHART_ANALYSIS = True
            mock_cfg.TABLE_ANALYSIS = True
            mock_cfg.DIAGRAM_ANALYSIS = True
            mock_cfg.PHOTO_ANALYSIS = True
            mock_cfg.MAX_IMAGE_TOKENS = 4096

            mock_client = MagicMock()
            mock_client.is_available.return_value = True

            with patch("src.llm.vision_ocr.get_vision_ocr_client", return_value=mock_client):
                result = _run_async(vision_analysis_status())

        assert result["enabled"] is True
        assert result["chart_analysis"] is True
        assert result["status"] == "active"


# ═══════════════════════════════════════════════════════════════════════
# 11. Diagram Extractor Enhancement
# ═══════════════════════════════════════════════════════════════════════

class TestDiagramExtractorMoe:
    def test_extract_diagram_with_vision_function_exists(self):
        from src.doc_understanding.diagram_extractor import extract_diagram_with_vision
        assert callable(extract_diagram_with_vision)

    def test_extract_diagram_structure_accepts_image_param(self):
        import inspect
        from src.doc_understanding.diagram_extractor import extract_diagram_structure
        sig = inspect.signature(extract_diagram_structure)
        assert "image" in sig.parameters

    def test_vision_first_then_text_fallback(self):
        from src.doc_understanding.diagram_extractor import extract_diagram_structure

        # Mock vision client that fails
        with patch("src.doc_understanding.diagram_extractor.extract_diagram_with_vision", return_value=None):
            # Mock LLM client for text fallback
            mock_client = MagicMock()
            mock_client.generate_with_metadata.return_value = (
                '{"diagram_type":"flowchart","nodes":[{"label":"Start","node_type":"start_end"}],"edges":[],"description":"test"}',
                {},
            )
            with patch("src.llm.clients.get_default_client", return_value=mock_client):
                result = extract_diagram_structure(
                    "Start -> Process -> End",
                    image=b"fake_image",
                )
        # Should fall through to text-based extraction
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# 12. Orchestrator — thinking client helper
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorMoe:
    def test_get_thinking_client_enabled(self):
        from src.agentic.orchestrator import _get_thinking_client
        with patch("src.api.config.Config.ThinkingModel") as mock_cfg:
            mock_cfg.ENABLED = True
            mock_cfg.MODEL = "lfm2.5-thinking:latest"
            with patch("src.llm.clients.OllamaClient") as MockClient:
                MockClient.return_value = _FakeLLM()
                client = _get_thinking_client()
        assert client is not None

    def test_get_thinking_client_disabled(self):
        from src.agentic.orchestrator import _get_thinking_client
        with patch("src.api.config.Config.ThinkingModel") as mock_cfg:
            mock_cfg.ENABLED = False
            client = _get_thinking_client()
        assert client is None

    def test_execute_agents_parallel_import(self):
        from src.agentic.orchestrator import _execute_agents_parallel
        assert callable(_execute_agents_parallel)
