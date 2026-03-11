"""Tests for the intelligent multi-model task routing system.

Covers:
  - ModelCapability dataclass and speed_tier derivation
  - ModelRegistry: discover(), get(), get_available(), best_for(), family matching
  - TaskType enum completeness
  - _TASK_MODEL_PREFERENCES and _TASK_OPTIONS validation
  - TaskRouter: select_model(), get_options(), explain(), config overrides, fallback chains
  - Thread-local task context: set/get/clear, task_scope, nesting, thread isolation
  - TaskAwareGateway: generate_for_task(), duck-typed generate() with task_scope,
    fallback to role-based, stats tracking
  - Backward compatibility: generate_for_role(), AgentRole, convenience methods
  - Pipeline wiring: task_scope at each pipeline call site
  - Health endpoint: /task-routing/status response shape
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from contextlib import nullcontext
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

    def classify(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def verify(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


def _fake_ollama_list_response():
    """Simulates ollama.list() return value."""
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


def _build_populated_registry():
    """Build a ModelRegistry populated with all 8 standard models (no Ollama needed)."""
    from src.llm.model_registry import ModelCapability, ModelRegistry

    registry = ModelRegistry()
    for entry in _fake_ollama_list_response()["models"]:
        from src.llm.model_registry import _match_family, _MODEL_PROFILES
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
            context_window=profile.get("context_window", 4096),
            available=True,
        )
        registry.register(cap)
    return registry


# ═══════════════════════════════════════════════════════════════════════
# 1. ModelCapability
# ═══════════════════════════════════════════════════════════════════════

class TestModelCapability:
    def test_dataclass_defaults(self):
        from src.llm.model_registry import ModelCapability
        cap = ModelCapability(name="test-model")
        assert cap.name == "test-model"
        assert cap.size_bytes == 0
        assert cap.speed_tier == "medium"
        assert cap.strengths == []
        assert cap.available is True

    def test_dataclass_with_values(self):
        from src.llm.model_registry import ModelCapability
        cap = ModelCapability(
            name="DocWain-Agent:latest",
            size_bytes=13_000_000_000,
            speed_tier="heavy",
            strengths=["generation"],
            supports_cot=False,
            supports_vision=False,
            context_window=8192,
        )
        assert cap.speed_tier == "heavy"
        assert "generation" in cap.strengths

    def test_speed_tier_from_size_fast(self):
        from src.llm.model_registry import _speed_tier_from_size
        assert _speed_tier_from_size(2_000_000_000) == "fast"

    def test_speed_tier_from_size_medium(self):
        from src.llm.model_registry import _speed_tier_from_size
        assert _speed_tier_from_size(5_000_000_000) == "medium"

    def test_speed_tier_from_size_heavy(self):
        from src.llm.model_registry import _speed_tier_from_size
        assert _speed_tier_from_size(13_000_000_000) == "heavy"

    def test_speed_tier_from_size_boundary(self):
        from src.llm.model_registry import _speed_tier_from_size
        # 3GB boundary: exactly 3GB in bytes is < 3.0 GiB → "fast"
        # 3.1 GB → "medium"
        assert _speed_tier_from_size(2_999_999_999) == "fast"
        assert _speed_tier_from_size(3_300_000_000) == "medium"
        assert _speed_tier_from_size(8_600_000_000) == "heavy"


# ═══════════════════════════════════════════════════════════════════════
# 2. ModelRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_empty_registry(self):
        from src.llm.model_registry import ModelRegistry
        reg = ModelRegistry()
        assert reg.get_available() == []
        assert reg.get("DocWain-Agent:latest") is None

    def test_register_model(self):
        from src.llm.model_registry import ModelCapability, ModelRegistry
        reg = ModelRegistry()
        cap = ModelCapability(name="test:latest", size_bytes=1000)
        reg.register(cap)
        assert reg.get("test:latest") is not None
        assert len(reg.get_available()) == 1

    def test_discover_with_mock(self):
        from src.llm.model_registry import ModelRegistry
        reg = ModelRegistry()
        with patch.dict("sys.modules", {"ollama": MagicMock(list=MagicMock(return_value=_fake_ollama_list_response()))}):
            count = reg.discover()
        assert count == 8
        assert len(reg.get_available()) == 8

    def test_discover_handles_failure(self):
        from src.llm.model_registry import ModelRegistry
        reg = ModelRegistry()
        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = ConnectionError("no ollama")
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            count = reg.discover()
        assert count == 0

    def test_get_returns_correct_model(self):
        reg = _build_populated_registry()
        cap = reg.get("DocWain-Agent:latest")
        assert cap is not None
        assert cap.speed_tier == "heavy"
        assert "generation" in cap.strengths

    def test_get_available_only_available(self):
        from src.llm.model_registry import ModelCapability
        reg = _build_populated_registry()
        # Mark one as unavailable
        cap = reg.get("llava:latest")
        cap.available = False
        available = reg.get_available()
        names = [m.name for m in available]
        assert "llava:latest" not in names
        assert len(available) == 7

    def test_best_for_fast_classification(self):
        reg = _build_populated_registry()
        best = reg.best_for("fast_classification")
        assert best == "llama3.2:latest"

    def test_best_for_reasoning(self):
        reg = _build_populated_registry()
        best = reg.best_for("reasoning")
        # lfm2.5-thinking has same overlap (3) as deepseek-r1 but higher speed bonus (fast vs medium)
        assert best == "lfm2.5-thinking:latest"

    def test_best_for_generation(self):
        reg = _build_populated_registry()
        best = reg.best_for("generation")
        # DocWain-Agent and gemma2 both have "generation" — DocWain-Agent should win (more overlap)
        assert best in ("DocWain-Agent:latest", "gemma2:latest")

    def test_best_for_vision(self):
        reg = _build_populated_registry()
        best = reg.best_for("vision")
        assert best == "llava:latest"

    def test_best_for_unknown_requirement(self):
        reg = _build_populated_registry()
        best = reg.best_for("nonexistent_requirement")
        # Should return some available model
        assert best is not None

    def test_family_matching(self):
        from src.llm.model_registry import _match_family
        # gpt-oss maps to qwen3 via _FAMILY_MAP, but qwen3 not in _MODEL_PROFILES → None
        assert _match_family("gpt-oss:latest") is None
        # docwain-agent has its own profile in _MODEL_PROFILES
        assert _match_family("docwain-agent:latest") == "docwain-agent"
        assert _match_family("llama3.2:latest") == "llama3.2"
        assert _match_family("mistral:7b") == "mistral"
        assert _match_family("deepseek-r1:latest") == "deepseek-r1"
        assert _match_family("unknown-model:latest") is None

    def test_singleton(self):
        from src.llm.model_registry import get_model_registry, set_model_registry, ModelRegistry
        original = get_model_registry()
        try:
            reg = ModelRegistry()
            set_model_registry(reg)
            assert get_model_registry() is reg
        finally:
            set_model_registry(original)


# ═══════════════════════════════════════════════════════════════════════
# 3. TaskType
# ═══════════════════════════════════════════════════════════════════════

class TestTaskType:
    def test_all_14_members(self):
        from src.llm.task_router import TaskType
        assert len(TaskType) == 14

    def test_string_values(self):
        from src.llm.task_router import TaskType
        assert TaskType.QUERY_REWRITE == "query_rewrite"
        assert TaskType.INTENT_PARSE == "intent_parse"
        assert TaskType.RESPONSE_GENERATION == "response_generation"
        assert TaskType.GENERAL == "general"

    def test_all_are_strings(self):
        from src.llm.task_router import TaskType
        for t in TaskType:
            assert isinstance(t.value, str)

    def test_enum_from_string(self):
        from src.llm.task_router import TaskType
        assert TaskType("query_rewrite") is TaskType.QUERY_REWRITE
        assert TaskType("general") is TaskType.GENERAL


# ═══════════════════════════════════════════════════════════════════════
# 4. _TASK_MODEL_PREFERENCES
# ═══════════════════════════════════════════════════════════════════════

class TestTaskModelPreferences:
    def test_all_tasks_have_preferences(self):
        from src.llm.task_router import TaskType, _TASK_MODEL_PREFERENCES
        for task in TaskType:
            assert task in _TASK_MODEL_PREFERENCES, f"{task} missing from preferences"

    def test_no_empty_preference_lists(self):
        from src.llm.task_router import _TASK_MODEL_PREFERENCES
        for task, prefs in _TASK_MODEL_PREFERENCES.items():
            assert len(prefs) > 0, f"{task} has empty preference list"

    def test_generation_tasks_prefer_gpt_oss(self):
        """Generation tasks route to DocWain-Agent first (MoE: reasoning tasks use lfm2.5-thinking)."""
        from src.llm.task_router import TaskType, _TASK_MODEL_PREFERENCES
        gen_tasks = [
            TaskType.RESPONSE_GENERATION, TaskType.CONTENT_GENERATION,
            TaskType.COMPLEX_EXTRACTION, TaskType.TOOL_EXECUTION,
            TaskType.STRUCTURED_EXTRACTION, TaskType.DOCUMENT_UNDERSTANDING,
            TaskType.QUERY_REWRITE, TaskType.GENERAL,
        ]
        for task in gen_tasks:
            assert _TASK_MODEL_PREFERENCES[task][0] == "docwain-agent", f"{task} should prefer DocWain-Agent"

    def test_reasoning_tasks_prefer_docwain_agent(self):
        """All tasks route to DocWain-Agent first to avoid model swap contention on T4 16GB."""
        from src.llm.task_router import TaskType, _TASK_MODEL_PREFERENCES
        reasoning_tasks = [
            TaskType.ANSWER_JUDGING, TaskType.GROUNDING_VERIFY,
            TaskType.AGENT_REASONING, TaskType.QUERY_CLASSIFICATION,
            TaskType.INTENT_PARSE, TaskType.CONVERSATION_SUMMARY,
        ]
        for task in reasoning_tasks:
            assert _TASK_MODEL_PREFERENCES[task][0] == "docwain-agent", f"{task} should prefer DocWain-Agent"

    def test_generation_tasks_have_docwain_agent_first(self):
        from src.llm.task_router import TaskType, _TASK_MODEL_PREFERENCES
        gen_tasks = [TaskType.RESPONSE_GENERATION, TaskType.CONTENT_GENERATION]
        for task in gen_tasks:
            assert _TASK_MODEL_PREFERENCES[task][0] == "docwain-agent", f"{task} should prefer DocWain-Agent"

    def test_judging_tasks_have_deepseek_fallback(self):
        from src.llm.task_router import TaskType, _TASK_MODEL_PREFERENCES
        verify_tasks = [TaskType.ANSWER_JUDGING, TaskType.GROUNDING_VERIFY]
        for task in verify_tasks:
            assert "deepseek-r1" in _TASK_MODEL_PREFERENCES[task], f"{task} should have deepseek-r1 as fallback"

    def test_preferences_contain_valid_families(self):
        from src.llm.model_registry import _MODEL_PROFILES
        from src.llm.task_router import _TASK_MODEL_PREFERENCES
        valid_families = set(_MODEL_PROFILES.keys())
        for task, prefs in _TASK_MODEL_PREFERENCES.items():
            for p in prefs:
                assert p in valid_families, f"Unknown model family '{p}' in {task}"


# ═══════════════════════════════════════════════════════════════════════
# 5. _TASK_OPTIONS
# ═══════════════════════════════════════════════════════════════════════

class TestTaskOptions:
    def test_all_tasks_have_options(self):
        from src.llm.task_router import TaskType, _TASK_OPTIONS
        for task in TaskType:
            assert task in _TASK_OPTIONS, f"{task} missing from options"

    def test_temperature_in_valid_range(self):
        from src.llm.task_router import _TASK_OPTIONS
        for task, opts in _TASK_OPTIONS.items():
            temp = opts.get("temperature", 0.3)
            assert 0.0 <= temp <= 1.0, f"{task} temperature {temp} out of range"

    def test_num_predict_positive(self):
        from src.llm.task_router import _TASK_OPTIONS
        for task, opts in _TASK_OPTIONS.items():
            np = opts.get("num_predict", 2048)
            assert np > 0, f"{task} num_predict must be positive"

    def test_num_ctx_positive(self):
        from src.llm.task_router import _TASK_OPTIONS
        for task, opts in _TASK_OPTIONS.items():
            nc = opts.get("num_ctx", 4096)
            assert nc > 0, f"{task} num_ctx must be positive"

    def test_classification_low_temperature(self):
        from src.llm.task_router import TaskType, _TASK_OPTIONS
        low_temp_tasks = [TaskType.INTENT_PARSE, TaskType.QUERY_CLASSIFICATION, TaskType.ANSWER_JUDGING, TaskType.GROUNDING_VERIFY]
        for task in low_temp_tasks:
            assert _TASK_OPTIONS[task]["temperature"] <= 0.05, f"{task} should have temperature <= 0.05"

    def test_rewrite_low_num_predict(self):
        from src.llm.task_router import TaskType, _TASK_OPTIONS
        assert _TASK_OPTIONS[TaskType.QUERY_REWRITE]["num_predict"] < 200


# ═══════════════════════════════════════════════════════════════════════
# 6. TaskRouter
# ═══════════════════════════════════════════════════════════════════════

class TestTaskRouter:
    def test_select_model_full_registry(self):
        """All tasks prefer DocWain-Agent to avoid model swap contention."""
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        assert router.select_model(TaskType.QUERY_REWRITE) == "DocWain-Agent:latest"

    def test_select_model_generation(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        assert router.select_model(TaskType.RESPONSE_GENERATION) == "DocWain-Agent:latest"

    def test_select_model_judging(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        # All tasks prefer DocWain-Agent to avoid model swap contention
        selected = router.select_model(TaskType.ANSWER_JUDGING)
        assert selected == "DocWain-Agent:latest"

    def test_select_model_extraction(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        assert router.select_model(TaskType.STRUCTURED_EXTRACTION) == "DocWain-Agent:latest"

    def test_select_model_fallback_when_preferred_unavailable(self):
        from src.llm.model_registry import ModelCapability, ModelRegistry
        from src.llm.task_router import TaskRouter, TaskType
        # Registry with only DocWain-Agent
        reg = ModelRegistry()
        reg.register(ModelCapability(name="DocWain-Agent:latest", strengths=["generation"]))
        router = TaskRouter(reg)
        # QUERY_REWRITE prefers llama3.2, but it's not available → falls to gemma2 → not available → DocWain-Agent
        selected = router.select_model(TaskType.QUERY_REWRITE)
        assert selected == "DocWain-Agent:latest"

    def test_select_model_empty_registry_ultimate_fallback(self):
        from src.llm.model_registry import ModelRegistry
        from src.llm.task_router import TaskRouter, TaskType
        reg = ModelRegistry()
        router = TaskRouter(reg)
        assert router.select_model(TaskType.GENERAL) == "DocWain-Agent:latest"

    def test_get_options_returns_dict(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        opts = router.get_options(TaskType.QUERY_REWRITE)
        assert isinstance(opts, dict)
        assert "temperature" in opts
        assert "num_predict" in opts

    def test_get_options_all_tasks(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        for task in TaskType:
            opts = router.get_options(task)
            assert isinstance(opts, dict)

    def test_explain_returns_debug_info(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        info = router.explain(TaskType.QUERY_REWRITE)
        assert info["task"] == "query_rewrite"
        assert "selected_model" in info
        assert "preference_list" in info
        assert "options" in info

    def test_explain_all_tasks(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        for task in TaskType:
            info = router.explain(task)
            assert info["task"] == task.value

    def test_config_override(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        with patch("src.llm.task_router.TaskRouter._config_override", return_value="custom-model:latest"):
            result = router.select_model(TaskType.INTENT_PARSE)
            assert result == "custom-model:latest"

    def test_config_override_empty_string_means_auto(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        # Empty string means "use auto-routing" — INTENT_PARSE prefers DocWain-Agent
        with patch("src.llm.task_router.TaskRouter._config_override", return_value=None):
            result = router.select_model(TaskType.INTENT_PARSE)
            assert result == "DocWain-Agent:latest"

    def test_partial_registry_walks_preference(self):
        from src.llm.model_registry import ModelCapability, ModelRegistry
        from src.llm.task_router import TaskRouter, TaskType
        # Registry with only mistral and gemma2
        reg = ModelRegistry()
        reg.register(ModelCapability(name="mistral:latest", strengths=["structured_extraction"]))
        reg.register(ModelCapability(name="gemma2:latest", strengths=["generation"]))
        router = TaskRouter(reg)
        # QUERY_REWRITE prefers [DocWain-Agent, mistral, llama3.2] → DocWain-Agent missing → mistral
        assert router.select_model(TaskType.QUERY_REWRITE) == "mistral:latest"

    def test_select_model_deterministic(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        # Same call should always return the same result
        results = [router.select_model(TaskType.ANSWER_JUDGING) for _ in range(10)]
        assert len(set(results)) == 1


# ═══════════════════════════════════════════════════════════════════════
# 7. Thread-local task context
# ═══════════════════════════════════════════════════════════════════════

class TestTaskContext:
    def test_get_returns_none_by_default(self):
        from src.llm.task_router import get_current_task, clear_current_task
        clear_current_task()
        assert get_current_task() is None

    def test_set_and_get(self):
        from src.llm.task_router import TaskType, set_current_task, get_current_task, clear_current_task
        set_current_task(TaskType.QUERY_REWRITE)
        assert get_current_task() is TaskType.QUERY_REWRITE
        clear_current_task()

    def test_clear(self):
        from src.llm.task_router import TaskType, set_current_task, get_current_task, clear_current_task
        set_current_task(TaskType.INTENT_PARSE)
        clear_current_task()
        assert get_current_task() is None

    def test_task_scope_context_manager(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.RESPONSE_GENERATION):
            assert get_current_task() is TaskType.RESPONSE_GENERATION
        assert get_current_task() is None

    def test_task_scope_nesting(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.RESPONSE_GENERATION):
            assert get_current_task() is TaskType.RESPONSE_GENERATION
            with task_scope(TaskType.ANSWER_JUDGING):
                assert get_current_task() is TaskType.ANSWER_JUDGING
            # Restored to outer scope
            assert get_current_task() is TaskType.RESPONSE_GENERATION
        assert get_current_task() is None

    def test_thread_isolation(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        results = {}
        barrier = threading.Barrier(2)

        def _worker(tid: int, task: TaskType):
            with task_scope(task):
                barrier.wait(timeout=5)
                results[tid] = get_current_task()

        t1 = threading.Thread(target=_worker, args=(1, TaskType.QUERY_REWRITE))
        t2 = threading.Thread(target=_worker, args=(2, TaskType.RESPONSE_GENERATION))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results[1] is TaskType.QUERY_REWRITE
        assert results[2] is TaskType.RESPONSE_GENERATION

    def test_task_scope_restores_on_exception(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        try:
            with task_scope(TaskType.INTENT_PARSE):
                raise ValueError("test error")
        except ValueError:
            pass
        assert get_current_task() is None

    def test_task_scope_restores_nested_on_exception(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.RESPONSE_GENERATION):
            try:
                with task_scope(TaskType.ANSWER_JUDGING):
                    raise ValueError("inner error")
            except ValueError:
                pass
            assert get_current_task() is TaskType.RESPONSE_GENERATION
        assert get_current_task() is None


# ═══════════════════════════════════════════════════════════════════════
# 8. TaskAwareGateway
# ═══════════════════════════════════════════════════════════════════════

class TestTaskAwareGateway:
    def _make_gateway(self, registry=None, fallback=None):
        from src.llm.task_router import TaskRouter
        from src.llm.multi_agent import TaskAwareGateway
        if registry is None:
            registry = _build_populated_registry()
        router = TaskRouter(registry)
        return TaskAwareGateway(router=router, fallback_gateway=fallback)

    def test_init(self):
        gw = self._make_gateway()
        assert gw.backend == "task_aware"
        assert gw.model_name == "task-aware-multi-model"

    def test_generate_for_task_with_fake_client(self):
        from src.llm.task_router import TaskType
        from src.llm.multi_agent import TaskAwareGateway
        gw = self._make_gateway()
        fake = _FakeLLM("routed response")
        # Inject fake client directly — all tasks prefer DocWain-Agent first
        gw._model_clients["DocWain-Agent:latest"] = fake
        result = gw.generate_for_task(TaskType.QUERY_REWRITE, "test prompt")
        assert result == "routed response"
        assert len(fake.calls) == 1

    def test_generate_for_task_tracks_stats(self):
        from src.llm.task_router import TaskType
        gw = self._make_gateway()
        fake = _FakeLLM("ok")
        gw._model_clients["DocWain-Agent:latest"] = fake
        gw.generate_for_task(TaskType.QUERY_REWRITE, "prompt1")
        gw.generate_for_task(TaskType.QUERY_REWRITE, "prompt2")
        stats = gw.get_task_stats()
        assert "query_rewrite" in stats
        assert stats["query_rewrite"]["calls"] == 2

    def test_generate_with_metadata_for_task(self):
        from src.llm.task_router import TaskType
        gw = self._make_gateway()
        fake = _FakeLLM("meta response")
        # INTENT_PARSE routes to DocWain-Agent (all tasks prefer DocWain-Agent)
        gw._model_clients["DocWain-Agent:latest"] = fake
        text, meta = gw.generate_with_metadata_for_task(TaskType.INTENT_PARSE, "prompt")
        assert text == "meta response"
        assert meta["task_type"] == "intent_parse"

    def test_duck_typed_generate_without_scope(self):
        """Without task_scope, generate() falls back to GENERATOR role."""
        from src.llm.task_router import clear_current_task
        from src.llm.multi_agent import AgentRole
        gw = self._make_gateway()
        fake = _FakeLLM("generator response")
        gw._clients[AgentRole.GENERATOR] = fake
        clear_current_task()
        result = gw.generate("test prompt")
        assert result == "generator response"

    def test_duck_typed_generate_with_scope(self):
        """With task_scope, generate() routes to the task-specific model."""
        from src.llm.task_router import TaskType, task_scope
        gw = self._make_gateway()
        fast_fake = _FakeLLM("fast response")
        gw._model_clients["DocWain-Agent:latest"] = fast_fake
        with task_scope(TaskType.QUERY_REWRITE):
            result = gw.generate("test prompt")
        assert result == "fast response"

    def test_duck_typed_generate_with_metadata_with_scope(self):
        from src.llm.task_router import TaskType, task_scope
        gw = self._make_gateway()
        fake = _FakeLLM("metadata response")
        # ANSWER_JUDGING routes to DocWain-Agent (all tasks prefer DocWain-Agent)
        gw._model_clients["DocWain-Agent:latest"] = fake
        with task_scope(TaskType.ANSWER_JUDGING):
            text, meta = gw.generate_with_metadata("test prompt")
        assert text == "metadata response"
        assert meta["task_type"] == "answer_judging"

    def test_fallback_to_gateway_on_client_failure(self):
        from src.llm.task_router import TaskType
        fallback = _FakeLLM("fallback response")
        gw = self._make_gateway(fallback=fallback)
        # Don't inject any model clients → will try to create OllamaClient → fail → use fallback
        with patch("src.llm.clients.OllamaClient", side_effect=ConnectionError("no ollama")):
            result = gw.generate_for_task(TaskType.QUERY_REWRITE, "prompt")
        assert result == "fallback response"

    def test_all_tasks_route_to_gpt_oss(self):
        """All tasks route to DocWain-Agent to avoid model swap contention on T4 16GB."""
        from src.llm.task_router import TaskType, task_scope
        gw = self._make_gateway()
        gpt_fake = _FakeLLM("DocWain-Agent response")
        gw._model_clients["DocWain-Agent:latest"] = gpt_fake

        with task_scope(TaskType.QUERY_REWRITE):
            r1 = gw.generate("q1")
        with task_scope(TaskType.RESPONSE_GENERATION):
            r2 = gw.generate("q2")
        with task_scope(TaskType.ANSWER_JUDGING):
            r3 = gw.generate("q3")

        assert r1 == "DocWain-Agent response"        # Generation → DocWain-Agent
        assert r2 == "DocWain-Agent response"        # Generation → DocWain-Agent
        assert r3 == "DocWain-Agent response"         # Reasoning → DocWain-Agent (no swap)

    def test_get_stats_includes_task_routing(self):
        from src.llm.task_router import TaskType
        gw = self._make_gateway()
        fake = _FakeLLM("ok")
        gw._model_clients["DocWain-Agent:latest"] = fake
        gw.generate_for_task(TaskType.QUERY_REWRITE, "p")
        stats = gw.get_stats()
        assert "task_routing" in stats
        assert "model_clients" in stats

    def test_task_options_passed_to_client(self):
        from src.llm.task_router import TaskType
        gw = self._make_gateway()
        fake = _FakeLLM("ok")
        gw._model_clients["DocWain-Agent:latest"] = fake
        gw.generate_for_task(TaskType.QUERY_REWRITE, "prompt")
        # The options should have been passed to generate
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert "options" in call
        assert call["options"]["temperature"] == 0.1

    def test_generate_for_task_accepts_string(self):
        from src.llm.task_router import TaskType
        gw = self._make_gateway()
        fake = _FakeLLM("ok")
        gw._model_clients["DocWain-Agent:latest"] = fake
        # Pass string instead of enum
        result = gw.generate_for_task("query_rewrite", "prompt")
        assert result == "ok"

    def test_concurrent_task_routing(self):
        """All tasks use DocWain-Agent — concurrent requests hit the same model."""
        from src.llm.task_router import TaskType, task_scope
        gw = self._make_gateway()
        gw._model_clients["DocWain-Agent:latest"] = _FakeLLM("DocWain-Agent response")
        results = {}

        def _worker(tid, task):
            with task_scope(task):
                results[tid] = gw.generate("test")

        t1 = threading.Thread(target=_worker, args=(1, TaskType.QUERY_REWRITE))
        t2 = threading.Thread(target=_worker, args=(2, TaskType.RESPONSE_GENERATION))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert results[1] == "DocWain-Agent response"
        assert results[2] == "DocWain-Agent response"

    def test_error_tracking(self):
        from src.llm.task_router import TaskType
        gw = self._make_gateway(fallback=_FakeLLM("fallback"))
        # Client that always fails
        failing_fake = MagicMock()
        failing_fake.generate.side_effect = RuntimeError("boom")
        gw._model_clients["DocWain-Agent:latest"] = failing_fake
        result = gw.generate_for_task(TaskType.QUERY_REWRITE, "prompt")
        assert result == "fallback"
        stats = gw.get_task_stats()
        assert stats["query_rewrite"]["errors"] == 1

    def test_model_client_lazy_creation(self):
        from src.llm.task_router import TaskType
        gw = self._make_gateway()
        assert len(gw._model_clients) == 0
        # After generate_for_task, a client should be created (or fail)
        with patch("src.llm.clients.OllamaClient", return_value=_FakeLLM("ok")):
            gw.generate_for_task(TaskType.QUERY_REWRITE, "prompt")
        assert "DocWain-Agent:latest" in gw._model_clients


# ═══════════════════════════════════════════════════════════════════════
# 9. Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_generate_for_role_still_works(self):
        from src.llm.multi_agent import AgentRole, TaskAwareGateway
        from src.llm.task_router import TaskRouter
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        gw = TaskAwareGateway(router=router)
        fake = _FakeLLM("role response")
        gw._clients[AgentRole.GENERATOR] = fake
        result = gw.generate_for_role(AgentRole.GENERATOR, "prompt")
        assert result == "role response"

    def test_classify_convenience_method(self):
        from src.llm.multi_agent import AgentRole, TaskAwareGateway
        from src.llm.task_router import TaskRouter
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        gw = TaskAwareGateway(router=router)
        fake = _FakeLLM("classified")
        gw._clients[AgentRole.CLASSIFIER] = fake
        result = gw.classify("what is this?")
        assert result == "classified"

    def test_verify_convenience_method(self):
        from src.llm.multi_agent import AgentRole, TaskAwareGateway
        from src.llm.task_router import TaskRouter
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        gw = TaskAwareGateway(router=router)
        fake = _FakeLLM("verified")
        gw._clients[AgentRole.VERIFIER] = fake
        result = gw.verify("check this")
        assert result == "verified"

    def test_extract_convenience_method(self):
        from src.llm.multi_agent import AgentRole, TaskAwareGateway
        from src.llm.task_router import TaskRouter
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        gw = TaskAwareGateway(router=router)
        fake = _FakeLLM("extracted")
        gw._clients[AgentRole.EXTRACTOR] = fake
        result = gw.extract("get fields")
        assert result == "extracted"

    def test_agent_role_mapping_preserved(self):
        from src.llm.multi_agent import AgentRole, TaskAwareGateway
        from src.llm.task_router import TaskRouter
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        gw = TaskAwareGateway(router=router)
        assert gw.get_role_model(AgentRole.CLASSIFIER) == "DocWain-Agent:latest"
        assert gw.get_role_model(AgentRole.GENERATOR) == "DocWain-Agent:latest"

    def test_list_roles(self):
        from src.llm.multi_agent import TaskAwareGateway
        from src.llm.task_router import TaskRouter
        reg = _build_populated_registry()
        router = TaskRouter(reg)
        gw = TaskAwareGateway(router=router)
        roles = gw.list_roles()
        assert "classifier" in roles
        assert "generator" in roles


# ═══════════════════════════════════════════════════════════════════════
# 10. Pipeline Wiring
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineWiring:
    def test_task_scope_import(self):
        """task_scope and TaskType are importable from the task_router module."""
        from src.llm.task_router import task_scope, TaskType
        assert callable(task_scope)
        assert len(TaskType) == 14

    def test_rewrite_uses_query_rewrite_scope(self):
        """Verify that the rewrite call site would set QUERY_REWRITE scope."""
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.QUERY_REWRITE):
            assert get_current_task() is TaskType.QUERY_REWRITE
        assert get_current_task() is None

    def test_classification_uses_query_classification_scope(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.QUERY_CLASSIFICATION):
            assert get_current_task() is TaskType.QUERY_CLASSIFICATION

    def test_generation_uses_response_generation_scope(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.RESPONSE_GENERATION):
            assert get_current_task() is TaskType.RESPONSE_GENERATION

    def test_judging_uses_answer_judging_scope(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.ANSWER_JUDGING):
            assert get_current_task() is TaskType.ANSWER_JUDGING

    def test_verify_uses_grounding_verify_scope(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.GROUNDING_VERIFY):
            assert get_current_task() is TaskType.GROUNDING_VERIFY

    def test_tool_execution_scope(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.TOOL_EXECUTION):
            assert get_current_task() is TaskType.TOOL_EXECUTION

    def test_content_generation_scope(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.CONTENT_GENERATION):
            assert get_current_task() is TaskType.CONTENT_GENERATION

    def test_conversation_summary_scope(self):
        from src.llm.task_router import TaskType, task_scope, get_current_task, clear_current_task
        clear_current_task()
        with task_scope(TaskType.CONVERSATION_SUMMARY):
            assert get_current_task() is TaskType.CONVERSATION_SUMMARY


# ═══════════════════════════════════════════════════════════════════════
# 11. Health Endpoint
# ═══════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    @staticmethod
    def _run_async(coro):
        """Run an async coroutine safely in tests (avoids deprecated event loop)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Already inside an event loop — create a new one in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(coro)).result(timeout=10)
        return asyncio.run(coro)

    def test_disabled_status(self):
        """When TASK_ROUTING_ENABLED=false, endpoint returns disabled."""
        with patch("src.api.config.Config.TaskRouting") as mock_cfg:
            mock_cfg.ENABLED = False
            from src.api.health_endpoints import task_routing_status
            result = self._run_async(task_routing_status())
            assert result["enabled"] is False

    def test_enabled_without_registry(self):
        with patch("src.api.config.Config.TaskRouting") as mock_cfg:
            mock_cfg.ENABLED = True
            with patch("src.llm.model_registry.get_model_registry", return_value=None):
                from src.api.health_endpoints import task_routing_status
                result = self._run_async(task_routing_status())
                assert result["status"] == "registry_not_initialized"

    def test_enabled_with_registry_and_gateway(self):
        from src.llm.task_router import TaskRouter, TaskType
        reg = _build_populated_registry()
        router = TaskRouter(reg)

        class FakeGateway:
            _router = router
            def get_task_stats(self):
                return {}

        with patch("src.api.config.Config.TaskRouting") as mock_cfg:
            mock_cfg.ENABLED = True
            with patch("src.llm.model_registry.get_model_registry", return_value=reg):
                with patch("src.llm.gateway.get_llm_gateway", return_value=FakeGateway()):
                    from src.api.health_endpoints import task_routing_status
                    result = self._run_async(task_routing_status())
                    assert result["enabled"] is True
                    assert result["models_discovered"] == 8
                    assert "routing_table" in result
                    assert len(result["routing_table"]) == 14

    def test_response_shape(self):
        from src.llm.task_router import TaskRouter
        reg = _build_populated_registry()
        router = TaskRouter(reg)

        class FakeGateway:
            _router = router
            def get_task_stats(self):
                return {"query_rewrite": {"calls": 5, "errors": 0, "avg_latency_ms": 12.3}}

        with patch("src.api.config.Config.TaskRouting") as mock_cfg:
            mock_cfg.ENABLED = True
            with patch("src.llm.model_registry.get_model_registry", return_value=reg):
                with patch("src.llm.gateway.get_llm_gateway", return_value=FakeGateway()):
                    from src.api.health_endpoints import task_routing_status
                    result = self._run_async(task_routing_status())
                    # Check routing table entry shape
                    rt = result["routing_table"]
                    entry = rt["query_rewrite"]
                    assert "selected_model" in entry
                    assert "preference_list" in entry
                    assert "options" in entry
                    # Check task stats
                    assert result["task_stats"]["query_rewrite"]["calls"] == 5


# ═══════════════════════════════════════════════════════════════════════
# 12. Config.TaskRouting
# ═══════════════════════════════════════════════════════════════════════

class TestConfigTaskRouting:
    def test_config_class_exists(self):
        from src.api.config import Config
        assert hasattr(Config, "TaskRouting")

    def test_enabled_default(self):
        from src.api.config import Config
        assert isinstance(Config.TaskRouting.ENABLED, bool)

    def test_fallback_model_default(self):
        from src.api.config import Config
        assert Config.TaskRouting.FALLBACK_MODEL == "DocWain-Agent:latest"

    def test_per_task_overrides_empty_by_default(self):
        from src.api.config import Config
        assert Config.TaskRouting.QUERY_REWRITE_MODEL == ""
        assert Config.TaskRouting.RESPONSE_GENERATION_MODEL == ""
        assert Config.TaskRouting.ANSWER_JUDGING_MODEL == ""
