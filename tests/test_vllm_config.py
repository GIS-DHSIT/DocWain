"""Tests for vLLM and adaptive engine config defaults."""

from src.api.config import Config


class TestVLLMDefaults:
    def test_vllm_enabled_defaults_true(self):
        assert Config.VLLM.ENABLED is True

    def test_vllm_model_name_defaults_to_qwen(self):
        assert Config.VLLM.MODEL_NAME == "Qwen2.5-14B-Instruct-AWQ"


class TestIntelligenceAdaptiveEngine:
    def test_reasoning_fast_path_enabled_defaults_true(self):
        assert Config.Intelligence.REASONING_FAST_PATH_ENABLED is True

    def test_verify_confidence_threshold_defaults_0_8(self):
        assert Config.Intelligence.VERIFY_CONFIDENCE_THRESHOLD == 0.8


class TestLLMConcurrency:
    def test_max_concurrency_defaults_to_8(self):
        assert Config.LLM.MAX_CONCURRENCY == 8
