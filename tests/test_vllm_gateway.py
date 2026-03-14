"""Tests that create_llm_gateway() creates vLLM as primary backend.

Verifies:
1. When Config.VLLM.ENABLED=True, primary is OpenAICompatibleClient with backend="vllm"
2. When vLLM init raises, gateway falls back to Ollama as primary
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_vllm_config(enabled=True):
    """Build a mock Config with VLLM settings."""
    cfg = MagicMock()
    cfg.VLLM.ENABLED = enabled
    cfg.VLLM.ENDPOINT = "http://localhost:8001/v1"
    cfg.VLLM.API_KEY = "test-key"
    cfg.VLLM.MODEL_NAME = "my-model"
    cfg.LLM.MAX_CONCURRENCY = 2
    cfg.LLM.DISABLE_EXTERNAL = True
    return cfg


@patch("src.llm.gateway.OllamaClient")
@patch("src.llm.gateway.OpenAICompatibleClient")
def test_vllm_enabled_creates_openai_compatible_primary(
    mock_openai_cls, mock_ollama_cls
):
    """When VLLM.ENABLED=True, primary should be OpenAICompatibleClient."""
    from src.llm.gateway import create_llm_gateway

    mock_primary = MagicMock()
    mock_primary.backend = "vllm"
    mock_openai_cls.return_value = mock_primary

    mock_fallback = MagicMock()
    mock_fallback.backend = "ollama"
    mock_ollama_cls.return_value = mock_fallback

    mock_cfg = _make_vllm_config(enabled=True)

    with patch("src.api.config.Config", mock_cfg), \
         patch("src.llm.clients._resolve_model_alias", return_value=None):
        gateway = create_llm_gateway()

    # OpenAICompatibleClient was instantiated with vLLM config values
    mock_openai_cls.assert_called_once_with(
        model_name="my-model",
        endpoint="http://localhost:8001/v1",
        api_key="test-key",
    )
    # Gateway reports vllm backend (via LLMClientWrapper -> __getattr__)
    assert gateway.backend == "vllm"
    assert gateway.name != "ollama-only"


@patch("src.llm.gateway.OllamaClient")
@patch("src.llm.gateway.OpenAICompatibleClient")
def test_vllm_init_failure_falls_back_to_ollama(
    mock_openai_cls, mock_ollama_cls
):
    """When vLLM client init raises, gateway should fall back to Ollama as primary."""
    from src.llm.gateway import create_llm_gateway

    mock_openai_cls.side_effect = RuntimeError("vLLM server unreachable")

    mock_ollama = MagicMock()
    mock_ollama.backend = "ollama"
    mock_ollama_cls.return_value = mock_ollama

    mock_cfg = _make_vllm_config(enabled=True)

    with patch("src.api.config.Config", mock_cfg), \
         patch("src.llm.clients._resolve_model_alias", return_value=None):
        gateway = create_llm_gateway()

    # Should have attempted vLLM
    mock_openai_cls.assert_called_once()
    # Fell back to ollama-only
    assert gateway.backend == "ollama"
    assert gateway.name == "ollama-only"
