"""vLLM configuration — added to Config at import time."""
from __future__ import annotations

import os


class VLLMConfig:
    """vLLM serving configuration.

    vLLM serves SafeTensor models via an OpenAI-compatible API.
    Ollama (GGUF) is the automatic fallback when vLLM is unavailable.

    Setup:
        vllm serve openai/gpt-oss-20b --port 8000 --dtype auto --max-model-len 4096

    Environment variables:
        VLLM_ENABLED=true
        VLLM_ENDPOINT=http://localhost:8000/v1/chat/completions
        VLLM_MODEL_NAME=gpt-oss
        VLLM_API_KEY=  (optional)
        VLLM_TIMEOUT=30
    """
    ENABLED = os.getenv("VLLM_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "gpt-oss")
    API_KEY = os.getenv("VLLM_API_KEY", "")
    TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "30"))
