"""Centralized LLM gateway — single entry point for all LLM calls.

Priority chain: vLLM (SafeTensor) → Gemini → Ollama (GGUF fallback).
All 16+ files that previously did `import ollama` route through here.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from src.llm.clients import (
    GeminiClient,
    LLMClientWrapper,
    OllamaClient,
    OpenAICompatibleClient,
    ResilientLLMClient,
)

logger = get_logger(__name__)

_GATEWAY: Optional["LLMGateway"] = None
_GATEWAY_LOCK = threading.Lock()

class LLMGateway:
    """Single entry point for ALL LLM calls.

    Wraps a ResilientLLMClient with health tracking and backend info.
    Duck-types the same interface as OllamaClient so existing code works unchanged.
    """

    def __init__(self, primary: Any, fallback: Any = None, name: str = "default"):
        if fallback is not None:
            self._client = ResilientLLMClient(primary, fallback)
        else:
            self._client = primary
        self.model_name = getattr(self._client, "model_name", None)
        self.backend = getattr(primary, "backend", None) or "unknown"
        self.name = name
        self._created_at = time.time()
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {"calls": 0, "errors": 0}
        logger.info(
            "LLMGateway '%s' initialized: primary=%s fallback=%s",
            name,
            getattr(primary, "backend", type(primary).__name__),
            getattr(fallback, "backend", type(fallback).__name__) if fallback else "none",
        )

    def generate(self, prompt: str, **kwargs) -> str:
        with self._stats_lock:
            self._stats["calls"] += 1
            self._stats.setdefault(self.backend, 0)
            self._stats[self.backend] += 1
        try:
            return self._client.generate(prompt, **kwargs)
        except Exception:
            with self._stats_lock:
                self._stats["errors"] += 1
            raise

    def generate_with_metadata(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        with self._stats_lock:
            self._stats["calls"] += 1
            self._stats.setdefault(self.backend, 0)
            self._stats[self.backend] += 1
        try:
            if hasattr(self._client, "generate_with_metadata"):
                return self._client.generate_with_metadata(prompt, **kwargs)
            text = self._client.generate(prompt, **kwargs)
            return text, {"response": text}
        except Exception:
            with self._stats_lock:
                self._stats["errors"] += 1
            raise

    def chat_with_metadata(self, messages, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Chat-based generation with system/user role separation.

        Delegates to underlying client's chat_with_metadata for proper
        role-separated prompts (produces better results than raw generate).
        """
        with self._stats_lock:
            self._stats["calls"] += 1
            self._stats.setdefault(self.backend, 0)
            self._stats[self.backend] += 1
        try:
            if hasattr(self._client, "chat_with_metadata"):
                return self._client.chat_with_metadata(messages, **kwargs)
            # Fallback: concatenate messages into a single prompt
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            )
            return self.generate_with_metadata(prompt, **kwargs)
        except Exception:
            with self._stats_lock:
                self._stats["errors"] += 1
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return per-backend call statistics."""
        with self._stats_lock:
            return {
                **dict(self._stats),
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }

    def classify(self, prompt: str, **kwargs) -> str:
        """Low-latency classification call (intent, domain, sentiment).

        Uses the same backend but callers can use this for semantic distinction.
        """
        return self.generate(prompt, max_retries=1, backoff=0.2, **kwargs)

    def warm_up(self):
        warm = getattr(self._client, "warm_up", None)
        if callable(warm):
            warm()

    def health_check(self) -> Dict[str, Any]:
        """Check if the gateway can generate responses."""
        start = time.time()
        try:
            text = self.generate("Respond with OK.", max_retries=1, backoff=0.0)
            latency_ms = (time.time() - start) * 1000
            return {
                "status": "healthy" if text else "degraded",
                "backend": self.backend,
                "model": self.model_name,
                "latency_ms": round(latency_ms, 1),
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "backend": self.backend,
                "model": self.model_name,
                "error": str(exc),
            }

    # Backward compatibility — some code checks hasattr for in_cooldown
    def in_cooldown(self) -> bool:
        checker = getattr(self._client, "in_cooldown", None)
        if callable(checker):
            return checker()
        return False

def create_llm_gateway(
    model_name: Optional[str] = None,
    backend_override: Optional[str] = None,
) -> LLMGateway:
    """Factory: creates an LLMGateway with vLLM primary → Ollama fallback.

    Reads Config.VLLM for vLLM settings. Falls back to Gemini or Ollama
    based on available configuration.
    """
    from src.api.config import Config
    from src.llm.clients import _resolve_model_alias

    model_name = _resolve_model_alias(model_name)
    backend = (backend_override or os.getenv("LLM_BACKEND", "")).lower().strip()

    primary = None
    fallback = None

    # Always create Ollama as the ultimate fallback
    try:
        fallback = OllamaClient(model_name)
    except Exception as exc:
        logger.warning("Ollama fallback init failed: %s", exc)

    # Check for vLLM first (highest priority for speed)
    vllm_enabled = getattr(Config, "VLLM", None) and getattr(Config.VLLM, "ENABLED", False)
    if vllm_enabled or backend in ("vllm", "openai", "openai_compatible", "local_http"):
        try:
            vllm_cfg = getattr(Config, "VLLM", None)
            primary = OpenAICompatibleClient(
                model_name=getattr(vllm_cfg, "MODEL_NAME", None) or model_name,
                endpoint=getattr(vllm_cfg, "ENDPOINT", None) if vllm_cfg else None,
                api_key=getattr(vllm_cfg, "API_KEY", None) if vllm_cfg else None,
            )
        except Exception as exc:
            logger.warning("vLLM/OpenAI-compatible client init failed: %s", exc)

    # Gemini as secondary option
    if primary is None and (backend == "gemini" or (model_name or "").lower().startswith("gemini")):
        disable_external = getattr(Config.LLM, "DISABLE_EXTERNAL", False)
        if not disable_external:
            try:
                primary = GeminiClient(model_name)
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

    # If no primary was created, use Ollama as primary (no fallback needed)
    if primary is None:
        return LLMGateway(fallback or OllamaClient(model_name), fallback=None, name="ollama-only")

    # Apply concurrency semaphore
    max_concurrency = getattr(Config.LLM, "MAX_CONCURRENCY", 2)
    semaphore = threading.Semaphore(max_concurrency)
    primary = LLMClientWrapper(primary, semaphore)

    return LLMGateway(primary, fallback, name=f"{getattr(primary, 'backend', 'primary')}-with-ollama-fallback")

def get_llm_gateway() -> LLMGateway:
    """Get the singleton LLMGateway, creating it if needed."""
    global _GATEWAY
    if _GATEWAY is not None:
        return _GATEWAY
    with _GATEWAY_LOCK:
        if _GATEWAY is not None:
            return _GATEWAY
        _GATEWAY = create_llm_gateway()
        return _GATEWAY

def set_llm_gateway(gateway: LLMGateway) -> None:
    """Set the singleton LLMGateway (called during app startup)."""
    global _GATEWAY
    _GATEWAY = gateway
