"""
LLM Gateway - unified interface to language model backends.

Primary: vLLM via OpenAICompatibleClient
Fallback (dev only): Ollama via OllamaClient

Public API (unchanged):
    create_llm_gateway() -> LLMGateway
    get_llm_gateway()    -> LLMGateway
    set_llm_gateway(gw)  -> None
"""

from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.utils.logging_utils import get_logger
from src.llm.health import VLLMHealthMonitor

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    text: str
    thinking: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Thinking-block parser
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking(raw: str) -> Tuple[str, Optional[str]]:
    """Split ``<think>...</think>`` blocks from Qwen3 output.

    Returns:
        (answer_text, thinking_text_or_None)
    """
    match = _THINK_RE.search(raw)
    if not match:
        return raw.strip(), None
    thinking = match.group(1).strip()
    answer = _THINK_RE.sub("", raw).strip()
    return answer, thinking or None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_gateway_instance: Optional[LLMGateway] = None
_gateway_lock = threading.Lock()


# ---------------------------------------------------------------------------
# LLMGateway
# ---------------------------------------------------------------------------

class LLMGateway:
    """Unified gateway to LLM backends.

    Prioritises vLLM (via ``OpenAICompatibleClient``).  Falls back to Ollama
    only when vLLM is disabled or unhealthy (intended for local dev).
    """

    def __init__(self) -> None:
        self._primary = None  # OpenAICompatibleClient
        self._fallback = None  # OllamaClient (dev only)
        self._health_monitor: Optional[VLLMHealthMonitor] = None

        # Expose for backward compat
        self.model_name: Optional[str] = None
        self.backend: str = "unknown"

        # Stats
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "requests": 0,
            "failures": 0,
            "fallback_used": 0,
            "last_error": None,
            "last_request_ts": None,
        }
        self._created_at = time.time()

        # Cooldown tracking (populated on repeated failures)
        self._cooldown_until: float = 0.0

        self._init_clients()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_clients(self) -> None:
        """Create backend clients based on configuration.

        Uses Ollama as the sole backend (GPU-efficient, single-server).
        vLLM support removed to eliminate dual-backend complexity.
        """
        # --- Ollama (primary) ---
        try:
            from src.llm.clients import OllamaClient
            self._primary = OllamaClient()
            self.backend = "ollama"
            self.model_name = self._primary.model_name
            logger.info("Ollama primary client initialised (model=%s)", self.model_name)
        except Exception as exc:
            logger.error("Failed to create Ollama client: %s", exc)
            self._primary = None

        if self._primary is None:
            logger.error("No LLM backend available - all calls will fail")

    def _pick_client(self):
        """Return the primary Ollama client."""
        if self._primary is not None:
            return self._primary

        # Last resort: return primary even if unhealthy so caller gets a real error
        if self._primary is not None:
            return self._primary

        raise RuntimeError("No LLM backend configured")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text. Returns the answer string (backward compatible).

        Args:
            prompt: User prompt text.
            system: Optional system prompt.
            think: Enable Qwen3 thinking mode (``<think>`` tags).
            temperature: Sampling temperature (default from Config.LLM).
            max_tokens: Max generation tokens (default from Config.LLM).
            **kwargs: Forwarded to the underlying client.

        Returns:
            Generated answer text (thinking blocks stripped).
        """
        resp = self._do_generate(
            prompt, system=system, think=think,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )
        return resp.text

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system: str = "",
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text and return ``(text, metadata_dict)``.

        Backward-compatible with the old gateway signature.
        """
        if options:
            kwargs.setdefault("options", options)
        resp = self._do_generate(
            prompt, system=system, think=think,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )
        meta: Dict[str, Any] = {
            "usage": resp.usage,
            "backend": self._active_backend_name(),
        }
        if resp.thinking:
            meta["thinking"] = resp.thinking
        return resp.text, meta

    def chat_with_metadata(
        self,
        messages: List[Dict[str, str]],
        *,
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Chat-based generation with system/user messages.

        When *think* is True and the primary client is vLLM, passes
        ``extra_body={"chat_template_kwargs": {"enable_thinking": True}}``
        so the server can activate Qwen3 thinking mode.
        """
        client = self._pick_client()

        temperature = temperature if temperature is not None else Config.LLM.TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else Config.LLM.MAX_TOKENS

        opts = dict(options or {})
        opts.setdefault("temperature", temperature)
        opts.setdefault("max_tokens", max_tokens)
        opts.setdefault("top_p", Config.LLM.TOP_P)

        self._record_request()

        raw, usage_meta = client.chat_with_metadata(
            messages, options=opts, thinking=think, **kwargs,
        )

        answer, thinking = _split_thinking(raw)

        meta: Dict[str, Any] = {
            "usage": usage_meta,
            "backend": "ollama",
        }
        if thinking:
            meta["thinking"] = thinking

        return answer, meta

    # ------------------------------------------------------------------
    # Classification helper
    # ------------------------------------------------------------------

    def classify(self, prompt: str, **kwargs: Any) -> str:
        """Convenience method for classification tasks (low temperature)."""
        return self.generate(prompt, temperature=0.05, max_tokens=256, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Send a trivial request to warm up the backend."""
        try:
            self.generate("Say OK.", max_tokens=8)
            logger.info("LLM warm-up complete")
        except Exception as exc:
            logger.warning("LLM warm-up failed: %s", exc)

    def health_check(self) -> Dict[str, Any]:
        """Return a health summary dict."""
        return {
            "healthy": self._primary is not None,
            "primary": {
                "available": self._primary is not None,
                "backend": getattr(self._primary, "backend", None),
                "model": getattr(self._primary, "model_name", None),
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                **dict(self._stats),
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }

    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def shutdown(self) -> None:
        """Stop background threads."""
        if self._health_monitor:
            self._health_monitor.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_generate(
        self,
        prompt: str,
        *,
        system: str = "",
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Unified generation logic shared by generate() and generate_with_metadata()."""
        client = self._pick_client()

        temperature = temperature if temperature is not None else Config.LLM.TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else Config.LLM.MAX_TOKENS

        # Extract extra options to merge later (avoids duplicate kwarg for 'options')
        extra_options = kwargs.pop("options", None)

        self._record_request()

        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        opts = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_options:
            opts.update(extra_options)

        raw, usage_meta = client.generate_with_metadata(
            full_prompt, options=opts, thinking=think, **kwargs,
        )

        answer, thinking = _split_thinking(raw)
        return LLMResponse(text=answer, thinking=thinking, usage=usage_meta)

    def _vllm_chat(
        self,
        client,
        messages: List[Dict[str, str]],
        *,
        think: bool = False,
        opts: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict]:
        """Call vLLM via OpenAICompatibleClient, injecting thinking kwargs when needed."""
        call_kwargs: Dict[str, Any] = dict(kwargs)
        call_opts = dict(opts or {})

        if think:
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True}
            }

        # Format messages into a single prompt for the client
        prompt = self._messages_to_prompt(messages)

        return client.generate_with_metadata(prompt, options=call_opts, **call_kwargs)

    @staticmethod
    def _build_messages(prompt: str, system: str = "") -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Flatten chat messages into a single prompt string for clients
        that only accept a prompt argument."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def _active_backend_name(self) -> str:
        client = self._pick_client()
        return getattr(client, "backend", "unknown")

    def _record_request(self) -> None:
        with self._stats_lock:
            self._stats["requests"] += 1
            self._stats["last_request_ts"] = time.time()

    def _record_failure(self, exc: Exception) -> None:
        with self._stats_lock:
            self._stats["failures"] += 1
            self._stats["last_error"] = str(exc)

    def _record_fallback(self) -> None:
        with self._stats_lock:
            self._stats["fallback_used"] += 1


# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------

def create_llm_gateway(
    model_name: Optional[str] = None,
    backend_override: Optional[str] = None,
) -> LLMGateway:
    """Create (or recreate) the global LLMGateway singleton.

    Args are accepted for backward compatibility but ignored — the gateway
    reads its configuration from ``Config.VLLM`` and ``Config.LLM``.
    """
    global _gateway_instance
    with _gateway_lock:
        if _gateway_instance is not None:
            _gateway_instance.shutdown()
        _gateway_instance = LLMGateway()
        return _gateway_instance


def get_llm_gateway() -> LLMGateway:
    """Return the existing singleton, creating it on first call."""
    global _gateway_instance
    if _gateway_instance is None:
        with _gateway_lock:
            if _gateway_instance is None:
                _gateway_instance = LLMGateway()
    return _gateway_instance


def set_llm_gateway(gateway: LLMGateway) -> None:
    """Replace the global singleton (useful for testing)."""
    global _gateway_instance
    with _gateway_lock:
        _gateway_instance = gateway
