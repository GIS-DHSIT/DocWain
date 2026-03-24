"""LLM client implementations.

Each client speaks a common duck-typed interface:
    .generate(prompt, ...) -> str
    .generate_with_metadata(prompt, ...) -> Tuple[str, dict]
    .warm_up() -> None  (optional)
"""
from __future__ import annotations

import hashlib
import json
from src.utils.logging_utils import get_logger
import os
import random
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

logger = get_logger(__name__)

# ── Lazy imports from dw_newron to avoid circular deps ─────────────

def _get_metrics_store():
    from src.api.dw_newron import get_metrics_store
    return get_metrics_store()

def _resolve_model_alias(name):
    from src.api.dw_newron import _resolve_model_alias
    return _resolve_model_alias(name)

def _configure_gemini():
    from src.api.dw_newron import configure_gemini
    return configure_gemini()

def _generate_text_gemini(**kwargs):
    from src.api.dw_newron import generate_text
    return generate_text(**kwargs)

def _get_redis_client():
    from src.api.dw_newron import get_redis_client
    return get_redis_client()

def _get_config():
    from src.api.config import Config
    return Config

# ── Thinking mode helpers ──────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def _split_thinking(text: str) -> Tuple[str, str]:
    """Extract <think>...</think> block from response, returning (reasoning, cleaned_text)."""
    match = _THINK_RE.search(text)
    if not match:
        return "", text
    reasoning = match.group(1).strip()
    cleaned = _THINK_RE.sub("", text).strip()
    return reasoning, cleaned

# ── Default client singleton ───────────────────────────────────────

_default_client = None
_default_client_lock = threading.Lock()

def get_default_client():
    """Return the local document processing client.

    Document processing (classification, entity extraction, summarisation)
    always uses the fast local model to avoid burning cloud API quota.
    """
    return get_local_client()


# ── Local client for document processing ──────────────────────────

_local_client = None
_local_client_lock = threading.Lock()

def get_local_client():
    """Return a local Ollama client for document processing (qwen3:14b).

    This client always talks to the LOCAL Ollama instance (no cloud),
    using a lightweight model optimised for fast extraction, classification,
    summarisation and entity extraction during document ingestion.
    """
    global _local_client
    if _local_client is None:
        with _local_client_lock:
            if _local_client is None:
                local_model = os.getenv("OLLAMA_LOCAL_MODEL", "qwen3:14b")
                _local_client = OllamaClient(model_name=local_model)
                # Override to ensure local-only (no cloud auth headers)
                try:
                    import ollama as _ollama
                    import httpx as _httpx
                    _local_client._client = _ollama.Client(
                        host=os.getenv("OLLAMA_LOCAL_HOST", "http://localhost:11434"),
                        timeout=_httpx.Timeout(OllamaClient._OLLAMA_HTTP_TIMEOUT_S),
                    )
                except Exception:
                    pass
                logger.info("Local document processing client ready (model=%s)", local_model)
    return _local_client

# ── OllamaClient ───────────────────────────────────────────────────

class OllamaClient:
    """Handles local Ollama model calls with controlled generation."""

    # HTTP-level timeout for Ollama requests.  When a pipeline-level timeout
    # fires, future.cancel() does NOT kill the HTTP connection — the zombie
    # request keeps running until Ollama finishes.
    # Must be >= LLM_EXTRACT_TIMEOUT_S (90s) to avoid premature cancellation.
    # Qwen3 generates ~2048 tokens (1K thinking + 1K content) at ~45tok/s = 45s
    # + prompt processing overhead, so 100s gives comfortable margin.
    _OLLAMA_HTTP_TIMEOUT_S = 300.0

    def __init__(self, model_name: Optional[str] = None):
        resolved = _resolve_model_alias(model_name) or _resolve_model_alias(os.getenv("OLLAMA_MODEL")) or "qwen3:14b"
        self.model_name = resolved
        if not self.model_name:
            raise ValueError("OLLAMA_MODEL environment variable is not set")
        self.backend = "ollama"
        # Create a client with bounded HTTP timeout to prevent zombie requests
        # Support Ollama Cloud via OLLAMA_HOST + OLLAMA_API env vars
        try:
            import ollama as _ollama
            import httpx as _httpx
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            ollama_api_key = os.getenv("OLLAMA_API", "")
            client_kwargs: Dict[str, Any] = {
                "host": ollama_host,
                "timeout": _httpx.Timeout(self._OLLAMA_HTTP_TIMEOUT_S),
            }
            self._is_cloud = bool(ollama_api_key)
            if ollama_api_key:
                client_kwargs["headers"] = {"Authorization": f"Bearer {ollama_api_key}"}
                logger.info("Ollama Cloud mode enabled (host=%s)", ollama_host)
            self._client = _ollama.Client(**client_kwargs)
        except Exception:
            self._client = None
            self._is_cloud = False
        logger.info("Initialized OllamaClient with model: %s (cloud=%s)", self.model_name, self._is_cloud)

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff: float = 1.0,
        thinking: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        import ollama
        Config = _get_config()
        metrics_store = _get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        generation_options = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.2),
            "top_p": getattr(Config.LLM, "TOP_P", 0.85),
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
            "num_predict": getattr(Config.LLM, "MAX_TOKENS", 4096),
        }
        if options:
            merged = {k: v for k, v in options.items() if v is not None}
            # Map max_tokens to Ollama's num_predict
            if "max_tokens" in merged:
                merged.setdefault("num_predict", merged.pop("max_tokens"))
            generation_options.update(merged)

        # Explicitly set think=False to avoid thinking token overhead.
        # qwen3:14b thinking tokens consume from num_predict budget.
        extra_kwargs: Dict[str, Any] = {"think": thinking}

        last_response: Dict[str, Any] = {}
        for attempt in range(1, max_retries + 1):
            try:
                _gen_fn = self._client.generate if self._client else ollama.generate
                call_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": generation_options,
                }
                # keep_alive is only for local Ollama — cloud returns 500
                if not self._is_cloud:
                    call_kwargs["keep_alive"] = "24h"
                try:
                    response = _gen_fn(**call_kwargs, **extra_kwargs)
                except TypeError:
                    # Older ollama client doesn't support 'think' param
                    response = _gen_fn(**call_kwargs)
                last_response = response or {}
                # Handle both dict and Pydantic GenerateResponse objects
                if hasattr(last_response, "response"):
                    text = (last_response.response or "").strip()
                else:
                    text = (last_response.get("response") or "").strip()
                reasoning = ""
                if thinking and "<think>" in text:
                    reasoning, text = _split_thinking(text)
                if metrics_store.available:
                    latency_ms = (time.time() - request_started) * 1000
                    metrics_store.record(
                        values={"llm_latency_ms": latency_ms},
                        histograms={"llm_latency_ms": latency_ms},
                        model_id=self.model_name,
                    )
                    if attempt > 1:
                        metrics_store.record(
                            counters={"llm_retry_count": attempt - 1},
                            model_id=self.model_name,
                        )
                try:
                    meta = dict(last_response)
                except (TypeError, ValueError):
                    meta = {"model": self.model_name, "backend": "ollama"}
                if reasoning:
                    meta["reasoning"] = reasoning
                return text, meta
            except Exception as exc:
                exc_msg = str(exc).lower()
                is_loading = "loading model" in exc_msg or "llm server loading model" in exc_msg
                if is_loading and attempt == max_retries:
                    # Model is loading into GPU — wait and give it one extra retry
                    logger.warning(
                        "Ollama model loading detected on attempt %d/%d; waiting 15s for model load",
                        attempt, max_retries,
                    )
                    time.sleep(15)
                    try:
                        _gen_fn = self._client.generate if self._client else ollama.generate
                        response = _gen_fn(
                            model=self.model_name,
                            prompt=prompt,
                            options=generation_options,
                            keep_alive="24h",
                            **extra_kwargs,
                        )
                        last_response = response or {}
                        text = (last_response.get("response") or "").strip()
                        reasoning = ""
                        if thinking and "<think>" in text:
                            reasoning, text = _split_thinking(text)
                        meta = dict(last_response)
                        if reasoning:
                            meta["reasoning"] = reasoning
                        logger.info("Ollama model-loading retry succeeded")
                        return text, meta
                    except Exception as retry_exc:
                        logger.warning("Ollama model-loading retry also failed: %s", retry_exc)
                        # Fall through to normal failure path

                logger.warning("Ollama attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error("All Ollama retries failed")
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "", last_response

    def generate(self, prompt: str, max_retries: int = 1, backoff: float = 0.5, **kwargs) -> str:
        # Accept and forward options kwarg (TaskAwareGateway passes it)
        extra = {}
        if "options" in kwargs:
            extra["options"] = kwargs.pop("options")
        text, response = self.generate_with_metadata(prompt, max_retries=max_retries, backoff=backoff, **extra)
        if not text:
            logger.debug("Ollama returned empty response: %s", response)
            return "I don't have enough information in the documents to answer that."
        return text

    def chat_with_metadata(
        self,
        messages: List[Dict[str, str]],
        *,
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff: float = 1.0,
        thinking: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Chat-based generation with proper system/user/assistant role separation.

        Uses ``ollama.chat(messages=...)`` instead of ``ollama.generate(prompt=...)``.
        This gives the model proper role context, improving instruction following
        and evidence separation.
        """
        import ollama
        Config = _get_config()
        request_started = time.time()

        generation_options = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.2),
            "top_p": getattr(Config.LLM, "TOP_P", 0.85),
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
            "num_predict": getattr(Config.LLM, "MAX_TOKENS", 4096),
        }
        if options:
            merged = {k: v for k, v in options.items() if v is not None}
            if "max_tokens" in merged:
                merged.setdefault("num_predict", merged.pop("max_tokens"))
            generation_options.update(merged)

        # Explicitly set think=False to avoid thinking token overhead.
        extra_kwargs: Dict[str, Any] = {"think": thinking}

        last_response: Dict[str, Any] = {}
        for attempt in range(1, max_retries + 1):
            try:
                _chat_fn = self._client.chat if self._client else ollama.chat
                call_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "options": generation_options,
                }
                if not self._is_cloud:
                    call_kwargs["keep_alive"] = "24h"
                try:
                    response = _chat_fn(**call_kwargs, **extra_kwargs)
                except TypeError:
                    response = _chat_fn(**call_kwargs)
                last_response = response or {}
                # Chat API returns message.content instead of response
                # Handle both dict and Pydantic ChatResponse objects
                if hasattr(last_response, "message"):
                    msg = last_response.message or {}
                else:
                    msg = last_response.get("message") or {}
                if hasattr(msg, "content"):
                    text = (msg.content or "").strip()
                else:
                    text = (msg.get("content") or "").strip()
                reasoning = ""
                if thinking and "<think>" in text:
                    reasoning, text = _split_thinking(text)

                # Qwen3 thinking fallback: even with think=False, the model
                # generates thinking tokens that consume num_predict budget.
                # When content is empty but thinking has substance, log the
                # issue but return empty — raw thinking text is ungrounded
                # internal reasoning that can hallucinate.  The pipeline's
                # emergency chunk summary will handle the empty response.
                if not text:
                    thinking_text = (getattr(msg, "thinking", None) or (msg.get("thinking") if isinstance(msg, dict) else "") or "").strip()
                    if thinking_text and len(thinking_text) > 100:
                        logger.warning(
                            "Content empty but thinking has %d chars — returning empty "
                            "(thinking text is ungrounded, pipeline will use fallback)",
                            len(thinking_text),
                        )

                try:
                    meta = dict(last_response)
                except (TypeError, ValueError):
                    meta = {"model": self.model_name, "backend": "ollama"}
                if reasoning:
                    meta["reasoning"] = reasoning
                return text, meta

            except Exception as exc:
                exc_str = str(exc).lower()
                if "model" in exc_str and ("loading" in exc_str or "not found" in exc_str):
                    if attempt <= max_retries:
                        logger.info("Ollama chat model loading, waiting 15s (attempt %d/%d)", attempt, max_retries)
                        time.sleep(15)
                        continue
                logger.warning("Ollama chat attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    raise

        return "", last_response

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Generate with native tool calling support.

        Returns (text, tool_calls, raw_response).
        """
        import ollama
        Config = _get_config()
        generation_options = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.2),
            "num_ctx": 4096,
            "num_predict": getattr(Config.LLM, "MAX_TOKENS", 2048),
        }
        if options:
            generation_options.update({k: v for k, v in options.items() if v is not None})
        try:
            _gen_fn = self._client.generate if self._client else ollama.generate
            raw = _gen_fn(
                model=self.model_name,
                prompt=prompt,
                tools=tools,
                options=generation_options,
                keep_alive="24h",
            )
        except Exception as exc:
            logger.warning("Ollama tool-calling generate failed: %s", exc)
            raise
        raw = raw or {}
        tool_calls = raw.get("tool_calls", [])
        text = (raw.get("response") or "").strip()
        return text, tool_calls, raw

    def generate_stream(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Generate with true token streaming via ollama.generate(stream=True)."""
        import ollama
        Config = _get_config()
        generation_options = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.2),
            "top_p": getattr(Config.LLM, "TOP_P", 0.85),
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
            "num_predict": getattr(Config.LLM, "MAX_TOKENS", 2048),
        }
        if options:
            generation_options.update({k: v for k, v in options.items() if v is not None})
        try:
            _gen_fn = self._client.generate if self._client else ollama.generate
            for chunk in _gen_fn(
                model=self.model_name,
                prompt=prompt,
                options=generation_options,
                stream=True,
                keep_alive="24h",
            ):
                token = chunk.get("response", "") if isinstance(chunk, dict) else getattr(chunk, "response", "")
                if token:
                    yield token
        except Exception as exc:
            logger.warning("Ollama streaming failed: %s", exc)
            raise

    def warm_up(self):
        try:
            self.generate("ping", max_retries=1, backoff=0.0)
        except Exception as exc:
            logger.warning("Ollama warm-up failed (continuing): %s", exc)

# ── RateLimitCooldownError ─────────────────────────────────────────

class RateLimitCooldownError(RuntimeError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.code = 429
        self.retry_after = retry_after

# ── GeminiClient ───────────────────────────────────────────────────

class GeminiClient:
    """Handles Google Gemini API calls with circuit breaker and rate limiting."""

    def __init__(self, model_name: Optional[str] = None):
        Config = _get_config()
        self.api_key = _configure_gemini()
        self.model_name = model_name or Config.Model.GEMINI_MODEL_NAME
        if not self.model_name:
            raise ValueError("Gemini model name is not configured")
        self.generation_config = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.3),
            "top_p": getattr(Config.LLM, "TOP_P", 0.95),
            "top_k": 40,
            "max_output_tokens": getattr(Config.LLM, "MAX_TOKENS", 2048),
        }
        self.backend = "gemini"
        logger.info("Initialized GeminiClient with model: %s", self.model_name)
        self._cache: Dict[str, Tuple[float, str]] = {}
        self._cache_ttl = int(os.getenv("GEMINI_CACHE_TTL_SECONDS", "900"))
        self._circuit_open_until = 0.0
        self._circuit_failures = 0
        self._circuit_threshold = int(os.getenv("GEMINI_CIRCUIT_BREAKER_THRESHOLD", "2"))
        self._circuit_timeout = int(os.getenv("GEMINI_CIRCUIT_BREAKER_TIMEOUT", "60"))
        self._rate_limit_window = int(os.getenv("GEMINI_RATE_LIMIT_WINDOW", "60"))
        self._rate_limit_threshold = int(os.getenv("GEMINI_RATE_LIMIT_THRESHOLD", str(self._circuit_threshold)))
        self._rate_limit_cooldown = int(os.getenv("GEMINI_RATE_LIMIT_COOLDOWN", str(self._circuit_timeout)))
        self._rate_limit_hits = 0
        self._rate_limit_last_ts = 0.0
        self._redis_client = None
        self._cooldown_key = f"llm:cooldown:gemini:{self.model_name}"

    def generate_with_metadata(
        self, prompt: str, max_retries: int = 3, backoff: float = 1.0, options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        # Extract system_instruction from kwargs (passed by gateway)
        system_instruction = kwargs.pop("system_instruction", None)

        # Apply per-call overrides to generation_config
        saved = dict(self.generation_config) if options else None
        if options:
            if "temperature" in options:
                self.generation_config["temperature"] = options["temperature"]
            if "max_tokens" in options:
                self.generation_config["max_output_tokens"] = options["max_tokens"]
            if "top_p" in options:
                self.generation_config["top_p"] = options["top_p"]
        try:
            text = self.generate(prompt, max_retries=max_retries, backoff=backoff,
                                 system_instruction=system_instruction)
        finally:
            if saved is not None:
                self.generation_config = saved
        return text, {"response": text, "backend": "gemini", "model": self.model_name}

    def _get_redis_client_inner(self):
        if self._redis_client is None:
            try:
                self._redis_client = _get_redis_client()
            except Exception:
                self._redis_client = None
        return self._redis_client

    def in_cooldown(self) -> bool:
        now = time.time()
        if self._circuit_open_until and now < self._circuit_open_until:
            return True
        redis_until = self._get_cooldown_until()
        if redis_until and redis_until > now:
            self._circuit_open_until = max(self._circuit_open_until, redis_until)
            return True
        return False

    def _get_cooldown_until(self) -> Optional[float]:
        redis_client = self._get_redis_client_inner()
        if not redis_client:
            return None
        try:
            raw = redis_client.get(self._cooldown_key)
        except Exception:
            return None
        if not raw:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return float(raw)
        except Exception:
            return None

    def _set_cooldown(self, until_ts: float) -> None:
        self._circuit_open_until = max(self._circuit_open_until, until_ts)
        redis_client = self._get_redis_client_inner()
        if redis_client:
            try:
                ttl = max(1, int(until_ts - time.time()))
                redis_client.setex(self._cooldown_key, ttl, str(int(until_ts)))
            except Exception:
                pass

    def _record_rate_limit_hit(self) -> bool:
        now = time.time()
        if now - self._rate_limit_last_ts > self._rate_limit_window:
            self._rate_limit_hits = 0
        self._rate_limit_hits += 1
        self._rate_limit_last_ts = now
        if self._rate_limit_hits >= self._rate_limit_threshold:
            self._set_cooldown(now + self._rate_limit_cooldown)
            return True
        return False

    @staticmethod
    def _extract_status(exc: Exception) -> Optional[int]:
        for attr in ("code", "status", "status_code"):
            value = getattr(exc, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if isinstance(value, int):
                return value
            try:
                if hasattr(value, "value"):
                    return int(value.value)
            except Exception:
                continue
        msg = str(exc)
        if "429" in msg or "too many requests" in msg.lower():
            return 429
        return None

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        from email.utils import parsedate_to_datetime
        from datetime import datetime
        response = getattr(exc, "response", None) or getattr(exc, "resp", None) or getattr(exc, "http_response", None)
        headers = None
        if response is not None:
            headers = getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    if isinstance(retry_after, (int, float)):
                        return float(retry_after)
                    retry_after = str(retry_after).strip()
                    if retry_after.isdigit():
                        return float(retry_after)
                    parsed = parsedate_to_datetime(retry_after)
                    if parsed:
                        delta = (parsed - datetime.now(parsed.tzinfo)).total_seconds()
                        return max(0.0, delta)
                except Exception:
                    return None
        retry_delay = getattr(exc, "retry_delay", None)
        try:
            if retry_delay is not None:
                return float(retry_delay)
        except Exception:
            pass
        return None

    def generate(self, prompt: str, max_retries: int = 1, backoff: float = 0.5,
                 system_instruction: Optional[str] = None) -> str:
        metrics_store = _get_metrics_store()
        request_started = time.time()
        cache_key = hashlib.sha256(
            ((system_instruction or "") + (prompt or "")).encode("utf-8")
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            ts, text = cached
            if (time.time() - ts) <= self._cache_ttl:
                logger.info("Gemini cache hit")
                return text
            self._cache.pop(cache_key, None)
        if self.in_cooldown():
            raise RateLimitCooldownError("Gemini cooldown active; skipping request")

        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        max_retries = max(1, min(int(max_retries or 1), 3))
        total_sleep = 0.0
        max_sleep = float(os.getenv("GEMINI_MAX_BACKOFF_SECONDS", "2.5"))
        total_cap = float(os.getenv("GEMINI_TOTAL_BACKOFF_CAP_SECONDS", "3.5"))
        rate_limit_seen = False
        for attempt in range(1, max_retries + 1):
            try:
                text, response = _generate_text_gemini(
                    api_key=self.api_key,
                    model=self.model_name,
                    prompt=prompt,
                    generation_config=self.generation_config,
                    system_instruction=system_instruction,
                )
                if text:
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                        if attempt > 1:
                            metrics_store.record(
                                counters={"llm_retry_count": attempt - 1},
                                model_id=self.model_name,
                            )
                    self._circuit_failures = 0
                    self._circuit_open_until = 0.0
                    self._rate_limit_hits = 0
                    self._cache[cache_key] = (time.time(), text)
                    return text
                logger.warning("No text in response: %s", response)
                return "I apologize, but I couldn't generate a proper response."
            except Exception as e:
                status = self._extract_status(e)
                retry_after = self._extract_retry_after(e)
                if status == 429 and not rate_limit_seen:
                    rate_limit_seen = True
                    triggered = self._record_rate_limit_hit()
                    logger.warning(
                        "Gemini rate limit encountered (attempt %s/%s)",
                        attempt, max_retries,
                        extra={"stage": "generate", "provider": "gemini", "model": self.model_name, "retry_after": retry_after},
                    )
                    if triggered:
                        raise RateLimitCooldownError("Gemini rate limit cooldown activated", retry_after=self._rate_limit_cooldown)
                elif status is not None:
                    logger.warning(
                        "Gemini API attempt %s/%s failed (status=%s): %s",
                        attempt, max_retries, status, e,
                        extra={"stage": "generate", "provider": "gemini", "model": self.model_name},
                    )
                else:
                    logger.warning(
                        "Gemini API attempt %s/%s failed: %s",
                        attempt, max_retries, e,
                        extra={"stage": "generate", "provider": "gemini", "model": self.model_name},
                    )
                if status in {500, 502, 503, 504}:
                    self._circuit_failures += 1
                    if self._circuit_failures >= self._circuit_threshold:
                        self._set_cooldown(time.time() + self._circuit_timeout)
                if attempt < max_retries:
                    sleep_for = None
                    if retry_after is not None:
                        sleep_for = min(max_sleep, max(0.0, float(retry_after)))
                    else:
                        base = max(0.1, float(backoff))
                        sleep_for = min(max_sleep, base * (2 ** (attempt - 1)))
                        sleep_for += random.uniform(0, sleep_for * 0.25)
                    remaining = max(0.0, total_cap - total_sleep)
                    if sleep_for and remaining > 0:
                        sleep_for = min(sleep_for, remaining)
                        time.sleep(sleep_for)
                        total_sleep += sleep_for
                else:
                    logger.error("All retry attempts failed: %s", e)
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "I apologize, but I encountered an error generating a response."

    def warm_up(self):
        try:
            self.generate("ping", max_retries=1, backoff=0.0)
        except Exception as exc:
            logger.warning("Gemini warm-up failed (continuing): %s", exc)

# ── OpenAICompatibleClient ─────────────────────────────────────────

class OpenAICompatibleClient:
    """Handles OpenAI-compatible local LLM endpoints (vLLM, TGI, etc.)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        Config = _get_config()
        self.endpoint = endpoint or os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        if self.endpoint.rstrip("/").endswith("/v1"):
            self.endpoint = self.endpoint.rstrip("/") + "/chat/completions"
        self.model_name = model_name or os.getenv("LOCAL_LLM_MODEL", "local-model")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "")
        self.temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", str(getattr(Config.LLM, "TEMPERATURE", 0.0))))
        self.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", str(getattr(Config.LLM, "MAX_TOKENS", 2048))))
        self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "30"))
        self.backend = "vllm"
        logger.info("Initialized OpenAICompatibleClient at %s with model %s", self.endpoint, self.model_name)

    def warm_up(self):
        try:
            self.generate("ping", max_retries=1, backoff=0.0)
        except Exception as exc:
            logger.warning("Local LLM warm-up failed (continuing): %s", exc)

    def generate(self, prompt: str, max_retries: int = 1, backoff: float = 0.5, **kwargs) -> str:
        metrics_store = _get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        call_temp = kwargs.get("temperature", self.temperature)
        call_max_tokens = kwargs.get("max_tokens", self.max_tokens)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": call_temp,
            "max_tokens": call_max_tokens,
        }
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, max_retries + 1):
            try:
                req = request.Request(self.endpoint, data=data, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                response = json.loads(body)
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or choice.get("text") or ""
                text = text.strip()
                if text:
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                        if attempt > 1:
                            metrics_store.record(
                                counters={"llm_retry_count": attempt - 1},
                                model_id=self.model_name,
                            )
                    return text
                logger.warning("No text in local LLM response: %s", response)
                return "I apologize, but I couldn't generate a proper response."
            except (HTTPError, URLError, ValueError) as e:
                logger.warning("Local LLM attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error("All local LLM retry attempts failed: %s", e)
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "I apologize, but I encountered an error generating a response."

    def generate_with_metadata(
        self, prompt: str, *, options: Optional[Dict[str, Any]] = None, max_retries: int = 3, backoff: float = 1.0,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate with metadata for pipeline compatibility."""
        call_kwargs = {}
        if options:
            if "temperature" in options:
                call_kwargs["temperature"] = options["temperature"]
            if "max_tokens" in options or "num_predict" in options:
                val = options.get("max_tokens")
                if val is None:
                    val = options.get("num_predict")
                call_kwargs["max_tokens"] = val
            if "top_p" in options:
                call_kwargs["top_p"] = options["top_p"]
        text = self.generate(prompt, max_retries=max_retries, backoff=backoff, **call_kwargs)
        return text, {"response": text, "model": self.model_name, "backend": "vllm"}

# ── _LLMClientWrapper (semaphore-based concurrency control) ────────

class LLMClientWrapper:
    """Wraps any LLM client with a semaphore for concurrency control."""

    def __init__(self, client, semaphore):
        self._client = client
        self._semaphore = semaphore

    def __getattr__(self, item):
        return getattr(self._client, item)

    def generate(self, *args, **kwargs):
        with self._semaphore:
            return self._client.generate(*args, **kwargs)

    def generate_with_metadata(self, *args, **kwargs):
        with self._semaphore:
            if hasattr(self._client, "generate_with_metadata"):
                return self._client.generate_with_metadata(*args, **kwargs)
            text = self._client.generate(*args, **kwargs)
            return text, {"response": text}

# ── ResilientLLMClient ─────────────────────────────────────────────

class ResilientLLMClient:
    """Fallback to a secondary client when the primary fails with rate/timeout errors."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.model_name = getattr(primary, "model_name", None) or (getattr(fallback, "model_name", None) if fallback else None)
        self.backend = getattr(primary, "backend", None) or "unknown"

    @staticmethod
    def _should_fallback(exc: Exception) -> bool:
        status = getattr(exc, "code", None) or getattr(exc, "status", None)
        if status in {408, 429, 500, 502, 503, 504}:
            return True
        msg = str(exc).lower()
        return "timeout" in msg or "timed out" in msg

    def _primary_in_cooldown(self) -> bool:
        primary = self.primary
        if not primary:
            return False
        checker = getattr(primary, "in_cooldown", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    def generate(self, *args, **kwargs):
        if self._primary_in_cooldown() and self.fallback:
            logger.warning(
                "Primary LLM in cooldown; using fallback",
                extra={"stage": "generate", "provider": getattr(self.primary, "backend", "unknown")},
            )
            return self.fallback.generate(*args, **kwargs)
        try:
            return self.primary.generate(*args, **kwargs)
        except Exception as exc:
            if self._should_fallback(exc) and self.fallback:
                logger.warning(
                    "Primary LLM failed; falling back: %s", exc,
                    extra={"stage": "generate", "provider": getattr(self.primary, "backend", "unknown")},
                )
                return self.fallback.generate(*args, **kwargs)
            raise

    def generate_with_metadata(self, *args, **kwargs):
        if self._primary_in_cooldown() and self.fallback:
            logger.warning(
                "Primary LLM in cooldown; using fallback",
                extra={"stage": "generate", "provider": getattr(self.primary, "backend", "unknown")},
            )
            if hasattr(self.fallback, "generate_with_metadata"):
                return self.fallback.generate_with_metadata(*args, **kwargs)
            text = self.fallback.generate(*args, **kwargs)
            return text, {"response": text}
        try:
            if hasattr(self.primary, "generate_with_metadata"):
                return self.primary.generate_with_metadata(*args, **kwargs)
            text = self.primary.generate(*args, **kwargs)
            return text, {"response": text}
        except Exception as exc:
            if self._should_fallback(exc) and self.fallback:
                logger.warning(
                    "Primary LLM failed; falling back: %s", exc,
                    extra={"stage": "generate", "provider": getattr(self.primary, "backend", "unknown")},
                )
                if hasattr(self.fallback, "generate_with_metadata"):
                    return self.fallback.generate_with_metadata(*args, **kwargs)
                text = self.fallback.generate(*args, **kwargs)
                return text, {"response": text}
            raise

    def warm_up(self):
        warm = getattr(self.primary, "warm_up", None)
        if callable(warm):
            try:
                warm()
            except Exception:
                pass
        if self.fallback:
            warm_fb = getattr(self.fallback, "warm_up", None)
            if callable(warm_fb):
                try:
                    warm_fb()
                except Exception:
                    pass

# ── OpenAIClient (Azure OpenAI / GPT-4o) ──────────────────────────

class OpenAIClient:
    """Azure OpenAI client with circuit breaker."""

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        deployment: str = "gpt-4o",
        api_version: str = "2024-05-01-preview",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self.model_name = f"azure/{deployment}"
        self.backend = "azure_openai"
        self._circuit_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_threshold = 3
        self._circuit_cooldown = 60

    def in_cooldown(self) -> bool:
        return time.time() < self._circuit_open_until

    def generate(self, prompt: str, max_retries: int = 2, backoff: float = 1.0) -> str:
        text, _ = self.generate_with_metadata(prompt, max_retries=max_retries, backoff=backoff)
        return text or "I couldn't generate a response."

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        backoff: float = 1.0,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        if self.in_cooldown():
            raise RateLimitCooldownError("Azure OpenAI circuit breaker open")
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint/key not configured")

        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        opts = options or {}
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": opts.get("temperature", 0.3),
            "max_tokens": opts.get("max_tokens", opts.get("num_predict", 4096)),
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        for attempt in range(1, max_retries + 1):
            try:
                req = request.Request(url, data=data, headers=headers, method="POST")
                with request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choice = (body.get("choices") or [{}])[0]
                text = (choice.get("message", {}).get("content") or "").strip()
                self._circuit_failures = 0
                return text, {"response": text, "model": self.deployment, "backend": self.backend, "raw": body}
            except Exception as exc:
                logger.warning("Azure OpenAI attempt %d/%d failed: %s", attempt, max_retries, exc)
                self._circuit_failures += 1
                if self._circuit_failures >= self._circuit_threshold:
                    self._circuit_open_until = time.time() + self._circuit_cooldown
                    logger.warning("Azure OpenAI circuit breaker opened for %ds", self._circuit_cooldown)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    raise
        return "", {"backend": self.backend, "error": "max_retries_exceeded"}

    def chat_with_metadata(
        self,
        messages: list,
        *,
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        backoff: float = 1.0,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Chat-style API — accepts list of {role, content} messages."""
        if self.in_cooldown():
            raise RateLimitCooldownError("Azure OpenAI circuit breaker open")
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint/key not configured")

        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        opts = options or {}
        payload = {
            "messages": messages,
            "temperature": opts.get("temperature", 0.3),
            "max_tokens": opts.get("max_tokens", opts.get("num_predict", 4096)),
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        for attempt in range(1, max_retries + 1):
            try:
                req = request.Request(url, data=data, headers=headers, method="POST")
                with request.urlopen(req, timeout=90) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choice = (body.get("choices") or [{}])[0]
                text = (choice.get("message", {}).get("content") or "").strip()
                self._circuit_failures = 0
                return text, {"response": text, "model": self.deployment, "backend": self.backend, "raw": body}
            except Exception as exc:
                logger.warning("Azure OpenAI chat attempt %d/%d failed: %s", attempt, max_retries, exc)
                self._circuit_failures += 1
                if self._circuit_failures >= self._circuit_threshold:
                    self._circuit_open_until = time.time() + self._circuit_cooldown
                    logger.warning("Azure OpenAI circuit breaker opened for %ds", self._circuit_cooldown)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    raise
        return "", {"backend": self.backend, "error": "max_retries_exceeded"}

    def warm_up(self):
        pass  # No warm-up needed for cloud client

# ── ClaudeClient (Anthropic Claude) ───────────────────────────────

class ClaudeClient:
    """Anthropic Claude API client with circuit breaker."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key
        self.model_name = model
        self.backend = "claude"
        self._circuit_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_threshold = 3
        self._circuit_cooldown = 60

    def in_cooldown(self) -> bool:
        return time.time() < self._circuit_open_until

    def generate(self, prompt: str, max_retries: int = 2, backoff: float = 1.0) -> str:
        text, _ = self.generate_with_metadata(prompt, max_retries=max_retries, backoff=backoff)
        return text or "I couldn't generate a response."

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        backoff: float = 1.0,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        if self.in_cooldown():
            raise RateLimitCooldownError("Claude circuit breaker open")
        if not self.api_key:
            raise ValueError("Claude API key not configured")

        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model_name,
            "max_tokens": (options or {}).get("num_predict", 2048),
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        for attempt in range(1, max_retries + 1):
            try:
                req = request.Request(url, data=data, headers=headers, method="POST")
                with request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content_blocks = body.get("content", [])
                text = ""
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
                text = text.strip()
                self._circuit_failures = 0
                return text, {"response": text, "model": self.model_name, "backend": self.backend, "raw": body}
            except Exception as exc:
                logger.warning("Claude attempt %d/%d failed: %s", attempt, max_retries, exc)
                self._circuit_failures += 1
                if self._circuit_failures >= self._circuit_threshold:
                    self._circuit_open_until = time.time() + self._circuit_cooldown
                    logger.warning("Claude circuit breaker opened for %ds", self._circuit_cooldown)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    raise
        return "", {"backend": "claude", "error": "max_retries_exceeded"}

    def warm_up(self):
        pass  # No warm-up needed for cloud client
