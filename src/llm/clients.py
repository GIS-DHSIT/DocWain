"""LLM client implementations.

Each client speaks a common duck-typed interface:
    .generate(prompt, ...) -> str
    .generate_with_metadata(prompt, ...) -> Tuple[str, dict]
    .warm_up() -> None  (optional)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


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


# ── OllamaClient ───────────────────────────────────────────────────

class OllamaClient:
    """Handles local Ollama model calls with controlled generation."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = _resolve_model_alias(model_name) or os.getenv("OLLAMA_MODEL", "gpt-oss:latest")
        if not self.model_name:
            raise ValueError("OLLAMA_MODEL environment variable is not set")
        self.backend = "ollama"
        logger.info("Initialized OllamaClient with model: %s", self.model_name)

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 1,
        backoff: float = 0.5,
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
            "num_ctx": 4096,
            "num_predict": getattr(Config.LLM, "MAX_TOKENS", 2048),
        }
        if options:
            generation_options.update({k: v for k, v in options.items() if v is not None})

        last_response: Dict[str, Any] = {}
        for attempt in range(1, max_retries + 1):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=generation_options,
                )
                last_response = response or {}
                text = (last_response.get("response") or "").strip()
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
                return text, last_response
            except Exception as exc:
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

    def generate(self, prompt: str, max_retries: int = 3, backoff: float = 1.0) -> str:
        text, response = self.generate_with_metadata(prompt, max_retries=max_retries, backoff=backoff)
        if not text:
            logger.warning("Ollama returned empty response: %s", response)
            return "I don't have enough information in the documents to answer that."
        return text

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
    ) -> Tuple[str, Dict[str, Any]]:
        text = self.generate(prompt, max_retries=max_retries, backoff=backoff)
        return text, {"response": text}

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

    def generate(self, prompt: str, max_retries: int = 3, backoff: float = 1.0) -> str:
        metrics_store = _get_metrics_store()
        request_started = time.time()
        cache_key = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
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

    def generate(self, prompt: str, max_retries: int = 3, backoff: float = 1.0) -> str:
        metrics_store = _get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
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
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate with metadata for pipeline compatibility."""
        text = self.generate(prompt, max_retries=max_retries, backoff=backoff)
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
