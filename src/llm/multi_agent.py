"""Multi-agent LLM gateway — routes pipeline calls to role-specific Ollama models.

Each pipeline role (classifier, extractor, generator, verifier) maps to a
specialized local model selected for that task:
  - classifier  → llama3.2  (2GB, fast intent/domain classification)
  - extractor   → mistral   (4.1GB, structured extraction)
  - generator   → gpt-oss   (13GB, response synthesis)
  - verifier    → deepseek-r1 (4.7GB, chain-of-thought grounding)

Duck-types the same interface as LLMGateway so existing code works unchanged.
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    GENERATOR = "generator"
    VERIFIER = "verifier"
    DEFAULT = "default"


# Default model assignments per role
_DEFAULT_ROLE_MODELS: Dict[str, str] = {
    AgentRole.CLASSIFIER: "llama3.2:latest",
    AgentRole.EXTRACTOR: "mistral:latest",
    AgentRole.GENERATOR: "gpt-oss:latest",
    AgentRole.VERIFIER: "deepseek-r1:latest",
    AgentRole.DEFAULT: "llama3.2:latest",
}


class _RoleStats:
    """Thread-safe per-role call statistics."""
    __slots__ = ("calls", "errors", "total_latency_ms", "_lock")

    def __init__(self):
        self.calls = 0
        self.errors = 0
        self.total_latency_ms = 0.0
        self._lock = threading.Lock()

    def record_call(self, latency_ms: float):
        with self._lock:
            self.calls += 1
            self.total_latency_ms += latency_ms

    def record_error(self):
        with self._lock:
            self.errors += 1

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            avg = (self.total_latency_ms / self.calls) if self.calls else 0.0
            return {
                "calls": self.calls,
                "errors": self.errors,
                "avg_latency_ms": round(avg, 1),
            }


class MultiAgentGateway:
    """Routes LLM calls to role-specific Ollama models.

    Parameters
    ----------
    role_models : dict, optional
        Mapping of AgentRole → Ollama model name. Missing roles use defaults.
    fallback_gateway : optional
        Singleton LLMGateway to fall back to on per-role failure.
    """

    def __init__(
        self,
        role_models: Optional[Dict[str, str]] = None,
        fallback_gateway: Any = None,
    ):
        self._role_models: Dict[str, str] = {**_DEFAULT_ROLE_MODELS}
        if role_models:
            self._role_models.update(role_models)

        self._fallback = fallback_gateway
        self._clients: Dict[str, Any] = {}
        self._client_locks: Dict[str, threading.Lock] = {
            role: threading.Lock() for role in AgentRole
        }
        self._stats: Dict[str, _RoleStats] = {
            role: _RoleStats() for role in AgentRole
        }
        self._created_at = time.time()
        self.backend = "multi_agent"
        self.model_name = "multi-agent"

        logger.info(
            "MultiAgentGateway initialized: %s",
            {r: m for r, m in self._role_models.items()},
        )

    def _get_client(self, role: str) -> Any:
        """Lazy-create an OllamaClient for the given role."""
        if role in self._clients:
            return self._clients[role]

        lock = self._client_locks.get(role) or self._client_locks[AgentRole.DEFAULT]
        with lock:
            if role in self._clients:
                return self._clients[role]
            model_name = self._role_models.get(role, self._role_models[AgentRole.DEFAULT])
            try:
                from src.llm.clients import OllamaClient
                client = OllamaClient(model_name=model_name)
                self._clients[role] = client
                logger.info("Created OllamaClient for role=%s model=%s", role, model_name)
                return client
            except Exception as exc:
                logger.warning("Failed to create client for role=%s: %s", role, exc)
                return None

    def generate_for_role(
        self,
        role: str,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate text using the model assigned to the given role."""
        start = time.time()
        stats = self._stats.get(role) or self._stats[AgentRole.DEFAULT]

        client = self._get_client(role)
        if client is not None:
            try:
                text = client.generate(prompt, **kwargs)
                stats.record_call((time.time() - start) * 1000)
                return text
            except Exception as exc:
                stats.record_error()
                logger.warning("Role %s generation failed: %s — trying fallback", role, exc)

        # Fallback to DEFAULT role
        if role != AgentRole.DEFAULT:
            default_client = self._get_client(AgentRole.DEFAULT)
            if default_client is not None:
                try:
                    text = default_client.generate(prompt, **kwargs)
                    stats.record_call((time.time() - start) * 1000)
                    return text
                except Exception as exc:
                    stats.record_error()
                    logger.warning("DEFAULT role also failed: %s", exc)

        # Ultimate fallback to singleton gateway
        if self._fallback is not None:
            text = self._fallback.generate(prompt, **kwargs)
            stats.record_call((time.time() - start) * 1000)
            return text

        return ""

    def generate_with_metadata_for_role(
        self,
        role: str,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text + metadata using the model assigned to the given role."""
        start = time.time()
        stats = self._stats.get(role) or self._stats[AgentRole.DEFAULT]

        client = self._get_client(role)
        if client is not None:
            try:
                if hasattr(client, "generate_with_metadata"):
                    text, meta = client.generate_with_metadata(prompt, options=options, **kwargs)
                else:
                    text = client.generate(prompt, **kwargs)
                    meta = {"response": text}
                meta["agent_role"] = role
                meta["agent_model"] = self._role_models.get(role, "unknown")
                stats.record_call((time.time() - start) * 1000)
                return text, meta
            except Exception as exc:
                stats.record_error()
                logger.warning("Role %s generation failed: %s — trying fallback", role, exc)

        # Fallback chain: DEFAULT → singleton gateway
        if role != AgentRole.DEFAULT:
            default_client = self._get_client(AgentRole.DEFAULT)
            if default_client is not None:
                try:
                    if hasattr(default_client, "generate_with_metadata"):
                        text, meta = default_client.generate_with_metadata(prompt, options=options, **kwargs)
                    else:
                        text = default_client.generate(prompt, **kwargs)
                        meta = {"response": text}
                    meta["agent_role"] = AgentRole.DEFAULT
                    stats.record_call((time.time() - start) * 1000)
                    return text, meta
                except Exception as exc:
                    stats.record_error()
                    logger.warning("DEFAULT role also failed: %s", exc)

        if self._fallback is not None:
            if hasattr(self._fallback, "generate_with_metadata"):
                text, meta = self._fallback.generate_with_metadata(prompt, options=options, **kwargs)
            else:
                text = self._fallback.generate(prompt, **kwargs)
                meta = {"response": text}
            meta["agent_role"] = "fallback"
            stats.record_call((time.time() - start) * 1000)
            return text, meta

        return "", {"agent_role": role, "error": "no_client_available"}

    # ── Duck-typed interface (same as LLMGateway) ──────────────────

    def generate(self, prompt: str, **kwargs) -> str:
        """Default generation uses GENERATOR role."""
        return self.generate_for_role(AgentRole.GENERATOR, prompt, **kwargs)

    def generate_with_metadata(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Default generation with metadata uses GENERATOR role."""
        return self.generate_with_metadata_for_role(AgentRole.GENERATOR, prompt, **kwargs)

    # ── Convenience methods ────────────────────────────────────────

    def classify(self, prompt: str, **kwargs) -> str:
        """Classification uses the fast CLASSIFIER role (llama3.2)."""
        return self.generate_for_role(AgentRole.CLASSIFIER, prompt, max_retries=1, backoff=0.2, **kwargs)

    def extract(self, prompt: str, **kwargs) -> str:
        """Extraction uses the EXTRACTOR role (mistral)."""
        return self.generate_for_role(AgentRole.EXTRACTOR, prompt, **kwargs)

    def verify(self, prompt: str, **kwargs) -> str:
        """Verification uses the VERIFIER role (deepseek-r1)."""
        return self.generate_for_role(AgentRole.VERIFIER, prompt, **kwargs)

    # ── Stats & health ─────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return per-role call statistics."""
        return {
            "roles": {role: stats.to_dict() for role, stats in self._stats.items()},
            "models": dict(self._role_models),
            "active_clients": list(self._clients.keys()),
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }

    def get_role_model(self, role: str) -> str:
        """Return the model name assigned to a role."""
        return self._role_models.get(role, self._role_models.get(AgentRole.DEFAULT, "unknown"))

    def list_roles(self) -> List[str]:
        """Return all available role names."""
        return [r.value for r in AgentRole]

    def warm_up(self):
        """No-op — models are loaded on demand."""
        pass


def create_multi_agent_gateway(
    fallback_gateway: Any = None,
) -> MultiAgentGateway:
    """Factory: creates a MultiAgentGateway from Config.MultiAgent settings."""
    from src.api.config import Config

    ma_cfg = getattr(Config, "MultiAgent", None)
    if ma_cfg is None:
        return MultiAgentGateway(fallback_gateway=fallback_gateway)

    role_models = {
        AgentRole.CLASSIFIER: getattr(ma_cfg, "CLASSIFIER_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.CLASSIFIER]),
        AgentRole.EXTRACTOR: getattr(ma_cfg, "EXTRACTOR_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.EXTRACTOR]),
        AgentRole.GENERATOR: getattr(ma_cfg, "GENERATOR_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.GENERATOR]),
        AgentRole.VERIFIER: getattr(ma_cfg, "VERIFIER_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.VERIFIER]),
        AgentRole.DEFAULT: getattr(ma_cfg, "DEFAULT_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.DEFAULT]),
    }

    return MultiAgentGateway(role_models=role_models, fallback_gateway=fallback_gateway)
