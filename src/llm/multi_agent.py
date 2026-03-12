"""Multi-agent LLM gateway — routes pipeline calls to role-specific Ollama models.

Each pipeline role (classifier, extractor, generator, verifier) maps to a
specialized local model selected for that task:
  - classifier  → llama3.2  (2GB, fast intent/domain classification)
  - extractor   → mistral   (4.1GB, structured extraction)
  - generator   → DocWain-Agent   (13GB, response synthesis)
  - verifier    → deepseek-r1 (4.7GB, chain-of-thought grounding)

Duck-types the same interface as LLMGateway so existing code works unchanged.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

class AgentRole(str, Enum):
    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    GENERATOR = "generator"
    VERIFIER = "verifier"
    REASONER = "reasoner"
    VISION = "vision"
    DEFAULT = "default"

# Default model assignments per role — MoE routing
# CRITICAL: On a single T4 16GB GPU, Ollama can only keep ONE large model loaded.
# Routing CLASSIFIER/DEFAULT to a different model than GENERATOR triggers model
# swaps (500 "loading model" errors + 10-30s swap delay).  Route ALL roles to
# DocWain-Agent to eliminate swap contention.  lfm2.5-thinking is only used for
# REASONER/VERIFIER (infrequent, can absorb the swap cost).
_DEFAULT_ROLE_MODELS: Dict[str, str] = {
    AgentRole.CLASSIFIER: "DocWain-Agent:latest",            # Same as generator — no model swap
    AgentRole.EXTRACTOR: "DocWain-Agent:latest",             # Same as generator — no model swap
    AgentRole.GENERATOR: "DocWain-Agent:latest",
    AgentRole.VERIFIER: "DocWain-Agent:latest",              # Avoid swap on verification
    AgentRole.REASONER: "DocWain-Agent:latest",              # Avoid swap on reasoning
    AgentRole.VISION: "glm-ocr:latest",                # Vision/OCR sub-agent
    AgentRole.DEFAULT: "DocWain-Agent:latest",               # Default — no model swap
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
        """Classification uses CLASSIFIER role (DocWain-Agent — same model, no swap)."""
        return self.generate_for_role(AgentRole.CLASSIFIER, prompt, max_retries=2, backoff=0.5, **kwargs)

    def extract(self, prompt: str, **kwargs) -> str:
        """Extraction uses the EXTRACTOR role (mistral)."""
        return self.generate_for_role(AgentRole.EXTRACTOR, prompt, **kwargs)

    def verify(self, prompt: str, **kwargs) -> str:
        """Verification uses the VERIFIER role (lfm2.5-thinking)."""
        return self.generate_for_role(AgentRole.VERIFIER, prompt, **kwargs)

    def reason(self, prompt: str, **kwargs) -> str:
        """Reasoning uses the REASONER role (lfm2.5-thinking)."""
        return self.generate_for_role(AgentRole.REASONER, prompt, **kwargs)

    def analyze_vision(self, prompt: str, *, images: Optional[List[Any]] = None, **kwargs) -> str:
        """Vision analysis uses the VISION role (glm-ocr).

        If ``images`` is provided, passes them to the Ollama images parameter
        for multimodal analysis.
        """
        if images:
            kwargs["images"] = images
        return self.generate_for_role(AgentRole.VISION, prompt, **kwargs)

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

class TaskAwareGateway(MultiAgentGateway):
    """Extends MultiAgentGateway with task-type-aware model routing.

    When a ``task_scope`` context is active (via thread-local), ``generate()``
    automatically selects the optimal model for the current task.  Otherwise
    falls back to the standard role-based routing.
    """

    def __init__(
        self,
        router: Any,
        fallback_gateway: Any = None,
        role_models: Optional[Dict[str, str]] = None,
    ):
        super().__init__(role_models=role_models, fallback_gateway=fallback_gateway)
        self._router = router
        self._task_stats: Dict[str, _RoleStats] = {}
        self._model_clients: Dict[str, Any] = {}
        self._model_client_lock = threading.Lock()
        self.backend = "task_aware"
        self.model_name = "task-aware-multi-model"
        logger.info("TaskAwareGateway initialized with TaskRouter")

    def _get_client_by_model(self, model_name: str) -> Any:
        """Lazy-create an OllamaClient keyed by model name."""
        if model_name in self._model_clients:
            return self._model_clients[model_name]

        with self._model_client_lock:
            if model_name in self._model_clients:
                return self._model_clients[model_name]
            try:
                from src.llm.clients import OllamaClient
                client = OllamaClient(model_name=model_name)
                self._model_clients[model_name] = client
                logger.info("TaskAwareGateway created OllamaClient for model=%s", model_name)
                return client
            except Exception as exc:
                logger.warning("TaskAwareGateway failed to create client for model=%s: %s", model_name, exc)
                return None

    def _get_or_create_task_stats(self, task_value: str) -> _RoleStats:
        if task_value not in self._task_stats:
            self._task_stats[task_value] = _RoleStats()
        return self._task_stats[task_value]

    def generate_for_task(self, task: Any, prompt: str, **kwargs) -> str:
        """Generate text using the model selected for *task*."""
        from src.llm.task_router import TaskType
        if not isinstance(task, TaskType):
            task = TaskType(task)

        start = time.time()
        stats = self._get_or_create_task_stats(task.value)
        model = self._router.select_model(task)
        options = {**self._router.get_options(task), **kwargs.pop("options", {})}

        client = self._get_client_by_model(model)
        if client is not None:
            try:
                text = client.generate(prompt, options=options, **kwargs)
                stats.record_call((time.time() - start) * 1000)
                logger.debug("TaskRouter: %s → %s (%dms)", task.value, model,
                             int((time.time() - start) * 1000))
                return text
            except Exception as exc:
                stats.record_error()
                logger.warning("Task %s (model=%s) failed: %s — trying fallback", task.value, model, exc)

        # Fallback to parent's GENERATOR role
        if self._fallback is not None:
            text = self._fallback.generate(prompt, **kwargs)
            stats.record_call((time.time() - start) * 1000)
            return text

        return super().generate(prompt, **kwargs)

    def generate_with_metadata_for_task(
        self, task: Any, prompt: str, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text + metadata using the model selected for *task*."""
        from src.llm.task_router import TaskType
        if not isinstance(task, TaskType):
            task = TaskType(task)

        start = time.time()
        stats = self._get_or_create_task_stats(task.value)
        model = self._router.select_model(task)
        options = {**self._router.get_options(task), **kwargs.pop("options", {})}

        client = self._get_client_by_model(model)
        if client is not None:
            try:
                if hasattr(client, "generate_with_metadata"):
                    text, meta = client.generate_with_metadata(prompt, options=options, **kwargs)
                    if not isinstance(meta, dict):
                        meta = {"_raw": str(meta)}
                else:
                    text = client.generate(prompt, options=options, **kwargs)
                    meta = {"response": text}
                meta["task_type"] = task.value
                meta["task_model"] = model
                stats.record_call((time.time() - start) * 1000)
                return text, meta
            except Exception as exc:
                stats.record_error()
                logger.warning("Task %s (model=%s) failed: %s", task.value, model, exc)

        # Fallback
        if self._fallback is not None:
            if hasattr(self._fallback, "generate_with_metadata"):
                text, meta = self._fallback.generate_with_metadata(prompt, **kwargs)
                if not isinstance(meta, dict):
                    meta = {"_raw": str(meta)}
            else:
                text = self._fallback.generate(prompt, **kwargs)
                meta = {"response": text}
            meta["task_type"] = task.value
            meta["task_model"] = "fallback"
            stats.record_call((time.time() - start) * 1000)
            return text, meta

        return "", {"task_type": task.value, "error": "no_client_available"}

    # ── Override duck-typed interface to use task context ──────────

    def generate(self, prompt: str, **kwargs) -> str:
        from src.llm.task_router import get_current_task
        task = get_current_task()
        if task is not None:
            return self.generate_for_task(task, prompt, **kwargs)
        return super().generate(prompt, **kwargs)

    def generate_with_metadata(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        from src.llm.task_router import get_current_task
        task = get_current_task()
        if task is not None:
            return self.generate_with_metadata_for_task(task, prompt, **kwargs)
        return super().generate_with_metadata(prompt, **kwargs)

    # ── Stats ─────────────────────────────────────────────────────

    def get_task_stats(self) -> Dict[str, Any]:
        """Return per-task call statistics."""
        return {t: s.to_dict() for t, s in self._task_stats.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Return combined role + task statistics."""
        base = super().get_stats()
        base["task_routing"] = self.get_task_stats()
        base["model_clients"] = list(self._model_clients.keys())
        return base

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
        AgentRole.REASONER: getattr(ma_cfg, "REASONER_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.REASONER]),
        AgentRole.VISION: getattr(ma_cfg, "VISION_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.VISION]),
        AgentRole.DEFAULT: getattr(ma_cfg, "DEFAULT_MODEL", _DEFAULT_ROLE_MODELS[AgentRole.DEFAULT]),
    }

    return MultiAgentGateway(role_models=role_models, fallback_gateway=fallback_gateway)
