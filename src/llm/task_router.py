"""Task-aware model routing — selects the optimal Ollama model for each LLM task.

Maps 12 task types to ordered model preference lists.  ``TaskRouter`` walks the
preference list and returns the first model available in the ``ModelRegistry``.
A thread-local ``task_scope()`` context manager lets callers declare the current
task so the gateway can route transparently.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import os
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

from src.llm.model_registry import ModelRegistry, _match_family

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Task type taxonomy
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    QUERY_REWRITE = "query_rewrite"
    INTENT_PARSE = "intent_parse"
    RESPONSE_GENERATION = "response_generation"
    STRUCTURED_EXTRACTION = "structured_extraction"
    TOOL_EXECUTION = "tool_execution"
    ANSWER_JUDGING = "answer_judging"
    GROUNDING_VERIFY = "grounding_verify"
    CONTENT_GENERATION = "content_generation"
    QUERY_CLASSIFICATION = "query_classification"
    CONVERSATION_SUMMARY = "conversation_summary"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    COMPLEX_EXTRACTION = "complex_extraction"
    AGENT_REASONING = "agent_reasoning"
    GENERAL = "general"

# ---------------------------------------------------------------------------
# Ordered preference lists (first available wins)
# ---------------------------------------------------------------------------

_TASK_MODEL_PREFERENCES: Dict[TaskType, List[str]] = {
    # --- Generation tasks: docwain-agent first (1.2B too small for long text) ---
    TaskType.RESPONSE_GENERATION:    ["docwain-agent", "gemma2", "mistral"],
    TaskType.CONTENT_GENERATION:     ["docwain-agent", "gemma2", "mistral"],
    TaskType.COMPLEX_EXTRACTION:     ["docwain-agent", "mistral", "gemma2"],
    TaskType.TOOL_EXECUTION:         ["docwain-agent", "mistral", "gemma2"],
    TaskType.STRUCTURED_EXTRACTION:  ["docwain-agent", "mistral", "gemma2"],
    TaskType.DOCUMENT_UNDERSTANDING: ["docwain-agent", "mistral", "gemma2"],
    TaskType.QUERY_REWRITE:          ["docwain-agent", "mistral", "llama3.2"],
    # --- Reasoning/analysis tasks: docwain-agent first to avoid model swap contention ---
    # On T4 16GB, model swaps cost 10-30s + cause 500 errors. Keep everything
    # on DocWain-Agent until a second GPU is available.
    TaskType.ANSWER_JUDGING:         ["docwain-agent", "lfm2.5-thinking", "deepseek-r1"],
    TaskType.GROUNDING_VERIFY:       ["docwain-agent", "lfm2.5-thinking", "deepseek-r1"],
    TaskType.AGENT_REASONING:        ["docwain-agent", "lfm2.5-thinking", "deepseek-r1"],
    TaskType.QUERY_CLASSIFICATION:   ["docwain-agent", "lfm2.5-thinking", "llama3.2"],
    TaskType.INTENT_PARSE:           ["docwain-agent", "lfm2.5-thinking", "mistral"],
    TaskType.CONVERSATION_SUMMARY:   ["docwain-agent", "lfm2.5-thinking", "gemma2"],
    # --- General fallback ---
    TaskType.GENERAL:                ["docwain-agent", "mistral", "gemma2"],
}

# ---------------------------------------------------------------------------
# Per-task generation parameter overrides
# ---------------------------------------------------------------------------

_TASK_OPTIONS: Dict[TaskType, Dict[str, Any]] = {
    TaskType.QUERY_REWRITE:         {"temperature": 0.1,  "num_predict": 96,   "num_ctx": 4096},
    TaskType.INTENT_PARSE:          {"temperature": 0.0,  "num_predict": 256,  "num_ctx": 4096},
    TaskType.RESPONSE_GENERATION:   {"temperature": 0.3,  "num_predict": 8192, "num_ctx": 16384},
    TaskType.STRUCTURED_EXTRACTION: {"temperature": 0.05, "num_predict": 8192, "num_ctx": 16384},
    TaskType.TOOL_EXECUTION:        {"temperature": 0.2,  "num_predict": 4096, "num_ctx": 8192},
    TaskType.ANSWER_JUDGING:        {"temperature": 0.05, "num_predict": 2048, "num_ctx": 8192},
    TaskType.GROUNDING_VERIFY:      {"temperature": 0.05, "num_predict": 2048, "num_ctx": 8192},
    TaskType.CONTENT_GENERATION:    {"temperature": 0.7,  "num_predict": 8192, "num_ctx": 16384},
    TaskType.QUERY_CLASSIFICATION:  {"temperature": 0.0,  "num_predict": 512,  "num_ctx": 4096},
    TaskType.CONVERSATION_SUMMARY:  {"temperature": 0.2,  "num_predict": 1024, "num_ctx": 4096},
    TaskType.DOCUMENT_UNDERSTANDING: {"temperature": 0.1, "num_predict": 4096, "num_ctx": 8192},
    TaskType.COMPLEX_EXTRACTION:    {"temperature": 0.2, "num_predict": 8192, "num_ctx": 16384},
    TaskType.AGENT_REASONING:       {"temperature": 0.05, "num_predict": 4096, "num_ctx": 8192},
    TaskType.GENERAL:               {"temperature": 0.3,  "num_predict": 8192, "num_ctx": 8192},
}

# ---------------------------------------------------------------------------
# Thread-local task context
# ---------------------------------------------------------------------------

_task_context = threading.local()

def set_current_task(task: TaskType) -> None:
    _task_context.task = task

def get_current_task() -> Optional[TaskType]:
    return getattr(_task_context, "task", None)

def clear_current_task() -> None:
    _task_context.task = None

@contextmanager
def task_scope(task: TaskType):
    """Context manager: sets task type for the duration of a block."""
    prev = get_current_task()
    set_current_task(task)
    try:
        yield
    finally:
        # Restore previous task (supports nesting)
        if prev is not None:
            set_current_task(prev)
        else:
            clear_current_task()

# ---------------------------------------------------------------------------
# TaskRouter
# ---------------------------------------------------------------------------

class TaskRouter:
    """Selects the optimal Ollama model for a given task type."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def select_model(self, task: TaskType) -> str:
        """Walk preference list for *task*, return first available model.

        Falls back to first available model overall, then ``DocWain-Agent:latest``.
        """
        # Check config overrides first
        override = self._config_override(task)
        if override:
            return override

        preferences = _TASK_MODEL_PREFERENCES.get(task, _TASK_MODEL_PREFERENCES[TaskType.GENERAL])
        for family in preferences:
            # Try to find an available model matching this family
            for cap in self._registry.get_available():
                matched = _match_family(cap.name)
                if matched == family:
                    return cap.name

        # Fallback: first available model
        available = self._registry.get_available()
        if available:
            return available[0].name

        # Ultimate fallback
        return "DocWain-Agent:latest"

    def get_options(self, task: TaskType) -> Dict[str, Any]:
        """Return generation parameter overrides for *task*."""
        base = dict(_TASK_OPTIONS.get(task, _TASK_OPTIONS[TaskType.GENERAL]))
        # Merge config overrides
        cfg_overrides = self._config_option_overrides(task)
        if cfg_overrides:
            base.update(cfg_overrides)
        return base

    def explain(self, task: TaskType) -> Dict[str, Any]:
        """Return a debug-friendly dict explaining routing for *task*."""
        selected = self.select_model(task)
        preferences = _TASK_MODEL_PREFERENCES.get(task, [])
        options = self.get_options(task)
        return {
            "task": task.value,
            "selected_model": selected,
            "preference_list": list(preferences),
            "options": options,
        }

    @staticmethod
    def _config_override(task: TaskType) -> Optional[str]:
        """Check Config.TaskRouting for a per-task model override."""
        try:
            from src.api.config import Config
            cfg = getattr(Config, "TaskRouting", None)
            if cfg is None:
                return None
            attr_name = task.value.upper() + "_MODEL"
            val = getattr(cfg, attr_name, "")
            return val if val else None
        except Exception:
            return None

    @staticmethod
    def _config_option_overrides(task: TaskType) -> Optional[Dict[str, Any]]:
        """Load per-task option overrides from environment variables.

        Convention: TASK_{TASK_NAME}_{PARAM} (e.g., TASK_RESPONSE_GENERATION_TEMPERATURE=0.5)
        Supported params: TEMPERATURE, NUM_PREDICT, NUM_CTX, TOP_P
        """
        prefix = f"TASK_{task.value.upper()}_"
        overrides: Dict[str, Any] = {}
        _PARAM_TYPES = {
            "TEMPERATURE": float,
            "NUM_PREDICT": int,
            "NUM_CTX": int,
            "TOP_P": float,
        }
        for param, typ in _PARAM_TYPES.items():
            val = os.environ.get(f"{prefix}{param}")
            if val is not None:
                try:
                    overrides[param.lower()] = typ(val)
                except (TypeError, ValueError):
                    pass
        return overrides if overrides else None
