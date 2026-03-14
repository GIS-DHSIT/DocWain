"""Auto-discovers local Ollama models and catalogs their capabilities.

Provides a ``ModelRegistry`` singleton that knows which models are available,
their speed tier, strengths, and feature support.  Used by ``TaskRouter`` to
select the optimal model for each LLM task type.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded model knowledge base
# ---------------------------------------------------------------------------

_MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "docwain-agent": {
        "speed_tier": "heavy",
        "strengths": ["generation", "instruction_following", "creative", "tool_calling", "reasoning"],
        "supports_json_mode": True,
        "supports_cot": True,
        "supports_vision": False,
        "supports_tool_calling": True,
        "context_window": 131072,
    },
    "llama3.2": {
        "speed_tier": "fast",
        "strengths": ["classification", "intent_parsing", "short_response"],
        "supports_json_mode": True,
        "supports_cot": False,
        "supports_vision": False,
        "context_window": 4096,
    },
    "mistral": {
        "speed_tier": "medium",
        "strengths": ["structured_extraction", "json_generation", "instruction_following"],
        "supports_json_mode": True,
        "supports_cot": False,
        "supports_vision": False,
        "context_window": 8192,
    },
    "deepseek-r1": {
        "speed_tier": "medium",
        "strengths": ["reasoning", "verification", "chain_of_thought", "math"],
        "supports_json_mode": True,
        "supports_cot": True,
        "supports_vision": False,
        "context_window": 8192,
    },
    "gemma2": {
        "speed_tier": "medium",
        "strengths": ["generation", "summarization", "qa"],
        "supports_json_mode": True,
        "supports_cot": False,
        "supports_vision": False,
        "context_window": 8192,
    },
    "llava": {
        "speed_tier": "medium",
        "strengths": ["vision", "image_understanding"],
        "supports_json_mode": False,
        "supports_cot": False,
        "supports_vision": True,
        "context_window": 4096,
    },
    "glm-ocr": {
        "speed_tier": "medium",
        "strengths": ["vision", "ocr", "image_understanding", "text_extraction", "chart_analysis", "table_analysis"],
        "supports_json_mode": False,
        "supports_cot": False,
        "supports_vision": True,
        "context_window": 4096,
    },
    "lfm2.5-thinking": {
        "speed_tier": "fast",
        "strengths": ["reasoning", "chain_of_thought", "verification", "analysis", "tool_calling", "classification"],
        "supports_json_mode": True,
        "supports_cot": True,
        "supports_vision": False,
        "supports_tool_calling": True,
        "context_window": 131072,
    },
    "docwain-agent-v2": {
        "speed_tier": "heavy",
        "strengths": ["classification", "intent_parsing", "structured_extraction", "json_generation"],
        "supports_json_mode": True,
        "supports_cot": False,
        "supports_vision": False,
        "supports_tool_calling": False,
        "context_window": 4096,
    },
}

# Requirement → preferred strength tags (ordered by relevance)
_REQUIREMENT_STRENGTHS: Dict[str, List[str]] = {
    "fast_classification": ["classification", "intent_parsing", "short_response"],
    "structured_extraction": ["structured_extraction", "json_generation"],
    "reasoning": ["reasoning", "chain_of_thought", "verification"],
    "generation": ["generation", "instruction_following", "creative"],
    "summarization": ["summarization", "qa", "generation"],
    "vision": ["vision", "image_understanding"],
    "tool_calling": ["tool_calling", "instruction_following"],
}

@dataclass
class ModelCapability:
    """Describes the capabilities of a single local model."""

    name: str
    size_bytes: int = 0
    speed_tier: str = "medium"
    strengths: List[str] = field(default_factory=list)
    supports_json_mode: bool = True
    supports_cot: bool = False
    supports_vision: bool = False
    supports_tool_calling: bool = False
    context_window: int = 4096
    available: bool = True

def _match_family(model_name: str) -> Optional[str]:
    """Strip tag suffixes and match against known model families."""
    base = model_name.split(":")[0].lower().strip()
    # Exact match first
    if base in _MODEL_PROFILES:
        return base
    # Prefix match (e.g. "llama3.2-vision" → "llama3.2", "lfm2.5-thinking" → "lfm2.5-thinking")
    _FAMILY_MAP = {"lfm2": "lfm2.5-thinking", "gpt-oss": "qwen3", "docwain-agent": "qwen3"}
    for alias, family in _FAMILY_MAP.items():
        if base.startswith(alias) and family in _MODEL_PROFILES:
            return family
    for family in sorted(_MODEL_PROFILES.keys(), key=len, reverse=True):
        if base.startswith(family):
            return family
    return None

def _speed_tier_from_size(size_bytes: int) -> str:
    """Derive speed tier from model file size."""
    gb = size_bytes / (1024 ** 3)
    if gb < 3.0:
        return "fast"
    if gb <= 8.0:
        return "medium"
    return "heavy"

class ModelRegistry:
    """Auto-discovers available Ollama models and catalogs their capabilities."""

    def __init__(self):
        self._catalog: Dict[str, ModelCapability] = {}
        self._lock = threading.Lock()

    def discover(self) -> int:
        """Query Ollama for available models and populate the catalog.

        Returns the number of models discovered.
        """
        with self._lock:
            self._catalog.clear()
            try:
                import ollama
                response = ollama.list()
                models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
            except Exception as exc:
                logger.warning("ModelRegistry.discover() — ollama.list() failed: %s", exc)
                return 0

            for entry in models:
                if isinstance(entry, dict):
                    name = entry.get("name", "") or entry.get("model", "")
                else:
                    # ollama>=0.4 uses .model instead of .name
                    name = getattr(entry, "name", "") or getattr(entry, "model", "")
                if not name:
                    continue
                size = entry.get("size", 0) if isinstance(entry, dict) else getattr(entry, "size", 0)
                family = _match_family(name)
                if family and family in _MODEL_PROFILES:
                    profile = _MODEL_PROFILES[family]
                    cap = ModelCapability(
                        name=name,
                        size_bytes=size,
                        speed_tier=profile["speed_tier"],
                        strengths=list(profile["strengths"]),
                        supports_json_mode=profile["supports_json_mode"],
                        supports_cot=profile["supports_cot"],
                        supports_vision=profile["supports_vision"],
                        supports_tool_calling=profile.get("supports_tool_calling", False),
                        context_window=profile["context_window"],
                        available=True,
                    )
                else:
                    cap = ModelCapability(
                        name=name,
                        size_bytes=size,
                        speed_tier=_speed_tier_from_size(size),
                        strengths=["general"],
                        available=True,
                    )
                self._catalog[name] = cap

            logger.info(
                "ModelRegistry discovered %d models: %s",
                len(self._catalog),
                list(self._catalog.keys()),
            )
            return len(self._catalog)

    def register(self, cap: ModelCapability) -> None:
        """Manually register a model capability (useful for testing)."""
        with self._lock:
            self._catalog[cap.name] = cap

    def get(self, model_name: str) -> Optional[ModelCapability]:
        """Return the capability for a specific model name, or None."""
        return self._catalog.get(model_name)

    def get_available(self) -> List[ModelCapability]:
        """Return all models that are marked available."""
        return [m for m in self._catalog.values() if m.available]

    def best_for(self, requirement: str) -> Optional[str]:
        """Return the best available model name for a requirement string.

        Matches requirement against ``_REQUIREMENT_STRENGTHS`` to find
        models whose strengths overlap, then prefers faster models when
        strength overlap is tied.
        """
        target_strengths = _REQUIREMENT_STRENGTHS.get(requirement, [requirement])
        available = self.get_available()
        if not available:
            return None

        def _score(cap: ModelCapability) -> float:
            overlap = sum(1 for s in target_strengths if s in cap.strengths)
            # Tiebreak: prefer faster models (fast=2, medium=1, heavy=0)
            speed_bonus = {"fast": 0.2, "medium": 0.1, "heavy": 0.0}.get(cap.speed_tier, 0.0)
            return overlap + speed_bonus

        ranked = sorted(available, key=_score, reverse=True)
        best = ranked[0]
        if _score(best) > 0:
            return best.name
        return ranked[0].name  # fallback to first available

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_REGISTRY: Optional[ModelRegistry] = None
_REGISTRY_LOCK = threading.Lock()

def get_model_registry() -> Optional[ModelRegistry]:
    """Return the singleton ModelRegistry (may be None if not initialized)."""
    return _REGISTRY

def set_model_registry(registry: ModelRegistry) -> None:
    """Set the singleton ModelRegistry."""
    global _REGISTRY
    _REGISTRY = registry
