from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

VERSION = "1.0.0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_json_dict(name: str) -> Dict[str, int]:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        clean: Dict[str, int] = {}
        for key, value in data.items():
            try:
                clean[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return clean
    return {}


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


@dataclass(frozen=True)
class AppConfig:
    ollama_base_url: str = "http://localhost:11434"
    redis_url: str = "redis://localhost:6379/0"
    registry_ttl_seconds: int = 3600
    refresh_interval_seconds: int = 300
    warmup_enabled: bool = True
    warmup_keep_alive: str = "15m"
    warmup_vram_budget_ratio: float = 0.7
    warmup_max_concurrency: int = 2
    warmup_backoff_seconds: float = 1.0
    warmup_max_retries: int = 2
    warmup_cpu_max_models: int = 1
    classifier_probes_enabled: bool = False
    ray_namespace: str = "docwain"
    ray_actor_name: str = "ollama_registry_service"
    ray_dashboard_enabled: Optional[bool] = None
    ray_metrics_export_port: Optional[int] = None
    model_vram_overrides_mb: Dict[str, int] = field(default_factory=dict)
    unknown_model_vram_mb: int = 4096

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            redis_url=os.getenv("REDIS_URL", cls.redis_url),
            registry_ttl_seconds=_env_int("REGISTRY_TTL_SECONDS", cls.registry_ttl_seconds),
            refresh_interval_seconds=_env_int("REFRESH_INTERVAL_SECONDS", cls.refresh_interval_seconds),
            warmup_enabled=_env_bool("WARMUP_ENABLED", cls.warmup_enabled),
            warmup_keep_alive=os.getenv("WARMUP_KEEP_ALIVE", cls.warmup_keep_alive),
            warmup_vram_budget_ratio=_env_float("WARMUP_VRAM_BUDGET_RATIO", cls.warmup_vram_budget_ratio),
            warmup_max_concurrency=_env_int("WARMUP_MAX_CONCURRENCY", cls.warmup_max_concurrency),
            warmup_backoff_seconds=_env_float("WARMUP_BACKOFF_SECONDS", cls.warmup_backoff_seconds),
            warmup_max_retries=_env_int("WARMUP_MAX_RETRIES", cls.warmup_max_retries),
            warmup_cpu_max_models=_env_int("WARMUP_CPU_MAX_MODELS", cls.warmup_cpu_max_models),
            classifier_probes_enabled=_env_bool("CLASSIFIER_PROBES_ENABLED", cls.classifier_probes_enabled),
            ray_namespace=os.getenv("RAY_NAMESPACE", cls.ray_namespace),
            ray_actor_name=os.getenv("RAY_ACTOR_NAME", cls.ray_actor_name),
            ray_dashboard_enabled=_env_optional_bool("RAY_DASHBOARD_ENABLED"),
            ray_metrics_export_port=_env_optional_int("RAY_METRICS_EXPORT_PORT"),
            model_vram_overrides_mb=_env_json_dict("MODEL_VRAM_OVERRIDES_MB"),
            unknown_model_vram_mb=_env_int("UNKNOWN_MODEL_VRAM_MB", cls.unknown_model_vram_mb),
        )
