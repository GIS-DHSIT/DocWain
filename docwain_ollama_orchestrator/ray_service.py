from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from typing import Any, Dict, List, Optional

import ray

from docwain_ollama_orchestrator.capability_classifier import classify_model
from docwain_ollama_orchestrator.config import AppConfig, VERSION
from docwain_ollama_orchestrator.gpu_info import get_gpu_info
from docwain_ollama_orchestrator.ollama_client import OllamaClient, OllamaUnavailable
from docwain_ollama_orchestrator.redis_store import RedisStore
from docwain_ollama_orchestrator.registry import Registry
from docwain_ollama_orchestrator.warmup_manager import decide_warmup_set, perform_warmups

logger = logging.getLogger(__name__)


def ensure_ray_initialized(config: Optional[AppConfig] = None) -> None:
    if ray.is_initialized():
        return
    cfg = config or AppConfig.from_env()
    metrics_port = cfg.ray_metrics_export_port if cfg.ray_metrics_export_port is not None else "auto"
    dashboard_setting = cfg.ray_dashboard_enabled if cfg.ray_dashboard_enabled is not None else "auto"
    logger.info(
        "Ray startup settings: dashboard_enabled=%s metrics_export_port=%s",
        dashboard_setting,
        metrics_port,
    )
    init_kwargs: Dict[str, Any] = {
        "namespace": cfg.ray_namespace,
        "ignore_reinit_error": True,
        "log_to_driver": False,
        "include_dashboard": cfg.ray_dashboard_enabled,
    }
    if cfg.ray_metrics_export_port is not None:
        init_kwargs["_metrics_export_port"] = cfg.ray_metrics_export_port
    ray.init(**init_kwargs)


def get_or_create_actor(config: Optional[AppConfig] = None) -> "ray.actor.ActorHandle":
    cfg = config or AppConfig.from_env()
    ensure_ray_initialized(cfg)
    try:
        return ray.get_actor(cfg.ray_actor_name, namespace=cfg.ray_namespace)
    except ValueError:
        actor = OllamaRegistryService.options(
            name=cfg.ray_actor_name,
            namespace=cfg.ray_namespace,
            lifetime="detached",
        ).remote(cfg.__dict__)
        return actor


@ray.remote
class OllamaRegistryService:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        self.config = AppConfig.from_env() if config_dict is None else AppConfig(**config_dict)
        self.registry = Registry()
        self.redis_store = RedisStore(self.config.redis_url)
        self.ollama_client = OllamaClient(self.config.ollama_base_url)
        self._init_task: Optional[asyncio.Task] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._last_refresh: Optional[str] = None
        self._ollama_reachable: bool = False
        self._init_task = asyncio.get_event_loop().create_task(self._async_init())

    async def _async_init(self) -> None:
        self.registry.meta = {
            "started_at": dt.datetime.utcnow().isoformat() + "Z",
            "ollama_base_url": self.config.ollama_base_url,
            "refresh_interval_seconds": self.config.refresh_interval_seconds,
            "version": VERSION,
        }
        await self.refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.refresh_interval_seconds)
            try:
                await self.refresh()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Periodic refresh failed: %s", exc)

    async def _ensure_initialized(self) -> None:
        if self._init_task and not self._init_task.done():
            current = asyncio.current_task()
            if current is not self._init_task:
                await self._init_task

    async def _refresh_from_ollama(self) -> None:
        models = await self.ollama_client.list_models()
        running = await self.ollama_client.list_running()
        running_map = {item.get("name"): item for item in running if item.get("name")}
        enriched: List[Dict[str, Any]] = []
        gpu_info = get_gpu_info()

        for model in models:
            name = model.get("name")
            if not name:
                continue
            details = model.get("details") or {}
            capabilities = await classify_model(name, details, self.config, self.ollama_client)
            existing = self.registry.get_model(name) or {}
            warmup_state = existing.get("warmup") or {
                "attempted": False,
                "status": "NOT_STARTED",
                "last_error": None,
                "keep_alive": self.config.warmup_keep_alive,
                "warmup_latency_ms": None,
            }
            enriched.append(
                {
                    "name": name,
                    "digest": model.get("digest"),
                    "modified_at": model.get("modified_at"),
                    "size": model.get("size"),
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization") or details.get("quantization_level"),
                    "details": details,
                    "capabilities": capabilities,
                    "warmup": warmup_state,
                    "runtime": {
                        "last_seen_ts": dt.datetime.utcnow().isoformat() + "Z",
                        "is_loaded": name in running_map,
                    },
                    "resources": existing.get("resources", {}),
                }
            )

        decisions, budget_mb = decide_warmup_set(enriched, gpu_info, self.config)
        decision_map = {decision.name: decision for decision in decisions}
        for model in enriched:
            decision = decision_map.get(model["name"])
            if decision:
                model.setdefault("resources", {})["warmup_eligible"] = decision.eligible
                model["resources"]["warmup_reason"] = decision.reason
                model["resources"]["estimated_vram_mb"] = decision.estimated_vram_mb
                if not decision.eligible and model.get("warmup", {}).get("status") in {None, "NOT_STARTED"}:
                    model["warmup"] = {
                        "attempted": False,
                        "status": "SKIPPED",
                        "last_error": decision.reason,
                        "keep_alive": self.config.warmup_keep_alive,
                        "warmup_latency_ms": None,
                    }

        if self.config.warmup_enabled:
            warmup_targets = [
                model
                for model in enriched
                if decision_map.get(model["name"]) and decision_map[model["name"]].eligible
                and model.get("warmup", {}).get("status") != "WARMED"
            ]
            if warmup_targets:
                warmup_results = await perform_warmups(self.ollama_client, warmup_targets, decisions, self.config)
                for model in enriched:
                    if model["name"] in warmup_results:
                        model["warmup"] = warmup_results[model["name"]]

        started_at = self.registry.meta.get("started_at") if self.registry.meta else None
        self.registry = Registry()
        self.registry.merge_models(enriched)
        self.registry.meta.update(
            {
                "started_at": started_at or dt.datetime.utcnow().isoformat() + "Z",
                "ollama_base_url": self.config.ollama_base_url,
                "gpu_info": {
                    "available": gpu_info.available,
                    "total_mb": gpu_info.total_mb,
                    "free_mb": gpu_info.free_mb,
                    "method": gpu_info.method,
                },
                "warmup_policy": {
                    "budget_mb": budget_mb,
                    "budget_ratio": self.config.warmup_vram_budget_ratio,
                    "max_concurrency": self.config.warmup_max_concurrency,
                    "enabled": self.config.warmup_enabled,
                },
                "refresh_interval_seconds": self.config.refresh_interval_seconds,
                "version": VERSION,
            }
        )
        self._last_refresh = dt.datetime.utcnow().isoformat() + "Z"
        self._ollama_reachable = True

        registry_json = self.registry.serialize()
        meta_json = json.dumps(self.registry.meta, ensure_ascii=True)
        self.redis_store.save_registry(
            registry_json,
            meta_json,
            self.config.registry_ttl_seconds,
            self.registry.models,
        )

    async def refresh(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        try:
            await self._refresh_from_ollama()
        except OllamaUnavailable as exc:
            logger.warning("Ollama unreachable, attempting cached registry: %s", exc)
            cached = self.redis_store.load_registry()
            if cached:
                registry_json, meta_json = cached
                self.registry = Registry.deserialize(registry_json)
                try:
                    self.registry.meta = json.loads(meta_json)
                except json.JSONDecodeError:
                    pass
            self._ollama_reachable = False
        return self.registry.meta

    async def warmup_models(self, strategy: str = "auto") -> Dict[str, Any]:
        await self._ensure_initialized()
        if strategy not in {"auto", "force"}:
            strategy = "auto"
        if strategy == "force":
            self.config = AppConfig(**{**self.config.__dict__, "warmup_enabled": True})
        await self._refresh_from_ollama()
        return {"status": "ok", "last_refresh": self._last_refresh}

    async def get_registry(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        return {
            "models": list(self.registry.models.values()),
            "meta": self.registry.meta,
        }

    async def list_models(self) -> List[str]:
        await self._ensure_initialized()
        return self.registry.list_models()

    async def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        return self.registry.get_model(name)

    async def health(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        return {
            "status": "ok",
            "ollama_reachable": self._ollama_reachable,
            "last_refresh": self._last_refresh,
            "model_count": len(self.registry.models),
        }
