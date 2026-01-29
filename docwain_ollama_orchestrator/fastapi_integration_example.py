from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import ray
from fastapi import FastAPI

from docwain_ollama_orchestrator.config import AppConfig
from docwain_ollama_orchestrator.ray_service import get_or_create_actor
from docwain_ollama_orchestrator.redis_store import RedisStore
from docwain_ollama_orchestrator.registry import Registry

logger = logging.getLogger(__name__)


async def ensure_ollama_registry_actor(config: Optional[AppConfig] = None) -> Optional["ray.actor.ActorHandle"]:
    loop = asyncio.get_running_loop()
    try:
        actor = await loop.run_in_executor(None, get_or_create_actor, config)
        return actor
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to start Ollama registry actor: %s", exc)
        return None


def attach_registry_routes(app: FastAPI) -> None:
    @app.get("/api/ollama/registry")
    async def get_registry() -> Dict[str, Any]:
        config = AppConfig.from_env()
        actor = await ensure_ollama_registry_actor(config)
        if actor is not None:
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: ray.get(actor.get_registry.remote()))
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ray registry fetch failed: %s", exc)
        store = RedisStore(config.redis_url)
        cached = store.load_registry()
        if cached:
            registry_json, meta_json = cached
            registry = Registry.deserialize(registry_json)
            return {"models": list(registry.models.values()), "meta": registry.meta}
        return {"models": [], "meta": {"status": "unavailable"}}
