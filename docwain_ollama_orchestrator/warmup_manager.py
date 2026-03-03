from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from docwain_ollama_orchestrator.config import AppConfig
from docwain_ollama_orchestrator.gpu_info import GPUInfo
from docwain_ollama_orchestrator.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class WarmupDecision:
    name: str
    estimated_vram_mb: Optional[int]
    eligible: bool
    reason: str


def _parse_param_size(param_size: Optional[str]) -> Optional[float]:
    if not param_size:
        return None
    value = param_size.strip().lower()
    if value.endswith("b"):
        try:
            return float(value[:-1])
        except ValueError:
            return None
    return None


def _quantization_bytes(quant: Optional[str]) -> float:
    if not quant:
        return 2.0
    q = quant.lower()
    if "q2" in q:
        return 0.25
    if "q3" in q:
        return 0.375
    if "q4" in q:
        return 0.5
    if "q5" in q:
        return 0.625
    if "q6" in q:
        return 0.75
    if "q8" in q:
        return 1.0
    if "f16" in q or "fp16" in q:
        return 2.0
    if "f32" in q or "fp32" in q:
        return 4.0
    return 2.0


def estimate_vram_mb(details: Dict[str, Any], config: AppConfig) -> Optional[int]:
    param_size = _parse_param_size(details.get("parameter_size"))
    quant = details.get("quantization") or details.get("quantization_level")
    if param_size is None:
        return None
    bytes_per_param = _quantization_bytes(quant)
    size_gb = param_size * bytes_per_param
    size_mb = int(size_gb * 1024 * 1.2)
    return max(size_mb, 256)


def _capability_priority(capabilities: List[Dict[str, Any]]) -> int:
    priority = 0
    for entry in capabilities:
        cap = entry.get("capability")
        if cap in {"fast_router", "judge", "coder"}:
            priority -= 10
        if cap == "math":
            priority -= 5
    return priority


def decide_warmup_set(
    models: List[Dict[str, Any]],
    gpu_info: GPUInfo,
    config: AppConfig,
) -> Tuple[List[WarmupDecision], int]:
    decisions: List[WarmupDecision] = []
    free_mb = gpu_info.free_mb
    budget_mb: Optional[int] = None
    if gpu_info.available and free_mb:
        budget_mb = int(free_mb * config.warmup_vram_budget_ratio)

    sorted_models = sorted(
        models,
        key=lambda m: (
            _capability_priority(m.get("capabilities", [])),
            m.get("resources", {}).get("estimated_vram_mb") or math.inf,
        ),
    )

    remaining_budget = budget_mb
    cpu_remaining = config.warmup_cpu_max_models

    for model in sorted_models:
        name = model.get("name")
        if not name:
            continue
        details = model.get("details", {})
        override = config.model_vram_overrides_mb.get(name)
        estimated = override if override is not None else estimate_vram_mb(details, config)
        if estimated is None:
            estimated = config.unknown_model_vram_mb
        model.setdefault("resources", {})["estimated_vram_mb"] = estimated

        if not config.warmup_enabled:
            decisions.append(WarmupDecision(name, estimated, False, "warmup_disabled"))
            continue

        if gpu_info.available and budget_mb is not None:
            if estimated <= remaining_budget:
                decisions.append(WarmupDecision(name, estimated, True, "within_budget"))
                remaining_budget -= estimated
            else:
                decisions.append(WarmupDecision(name, estimated, False, "vram_budget_exceeded"))
            continue

        if not gpu_info.available:
            if cpu_remaining > 0:
                decisions.append(WarmupDecision(name, estimated, True, "cpu_warmup"))
                cpu_remaining -= 1
            else:
                decisions.append(WarmupDecision(name, estimated, False, "cpu_warmup_limit"))
            continue

        decisions.append(WarmupDecision(name, estimated, False, "gpu_info_unavailable"))

    return decisions, budget_mb or 0


async def perform_warmups(
    client: OllamaClient,
    models: List[Dict[str, Any]],
    decisions: List[WarmupDecision],
    config: AppConfig,
) -> Dict[str, Dict[str, Any]]:
    decision_map = {decision.name: decision for decision in decisions}
    semaphore = asyncio.Semaphore(max(1, config.warmup_max_concurrency))
    results: Dict[str, Dict[str, Any]] = {}

    async def _warmup_one(model: Dict[str, Any]) -> None:
        name = model["name"]
        decision = decision_map.get(name)
        if not decision or not decision.eligible:
            results[name] = {
                "attempted": False,
                "status": "SKIPPED",
                "last_error": decision.reason if decision else "not_selected",
                "keep_alive": config.warmup_keep_alive,
                "warmup_latency_ms": None,
            }
            return
        async with semaphore:
            for attempt in range(config.warmup_max_retries + 1):
                try:
                    start = time.perf_counter()
                    response = await client.warmup_chat(name, config.warmup_keep_alive)
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    results[name] = {
                        "attempted": True,
                        "status": "WARMED",
                        "last_error": None,
                        "keep_alive": config.warmup_keep_alive,
                        "warmup_latency_ms": latency_ms or response.latency_ms,
                    }
                    return
                except Exception as exc:  # noqa: BLE001
                    backoff = config.warmup_backoff_seconds * (2 ** attempt)
                    jitter = random.uniform(0, 0.3)
                    await asyncio.sleep(backoff + jitter)
                    if attempt >= config.warmup_max_retries:
                        results[name] = {
                            "attempted": True,
                            "status": "FAILED",
                            "last_error": str(exc),
                            "keep_alive": config.warmup_keep_alive,
                            "warmup_latency_ms": None,
                        }
                        return

    tasks = [_warmup_one(model) for model in models]
    if tasks:
        await asyncio.gather(*tasks)
    return results
