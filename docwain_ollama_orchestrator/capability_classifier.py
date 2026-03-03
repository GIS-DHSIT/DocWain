from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from docwain_ollama_orchestrator.config import AppConfig
from docwain_ollama_orchestrator.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class CapabilityEntry:
    capability: str
    confidence: float
    method: str


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


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


def _heuristic_capabilities(name: str, details: Dict[str, Any]) -> List[CapabilityEntry]:
    lower = _normalize_name(name)
    entries: List[CapabilityEntry] = []
    if any(token in lower for token in ["code", "coder", "codellama", "deepseek-coder"]):
        entries.append(CapabilityEntry("coder", 0.65, "heuristic"))
    if "math" in lower:
        entries.append(CapabilityEntry("math", 0.65, "heuristic"))
    if any(token in lower for token in ["vision", "llava"]):
        entries.append(CapabilityEntry("vision", 0.7, "heuristic"))
    if any(token in lower for token in ["judge", "rerank", "critic"]):
        entries.append(CapabilityEntry("judge", 0.6, "heuristic"))

    param_size = _parse_param_size(details.get("parameter_size"))
    if param_size is not None and param_size < 4:
        entries.append(CapabilityEntry("fast_router", 0.6, "heuristic"))
    if any(token in lower for token in ["mini", "small"]):
        entries.append(CapabilityEntry("fast_router", 0.6, "heuristic"))

    if not entries:
        entries.append(CapabilityEntry("general", 0.5, "heuristic"))
    else:
        entries.append(CapabilityEntry("general", 0.3, "heuristic"))
    return entries


async def _run_probe(client: OllamaClient, model: str, prompt: str, expect: str) -> bool:
    try:
        start = time.perf_counter()
        response = await client.warmup_chat(model, keep_alive="1m", prompt=prompt)
        latency = int((time.perf_counter() - start) * 1000)
        content = response.data.get("message", {}).get("content", "")
        logger.debug("Probe %s latency=%sms", model, latency)
        return expect in content
    except Exception as exc:  # noqa: BLE001
        logger.debug("Probe failed for %s: %s", model, exc)
        return False


async def classify_model(
    name: str,
    details: Dict[str, Any],
    config: AppConfig,
    client: Optional[OllamaClient] = None,
) -> List[Dict[str, Any]]:
    entries = _heuristic_capabilities(name, details)
    if not config.classifier_probes_enabled or client is None:
        return [entry.__dict__ for entry in entries]

    lower = _normalize_name(name)
    updated: List[CapabilityEntry] = []

    for entry in entries:
        if entry.capability == "coder" and "code" in lower:
            ok = await _run_probe(client, name, "Write a Python function to reverse a list.", "def")
            if ok:
                updated.append(CapabilityEntry("coder", 0.85, "probe"))
                continue
        if entry.capability == "math" and "math" in lower:
            ok = await _run_probe(client, name, "What is 17*19? Provide only number.", "323")
            if ok:
                updated.append(CapabilityEntry("math", 0.85, "probe"))
                continue
        if entry.capability == "judge" and any(token in lower for token in ["judge", "rerank", "critic"]):
            prompt = "Answer A: The sky is blue. Answer B: The sky is green. Which answer is better and why?"
            ok = await _run_probe(client, name, prompt, "Answer")
            if ok:
                updated.append(CapabilityEntry("judge", 0.8, "probe"))
                continue
        updated.append(entry)

    return [entry.__dict__ for entry in updated]
