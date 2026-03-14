from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from typing import Set

import ollama

logger = get_logger(__name__)

def _extract_model_names(payload: dict) -> Set[str]:
    names: Set[str] = set()
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if name:
            names.add(str(name))
    return names

def ensure_model_ready(
    model_name: str,
    *,
    warmup: bool = True,
    allow_pull: bool = False,
    timeout_sec: int = 30,
) -> bool:
    """
    Ensure the requested model is available and optionally warm it up.
    Returns True when the model is available (and warmed if requested).
    """
    if not model_name:
        logger.info("Startup model check skipped: no model name provided.")
        return False

    start = time.time()
    try:
        models_payload = ollama.list()
        known_models = _extract_model_names(models_payload)
        if model_name not in known_models:
            if allow_pull:
                logger.info("Startup model pull requested.")
                ollama.pull(model_name)
                models_payload = ollama.list()
                known_models = _extract_model_names(models_payload)
            if model_name not in known_models:
                logger.warning("Startup model unavailable.")
                return False

        if warmup:
            if time.time() - start > timeout_sec:
                logger.warning("Startup model warmup skipped due to timeout budget.")
                return True
            try:
                ollama.generate(
                    model=model_name,
                    prompt="ping",
                    options={"temperature": 0},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Startup model warmup failed: %s", exc)
                return False

        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup model readiness check failed: %s", exc)
        return False
