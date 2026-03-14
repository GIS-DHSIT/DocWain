from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
from typing import Any, Dict

logger = get_logger(__name__)

def emit_quality_telemetry(payload: Dict[str, Any]) -> None:
    if not payload:
        return
    try:
        logger.info("quality_eval=%s", json.dumps(payload, ensure_ascii=True, default=str))
    except Exception:
        logger.info("quality_eval_payload=%s", payload)
