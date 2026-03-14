from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.prompting.evidence_synthesizer import build_evidence_synthesizer_prompt

logger = get_logger(__name__)

def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except Exception:
            return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

class EvidenceSynthesizer:
    def __init__(self, llm_client: Optional[Any] = None, *, enabled: Optional[bool] = None) -> None:
        self.llm_client = llm_client
        if enabled is None:
            enabled = bool(getattr(Config.Retrieval, "EVIDENCE_SYNTHESIZER_ENABLED", False))
        self.enabled = enabled

    def synthesize(
        self,
        *,
        user_query: str,
        plan_json: Dict[str, Any],
        evidence_packets: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self.llm_client:
            return None
        if not evidence_packets:
            return None

        prompt = build_evidence_synthesizer_prompt(
            user_query=user_query,
            plan_json=plan_json,
            evidence_packets=evidence_packets,
        )

        max_retries = int(getattr(Config.Retrieval, "EVIDENCE_SYNTHESIZER_MAX_RETRIES", 1))
        backoff = float(getattr(Config.Retrieval, "EVIDENCE_SYNTHESIZER_BACKOFF", 0.3))

        try:
            response = self.llm_client.generate(prompt, max_retries=max_retries, backoff=backoff)
            payload = _extract_json(response)
            if payload:
                return payload
        except Exception as exc:  # noqa: BLE001
            logger.debug("Evidence synthesizer failed: %s", exc)
        return None

__all__ = ["EvidenceSynthesizer"]
