from __future__ import annotations

from typing import Optional

DOCWAIN_AGENT_ALIAS = "DocWain-Agent"
DOCWAIN_AGENT_TARGET = "gpt-oss:latest"


def normalize_model_name(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return model_name
    if model_name.strip().lower() == DOCWAIN_AGENT_ALIAS.lower():
        return DOCWAIN_AGENT_TARGET
    return model_name


def is_docwain_agent(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    return model_name.strip().lower() == DOCWAIN_AGENT_ALIAS.lower()
