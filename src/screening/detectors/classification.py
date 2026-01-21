from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

CLASSIFICATION_FIELDS = [
    "classification",
    "confidentiality",
    "security_label",
    "sensitivity",
    "label",
]

NORMALIZATION_MAP = {
    "public": "PUBLIC",
    "internal": "INTERNAL",
    "confidential": "CONFIDENTIAL",
    "restricted": "RESTRICTED",
    "secret": "SECRET",
    "top secret": "SECRET",
    "private": "INTERNAL",
    "min": "MINIMAL_RISK",
    "minimal": "MINIMAL_RISK",
    "minimal_risk": "MINIMAL_RISK",
}


def _normalize_value(value: str) -> Optional[str]:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[cleaned]
    for key, normalized in NORMALIZATION_MAP.items():
        if key in cleaned:
            return normalized
    return None


def resolve_classification(metadata: Dict[str, Any]) -> Tuple[str, bool, Optional[str], List[str]]:
    for field in CLASSIFICATION_FIELDS:
        raw = metadata.get(field)
        if raw is None:
            continue
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if not isinstance(raw, str):
            raw = str(raw)
        normalized = _normalize_value(raw)
        if normalized:
            return normalized, False, field, []
    return "MINIMAL_RISK", True, None, CLASSIFICATION_FIELDS
