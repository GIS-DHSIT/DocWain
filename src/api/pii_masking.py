from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

# Patterns for basic PII detection/masking
PII_PATTERNS = [
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
    (r"\b(?:\+?[\d]{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b(?:\d[ -]*?){13,16}\b", "[CARD]"),
    (r"\b[A-Z]{2}\d{6}\b", "[PASSPORT]"),
]


def detect_pii_with_ai(text: str) -> List[Dict[str, str]]:
    """
    Best-effort AI-driven PII detection with regex fallback.
    Returns list of items: {"type": str, "value": str}
    """
    detected: List[Dict[str, str]] = []

    # Regex baseline
    for pattern, label in PII_PATTERNS:
        for match in re.findall(pattern, text):
            detected.append({"type": label.strip("[]"), "value": match})

    # Optional LLM refinement via centralized gateway
    try:
        from src.llm.gateway import get_llm_gateway
        prompt = (
            "Identify PII (emails, phone numbers, SSNs, credit cards, passport numbers, addresses, bank accounts) "
            "in the provided text. Respond ONLY with JSON array of objects "
            '[{\"type\": \"<pii_type>\", \"value\": \"<exact_snippet>\"}]. '
            "Text:\n"
            f"{text[:4000]}"
        )
        payload = get_llm_gateway().generate(prompt)
        if payload:
            parsed = json.loads(payload)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "value" in item:
                        detected.append(
                            {
                                "type": item.get("type", "PII"),
                                "value": str(item["value"]),
                            }
                        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("LLM PII detection fallback used due to: %s", exc)

    # Deduplicate by value/type
    unique: Dict[Tuple[str, str], Dict[str, str]] = {}
    for item in detected:
        key = (item.get("type", ""), item.get("value", ""))
        unique[key] = item
    return list(unique.values())


def _mask_text(text: str) -> Tuple[str, int, List[Dict[str, str]]]:
    """Mask PII in a string and return masked text, count, and detected items."""
    count = 0
    detected_items = detect_pii_with_ai(text)
    masked = text
    for pattern, replacement in PII_PATTERNS:
        def _sub(match):
            nonlocal count
            count += 1
            return replacement
        masked = re.sub(pattern, _sub, masked)
    count = count or len(detected_items)
    return masked, count, detected_items


def mask_document_content(doc_content: Any):
    """
    Mask PII across document content.
    Returns (masked_content, total_pii_count, high_confidential_flag, pii_items).
    """
    total = 0
    found_items: List[Dict[str, str]] = []

    def _mask_value(val):
        nonlocal total, found_items
        if isinstance(val, str):
            masked_val, c, items = _mask_text(val)
            total += c
            if items:
                found_items.extend(items)
            return masked_val
        if isinstance(val, list):
            return [_mask_value(item) for item in val]
        if isinstance(val, dict):
            new_dict = {}
            for k, v in val.items():
                if k == "embeddings":
                    # Drop embeddings here to force recompute after masking
                    continue
                new_dict[k] = _mask_value(v)
            return new_dict
        return val

    masked = _mask_value(doc_content)
    # Deduplicate found items
    unique: Dict[Tuple[str, str], Dict[str, str]] = {}
    for item in found_items:
        key = (item.get("type", ""), item.get("value", ""))
        unique[key] = item
    return masked, total, False, list(unique.values())
