from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from src.prompting.persona import sanitize_response

_FORBIDDEN_MODEL_TERMS = re.compile(
    r"\b(ollama|llama|gpt-oss|gpt|openai|anthropic|gemini|claude|qwen)\b",
    re.IGNORECASE,
)

_HEX_TOKEN = re.compile(r"[0-9a-fA-F]{8,}")


def _sanitize_text(text: str) -> str:
    sanitized = sanitize_response(text)
    sanitized = _FORBIDDEN_MODEL_TERMS.sub("[redacted]", sanitized)
    return sanitized.strip()


def sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {k: sanitize_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [sanitize_payload(v) for v in payload]
    if isinstance(payload, str):
        return _sanitize_text(payload)
    return payload


def _safe_source_name(name: Optional[str], fallback: str) -> str:
    base = os.path.basename(str(name or "")).strip()
    if not base:
        base = fallback
    base = _HEX_TOKEN.sub("[redacted]", base)
    return base.strip() or fallback


def _safe_chunk_id(value: Optional[str], fallback: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return fallback
    if re.fullmatch(r"SOURCE-\d+", candidate, flags=re.IGNORECASE):
        return candidate.upper()
    return fallback


def build_evidence_list(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    evidence: List[Dict[str, str]] = []
    for idx, src in enumerate(sources or [], 1):
        fallback = f"Document {idx}"
        source_name = _safe_source_name(
            src.get("source_name") or src.get("file_name") or src.get("source") or src.get("citation"),
            fallback,
        )
        quote = (
            src.get("quote")
            or src.get("excerpt")
            or src.get("text")
            or ""
        )
        quote = re.sub(r"\s+", " ", str(quote or "")).strip()
        if len(quote) > 240:
            quote = quote[:237].rstrip() + "..."
        evidence.append(
            {
                "source": source_name,
                "chunk_id": _safe_chunk_id(src.get("chunk_id"), f"SOURCE-{idx}"),
                "quote": quote,
            }
        )
    return evidence


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def coerce_agent_response(
    raw_response: Any,
    sources: Iterable[Dict[str, Any]],
    *,
    fallback_answer: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Optional[Dict[str, Any]] = None
    if isinstance(raw_response, dict):
        payload = raw_response
    elif isinstance(raw_response, str):
        payload = _extract_json(raw_response)

    evidence = build_evidence_list(list(sources or []))
    has_evidence = bool(evidence)

    if payload is None:
        answer_text = fallback_answer or (raw_response if isinstance(raw_response, str) else "")
        payload = {
            "answer": str(answer_text or "").strip(),
            "confidence": 0.6 if has_evidence else 0.2,
            "key_points": [],
            "evidence": evidence,
            "missing_info": [] if has_evidence else ["No supporting document excerpts were retrieved for this query."],
        }

    # Normalize schema keys
    if "answer" not in payload:
        payload["answer"] = payload.get("response") or payload.get("summary") or ""
    if "confidence" not in payload:
        payload["confidence"] = 0.6 if has_evidence else 0.2
    if "key_points" not in payload:
        payload["key_points"] = []
    if "evidence" not in payload:
        payload["evidence"] = evidence
    if "missing_info" not in payload:
        payload["missing_info"] = [] if has_evidence else ["No supporting document excerpts were retrieved for this query."]

    # Normalize field types
    try:
        payload["confidence"] = float(payload.get("confidence", 0.0))
    except Exception:
        payload["confidence"] = 0.6 if has_evidence else 0.2
    if payload["confidence"] < 0.0:
        payload["confidence"] = 0.0
    if payload["confidence"] > 1.0:
        payload["confidence"] = 1.0

    if not isinstance(payload.get("key_points"), list):
        payload["key_points"] = [str(payload.get("key_points"))] if payload.get("key_points") else []
    payload["key_points"] = [str(item) for item in payload.get("key_points", []) if str(item).strip()]

    if not isinstance(payload.get("missing_info"), list):
        payload["missing_info"] = [str(payload.get("missing_info"))] if payload.get("missing_info") else []
    payload["missing_info"] = [str(item) for item in payload.get("missing_info", []) if str(item).strip()]

    if not isinstance(payload.get("evidence"), list):
        payload["evidence"] = evidence

    if not payload.get("evidence"):
        payload["evidence"] = evidence

    # Enforce evidence item schema
    cleaned_evidence: List[Dict[str, str]] = []
    for idx, item in enumerate(payload.get("evidence") or [], 1):
        if not isinstance(item, dict):
            continue
        source = _safe_source_name(item.get("source"), f"Document {idx}")
        chunk_id = _safe_chunk_id(item.get("chunk_id"), f"SOURCE-{idx}")
        quote = re.sub(r"\s+", " ", str(item.get("quote") or "")).strip()
        if len(quote) > 240:
            quote = quote[:237].rstrip() + "..."
        cleaned_evidence.append({"source": source, "chunk_id": chunk_id, "quote": quote})
    payload["evidence"] = cleaned_evidence

    return sanitize_payload(payload)
