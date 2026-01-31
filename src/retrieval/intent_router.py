from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ollama

logger = logging.getLogger(__name__)

_INTENT_CACHE: Dict[str, Dict[str, Any]] = {}
_INTENT_CACHE_MAX = 512

_INTENT_PROMPT = (
    "Analyze the user query and return strict JSON with keys: "
    "intent (summarize|extract|compare|rank|list|qa), "
    "target_doc_types (array), constraints (object), need_tables (boolean). "
    "Do not include any extra keys. Query: {query}"
)

_DOC_TYPE_KEYWORDS = {
    "resume": ["resume", "cv", "candidate", "experience"],
    "invoice": ["invoice", "bill", "payment", "due"],
    "purchase_order": ["purchase order", "po"],
    "contract": ["contract", "agreement", "terms"],
    "policy": ["policy", "procedure"],
    "brochure": ["brochure", "catalog"],
    "report": ["report", "analysis"],
    "statement": ["statement", "balance"],
}


@dataclass(frozen=True)
class IntentResult:
    intent: str
    target_doc_types: List[str]
    constraints: Dict[str, Any]
    need_tables: bool
    source: str


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _heuristic_intent(query: str) -> IntentResult:
    lowered = query.lower()
    intent = "qa"
    if re.search(r"\b(summary|summarize|overview)\b", lowered):
        intent = "summarize"
    elif re.search(r"\b(compare|difference|vs\b)\b", lowered):
        intent = "compare"
    elif re.search(r"\b(rank|top|best)\b", lowered):
        intent = "rank"
    elif re.search(r"\b(list|show|all)\b", lowered):
        intent = "list"
    elif re.search(r"\b(extract|find|identify)\b", lowered):
        intent = "extract"

    target_types = []
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        if any(term in lowered for term in keywords):
            target_types.append(doc_type)

    need_tables = bool(re.search(r"\b(table|line item|rows)\b", lowered))
    return IntentResult(intent=intent, target_doc_types=target_types, constraints={}, need_tables=need_tables, source="heuristic")


def _ollama_intent(query: str, model_name: Optional[str]) -> Optional[IntentResult]:
    if not model_name:
        return None
    prompt = _INTENT_PROMPT.format(query=query)
    try:
        resp = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0})
        payload = json.loads(resp.get("response", "").strip())
        return IntentResult(
            intent=str(payload.get("intent", "qa")),
            target_doc_types=[str(t) for t in payload.get("target_doc_types", []) if t],
            constraints=payload.get("constraints") or {},
            need_tables=bool(payload.get("need_tables", False)),
            source="ollama",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama intent analysis failed: %s", exc)
        return None


def analyze_intent(query: str, model_name: Optional[str] = None) -> IntentResult:
    query = (query or "").strip()
    if not query:
        return IntentResult(intent="qa", target_doc_types=[], constraints={}, need_tables=False, source="empty")

    key = _cache_key(query)
    if key in _INTENT_CACHE:
        cached = _INTENT_CACHE[key]
        return IntentResult(**cached)

    result = _ollama_intent(query, model_name=model_name) or _heuristic_intent(query)

    if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
        _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
    _INTENT_CACHE[key] = {
        "intent": result.intent,
        "target_doc_types": result.target_doc_types,
        "constraints": result.constraints,
        "need_tables": result.need_tables,
        "source": result.source,
    }
    return result


__all__ = ["IntentResult", "analyze_intent"]
