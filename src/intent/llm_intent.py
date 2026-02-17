from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

INTENT_CACHE_MAX = 512
INTENT_CACHE_TTL_SEC = 60 * 60
INTENT_TIMEOUT_MS = 500
INTENT_MAX_TOKENS = 160

_INTENT_CACHE: Dict[str, Dict[str, Any]] = {}

_INTENT_PROMPT = (
    "You are an intent parser for document QA. "
    "Return strict JSON with keys: "
    "intent, output_format, requested_fields, domain, constraints, entity_hints. "
    "Rules: "
    "intent in [qa, summarize, compare, rank, list, extract, contact, generate]. "
    "output_format in [bullets, paragraph, table, json, markdown]. "
    "requested_fields is an array of field names. "
    "domain in [resume, invoice, legal, policy, report, generic]. "
    "constraints is a JSON object (empty if none). "
    "entity_hints is an array of keywords or names. "
    "Return JSON only.\n\n"
    "QUERY: {query}\n"
)


@dataclass(frozen=True)
class IntentParse:
    intent: str
    output_format: str
    requested_fields: List[str]
    domain: str
    constraints: Dict[str, Any]
    entity_hints: List[str]
    source: str


def parse_intent(
    *,
    query: str,
    llm_client: Optional[Any] = None,
    redis_client: Optional[Any] = None,
    timeout_ms: int = INTENT_TIMEOUT_MS,
    max_tokens: int = INTENT_MAX_TOKENS,
) -> IntentParse:
    normalized = _normalize(query)
    if not normalized:
        return IntentParse(
            intent="qa",
            output_format="bullets",
            requested_fields=[],
            domain="generic",
            constraints={},
            entity_hints=[],
            source="empty",
        )

    key = _cache_key(normalized)
    cached = _cache_get(redis_client, key) or _local_cache_get(key)
    if cached:
        return IntentParse(**cached, source="cache")

    parsed = _llm_parse(
        query=normalized,
        llm_client=llm_client,
        timeout_ms=timeout_ms,
        max_tokens=max_tokens,
    )
    if parsed:
        _cache_set(redis_client, key, parsed)
        _local_cache_set(key, parsed)
        return IntentParse(**parsed, source="llm")

    fallback = _heuristic_parse(normalized)
    _cache_set(redis_client, key, fallback)
    _local_cache_set(key, fallback)
    return IntentParse(**fallback, source="heuristic")


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def _cache_key(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def _cache_get(redis_client: Optional[Any], key: str) -> Optional[Dict[str, Any]]:
    if redis_client is None:
        return None
    try:
        cached = redis_client.get(f"intent:{key}")
        if isinstance(cached, bytes):
            cached = cached.decode("utf-8")
        if cached:
            payload = json.loads(cached)
            if isinstance(payload, dict):
                return payload
    except Exception:
        return None
    return None


def _cache_set(redis_client: Optional[Any], key: str, payload: Dict[str, Any]) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(f"intent:{key}", INTENT_CACHE_TTL_SEC, json.dumps(payload, ensure_ascii=True))
    except Exception:
        return


def _local_cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _INTENT_CACHE.get(key)
    if not entry:
        return None
    if entry.get("_expires_at", 0) < time.time():
        _INTENT_CACHE.pop(key, None)
        return None
    return entry.get("payload")


def _local_cache_set(key: str, payload: Dict[str, Any]) -> None:
    if len(_INTENT_CACHE) >= INTENT_CACHE_MAX:
        _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
    _INTENT_CACHE[key] = {"payload": payload, "_expires_at": time.time() + INTENT_CACHE_TTL_SEC}


def _llm_parse(
    *,
    query: str,
    llm_client: Optional[Any],
    timeout_ms: int,
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    if llm_client is None:
        return None
    prompt = _INTENT_PROMPT.format(query=query)
    options = {
        "num_predict": int(max_tokens),
        "max_output_tokens": int(max_tokens),
        "temperature": 0,
    }

    def _call() -> str:
        if hasattr(llm_client, "generate_with_metadata"):
            text, _meta = llm_client.generate_with_metadata(prompt, options=options, max_retries=1, backoff=0.2)
            return text or ""
        return llm_client.generate(prompt, max_retries=1, backoff=0.2) or ""

    deadline = max(0.05, float(timeout_ms) / 1000.0)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        raw = future.result(timeout=deadline)
    except Exception:
        executor.shutdown(wait=False)
        return None
    executor.shutdown(wait=False)
    payload = _safe_json(raw)
    if not payload:
        return None
    cleaned = _sanitize_payload(payload)
    if not cleaned:
        return None
    return cleaned


def _safe_json(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    if "{" in text and "}" in text:
        snippet = text[text.find("{") : text.rfind("}") + 1]
        try:
            payload = json.loads(snippet)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    return None


def _sanitize_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    intent = str(payload.get("intent") or "qa").strip().lower()
    if intent not in {"qa", "summarize", "compare", "rank", "list", "extract", "contact", "generate"}:
        intent = "qa"

    output_format = str(payload.get("output_format") or "bullets").strip().lower()
    if output_format not in {"bullets", "paragraph", "table", "json", "markdown"}:
        output_format = "bullets"

    domain = str(payload.get("domain") or "generic").strip().lower()
    if domain not in {"resume", "invoice", "legal", "policy", "report", "generic"}:
        domain = "generic"

    requested_fields = _clean_list(payload.get("requested_fields") or [])
    entity_hints = _clean_list(payload.get("entity_hints") or [])
    constraints = payload.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}

    return {
        "intent": intent,
        "output_format": output_format,
        "requested_fields": requested_fields,
        "domain": domain,
        "constraints": constraints,
        "entity_hints": entity_hints,
    }


def _clean_list(values: Any) -> List[str]:
    items = []
    for value in values if isinstance(values, (list, tuple)) else []:
        cleaned = str(value or "").strip()
        if cleaned:
            items.append(cleaned)
    deduped: List[str] = []
    seen = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:12]


def _heuristic_parse(query: str) -> Dict[str, Any]:
    lowered = query.lower()
    intent = "qa"
    if re.search(r"\b(summary|summarize|overview|recap)\b", lowered):
        intent = "summarize"
    elif re.search(r"\b(compare|difference|vs\b)\b", lowered):
        intent = "compare"
    elif re.search(r"\b(rank|top|best|most)\b", lowered):
        intent = "rank"
    elif re.search(r"\b(list|show|all)\b", lowered):
        intent = "list"
    elif re.search(r"\b(extract|find|identify)\b", lowered):
        intent = "extract"
    elif re.search(r"\b(contact|email|phone|reach|linkedin)\b", lowered):
        intent = "contact"
    elif re.search(r"\b(generate|write|draft|create|compose|prepare)\b", lowered):
        intent = "generate"

    domain = "generic"
    if any(tok in lowered for tok in ("resume", "cv", "candidate", "experience", "skills", "qualifications")):
        domain = "resume"
    elif any(tok in lowered for tok in ("invoice", "amount due", "billing", "payment", "total amount")):
        domain = "invoice"
    elif any(tok in lowered for tok in ("contract", "agreement", "clause", "legal", "terms and conditions")):
        domain = "legal"

    output_format = "bullets"
    if "table" in lowered:
        output_format = "table"
    elif "json" in lowered:
        output_format = "json"
    elif "paragraph" in lowered:
        output_format = "paragraph"
    elif "markdown" in lowered:
        output_format = "markdown"

    fields = []
    if re.search(r"\b(email|emails)\b", lowered):
        fields.append("email")
    if re.search(r"\b(phone|phones|contact)\b", lowered):
        fields.append("phone")
    if "linkedin" in lowered:
        fields.append("linkedin")
    if "skills" in lowered:
        fields.append("skills")
    if re.search(r"\b(education|degree|university|college)\b", lowered):
        fields.append("education")
    if re.search(r"\b(certification|certificate|certified)\b", lowered):
        fields.append("certifications")
    if re.search(r"\b(experience|work history|employment)\b", lowered):
        fields.append("experience")

    # Extract entity hints — find capitalized names that aren't common English words
    entity_hints = []
    _COMMON_STARTS = {
        "The", "This", "That", "What", "How", "Can", "Will", "Should", "Does",
        "Which", "Where", "When", "Who", "Why", "Are", "Is", "Do", "Has", "Have",
        "Get", "Set", "Show", "List", "Find", "All", "Any", "Each", "Every",
        "Compare", "Rank", "Top", "Best", "Most", "Summary", "Summarize",
        "Extract", "Identify", "Review", "Describe", "Fetch", "Please",
        "Document", "Resume", "Invoice", "Profile", "Report", "Details",
        "Contact", "Email", "Phone", "Skills", "Experience", "Education",
    }
    # Match capitalized words (2+ chars) that look like proper names
    name_matches = re.findall(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,2})\b', query)
    for name in name_matches:
        first_word = name.split()[0]
        if first_word not in _COMMON_STARTS:
            entity_hints.append(name)
            break

    return {
        "intent": intent,
        "output_format": output_format,
        "requested_fields": fields,
        "domain": domain,
        "constraints": {},
        "entity_hints": entity_hints,
    }


__all__ = ["IntentParse", "parse_intent"]
