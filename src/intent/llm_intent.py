from __future__ import annotations

import concurrent.futures
import hashlib
import json
from src.utils.logging_utils import get_logger
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

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

    fallback = _fallback_parse(normalized)
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
    if domain == "insurance":
        domain = "policy"
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

# ── Trained MLP intent + domain classifier ─────────────────────────────
#
# Uses a trained multi-head MLP (IntentDomainClassifier) for intent and
# domain classification.  The classifier learns decision boundaries from
# self-supervised training data instead of static cosine-similarity against
# prototype embeddings.
#
# Field detection delegates to FieldImportanceClassifier (separate module).
# Entity extraction uses NLP-based extractor (separate module).
#
# Falls back to minimal regex when the trained classifier is unavailable.

import threading

def _get_embedder():
    """Return the app-wide embedding model, or lazy-load a standalone one."""
    try:
        from src.api.rag_state import get_app_state
        state = get_app_state()
        if state and state.embedding_model is not None:
            return state.embedding_model
    except Exception:  # noqa: BLE001
        pass
    # Standalone fallback (e.g. during tests or when app_state not ready)
    return _get_standalone_embedder()

_standalone_embedder = None
_standalone_embedder_attempted = False
_standalone_embedder_lock = threading.Lock()

def _get_standalone_embedder():
    """Lazy-load sentence-transformer embedder as last resort."""
    global _standalone_embedder, _standalone_embedder_attempted
    if _standalone_embedder is not None:
        return _standalone_embedder
    if _standalone_embedder_attempted:
        return None
    with _standalone_embedder_lock:
        if _standalone_embedder is not None:
            return _standalone_embedder
        if _standalone_embedder_attempted:
            return None
        _standalone_embedder_attempted = True
        try:
            import warnings
            warnings.filterwarnings("ignore", message=r".*_target_device.*has been deprecated", category=FutureWarning)
            from sentence_transformers import SentenceTransformer
            _standalone_embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
            logger.info("Loaded standalone sentence-transformer for intent classification")
            return _standalone_embedder
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load sentence-transformer for intent classification: %s", exc)
            return None

def _extract_entity_hints(query: str) -> List[str]:
    """Extract entity hints from query using NLP extractor."""
    entity_hints = []
    try:
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query(query)
        if entity:
            entity_hints.append(entity)
    except Exception:  # noqa: BLE001
        pass
    return entity_hints

def _detect_fields_from_embedding(query: str, embedder: Optional[Any] = None) -> List[str]:
    """Detect requested fields using FieldImportanceClassifier."""
    try:
        from src.rag_v3.field_classifier import get_field_classifier
        fc = get_field_classifier()
        if fc is None or embedder is None:
            return []
        query_vec = embedder.encode([query], normalize_embeddings=True)[0]
        probs = fc.predict(query_vec, threshold=0.3)
        return list(probs.keys())
    except Exception:  # noqa: BLE001
        return []

def _detect_output_format(query: str) -> str:
    """Detect output format from query text (simple structural check)."""
    lowered = query.lower()
    if "table" in lowered:
        return "table"
    if "json" in lowered:
        return "json"
    if "paragraph" in lowered:
        return "paragraph"
    if "markdown" in lowered:
        return "markdown"
    return "bullets"

def _neural_parse(query: str) -> Optional[Dict[str, Any]]:
    """Classify intent + domain using the trained MLP classifier.

    Returns a fully populated parse dict or None if classifier unavailable.
    """
    try:
        from src.intent.intent_classifier import get_intent_classifier, ensure_intent_classifier
    except ImportError:
        return None

    clf = get_intent_classifier()
    embedder = _get_embedder()

    # Try to ensure classifier is ready (auto-train if needed)
    if clf is None and embedder is not None:
        try:
            clf = ensure_intent_classifier(embedder)
        except Exception:  # noqa: BLE001
            pass

    if clf is None or not getattr(clf, "_trained", False):
        return None
    if embedder is None:
        return None

    try:
        query_vec = embedder.encode([query], normalize_embeddings=True)[0]
    except Exception:  # noqa: BLE001
        return None

    result = clf.predict(query_vec)
    if not result:
        return None

    # Apply confidence gates
    intent = result.get("intent", "qa")
    intent_conf = result.get("intent_confidence", 0.0)
    domain = result.get("domain", "generic")
    domain_conf = result.get("domain_confidence", 0.0)

    if intent_conf < 0.30:
        intent = "qa"
    if domain_conf < 0.25:
        domain = "generic"

    logger.debug(
        "Trained intent classifier: intent=%s (%.2f), domain=%s (%.2f)",
        intent, intent_conf, domain, domain_conf,
    )

    return {
        "intent": intent,
        "output_format": _detect_output_format(query),
        "requested_fields": _detect_fields_from_embedding(query, embedder),
        "domain": domain,
        "constraints": {},
        "entity_hints": _extract_entity_hints(query),
    }

def _fallback_parse(query: str) -> Dict[str, Any]:
    """Parse intent using trained MLP first, minimal regex fallback.

    Strategy:
    1. Trained MLP classifier (primary)
    2. Minimal regex fallback only when classifier is unavailable
    """
    # ── Strategy 1: Trained MLP classification (primary) ──────────────
    neural = _neural_parse(query)
    if neural is not None:
        return neural

    # ── Strategy 2: Minimal regex fallback (degraded mode) ───────────
    # Only reached when trained classifier is not available.
    logger.debug("Trained intent classifier unavailable; using regex fallback")
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

    # Minimal domain detection for fallback
    domain = "generic"
    if re.search(r"\b(resume|candidate|skills|experience|qualifications?|certifications?|education)\b", lowered):
        domain = "resume"
    elif re.search(r"\b(invoice|billing|payment|amount\s+due|line\s+items?)\b", lowered):
        domain = "invoice"
    elif re.search(r"\b(contract|clause|agreement|legal|termination|liability)\b", lowered):
        domain = "legal"
    elif re.search(r"\b(insurance|policy|coverage|premium|deductible|claim|underwriting)\b", lowered):
        domain = "policy"
    elif re.search(r"\b(report|analysis|findings|quarterly|annual\s+report)\b", lowered):
        domain = "report"

    return {
        "intent": intent,
        "output_format": _detect_output_format(query),
        "requested_fields": [],
        "domain": domain,
        "constraints": {},
        "entity_hints": _extract_entity_hints(query),
    }

__all__ = ["IntentParse", "parse_intent"]
