from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.cache.redis_keys import RedisKeys
from src.cache.redis_store import RedisStore
from src.embed.entity_extractor import EntityExtractor
from src.intent.llm_intent import parse_intent
from .models import Plan


_GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "will",
    "your",
    "into",
    "have",
    "has",
    "are",
    "was",
    "were",
    "you",
    "our",
    "but",
    "not",
    "all",
    "any",
    "can",
    "use",
    "using",
    "used",
    "about",
    "more",
    "less",
    "than",
    "then",
    "also",
    "via",
    "what",
    "who",
    "which",
    "when",
    "where",
    "why",
    "how",
}


class Planner:
    def __init__(self, llm_client: Optional[Any] = None, redis_client: Optional[Any] = None) -> None:
        self.llm_client = llm_client
        self.redis_client = redis_client
        self.entity_extractor = EntityExtractor()

    def plan(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Plan:
        session_id = session_id or "default"
        cached = self._load_cached_plan(subscription_id, profile_id, session_id, query)
        if cached:
            return cached

        normalized = " ".join((query or "").strip().split())
        intent_parse = parse_intent(query=normalized, llm_client=self.llm_client, redis_client=self.redis_client)
        intent = intent_parse.intent or _detect_intent(normalized)
        if _is_greeting(normalized):
            intent = "greet"

        if intent == "greet":
            plan = Plan(
                intent="greet",
                scope=_scope(subscription_id, profile_id, document_id),
                query_rewrites=[normalized] if normalized else [query],
                entity_hints=[],
                expected_answer_shape="bullets",
                query=normalized or query,
            )
            self._cache_plan(subscription_id, profile_id, session_id, query, plan)
            return plan

        plan_payload = None
        if self.llm_client and intent != "greet" and intent_parse.source not in {"llm", "cache"}:
            plan_payload = _llm_plan(self.llm_client, normalized, subscription_id, profile_id, document_id)
        if not plan_payload:
            plan_payload = _fallback_plan(normalized, subscription_id, profile_id, document_id)

        plan = Plan(
            intent=plan_payload.get("intent") or intent,
            scope=plan_payload.get("scope") or _scope(subscription_id, profile_id, document_id),
            query_rewrites=plan_payload.get("query_rewrites") or _query_rewrites(normalized),
            entity_hints=plan_payload.get("entity_hints")
            or _merge_hints(_entity_hints(normalized, self.entity_extractor), intent_parse.entity_hints),
            expected_answer_shape=plan_payload.get("expected_answer_shape")
            or _shape_from_output_format(intent_parse.output_format)
            or _answer_shape(intent),
            query=normalized or query,
        )
        self._cache_plan(subscription_id, profile_id, session_id, query, plan)
        return plan

    def _load_cached_plan(self, subscription_id: str, profile_id: str, session_id: str, query: str) -> Optional[Plan]:
        if not self.redis_client:
            return None
        store = RedisStore(self.redis_client)
        keys = RedisKeys(subscription_id=str(subscription_id), profile_id=str(profile_id))
        cached = store.get_plan_cache(keys, session_id, query)
        if not cached:
            return None
        return Plan(
            intent=cached.get("intent") or "answer",
            scope=cached.get("scope") or _scope(subscription_id, profile_id, None),
            query_rewrites=cached.get("query_rewrites") or [query],
            entity_hints=cached.get("entity_hints") or [],
            expected_answer_shape=cached.get("expected_answer_shape") or "bullets",
            query=cached.get("query") or query,
        )

    def _cache_plan(self, subscription_id: str, profile_id: str, session_id: str, query: str, plan: Plan) -> None:
        if not self.redis_client:
            return
        store = RedisStore(self.redis_client)
        keys = RedisKeys(subscription_id=str(subscription_id), profile_id=str(profile_id))
        payload = {
            "intent": plan.intent,
            "scope": plan.scope,
            "query_rewrites": plan.query_rewrites,
            "entity_hints": plan.entity_hints,
            "expected_answer_shape": plan.expected_answer_shape,
            "query": plan.query,
        }
        store.set_plan_cache(keys, session_id, query, payload)


def _scope(subscription_id: str, profile_id: str, document_id: Optional[str]) -> Dict[str, Any]:
    return {
        "subscription_id": str(subscription_id),
        "profile_id": str(profile_id),
        "document_id": str(document_id) if document_id else None,
    }


def _detect_intent(query: str) -> str:
    lower = (query or "").lower()
    if any(token in lower for token in ["compare", "difference", "versus", "vs"]):
        return "compare"
    if any(token in lower for token in ["summarize", "summary", "overview"]):
        return "summarize"
    if any(token in lower for token in ["rank", "top", "best", "order"]):
        return "rank"
    if any(token in lower for token in ["draft", "write", "compose", "email", "letter"]):
        return "draft"
    if any(token in lower for token in ["extract", "list", "show", "who", "what"]):
        return "extract"
    return "answer"


def _is_greeting(query: str) -> bool:
    lower = (query or "").strip().lower()
    if not lower:
        return False
    if lower in _GREETINGS:
        return True
    for greet in _GREETINGS:
        if lower.startswith(greet + " "):
            return True
    return False


def _answer_shape(intent: str) -> str:
    if intent == "compare":
        return "table"
    if intent == "draft":
        return "letter"
    return "bullets"


def _shape_from_output_format(output_format: str) -> Optional[str]:
    normalized = (output_format or "").strip().lower()
    if not normalized:
        return None
    if normalized == "table":
        return "table"
    if normalized in {"markdown", "bullets"}:
        return "bullets"
    if normalized == "paragraph":
        return "paragraph"
    if normalized == "json":
        return "json"
    return None


def _merge_hints(primary: List[str], extra: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for hint in (primary or []) + (extra or []):
        key = hint.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(hint)
    return merged[:12]


def _query_rewrites(query: str) -> List[str]:
    rewrites = [query]
    lower = (query or "").lower()
    if "skills" in lower:
        rewrites.append(query + " experience")
    if "compare" in lower:
        rewrites.append(query.replace("compare", "differences"))
    return list(dict.fromkeys([r for r in rewrites if r]))


def _entity_hints(query: str, extractor: EntityExtractor) -> List[str]:
    tokens = [tok for tok in (query or "").split() if tok]
    hints: List[str] = []
    for token in tokens:
        cleaned = token.strip(" ,.:;!?()[]{}\"'").lower()
        if not cleaned or cleaned in _STOPWORDS:
            continue
        if len(cleaned) < 3:
            continue
        hints.append(cleaned)
    for ent in extractor.extract(query):
        if ent.entity_type in {"skill", "person", "organization"}:
            hints.append(ent.entity_norm)
    deduped: List[str] = []
    seen = set()
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        deduped.append(hint)
    return deduped[:8]


def _llm_plan(llm_client: Any, query: str, subscription_id: str, profile_id: str, document_id: Optional[str]) -> Optional[Dict[str, Any]]:
    prompt = (
        "Return JSON only. Plan the retrieval."
        "\nFields: intent, scope, query_rewrites, entity_hints, expected_answer_shape."
        f"\nQuery: {query}\n"
        f"\nScope subscription_id={subscription_id} profile_id={profile_id} document_id={document_id or ''}"
    )
    try:
        raw = llm_client.generate(prompt, max_retries=2, backoff=0.4)
    except Exception:
        return None
    payload = _extract_json(raw)
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return {}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


def _fallback_plan(query: str, subscription_id: str, profile_id: str, document_id: Optional[str]) -> Dict[str, Any]:
    intent = _detect_intent(query)
    return {
        "intent": intent,
        "scope": _scope(subscription_id, profile_id, document_id),
        "query_rewrites": _query_rewrites(query),
        "entity_hints": _entity_hints(query, EntityExtractor()),
        "expected_answer_shape": _answer_shape(intent),
    }


__all__ = ["Planner"]
