from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence


_CURRENCY_SYMBOLS = "$€£¥"
_DECIMAL_TOKEN = "decimalpoint"

_INTENT_TOKENS = {"compare", "summarize", "products", "services", "total", "subtotal", "documents"}
_SYNONYMS = {
    "products": ["items", "line items"],
    "services": ["work", "service lines"],
    "summarize": ["summary", "overview"],
    "compare": ["comparison", "differences"],
    "total": ["total", "grand total", "amount due"],
    "subtotal": ["subtotal", "total", "amount due"],
}
_INTENT_SYNONYMS = {
    "COMPARE": ["compare", "comparison", "differences"],
    "SUMMARIZE": ["summary", "overview"],
    "PRODUCTS_SERVICES": ["items", "line items", "services", "work"],
    "TOTALS": ["total", "subtotal", "grand total", "amount due", "balance"],
}


def normalize_query(query: str) -> str:
    text = (query or "").lower().strip()
    if not text:
        return ""
    text = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL_TOKEN, text)
    text = re.sub(rf"[^a-z0-9\s{re.escape(_CURRENCY_SYMBOLS)}]", " ", text)
    text = text.replace(_DECIMAL_TOKEN, ".")
    text = " ".join(text.split())
    text = re.sub(r"[.!?,:;]+$", "", text)
    return text


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    subscription_id: str,
    session_id: str,
    user_id: str,
    model_name: str,
    prompt_version: str,
    normalized_query: str,
    retrieval_fingerprint: str,
) -> str:
    payload = "|".join(
        [
            subscription_id or "",
            session_id or "",
            user_id or "",
            model_name or "",
            prompt_version or "",
            normalized_query or "",
            retrieval_fingerprint or "",
        ]
    )
    return _hash(payload)


def compute_retrieval_fingerprint(chunks: Sequence[Any]) -> str:
    parts = []
    for chunk in chunks or []:
        meta = getattr(chunk, "metadata", {}) or {}
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or "")
        source_file = (
            meta.get("source_file")
            or meta.get("source")
            or getattr(chunk, "source", None)
            or meta.get("file_name")
            or ""
        )
        chunk_id = meta.get("chunk_id") or getattr(chunk, "id", "")
        text_hash = meta.get("text_hash")
        text = getattr(chunk, "text", "") or ""
        if not text_hash:
            normalized = " ".join(text.split())
            if normalized:
                text_hash = _hash(normalized)
        if not chunk_id or not text_hash:
            page = meta.get("page") or meta.get("page_start") or meta.get("page_end") or ""
            chunk_index = meta.get("chunk_index") or ""
            length = len(text)
            part = f"{doc_id}:{source_file}:{page}:{chunk_index}:{length}"
        else:
            part = f"{doc_id}:{source_file}:{chunk_id}:{text_hash}"
        parts.append(part)
    fingerprint_string = "|".join(parts)
    return _hash(fingerprint_string)


def extract_intent_tokens(normalized_query: str) -> Iterable[str]:
    tokens = re.findall(r"[a-z0-9]+", normalized_query or "")
    filtered = [t for t in tokens if len(t) >= 4]
    for token in _INTENT_TOKENS:
        if token in normalized_query:
            filtered.append(token)
    return list(dict.fromkeys(filtered))


def is_query_answer_consistent(normalized_query: str, response_text: str, intent: str) -> bool:
    if not normalized_query or not response_text:
        return False
    response_lower = response_text.lower()
    tokens = extract_intent_tokens(normalized_query)
    matches = 0
    for token in tokens:
        if token in response_lower:
            matches += 1
            continue
        synonyms = _SYNONYMS.get(token, [])
        if any(syn in response_lower for syn in synonyms):
            matches += 1
    if matches >= 2:
        return True
    intent_terms = _INTENT_SYNONYMS.get(intent or "", [])
    if intent_terms and any(term in response_lower for term in intent_terms):
        return True
    return False


@dataclass
class QueryContextFingerprintCache:
    redis_client: Any
    prompt_version: str = "v1"
    ttl_seconds: int = 600

    @classmethod
    def build(cls, redis_client: Any, prompt_version: Optional[str] = None) -> "QueryContextFingerprintCache":
        ttl = int(os.getenv("DW_CACHE_TTL_SECONDS", "600"))
        version = (prompt_version or os.getenv("DW_PROMPT_VERSION") or "v1").strip() or "v1"
        return cls(redis_client=redis_client, prompt_version=version, ttl_seconds=ttl)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.redis_client or not cache_key:
            return None
        try:
            cached = self.redis_client.get(cache_key)
        except Exception:
            return None
        if not cached:
            return None
        try:
            return json.loads(cached)
        except Exception:
            return None

    def set(
        self,
        *,
        cache_key: str,
        response_obj: Dict[str, Any],
        normalized_query_hash: str,
        retrieval_fingerprint: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.redis_client or not cache_key:
            return
        payload = {
            "created_at": time.time(),
            "cache_key": cache_key,
            "normalized_query_hash": normalized_query_hash,
            "retrieval_fingerprint": retrieval_fingerprint,
            "metadata": metadata or {},
            "response": response_obj,
        }
        try:
            self.redis_client.setex(cache_key, max(1, int(self.ttl_seconds)), json.dumps(payload))
        except Exception:
            return


__all__ = [
    "QueryContextFingerprintCache",
    "normalize_query",
    "build_cache_key",
    "compute_retrieval_fingerprint",
    "is_query_answer_consistent",
]
