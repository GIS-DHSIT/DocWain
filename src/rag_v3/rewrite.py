from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import re
import time
from typing import Optional, Tuple

from .types import LLMBudget

logger = logging.getLogger(__name__)

REWRITE_TIMEOUT_MS = 2500
REWRITE_MAX_TOKENS = 96
REWRITE_CACHE_TTL_SEC = 24 * 60 * 60

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "at",
    "from",
    "by",
    "about",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "their",
    "its",
}


def rewrite_query(
    *,
    query: str,
    subscription_id: str,
    profile_id: Optional[str],
    redis_client: Optional[object],
    llm_client: Optional[object],
    budget: LLMBudget,
    correlation_id: Optional[str] = None,
    rewrite_timeout_ms: int = REWRITE_TIMEOUT_MS,
    max_tokens: int = REWRITE_MAX_TOKENS,
) -> str:
    normalized = _normalize(query)
    cache_key = _cache_key(subscription_id, profile_id, normalized)

    if redis_client is not None:
        cached = redis_client.get(cache_key)
        if isinstance(cached, bytes):
            cached = cached.decode("utf-8")
        if isinstance(cached, str) and cached.strip():
            return cached.strip()

    if not _should_rewrite(normalized):
        _cache(redis_client, cache_key, normalized)
        return normalized

    if not (llm_client and budget.consume()):
        _cache(redis_client, cache_key, normalized)
        return normalized

    prompt = (
        "Rewrite the user query into a concise, unambiguous search query. "
        "Do NOT introduce new entities or facts. Preserve all names and numbers. "
        "Return only the rewritten query.\n\n"
        f"USER QUERY: {normalized}\n"
        "REWRITTEN QUERY:"
    )

    try:
        rewritten, meta, timed_out = _generate_with_timeout(
            llm_client,
            prompt,
            max_tokens=max_tokens,
            timeout_ms=rewrite_timeout_ms,
        )
        if timed_out:
            logger.warning(
                "RAG v3 rewrite timed out; using normalized query",
                extra={"stage": "rewrite", "correlation_id": correlation_id},
            )
            _cache(redis_client, cache_key, normalized)
            return normalized
        rewritten = _normalize(str(rewritten or ""))
        if not rewritten or rewritten == normalized:
            _cache(redis_client, cache_key, normalized)
            return normalized
        if _is_safe_rewrite(original=normalized, rewritten=rewritten):
            _cache(redis_client, cache_key, rewritten)
            return rewritten
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG v3 rewrite failed; using normalized query: %s",
            exc,
            extra={"stage": "rewrite", "correlation_id": correlation_id},
        )

    _cache(redis_client, cache_key, normalized)
    return normalized


def _normalize(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\n", " ").replace("\t", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _should_rewrite(text: str) -> bool:
    if not text:
        return False
    tokens = text.split()
    if len(tokens) <= 5:
        return False
    lowered = text.lower()
    ambiguous = any(word in lowered for word in ("this", "that", "it", "they", "them", "those", "these"))
    long_query = len(tokens) > 12
    return ambiguous or long_query


def _is_safe_rewrite(*, original: str, rewritten: str) -> bool:
    if not rewritten or rewritten == original:
        return False
    if len(rewritten) < 3:
        return False

    original_lower = original.lower()
    rewritten_lower = rewritten.lower()

    original_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", original))
    rewritten_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", rewritten))
    if rewritten_numbers - original_numbers:
        return False

    original_caps = {w for w in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", original)}
    rewritten_caps = {w for w in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", rewritten)}
    if rewritten_caps - original_caps:
        return False

    orig_tokens = {t for t in re.findall(r"[a-z0-9]+", original_lower) if t not in _STOPWORDS}
    new_tokens = {t for t in re.findall(r"[a-z0-9]+", rewritten_lower) if t not in _STOPWORDS}
    if not new_tokens:
        return False
    if len(new_tokens - orig_tokens) > max(2, len(orig_tokens) // 2):
        return False

    return True


def _cache_key(subscription_id: str, profile_id: Optional[str], query: str) -> str:
    seed = f"{subscription_id}:{profile_id or ''}:{query}".encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()
    return f"ragv3:rewrite:{digest}"


def _cache(redis_client: Optional[object], key: str, value: str) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(key, REWRITE_CACHE_TTL_SEC, value)
    except Exception:
        try:
            redis_client.set(key, value)
        except Exception:
            return


def _generate_with_timeout(
    llm_client: object,
    prompt: str,
    *,
    max_tokens: int,
    timeout_ms: int,
) -> Tuple[str, dict, bool]:
    options = {
        "num_predict": int(max_tokens),
        "max_output_tokens": int(max_tokens),
    }

    def _call():
        if hasattr(llm_client, "generate_with_metadata"):
            text, meta = llm_client.generate_with_metadata(
                prompt,
                options=options,
                max_retries=1,
                backoff=0.3,
            )
            return text or "", meta or {}
        text = llm_client.generate(prompt, max_retries=1, backoff=0.3)
        return text or "", {}

    deadline = max(0.05, float(timeout_ms) / 1000.0)
    start = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        text, meta = future.result(timeout=deadline)
        _ = time.monotonic() - start
        executor.shutdown(wait=False)
        return text, meta, False
    except concurrent.futures.TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return "", {}, True
