from __future__ import annotations

import concurrent.futures
import hashlib
from src.utils.logging_utils import get_logger
import re
import threading
import time
from typing import Optional, Tuple

from .types import LLMBudget

logger = get_logger(__name__)

REWRITE_TIMEOUT_MS = 5000
REWRITE_MAX_TOKENS = 96
REWRITE_CACHE_TTL_SEC = 24 * 60 * 60

# Shared semaphore to avoid rewrite and extract competing for the same GPU
_ollama_semaphore: Optional[threading.Semaphore] = None

def _get_ollama_semaphore() -> threading.Semaphore:
    global _ollama_semaphore
    if _ollama_semaphore is None:
        try:
            from src.api.config import Config
            _ollama_semaphore = threading.Semaphore(getattr(Config.LLM, "MAX_CONCURRENCY", 2))
        except Exception:
            _ollama_semaphore = threading.Semaphore(2)
    return _ollama_semaphore

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

_DOMAIN_REWRITE_HINTS: dict[str, str] = {
    "medical": "For medical queries, preserve clinical terms (diagnoses, medications, dosages, lab values). ",
    "hr": "For HR/resume queries, preserve job titles, skill names, and company names. ",
    "invoice": "For invoice queries, preserve monetary amounts, invoice numbers, vendor names. ",
    "legal": "For legal queries, preserve clause references, party names, legal terms. ",
    "policy": "For policy queries, preserve coverage terms, policy numbers, premium amounts. ",
    "tax": "For tax queries, preserve tax form numbers, filing years, income amounts, deduction details. ",
}

_DOMAIN_DETECT_RE = re.compile(
    r"\b(?:patient|diagnosis|medication|prescription|clinical|symptoms)\b", re.I
)
_DOMAIN_DETECT_HR_RE = re.compile(
    r"\b(?:resume|candidate|skills?|experience|salary|compensation|employee)\b", re.I
)
_DOMAIN_DETECT_INVOICE_RE = re.compile(
    r"\b(?:invoice|payment|vendor|amount\s+due|billing|purchase\s+order)\b", re.I
)
_DOMAIN_DETECT_LEGAL_RE = re.compile(
    r"\b(?:contract|agreement|clause|liability|indemnification|warranty)\b", re.I
)
_DOMAIN_DETECT_POLICY_RE = re.compile(
    r"\b(?:policy|premium|deductible|coverage|policyholder|insured)\b", re.I
)
_DOMAIN_DETECT_TAX_RE = re.compile(
    r"\b(?:tax\s+return|taxable\s+income|w-?2|1099|schedule\s+c|irs|deduction|withholding)\b", re.I
)

def _detect_domain_guidance(query: str) -> str:
    """Detect domain from query text and return appropriate rewrite guidance."""
    lowered = query.lower()
    if _DOMAIN_DETECT_RE.search(lowered):
        return _DOMAIN_REWRITE_HINTS["medical"]
    if _DOMAIN_DETECT_HR_RE.search(lowered):
        return _DOMAIN_REWRITE_HINTS["hr"]
    if _DOMAIN_DETECT_INVOICE_RE.search(lowered):
        return _DOMAIN_REWRITE_HINTS["invoice"]
    if _DOMAIN_DETECT_LEGAL_RE.search(lowered):
        return _DOMAIN_REWRITE_HINTS["legal"]
    if _DOMAIN_DETECT_POLICY_RE.search(lowered):
        return _DOMAIN_REWRITE_HINTS["policy"]
    if _DOMAIN_DETECT_TAX_RE.search(lowered):
        return _DOMAIN_REWRITE_HINTS["tax"]
    return ""

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

    # Extract proper nouns for explicit preservation instruction
    _proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', normalized)
    _names_instruction = ""
    if _proper_nouns:
        # Filter out common sentence starters
        _starts = {"What", "How", "When", "Where", "Who", "Which", "Can",
                    "Does", "Did", "Show", "List", "Find", "Get", "Tell",
                    "Compare", "Rank", "The", "This", "That"}
        _real_names = [n for n in _proper_nouns if n not in _starts]
        if _real_names:
            _names_instruction = f"\nIMPORTANT: Preserve these exact names: {', '.join(_real_names)}"

    # Domain-aware rewrite guidance — helps the LLM understand query context
    _domain_guidance = _detect_domain_guidance(normalized)

    # Detect multi-aspect queries for decomposition hints
    _multi_aspect_hint = ""
    _and_clauses = re.split(r'\b(?:and\s+(?:also)?|as\s+well\s+as)\b', normalized, flags=re.IGNORECASE)
    _question_marks = normalized.count("?")
    if len(_and_clauses) >= 2 and any(len(c.strip().split()) >= 3 for c in _and_clauses):
        _multi_aspect_hint = "\nThis query has multiple aspects — preserve ALL aspects in the rewrite."
    elif _question_marks >= 2:
        _multi_aspect_hint = "\nThis query contains multiple questions — preserve each question."

    prompt = (
        "Rewrite the user query into a concise, unambiguous search query. "
        "Do NOT introduce new entities or facts. Preserve all names, numbers, and dates. "
        "Remove conversational filler but keep domain-specific terms intact. "
        f"{_domain_guidance}"
        "Return only the rewritten query."
        f"{_names_instruction}"
        f"{_multi_aspect_hint}\n\n"
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
            fallback = _smart_timeout_fallback(normalized)
            logger.warning(
                "RAG v3 rewrite timed out; using smart fallback (original=%r, fallback=%r)",
                normalized[:200], fallback[:200],
                extra={"stage": "rewrite", "correlation_id": correlation_id},
            )
            _cache(redis_client, cache_key, fallback)
            return fallback
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

# Conversational filler phrases to strip in smart fallback
_FILLER_PATTERNS = [
    re.compile(r"^(?:can\s+you\s+)?(?:please\s+)?tell\s+me\s+", re.I),
    re.compile(r"^(?:can\s+you\s+)?(?:please\s+)?show\s+me\s+", re.I),
    re.compile(r"^(?:i\s+would\s+like\s+to\s+know\s+)", re.I),
    re.compile(r"^(?:i\s+want\s+to\s+know\s+)", re.I),
    re.compile(r"^(?:could\s+you\s+(?:please\s+)?)", re.I),
    re.compile(r"^(?:can\s+you\s+(?:please\s+)?)", re.I),
    re.compile(r"^(?:please\s+)", re.I),
    re.compile(r"^(?:i\s+need\s+(?:to\s+know|information\s+(?:about|on))\s+)", re.I),
    re.compile(r"^(?:i['\u2019]?m\s+trying\s+to\s+(?:find|understand|figure\s+out)\s+)", re.I),
    re.compile(r"^(?:could\s+you\s+explain\s+)", re.I),
    re.compile(r"^(?:help\s+me\s+(?:understand|find|with)\s+)", re.I),
    re.compile(r"^(?:do\s+you\s+(?:know|have)\s+)", re.I),
    re.compile(r"^(?:what\s+about\s+)", re.I),
    re.compile(r"^(?:i\s+(?:was\s+)?wonder(?:ing)?\s+(?:if|about|what)\s+)", re.I),
]

# Intent-preserving words that must survive filler stripping
_INTENT_WORDS = frozenset({
    "compare", "rank", "list", "summarize", "generate", "create",
    "write", "draft", "extract", "analyze", "explain", "describe",
    "calculate", "count", "total", "average", "find", "search",
    "translate", "convert", "screen", "review", "assess", "evaluate",
})

# Pattern to detect quoted strings that should be preserved
_QUOTED_RE = re.compile(r'"[^"]+"|\'[^\']+\'')

def _smart_timeout_fallback(query: str) -> str:
    """Produce a smarter fallback than raw normalized query when rewrite times out.

    Strips conversational filler while preserving key phrases, proper nouns,
    and quoted strings.
    """
    if not query or len(query.split()) <= 5:
        return query  # Short queries don't need stripping

    # Preserve quoted strings by replacing with placeholders
    quoted_parts = _QUOTED_RE.findall(query)
    working = query
    for i, part in enumerate(quoted_parts):
        working = working.replace(part, f"__QUOTED_{i}__", 1)

    # Strip conversational filler from the beginning
    for pattern in _FILLER_PATTERNS:
        working = pattern.sub("", working, count=1)

    # Strip parenthetical clauses (non-essential context)
    working = re.sub(r"\([^)]{5,80}\)", "", working)

    # Restore quoted strings
    for i, part in enumerate(quoted_parts):
        working = working.replace(f"__QUOTED_{i}__", part, 1)

    result = working.strip()
    # Safety: if stripping removed too much, return original
    if not result or len(result) < 3:
        return query

    # Safety: if filler stripping removed intent words, re-add ALL of them
    _original_lower = set(query.lower().split())
    _result_lower = set(result.lower().split())
    _lost_intent = (_original_lower & _INTENT_WORDS) - _result_lower
    if _lost_intent:
        # Prepend all lost intent words to preserve multi-intent queries
        result = f"{' '.join(sorted(_lost_intent))} {result}"

    # For comparison queries, ensure both entities survive
    if _IMPLICIT_COMPARISON_RE.search(query):
        # Extract capitalized names from original
        orig_names = set(re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b', query))
        result_names = set(re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b', result))
        _common_starts = {"What", "How", "When", "Where", "Who", "Which", "Can",
                          "Does", "Show", "List", "Find", "Compare", "The"}
        lost_names = (orig_names - result_names) - _common_starts
        if lost_names:
            result = result + " " + " ".join(lost_names)

    return result

def _normalize(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\n", " ").replace("\t", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

_IMPLICIT_COMPARISON_RE = re.compile(
    r"\b(?:versus|vs\.?|compared?\s+to|differ(?:ence|ent)|better|worse|"
    r"more\s+than|less\s+than|highest|lowest|best|worst)\b",
    re.IGNORECASE,
)
_CONJUNCTION_CLAUSE_RE = re.compile(
    r"\b(?:and\s+also|and\s+then|as\s+well\s+as|in\s+addition)\b",
    re.IGNORECASE,
)

def _should_rewrite(text: str) -> bool:
    if not text:
        return False
    tokens = text.split()
    if len(tokens) <= 3:
        return False
    lowered = text.lower()

    # Ambiguous pronouns needing resolution — allow shorter queries (4-5 tokens)
    _pronoun_words = {"this", "that", "it", "they", "them", "those", "these",
                      "his", "her", "their", "its"}
    ambiguous = any(word in _pronoun_words for word in lowered.split())
    # Short pronoun queries (4-5 tokens) only rewrite if pronoun is present
    if len(tokens) <= 5 and not ambiguous:
        return False
    # Long queries benefit from simplification
    long_query = len(tokens) > 12
    # Multi-clause queries with conjunctions need restructuring
    has_multi_clause = bool(_CONJUNCTION_CLAUSE_RE.search(lowered))
    # Implicit comparisons need explicit reformulation
    has_implicit_comparison = bool(_IMPLICIT_COMPARISON_RE.search(lowered)) and len(tokens) > 8
    # Nested parentheticals should be simplified
    has_parenthetical = "(" in text and ")" in text

    # Negated queries benefit from reformulation: "who doesn't have X?" → "employees without X"
    has_negation = bool(re.search(
        r"\b(?:doesn['\u2019]?t|don['\u2019]?t|isn['\u2019]?t|aren['\u2019]?t|"
        r"without|lacking|missing|no\s+\w+)\b", lowered
    )) and len(tokens) > 6

    # Conditional/hypothetical queries need simplification: "if X then what Y"
    has_conditional = bool(re.search(
        r"\b(?:if\s+\w+.*?\b(?:then|what|would|could|should)\b|"
        r"assuming\s+|suppose\s+|in\s+case\s+|provided\s+that\b)",
        lowered,
    )) and len(tokens) > 8

    # Temporal queries with relative references: "last month", "recently", "since 2020"
    has_relative_temporal = bool(re.search(
        r"\b(?:last\s+(?:month|year|quarter|week)|"
        r"recent(?:ly)?|past\s+\d+\s+(?:months?|years?|days?)|"
        r"since\s+\d{4}|before\s+\d{4}|after\s+\d{4})\b",
        lowered,
    )) and len(tokens) > 6

    return (ambiguous or long_query or has_multi_clause or has_implicit_comparison
            or has_parenthetical or has_negation or has_conditional or has_relative_temporal)

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
    new_caps = rewritten_caps - original_caps
    if new_caps:
        # Allow new caps if they are: (a) the first word (sentence-starter capitalization),
        # or (b) expansions of abbreviations whose lowercase form appears in original
        _first_word = rewritten.split()[0] if rewritten else ""
        _orig_lower = original.lower()
        _unexplained = {
            w for w in new_caps
            if w != _first_word and not re.search(r'\b' + re.escape(w.lower()) + r'\b', _orig_lower)
        }
        if _unexplained:
            return False

    orig_tokens = {t for t in re.findall(r"[a-z0-9]+", original_lower) if t not in _STOPWORDS}
    new_tokens = {t for t in re.findall(r"[a-z0-9]+", rewritten_lower) if t not in _STOPWORDS}
    if not new_tokens:
        return False
    if len(new_tokens - orig_tokens) > 3:
        return False

    # Proper noun preservation — never generalize specific names
    _original_proper = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', original))
    _rewritten_proper = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', rewritten))
    # Exclude common sentence-start words
    _common_starts = {"What", "How", "When", "Where", "Who", "Which", "Can", "Does", "Did", "Show", "List", "Find", "Get", "Tell", "Give", "Compare", "Rank", "The", "This", "That", "Are", "Were", "Has", "Have", "Is"}
    _original_names = _original_proper - _common_starts
    _rewritten_names = _rewritten_proper - _common_starts
    if _original_names and not (_original_names & _rewritten_names):
        # All proper nouns were lost in rewrite — unsafe
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
    sem = _get_ollama_semaphore()

    def _call():
        acquired = sem.acquire(timeout=max(0.01, float(timeout_ms) / 1000.0))
        if not acquired:
            return "", {}
        try:
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
        finally:
            sem.release()

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
