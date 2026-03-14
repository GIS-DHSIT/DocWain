"""LLM-based query classifier using fast model (llama3.2).

Classifies intent, domain, entity, and scope in a single LLM call.
Falls back gracefully to None on any failure — never blocks the pipeline.
"""
from __future__ import annotations

import concurrent.futures
import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Optional

from .role_prompts import CLASSIFIER_INTENT_TEMPLATE, CLASSIFIER_SYSTEM

logger = get_logger(__name__)

_VALID_INTENTS = frozenset({
    "factual", "comparison", "summary", "ranking", "timeline",
    "reasoning", "multi_field", "cross_document", "contact", "extract",
})
_VALID_DOMAINS = frozenset({"hr", "invoice", "legal", "medical", "policy", "generic"})
_VALID_SCOPES = frozenset({"all_profile", "specific_document", "targeted"})

# ── Heuristic keyword signals for fallback classification ───────────
_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "hr": [
        "work experience", "resume", "curriculum vitae", "job title",
        "years of experience", "technical skills", "candidate",
        "employment history", "education background", "qualifications",
    ],
    "invoice": [
        "invoice number", "amount due", "bill to", "total due",
        "payment terms", "purchase order", "unit price", "invoice date",
        "remittance", "balance due",
    ],
    "legal": [
        "agreement between", "terms and conditions", "liability clause",
        "governing law", "indemnification", "breach of contract",
        "non-disclosure", "intellectual property",
    ],
    "medical": [
        "patient name", "diagnosis", "medication", "dosage",
        "medical history", "lab results", "blood pressure",
        "clinical findings", "prescription", "treatment plan",
    ],
    "policy": [
        "insurance policy", "policy number", "coverage period",
        "sum insured", "premium amount", "exclusions",
        "policyholder", "claim process", "deductible",
        "policy document", "coverage limit",
    ],
}

_INTENT_SIGNALS: dict[str, list[str]] = {
    "comparison": ["compare", "vs", "versus", "difference between", "differences"],
    "summary": ["summarize", "overview", "key points", "brief"],
    "ranking": ["rank", "best", "top", "worst", "highest", "lowest"],
    "timeline": ["when", "timeline", "chronolog", "date", "history of"],
    "extract": ["extract", "list all", "find all", "get all"],
    "contact": ["contact", "email", "phone", "reach"],
}

_SCOPE_ALL_SIGNALS = [
    "all", "every", "each", "across", "compare", "total",
    "how many", "which ones", "list all",
]

def _heuristic_classify(query: str, profile_domain: Optional[str] = None) -> QueryClassification:
    """Keyword-based fallback when LLM classifier times out or fails."""
    q = query.lower()

    # Domain detection via multi-word phrase matching
    domain_scores: dict[str, int] = {}
    for domain, phrases in _DOMAIN_SIGNALS.items():
        score = sum(1 for p in phrases if p in q)
        if score:
            domain_scores[domain] = score

    if domain_scores:
        domain = max(domain_scores, key=domain_scores.get)
    elif profile_domain and profile_domain in _VALID_DOMAINS:
        domain = profile_domain
    else:
        domain = "generic"

    # Intent detection
    intent = "factual"
    for intent_name, phrases in _INTENT_SIGNALS.items():
        if any(p in q for p in phrases):
            intent = intent_name
            break

    # Scope detection
    scope = "targeted"
    if any(s in q for s in _SCOPE_ALL_SIGNALS):
        scope = "all_profile"

    # Entity detection — simple quoted or capitalized name extraction
    entity = None
    quoted = re.findall(r'"([^"]+)"', query)
    if quoted:
        entity = quoted[0]

    return QueryClassification(
        intent=intent,
        domain=domain,
        entity=entity,
        scope=scope,
        confidence=0.3,
    )

@dataclass
class QueryClassification:
    """Result of LLM-based query classification."""
    intent: str
    domain: str
    entity: Optional[str]
    scope: str
    confidence: float

    def __post_init__(self):
        if self.intent not in _VALID_INTENTS:
            self.intent = "factual"
        if self.domain not in _VALID_DOMAINS:
            self.domain = "generic"
        if self.scope not in _VALID_SCOPES:
            self.scope = "targeted"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

def _parse_classification(raw: str) -> Optional[QueryClassification]:
    """Parse JSON from LLM output into a QueryClassification."""
    if not raw:
        return None

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try to extract JSON object
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None

    intent = str(data.get("intent", "factual")).lower().strip()
    domain = str(data.get("domain", "generic")).lower().strip()
    entity = data.get("entity")
    if entity is not None:
        entity = str(entity).strip() or None
        if entity and entity.lower() in ("null", "none", "n/a"):
            entity = None
    scope = str(data.get("scope", "targeted")).lower().strip()
    try:
        confidence = float(data.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5

    return QueryClassification(
        intent=intent,
        domain=domain,
        entity=entity,
        scope=scope,
        confidence=confidence,
    )

def classify_query(
    query: str,
    llm_client: Any,
    timeout_s: float = 25.0,
    profile_domain: Optional[str] = None,
) -> Optional[QueryClassification]:
    """Classify a query using the LLM classifier.

    Parameters
    ----------
    query : str
        The user query to classify.
    llm_client : Any
        An LLM client (MultiAgentGateway or any duck-typed client).
        If it has a `classify()` method, that will be used.
    timeout_s : float
        Maximum seconds to wait for classification.
    profile_domain : str, optional
        Profile-level domain hint used as fallback in heuristic classification.

    Returns
    -------
    QueryClassification or None if classification fails.
    """
    if not query or not query.strip():
        return None

    prompt = f"{CLASSIFIER_SYSTEM}\n\n{CLASSIFIER_INTENT_TEMPLATE.format(query=query)}"

    def _call() -> str:
        # Set task scope for task-aware routing
        try:
            from src.llm.task_router import task_scope, TaskType
            _ctx = task_scope(TaskType.QUERY_CLASSIFICATION)
        except ImportError:
            from contextlib import nullcontext
            _ctx = nullcontext()
        with _ctx:
            # Prefer the role-aware classify() method
            if hasattr(llm_client, "classify"):
                return llm_client.classify(prompt)
            # Fall back to standard generate
            if hasattr(llm_client, "generate_for_role"):
                from .multi_agent import AgentRole
                return llm_client.generate_for_role(AgentRole.CLASSIFIER, prompt, max_retries=1, backoff=0.2)
            return llm_client.generate(prompt, max_retries=1, backoff=0.2)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        raw = future.result(timeout=timeout_s)
        result = _parse_classification(raw)
        if result:
            logger.info(
                "Query classified: intent=%s domain=%s entity=%s scope=%s confidence=%.2f",
                result.intent, result.domain, result.entity, result.scope, result.confidence,
            )
        return result
    except concurrent.futures.TimeoutError:
        logger.warning("Query classification timed out after %.1fs — using heuristic fallback", timeout_s)
        return _heuristic_classify(query, profile_domain)
    except Exception as exc:
        logger.warning("Query classification failed: %s — using heuristic fallback", exc)
        return _heuristic_classify(query, profile_domain)
    finally:
        executor.shutdown(wait=False)
