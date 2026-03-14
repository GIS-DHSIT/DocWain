"""Single-entry-point query understanding via fine-tuned LLM + NLU fallback.

Usage::

    from src.intelligence.query_understanding import understand_query
    task_spec = understand_query("Compare all candidates for a Python role")
    # task_spec.intent == "compare", task_spec.domain == "hr", ...

If the fine-tuned model (DocWain-Agent-v2) is unavailable or returns garbage,
the system falls back to the existing NLU engine with confidence=0.4.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, Optional

from src.intelligence.task_spec import TaskSpec
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_MODEL_NAME = "docwain-agent-v2:q4_k_m"
_SYSTEM_PROMPT = (
    "Parse the user's document query into a structured TaskSpec JSON. "
    "Respond ONLY with valid JSON, no extra text.\n"
    "Fields: intent (factual|compare|rank|summarize|extract|generate|analyze|timeline|redirect|clarify), "
    "domain (hr|medical|legal|invoice|insurance|policy|general|content|translation|education), "
    "output_format (paragraph|table|bullets|numbered|chart_data|json), "
    "entities (list of key entities), constraints (dict of structured filters), "
    "scope (all_documents|specific_document|cross_document), "
    "complexity (simple|medium|complex), confidence (0.0-1.0)."
)
_FALLBACK_CONFIDENCE = 0.4
_LLM_TIMEOUT_S = 10.0
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# ── LLM client singleton ────────────────────────────────────────────────────

_llm_client = None
_llm_lock = threading.Lock()
# DISABLED: docwain-agent-v2 (4.9GB) + gpt-oss (13GB) exceeds T4 16GB VRAM.
# Loading two models causes GPU memory contention and 200+s model swap delays.
# NLU fallback is CPU-based, fast (<50ms), and produces accurate TaskSpecs.
# Re-enable once model is fine-tuned and GPU memory management is resolved.
_llm_available: Optional[bool] = False


def _get_llm_client():
    """Lazily create an Ollama client for the fine-tuned model."""
    global _llm_client, _llm_available
    if _llm_available is False:
        return None
    if _llm_client is not None:
        return _llm_client
    with _llm_lock:
        if _llm_client is not None:
            return _llm_client
        try:
            import ollama as _ollama
            # Check if model exists
            models = _ollama.list()
            model_names = [
                m.model for m in getattr(models, "models", [])
            ]
            if not any(_MODEL_NAME.split(":")[0] in n for n in model_names):
                log.info("TaskSpec model %s not found — using NLU fallback", _MODEL_NAME)
                _llm_available = False
                return None
            _llm_available = True
            _llm_client = _ollama
            return _llm_client
        except Exception as exc:
            log.warning("Cannot init TaskSpec LLM client: %s", exc)
            _llm_available = False
            return None


# ── Core entry point ─────────────────────────────────────────────────────────


def understand_query(query: str, *, domain_hint: str = "") -> TaskSpec:
    """Understand a user query and return a unified TaskSpec.

    1. Try the fine-tuned DocWain-Agent-v2 model for JSON TaskSpec
    2. On failure (model unavailable, bad JSON, low confidence) → NLU fallback
    """
    # Try LLM-based understanding first
    task_spec = _try_llm_understanding(query)
    if task_spec is not None and task_spec.confidence >= 0.5:
        return task_spec

    # Fallback to existing NLU engine
    return _nlu_fallback(query, domain_hint=domain_hint)


def _try_llm_understanding(query: str) -> Optional[TaskSpec]:
    """Call the fine-tuned model and parse its JSON response."""
    client = _get_llm_client()
    if client is None:
        return None
    try:
        t0 = time.monotonic()
        response = client.chat(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.1, "num_predict": 512},
        )
        elapsed = time.monotonic() - t0
        raw = response.get("message", {}).get("content", "")
        log.debug("TaskSpec LLM response (%.1fs): %s", elapsed, raw[:200])

        # Extract JSON from response (model may wrap in markdown fences)
        match = _JSON_RE.search(raw)
        if not match:
            log.warning("TaskSpec LLM returned non-JSON: %s", raw[:100])
            return None

        task_spec = TaskSpec.from_json(match.group())
        return task_spec

    except json.JSONDecodeError as exc:
        log.warning("TaskSpec JSON parse error: %s", exc)
        return None
    except Exception as exc:
        log.warning("TaskSpec LLM call failed: %s", exc)
        return None


# ── NLU Fallback ─────────────────────────────────────────────────────────────

# Intent mapping from NLU engine labels to TaskSpec intents
_NLU_INTENT_MAP: Dict[str, str] = {
    "factual": "factual",
    "summary": "summarize",
    "comparison": "compare",
    "ranking": "rank",
    "timeline": "timeline",
    "reasoning": "analyze",
    "cross_document": "analyze",
    "analytics": "analyze",
    "multi_field": "extract",
    "extraction": "extract",
    "generate": "generate",
}

# Domain mapping from NLU domain labels
_NLU_DOMAIN_MAP: Dict[str, str] = {
    "hr": "hr",
    "medical": "medical",
    "legal": "legal",
    "invoice": "invoice",
    "insurance": "insurance",
    "policy": "policy",
    "content": "content",
    "translation": "translation",
    "education": "education",
}

# Scope inference from intent
_SCOPE_BY_INTENT: Dict[str, str] = {
    "compare": "cross_document",
    "rank": "all_documents",
    "summarize": "all_documents",
    "analyze": "all_documents",
    "timeline": "all_documents",
    "generate": "all_documents",
}

# Complexity inference from intent
_COMPLEXITY_BY_INTENT: Dict[str, str] = {
    "factual": "simple",
    "extract": "simple",
    "summarize": "medium",
    "compare": "medium",
    "rank": "medium",
    "generate": "medium",
    "analyze": "complex",
    "timeline": "complex",
}


def _nlu_fallback(query: str, *, domain_hint: str = "") -> TaskSpec:
    """Build TaskSpec from existing NLU engine classifications."""
    intent = "factual"
    domain = domain_hint or "general"
    entities: list = []

    try:
        from src.nlp.nlu_engine import classify_intent, classify_domain_task
        nlu_intent = classify_intent(query)
        intent = _NLU_INTENT_MAP.get(nlu_intent, "factual")

        domain_task = classify_domain_task(query, domain=domain_hint)
        if domain_task:
            domain = _NLU_DOMAIN_MAP.get(
                domain_task.get("domain", ""), "general"
            )
    except Exception as exc:
        log.warning("NLU engine fallback error: %s", exc)

    try:
        from src.nlp.query_entity_extractor import extract_all_entities
        entities = extract_all_entities(query) or []
    except Exception as exc:
        log.debug("Entity extraction fallback error: %s", exc)

    # Infer scope from intent
    scope = _SCOPE_BY_INTENT.get(intent, "all_documents")

    # Infer complexity from intent
    complexity = _COMPLEXITY_BY_INTENT.get(intent, "medium")

    # Infer output format from query keywords
    output_format = _infer_output_format(query)

    return TaskSpec(
        intent=intent,
        domain=domain,
        output_format=output_format,
        entities=entities,
        constraints={},
        scope=scope,
        complexity=complexity,
        confidence=_FALLBACK_CONFIDENCE,
    )


def _infer_output_format(query: str) -> str:
    """Heuristic output-format detection from query text."""
    q = query.lower()
    if any(w in q for w in ("table", "matrix", "side by side", "grid")):
        return "table"
    if any(w in q for w in ("list", "bullet", "bullets")):
        return "bullets"
    if any(w in q for w in ("chart", "graph", "visualize", "plot")):
        return "chart_data"
    if any(w in q for w in ("number", "ranked", "top ")):
        return "numbered"
    if any(w in q for w in ("json", "structured data")):
        return "json"
    return "paragraph"
