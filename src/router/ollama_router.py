from __future__ import annotations

import json
import logging
from typing import Optional

import ollama

from src.observability.metrics import metrics_store
from src.router.schema import RouterDecision

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """
You are a routing engine for DocWain. Return ONLY valid JSON matching this schema:
{
  "intent": {
    "category": "meta|qa|summarize|extract|compare|rank|timeline|compute|transform|unknown",
    "confidence": 0.0,
    "user_goal": "string"
  },
  "scope": {
    "subscription_id": "string",
    "profile_id": "string",
    "profile_name": "string",
    "document_filters": {
      "document_type_hints": ["string"],
      "file_name_hints": ["string"],
      "must_use_tables": true,
      "must_use_images": false
    }
  },
  "retrieval_plan": {
    "strategy": "none|semantic|hybrid|summary_first|table_first|multi_doc",
    "chunk_kinds": ["section_text","table_text","image_caption","doc_summary","section_summary","structured_field"],
    "query_rewrites": ["string"],
    "filters": [
      {"field":"subscription_id","op":"==","value":"string"},
      {"field":"profile_id","op":"==","value":"string"}
    ]
  },
  "response_policy": {
    "include_persona": true,
    "no_questions": true,
    "no_refusals": true,
    "style": "direct|explanatory|executive"
  }
}
Rules:
- include_persona MUST be true ONLY for category="meta".
- retrieval_plan.strategy MUST be "none" for category="meta".
- profile_id filter MUST ALWAYS be present for non-meta categories.
- no_questions=true and no_refusals=true ALWAYS.
Return JSON only. Do not add commentary.
""".strip()


def _build_prompt(query: str, subscription_id: str, profile_id: str, profile_name: str) -> str:
    return (
        f"{ROUTER_PROMPT}\n\n"
        f"subscription_id: {subscription_id}\n"
        f"profile_id: {profile_id}\n"
        f"profile_name: {profile_name}\n"
        f"user_query: {query}"
    )


def route_with_ollama(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    profile_name: str,
    model_name: Optional[str],
) -> Optional[RouterDecision]:
    if not model_name:
        return None
    prompt = _build_prompt(query, subscription_id, profile_id, profile_name)
    try:
        response = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0})
        payload = json.loads((response.get("response") or "").strip())
        return RouterDecision.parse_obj(payload)
    except Exception as exc:  # noqa: BLE001
        metrics_store().increment("ollama_router_fail_count")
        logger.debug("Ollama router failed: %s", exc)
        return None


__all__ = ["route_with_ollama", "ROUTER_PROMPT"]
