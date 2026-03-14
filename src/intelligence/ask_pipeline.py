from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from typing import Any, Dict, Iterable, List, Optional

from src.intelligence.facts_store import FactsStore
from src.intelligence.deterministic_router import DeterministicRoute, route_query
from src.intelligence.formatter import format_facts_response, _collect_evidence_sources
from src.intelligence.response_composer import build_greeting_response, compose_task_response

logger = get_logger(__name__)

def _collect_evidence(facts: Iterable[Dict[str, Any]], catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _collect_evidence_sources(facts, catalog)

def _facts_for_section_focus(facts: List[Dict[str, Any]], section_focus: List[str]) -> List[Dict[str, Any]]:
    if not section_focus:
        return facts
    focus_set = {str(k) for k in section_focus if k}
    return [fact for fact in facts if str(fact.get("section_kind")) in focus_set]

def answer_with_section_intelligence(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    session_state: Dict[str, Any],
    catalog: Dict[str, Any],
    entities_cache: Optional[Dict[str, Any]] = None,
    redis_client: Optional[Any] = None,
    db: Optional[Any] = None,
    route_override: Optional[DeterministicRoute] = None,
) -> Optional[Dict[str, Any]]:
    start_time = time.time()
    route_plan = route_override or route_query(query, session_state, catalog, entities_cache or {})
    task_type = route_plan.task_type

    if task_type == "greet":
        response_text = build_greeting_response(catalog)
        try:
            from src.intelligence.conversational_nlp import generate_conversational_response
            _resp = generate_conversational_response(query, catalog=catalog)
            if _resp and _resp.text:
                response_text = _resp.text
        except Exception:
            pass
        return {
            "response": response_text,
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {"task": "greet", "route_plan": route_plan.to_dict()},
        }
    if task_type in {"rank", "summarize", "compare", "generate"}:
        return None

    facts_store = FactsStore(redis_client=redis_client, db=db)
    target_document = None
    if route_plan.target_document_ids:
        target_document = route_plan.target_document_ids[0]
    if not target_document:
        target_document = session_state.get("active_document_id")

    facts_payload: Optional[Dict[str, Any]] = None
    facts_scope = "profile"
    doc_version_hash = None
    if target_document and catalog.get("documents"):
        for doc in catalog.get("documents") or []:
            if str(doc.get("document_id")) == str(target_document):
                doc_version_hash = doc.get("doc_version_hash")
                break
    if route_plan.scope == "current_document" and target_document:
        facts_payload = facts_store.get_document_facts(
            subscription_id,
            profile_id,
            target_document,
            doc_version_hash=doc_version_hash,
        )
        facts_scope = "document"
    if not facts_payload:
        facts_payload = facts_store.get_profile_facts(subscription_id, profile_id)
        facts_scope = "profile"

    if not facts_payload:
        return None

    facts_list = facts_payload.get("section_facts") or []
    if not facts_list and facts_payload.get("aggregated_facts"):
        agg = facts_payload.get("aggregated_facts") or {}
        for kind, entry in (agg.get("by_section_kind") or {}).items():
            facts_list.append(
                {
                    "section_kind": kind,
                    "entities": entry.get("entities") or [],
                    "attributes": entry.get("attributes") or {},
                    "evidence_spans": [],
                }
            )

    relevant_facts = _facts_for_section_focus(facts_list, route_plan.section_focus)
    if not relevant_facts:
        return None

    sources = _collect_evidence(relevant_facts, catalog)
    response_text = format_facts_response(
        query=query,
        route=route_plan,
        facts=relevant_facts,
        catalog=catalog,
    )
    if not response_text:
        return None

    response_text = compose_task_response(
        response_text=response_text,
        route_plan=route_plan,
        query=query,
    )

    metadata = {
        "route_plan": route_plan.to_dict(),
        "task_type": task_type,
        "facts_scope": facts_scope,
        "facts_found": True,
        "facts_used": len(relevant_facts),
        "facts_lookup_time_ms": int((time.time() - start_time) * 1000),
        "acknowledged": True,
    }
    return {
        "response": response_text,
        "sources": sources,
        "grounded": bool(sources),
        "context_found": True,
        "metadata": metadata,
    }

__all__ = ["answer_with_section_intelligence"]
