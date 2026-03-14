from __future__ import annotations

from src.utils.logging_utils import get_logger
import random
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.vector_store import build_collection_name, QdrantVectorStore, REQUIRED_PAYLOAD_INDEX_FIELDS
from src.embedding.model_loader import get_embedding_model
from src.intelligence.kg_query import KGQueryService
from src.intelligence.redis_intel_cache import RedisIntelCache, SessionState
from src.intelligence.deterministic_router import DeterministicRoute, route_query
from src.intelligence.pipelines import generate_document_task, rank_task, summarize_task
from src.intelligence.ask_pipeline import answer_with_section_intelligence
from src.intelligence.formatter import format_facts_response, _collect_evidence_sources
from src.intelligence.section_intelligence_builder import SectionIntelligenceBuilder
from src.intelligence.response_composer import compose_task_response, generate_ack
from src.intelligence.facts_store import FactsStore
from src.intelligence.kg_updater import KGUpdater
from src.services.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig, RetrievalCandidate
from src.utils.payload_utils import get_content_text

logger = get_logger(__name__)

def _normalize_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()

def _extract_name_candidates(query: str) -> List[str]:
    if not query:
        return []
    candidates: List[str] = []
    patterns = [
        r"\bfor\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b",
        r"\bof\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b",
        r"\babout\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b",
        r"\bregarding\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b",
        r"\b(?:patient|vendor)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b",
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})'s\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query or "")
        if match:
            candidates.append(match.group(1).strip())
    return list(dict.fromkeys([c for c in candidates if c]))

def _resolve_name_documents(
    *,
    name: str,
    entities_cache: Dict[str, Any],
    catalog: Dict[str, Any],
    facts_store: FactsStore,
    subscription_id: str,
    profile_id: str,
) -> List[str]:
    if not name:
        return []
    name_norm = _normalize_value(name)
    doc_ids: List[str] = []

    for ent in entities_cache.get("entities") or []:
        ent_type = str(ent.get("type") or "").upper()
        if ent_type and ent_type not in {"PERSON", "ORG", "ORGANIZATION", "VENDOR", "PATIENT"}:
            continue
        value = ent.get("value") or ""
        if name_norm and name_norm in _normalize_value(value):
            doc_ids.extend([str(d) for d in (ent.get("document_ids") or []) if d])
    doc_ids = list(dict.fromkeys(doc_ids))
    if doc_ids:
        return doc_ids

    docs = catalog.get("documents") or []
    for doc in docs:
        doc_id = doc.get("document_id")
        if not doc_id:
            continue
        facts_payload = facts_store.get_document_facts(subscription_id, profile_id, str(doc_id))
        if not facts_payload:
            continue
        for fact in facts_payload.get("section_facts") or []:
            if fact.get("section_kind") != "identity_contact":
                continue
            attrs = fact.get("attributes") or {}
            names = attrs.get("names") or []
            for candidate in names:
                if name_norm and name_norm in _normalize_value(candidate):
                    doc_ids.append(str(doc_id))
                    break
            if doc_ids and str(doc_id) in doc_ids:
                break
    return list(dict.fromkeys(doc_ids))

def _doc_citation_from_facts(
    facts: List[Dict[str, Any]],
    *,
    source_name: str,
) -> str:
    page = None
    for fact in facts:
        for span in fact.get("evidence_spans") or []:
            if span.get("page"):
                page = span.get("page")
                break
        if page:
            break
    if page:
        return f"{source_name}, p. {page}"
    return f"{source_name}"

def _extract_resume_profile(
    facts: List[Dict[str, Any]],
    *,
    source_name: str,
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "name": None,
        "skills": [],
        "education": [],
        "experience": [],
        "projects": [],
        "achievements": [],
        "source_name": source_name,
    }
    for fact in facts:
        kind = fact.get("section_kind")
        attrs = fact.get("attributes") or {}
        entities = fact.get("entities") or []
        if kind == "identity_contact" and not profile["name"]:
            names = attrs.get("names") or [e.get("value") for e in entities if str(e.get("type")).upper() == "PERSON"]
            if names:
                profile["name"] = names[0]
        if kind in {"skills_technical", "skills_functional", "tools_technologies"}:
            items = attrs.get("items") or []
            profile["skills"].extend([str(i) for i in items if i])
        if kind == "education":
            orgs = [e.get("value") for e in entities if str(e.get("type")).upper() in {"ORG", "ORGANIZATION"}]
            profile["education"].extend([o for o in orgs if o])
        if kind == "experience":
            orgs = [e.get("value") for e in entities if str(e.get("type")).upper() in {"ORG", "ORGANIZATION"}]
            profile["experience"].extend([o for o in orgs if o])
        if kind == "projects":
            terms = [e.get("value") for e in entities if e.get("value")]
            profile["projects"].extend(terms)
        if kind == "achievements_awards":
            terms = [e.get("value") for e in entities if e.get("value")]
            profile["achievements"].extend(terms)

    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        output = []
        for v in values:
            key = str(v).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(str(v))
        return output

    profile["skills"] = _dedupe(profile["skills"])[:15]
    profile["education"] = _dedupe(profile["education"])[:8]
    profile["experience"] = _dedupe(profile["experience"])[:8]
    profile["projects"] = _dedupe(profile["projects"])[:8]
    profile["achievements"] = _dedupe(profile["achievements"])[:6]
    return profile

def _extract_invoice_profile(
    facts: List[Dict[str, Any]],
    *,
    source_name: str,
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "source_name": source_name,
        "invoice_number": None,
        "total_amount": None,
        "due_date": None,
        "line_items": [],
        "parties": [],
    }
    for fact in facts:
        kind = fact.get("section_kind")
        attrs = fact.get("attributes") or {}
        if kind == "invoice_metadata":
            if not profile["invoice_number"]:
                profile["invoice_number"] = attrs.get("invoice_number") or attrs.get("purchase_order_number")
        if kind == "financial_summary":
            profile["total_amount"] = profile["total_amount"] or attrs.get("total_amount")
            profile["due_date"] = profile["due_date"] or attrs.get("due_date")
        if kind == "line_items":
            items = attrs.get("items") or []
            profile["line_items"].extend([str(i) for i in items if i])
        if kind == "parties_addresses":
            parties = attrs.get("names") or attrs.get("orgs") or []
            profile["parties"].extend([str(p) for p in parties if p])

    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        output = []
        for v in values:
            key = str(v).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(str(v))
        return output

    profile["line_items"] = _dedupe(profile["line_items"])[:10]
    profile["parties"] = _dedupe(profile["parties"])[:6]
    return profile

def _extract_medical_profile(
    facts: List[Dict[str, Any]],
    *,
    source_name: str,
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "source_name": source_name,
        "patient_name": None,
        "diagnoses": [],
        "medications": [],
        "notes": [],
    }
    for fact in facts:
        kind = fact.get("section_kind")
        attrs = fact.get("attributes") or {}
        entities = fact.get("entities") or []
        if kind == "identity_contact" and not profile["patient_name"]:
            names = attrs.get("names") or [e.get("value") for e in entities if str(e.get("type")).upper() == "PERSON"]
            if names:
                profile["patient_name"] = names[0]
        if kind == "diagnoses_procedures":
            terms = attrs.get("terms") or []
            profile["diagnoses"].extend([str(t) for t in terms if t])
        if kind == "medications":
            terms = attrs.get("terms") or []
            profile["medications"].extend([str(t) for t in terms if t])
        if kind == "notes":
            terms = attrs.get("terms") or []
            profile["notes"].extend([str(t) for t in terms if t])

    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        output = []
        for v in values:
            key = str(v).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(str(v))
        return output

    profile["diagnoses"] = _dedupe(profile["diagnoses"])[:10]
    profile["medications"] = _dedupe(profile["medications"])[:10]
    profile["notes"] = _dedupe(profile["notes"])[:6]
    return profile

def _build_profile_task_response(
    *,
    query: str,
    route_plan: DeterministicRoute,
    catalog: Dict[str, Any],
    facts_store: FactsStore,
    subscription_id: str,
    profile_id: str,
) -> Optional[Dict[str, Any]]:
    doc_lookup = {str(d.get("document_id")): d for d in catalog.get("documents") or []}
    target_doc_ids = route_plan.target_document_ids or list(doc_lookup.keys())
    if not target_doc_ids:
        return None

    profiles: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    for doc_id in target_doc_ids:
        facts_payload = facts_store.get_document_facts(subscription_id, profile_id, str(doc_id))
        if not facts_payload:
            continue
        doc_entry = doc_lookup.get(str(doc_id)) or {}
        source_name = doc_entry.get("source_name") or "Document"
        doc_domain = (doc_entry.get("doc_domain") or route_plan.domain_hint or "unknown").lower()
        facts = facts_payload.get("section_facts") or []
        if doc_domain in {"invoice", "purchase_order"}:
            profile = _extract_invoice_profile(facts, source_name=source_name)
            profile["doc_domain"] = doc_domain
        elif doc_domain == "medical":
            profile = _extract_medical_profile(facts, source_name=source_name)
            profile["doc_domain"] = doc_domain
        else:
            profile = _extract_resume_profile(facts, source_name=source_name)
            profile["doc_domain"] = doc_domain
        profile["citation"] = _doc_citation_from_facts(facts, source_name=source_name)
        profiles.append(profile)
        sources.append(
            {
                "source_name": source_name,
                "doc_domain": doc_entry.get("doc_domain"),
                "page": None,
            }
        )

    if not profiles:
        return None

    def _summary_line(profile: Dict[str, Any]) -> str:
        domain = profile.get("doc_domain") or "unknown"
        name = profile.get("name") or profile.get("patient_name") or profile.get("source_name") or "Document"
        parts: List[str] = []
        if domain in {"invoice", "purchase_order"}:
            if profile.get("invoice_number"):
                parts.append(f"invoice {profile['invoice_number']}")
            if profile.get("total_amount"):
                parts.append(f"total {profile['total_amount']}")
            if profile.get("due_date"):
                parts.append(f"due {profile['due_date']}")
        elif domain == "medical":
            if profile.get("diagnoses"):
                parts.append(f"diagnoses include {', '.join(profile['diagnoses'][:3])}")
            if profile.get("medications"):
                parts.append(f"medications include {', '.join(profile['medications'][:3])}")
        else:
            if profile.get("experience"):
                parts.append(f"experience with {', '.join(profile['experience'][:2])}")
            if profile.get("skills"):
                parts.append(f"skills in {', '.join(profile['skills'][:6])}")
            if profile.get("education"):
                parts.append(f"education at {', '.join(profile['education'][:2])}")
        if not parts:
            parts.append("documented details available in the file")
        return f"{name}: {'; '.join(parts)} ({profile.get('citation')})"

    def _highlights(profile: Dict[str, Any]) -> List[str]:
        highlights: List[str] = []
        domain = profile.get("doc_domain") or "unknown"
        if domain in {"invoice", "purchase_order"}:
            if profile.get("line_items"):
                highlights.append(
                    f"Line items: {', '.join(profile['line_items'][:4])} ({profile.get('citation')})"
                )
        elif domain == "medical":
            if profile.get("diagnoses"):
                highlights.append(
                    f"Key diagnoses: {', '.join(profile['diagnoses'][:3])} ({profile.get('citation')})"
                )
            if profile.get("medications"):
                highlights.append(
                    f"Medications: {', '.join(profile['medications'][:3])} ({profile.get('citation')})"
                )
        else:
            if profile.get("skills"):
                highlights.append(
                    f"Key skills: {', '.join(profile['skills'][:6])} ({profile.get('citation')})"
                )
            if profile.get("projects"):
                highlights.append(
                    f"Projects or focus areas: {', '.join(profile['projects'][:4])} ({profile.get('citation')})"
                )
            if profile.get("achievements"):
                highlights.append(
                    f"Achievements: {', '.join(profile['achievements'][:3])} ({profile.get('citation')})"
                )
        return highlights

    if route_plan.task_type == "summarize":
        lines = [_summary_line(profile) for profile in profiles]
        highlights = []
        for profile in profiles:
            highlights.extend(_highlights(profile))
        response_text = "Summary:\n" + "\n".join([f"- {line}" for line in lines if line])
        if highlights:
            response_text += "\nHighlights:\n" + "\n".join([f"- {line}" for line in highlights[:8]])
        return {
            "response": response_text,
            "sources": sources,
            "grounded": True,
            "context_found": True,
            "metadata": {"task": "summarize"},
        }

    if route_plan.task_type == "compare":
        lines = []
        for profile in profiles:
            name = profile.get("name") or profile.get("source_name") or "Document"
            summary = _summary_line(profile)
            lines.append(f"- {summary}")
        response_text = "Comparison:\n" + "\n".join(lines)
        return {
            "response": response_text,
            "sources": sources,
            "grounded": True,
            "context_found": True,
            "metadata": {"task": "compare"},
        }

    if route_plan.task_type == "generate":
        primary = profiles[0]
        name = primary.get("name") or "Candidate"
        skills = primary.get("skills") or []
        experience = primary.get("experience") or []
        citation = primary.get("citation")
        intro = "Dear Hiring Manager,\n\nI am writing to express interest in the role."
        if experience:
            intro += f" The candidate’s experience includes work with {', '.join(experience[:2])} ({citation})."
        body = []
        if skills:
            body.append(f"Key skills include {', '.join(skills[:6])} ({citation}).")
        closing = "\nI would welcome the opportunity to discuss how these strengths align with your needs.\n\nSincerely,\n" + name
        response_text = "\n".join([intro, "\n".join(body), closing]).strip()
        return {
            "response": response_text,
            "sources": sources,
            "grounded": True,
            "context_found": True,
            "metadata": {"task": "generate"},
        }

    if route_plan.task_type == "rank":
        query_terms = [t for t in re.findall(r"[A-Za-z0-9]{3,}", query.lower()) if t]
        ranked = []
        for profile in profiles:
            hay = " ".join((profile.get("skills") or []) + (profile.get("experience") or [])).lower()
            score = sum(1 for term in query_terms if term in hay)
            ranked.append((score, profile))
        ranked.sort(key=lambda item: item[0], reverse=True)
        lines = [
            "| Rank | Document | Evidence | Citations |",
            "| --- | --- | --- | --- |",
        ]
        for idx, (_score, profile) in enumerate(ranked, start=1):
            name = profile.get("name") or profile.get("source_name") or f"Document {idx}"
            evidence = "Relevant skills/experience align with the requested criteria."
            lines.append(f"| {idx} | {name} | {evidence} | {profile.get('citation')} |")
        response_text = "\n".join(lines + ["Criteria: alignment of skills/experience with the query."])
        return {
            "response": response_text,
            "sources": sources,
            "grounded": True,
            "context_found": True,
            "metadata": {"task": "rank", "acknowledged": True},
        }

    return None

@dataclass
class IntelChunk:
    text: str
    score: float
    metadata: Dict[str, Any]

def _with_retries(fn, *, retries: int = 2, base_delay: float = 0.2, jitter: float = 0.1):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                raise
            delay = base_delay * (attempt + 1) + random.random() * jitter
            time.sleep(delay)

def _safe_error_response(
    code: str,
    message: str,
    request_id: Optional[str],
    *,
    details: Optional[str] = None,
) -> Dict[str, Any]:
    error_payload = {"code": code, "request_id": request_id}
    if details:
        error_payload["details"] = details
    return {
        "response": message,
        "sources": [],
        "grounded": False,
        "context_found": False,
        "metadata": {"error": error_payload},
    }

def _build_filters(
    *,
    subscription_id: str,
    profile_id: str,
    route_plan: DeterministicRoute,
    session_state: SessionState,
) -> Dict[str, Any]:
    filters: Dict[str, Any] = {"subscription_id": subscription_id}
    document_ids: List[str] = []
    if route_plan.scope == "current_document":
        if route_plan.target_document_ids:
            document_ids.extend(route_plan.target_document_ids)
        elif session_state.active_document_id:
            document_ids.append(session_state.active_document_id)
    if document_ids:
        filters["document_ids"] = sorted(set(document_ids))
    if route_plan.domain_hint and route_plan.domain_hint not in {"unknown", "mixed", "generic"}:
        filters["doc_domains"] = [route_plan.domain_hint]
    if route_plan.section_focus and route_plan.task_type in {"extract", "list", "qa"}:
        filters["section_kinds"] = list(route_plan.section_focus)
    return filters

def _infer_domain_from_hits(
    candidates: List[RetrievalCandidate],
    catalog: Dict[str, Any],
) -> Optional[str]:
    domain_counts: Dict[str, int] = {}
    for cand in candidates:
        payload = cand.metadata or {}
        domain = payload.get("doc_domain")
        if not domain:
            doc_id = payload.get("document_id")
            if doc_id and catalog.get("documents"):
                for doc in catalog.get("documents"):
                    if str(doc.get("document_id")) == str(doc_id):
                        domain = doc.get("doc_domain")
                        break
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if not domain_counts:
        return None
    ranked = sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0]

def _apply_kg_boost(
    candidates: List[RetrievalCandidate],
    *,
    doc_ids: List[str],
    chunk_ids: List[str],
    boost: float = 0.15,
) -> List[RetrievalCandidate]:
    if not candidates:
        return candidates
    boosted: List[RetrievalCandidate] = []
    doc_id_set = {str(d) for d in doc_ids}
    chunk_id_set = {str(c) for c in chunk_ids}
    for cand in candidates:
        payload = cand.metadata or {}
        score = float(cand.score)
        if doc_id_set and str(payload.get("document_id")) in doc_id_set:
            score += boost
        if chunk_id_set and str(payload.get("chunk_id")) in chunk_id_set:
            score += boost
        cand.score = score
        boosted.append(cand)
    boosted.sort(key=lambda c: c.score, reverse=True)
    return boosted

def _candidates_to_chunks(candidates: List[RetrievalCandidate]) -> List[IntelChunk]:
    chunks: List[IntelChunk] = []
    for cand in candidates:
        chunks.append(IntelChunk(text=cand.text, score=cand.score, metadata=cand.metadata or {}))
    return chunks

def _filter_candidates_by_profile(
    candidates: List[RetrievalCandidate],
    *,
    profile_id: str,
) -> List[RetrievalCandidate]:
    if not candidates:
        return candidates
    filtered: List[RetrievalCandidate] = []
    dropped = 0
    for cand in candidates:
        meta = cand.metadata or {}
        if str(meta.get("profile_id") or "") == str(profile_id):
            filtered.append(cand)
        else:
            dropped += 1
    if dropped:
        logger.warning("Dropped %s retrieval candidates outside profile scope", dropped)
    return filtered

def _facts_for_section_focus(facts: List[Dict[str, Any]], section_focus: List[str]) -> List[Dict[str, Any]]:
    if not section_focus:
        return facts
    focus_set = {str(k) for k in section_focus if k}
    return [fact for fact in facts if str(fact.get("section_kind")) in focus_set]

def _documents_searched(
    catalog: Dict[str, Any],
    *,
    target_doc_ids: Optional[List[str]] = None,
) -> List[str]:
    docs = catalog.get("documents") or []
    doc_lookup = {str(d.get("document_id")): d.get("source_name") for d in docs if d.get("source_name")}
    if target_doc_ids:
        return [
            doc_lookup.get(str(doc_id)) or "Unknown document"
            for doc_id in target_doc_ids
            if doc_id
        ]
    return [d.get("source_name") for d in docs if d.get("source_name")]

def _build_not_found_response(
    *,
    query: str,
    request_id: Optional[str],
    route_plan: DeterministicRoute,
    documents_searched: List[str],
    retrieval_attempted: bool,
    retrieval_succeeded: bool,
    details: Optional[str] = None,
    error_code: str = "EVIDENCE_NOT_FOUND",
) -> Dict[str, Any]:
    searched_line = ""
    if documents_searched:
        preview = ", ".join(documents_searched[:8])
        if len(documents_searched) > 8:
            preview += "..."
        searched_line = f" Documents searched: {preview}"
    ack = generate_ack(
        query=query,
        task_type=route_plan.task_type,
        domain_hint=route_plan.domain_hint,
        section_focus=route_plan.section_focus,
        target_entity=route_plan.target_person,
    )
    prefix = f"{ack}\n" if ack else ""
    if error_code == "NAME_NOT_FOUND" and details and details.startswith("name_not_found:"):
        missing_name = details.split("name_not_found:", 1)[-1].strip()
        response_text = (
            f"{prefix}I couldn't find {missing_name} in the current profile documents."
            + searched_line
        )
    else:
        response_text = (
            f"{prefix}I couldn't find evidence for '{query}' in the current profile documents."
            + searched_line
        )
    metadata = {
        "execution_trace": {
            "route_plan": route_plan.to_dict(),
            "facts_hit": False,
            "retrieval_attempted": retrieval_attempted,
            "retrieval_succeeded": retrieval_succeeded,
            "documents_searched": documents_searched,
        },
        "error": {"code": error_code, "details": details or ""},
        "correlation_id": request_id,
    }
    return {
        "response": response_text,
        "sources": [],
        "grounded": False,
        "context_found": False,
        "metadata": metadata,
        "request_id": request_id,
    }

def _build_retrieval_filter_failed_response(
    *,
    query: str,
    request_id: Optional[str],
    route_plan: DeterministicRoute,
    documents_searched: List[str],
    details: Optional[str] = None,
    error_code: str = "RETRIEVAL_FILTER_FAILED",
) -> Dict[str, Any]:
    ack = generate_ack(
        query=query,
        task_type=route_plan.task_type,
        domain_hint=route_plan.domain_hint,
        section_focus=route_plan.section_focus,
        target_entity=route_plan.target_person,
    )
    prefix = f"{ack}\n" if ack else ""
    if error_code == "RETRIEVAL_INDEX_MISSING":
        field_label = None
        if details and "missing_index=" in details:
            try:
                field_label = details.split("missing_index=", 1)[1].split()[0].strip()
            except Exception:
                field_label = None
        field_label = field_label or "required field"
        response_text = f"{prefix}I couldn’t retrieve evidence due to an indexing issue ({field_label} not indexed)."
    elif error_code == "RETRIEVAL_QDRANT_UNAVAILABLE":
        response_text = f"{prefix}I couldn’t retrieve evidence because the document index is unavailable."
    elif error_code == "RETRIEVAL_INDEX_BOOTSTRAP_FAILED":
        response_text = f"{prefix}I couldn’t retrieve evidence due to an indexing issue."
    else:
        response_text = f"{prefix}Profile isolation enforced; cannot search outside profile."
    metadata = {
        "execution_trace": {
            "route_plan": route_plan.to_dict(),
            "facts_hit": False,
            "retrieval_attempted": True,
            "retrieval_succeeded": False,
            "documents_searched": documents_searched,
        },
        "error": {"code": error_code, "details": details or ""},
        "correlation_id": request_id,
    }
    return {
        "ok": False,
        "response": response_text,
        "sources": [],
        "grounded": False,
        "context_found": False,
        "metadata": metadata,
        "documents_searched": documents_searched,
        "request_id": request_id,
    }

def _is_filter_failure(exc: Exception) -> bool:
    if getattr(exc, "code", None) in {
        "RETRIEVAL_FILTER_FAILED",
        "RETRIEVAL_INDEX_MISSING",
        "RETRIEVAL_INDEX_BOOTSTRAP_FAILED",
        "RETRIEVAL_QDRANT_UNAVAILABLE",
    }:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "index required",
            "payload index",
            "payload indexes",
            "missing required qdrant payload",
            "query_points",
            "has no attribute 'query_points'",
            "filter",
        )
    )

def _build_missing_profile_scope_response(
    *,
    request_id: Optional[str],
    query: str,
) -> Dict[str, Any]:
    return {
        "response": "Profile scope is required to answer this question.",
        "sources": [],
        "grounded": False,
        "context_found": False,
        "metadata": {
            "error": {"code": "MISSING_PROFILE_SCOPE", "details": query or ""},
            "correlation_id": request_id,
        },
        "request_id": request_id,
    }

def run_intelligent_pipeline(
    *,
    query: str,
    subscription_id: str,
    profile_id: str,
    session_id: Optional[str],
    user_id: str,
    request_id: Optional[str] = None,
    redis_client: Optional[Any] = None,
    qdrant_client: Optional[QdrantClient] = None,
    embedder: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    if not getattr(Config.Intelligence, "ENABLED", True):
        return None
    if not profile_id:
        return _build_missing_profile_scope_response(request_id=request_id, query=query)

    cache = RedisIntelCache(redis_client)
    session_id = session_id or "default"
    session_state = cache.get_session_state(subscription_id, session_id)
    catalog = cache.get_json(cache.catalog_key(subscription_id, profile_id)) or {}
    entities_cache = cache.get_json(cache.entities_key(subscription_id, profile_id)) or {}

    route_plan = route_query(query, session_state.to_dict(), catalog, entities_cache)

    # Track entities in session
    kg_service = KGQueryService()
    query_entities = kg_service.extract_entities(query)

    db_client = None
    try:
        import sys

        data_handler = sys.modules.get("src.api.dataHandler")
        if data_handler is not None:
            db_client = getattr(data_handler, "db", None)
    except Exception:
        db_client = None

    facts_store = FactsStore(redis_client=redis_client, db=db_client)

    # Name resolution enforcement
    name_candidate = route_plan.target_person
    if not name_candidate:
        for ent in query_entities:
            if str(ent.get("type") or "").upper() in {"PERSON", "ORG", "ORGANIZATION", "VENDOR", "PATIENT"}:
                name_candidate = ent.get("value") or ent.get("normalized_value")
                break
    if not name_candidate:
        extracted = _extract_name_candidates(query)
        if extracted:
            name_candidate = extracted[0]
    if name_candidate and (catalog.get("documents") or (entities_cache.get("entities") or [])):
        resolved_doc_ids = _resolve_name_documents(
            name=str(name_candidate),
            entities_cache=entities_cache,
            catalog=catalog,
            facts_store=facts_store,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )
        if route_plan.domain_hint and route_plan.domain_hint not in {"unknown", "mixed", "generic"}:
            domain_target = route_plan.domain_hint
            doc_domain_lookup = {
                str(d.get("document_id")): d.get("doc_domain")
                for d in (catalog.get("documents") or [])
                if d.get("document_id")
            }
            resolved_doc_ids = [
                doc_id
                for doc_id in resolved_doc_ids
                if doc_domain_lookup.get(str(doc_id)) == domain_target
            ]
        if route_plan.target_document_ids:
            resolved_doc_ids = [doc_id for doc_id in resolved_doc_ids if doc_id in route_plan.target_document_ids]
        if not resolved_doc_ids:
            return _build_not_found_response(
                query=query,
                request_id=request_id,
                route_plan=route_plan,
                documents_searched=_documents_searched(catalog),
                retrieval_attempted=False,
                retrieval_succeeded=False,
                details=f"name_not_found:{name_candidate}",
                error_code="NAME_NOT_FOUND",
            )
        route_plan.target_person = name_candidate
        route_plan.target_document_ids = resolved_doc_ids
        route_plan.scope = "current_document"

    cache.touch_session_state(
        subscription_id,
        session_id,
        active_profile_id=profile_id,
        active_document_id=(route_plan.target_document_ids[0] if route_plan.target_document_ids else session_state.active_document_id),
        active_domain=route_plan.domain_hint,
        recent_route=route_plan.to_dict(),
        recent_entities=[{**ent, "ts": time.time()} for ent in query_entities][:5],
    )

    facts_answer = answer_with_section_intelligence(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        session_state=session_state.to_dict(),
        catalog=catalog,
        entities_cache=entities_cache,
        redis_client=redis_client,
        db=db_client,
        route_override=route_plan,
    )
    if facts_answer:
        facts_answer.setdefault("metadata", {})
        facts_answer["metadata"]["route_plan"] = route_plan.to_dict()
        facts_answer["metadata"]["execution_trace"] = {
            "route_plan": route_plan.to_dict(),
            "facts_hit": True,
            "retrieval_attempted": False,
            "retrieval_succeeded": False,
            "documents_searched": _documents_searched(
                catalog,
                target_doc_ids=route_plan.target_document_ids or None,
            ),
        }
        facts_answer["metadata"]["user_id"] = user_id
        return facts_answer

    if route_plan.task_type in {"summarize", "compare", "generate", "rank"}:
        profile_answer = _build_profile_task_response(
            query=query,
            route_plan=route_plan,
            catalog=catalog,
            facts_store=facts_store,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )
        if profile_answer:
            profile_answer.setdefault("metadata", {})
            profile_answer["metadata"]["route_plan"] = route_plan.to_dict()
            profile_answer["metadata"]["execution_trace"] = {
                "route_plan": route_plan.to_dict(),
                "facts_hit": True,
                "retrieval_attempted": False,
                "retrieval_succeeded": False,
                "documents_searched": _documents_searched(
                    catalog,
                    target_doc_ids=route_plan.target_document_ids or None,
                ),
            }
            profile_answer["metadata"]["user_id"] = user_id
            return profile_answer

    collection_name = build_collection_name(subscription_id)
    qdrant = qdrant_client
    if qdrant is None:
        try:
            from src.api.dataHandler import get_qdrant_client as _get_qdrant

            qdrant = _get_qdrant()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant unavailable for intelligent pipeline: %s", exc)
            return _build_not_found_response(
                query=query,
                request_id=request_id,
                route_plan=route_plan,
                documents_searched=_documents_searched(
                    catalog,
                    target_doc_ids=route_plan.target_document_ids or None,
                ),
                retrieval_attempted=True,
                retrieval_succeeded=False,
                details="qdrant_unavailable",
            )

    # --- ReasoningEngine: unified THINK→SEARCH→REASON→GENERATE→VERIFY ---
    _use_reasoning_engine = getattr(Config.Intelligence, "REASONING_ENGINE_ENABLED", True)
    if _use_reasoning_engine:
        try:
            from src.intelligence.reasoning_engine import ReasoningEngine
            from src.llm.gateway import get_llm_gateway

            _emb = embedder or get_embedding_model()[0]
            _llm = get_llm_gateway()

            # Build thinking client — use same model to avoid GPU swap on single-GPU
            _thinker = _llm

            # Build profile context summary for the THINK step
            _profile_docs = catalog.get("documents") or []
            _domain_counts: Dict[str, int] = {}
            _profile_ctx_lines = []
            for _doc in _profile_docs[:20]:
                _dname = _doc.get("source_name") or _doc.get("document_name") or "unknown"
                _ddomain = _doc.get("doc_domain") or "generic"
                _profile_ctx_lines.append(f"- {_dname} (domain: {_ddomain})")
                _domain_counts[_ddomain] = _domain_counts.get(_ddomain, 0) + 1
            _domain_summary = ", ".join(f"{count} {dom}" for dom, count in sorted(_domain_counts.items(), key=lambda x: -x[1]))
            _profile_ctx = f"PROFILE DOMAIN: {route_plan.domain_hint or 'unknown'} ({len(_profile_docs)} documents: {_domain_summary})\n"
            _profile_ctx += "\n".join(_profile_ctx_lines) if _profile_ctx_lines else "No documents listed."

            # Build conversation context from session state
            _conv_parts = []
            if session_state:
                if hasattr(session_state, "active_document_id") and session_state.active_document_id:
                    _conv_parts.append(f"Active document: {session_state.active_document_id}")
                if hasattr(session_state, "active_domain") and session_state.active_domain:
                    _conv_parts.append(f"Domain focus: {session_state.active_domain}")
                if hasattr(session_state, "recent_queries"):
                    _recent = getattr(session_state, "recent_queries", []) or []
                    if _recent:
                        _conv_parts.append("Previous queries: " + " | ".join(str(q) for q in _recent[-3:]))
            _conv_history = "\n".join(_conv_parts) if _conv_parts else ""

            engine = ReasoningEngine(
                llm_client=_llm,
                thinking_client=_thinker,
                qdrant_client=qdrant,
                embedder=_emb,
                collection_name=collection_name,
                subscription_id=subscription_id,
                profile_id=profile_id,
                max_iterations=2 if route_plan.task_type in {"extract", "list", "qa"} else 3,
            )

            engine_result = engine.answer(
                query=query,
                profile_context=_profile_ctx,
                conversation_history=_conv_history,
                task_type=route_plan.task_type or "",
            )

            if engine_result and engine_result.get("context_found"):
                engine_result.setdefault("metadata", {})
                engine_result["metadata"]["route_plan"] = route_plan.to_dict()
                engine_result["metadata"]["execution_trace"] = {
                    "route_plan": route_plan.to_dict(),
                    "facts_hit": False,
                    "retrieval_attempted": True,
                    "retrieval_succeeded": True,
                    "documents_searched": _documents_searched(
                        catalog,
                        target_doc_ids=route_plan.target_document_ids or None,
                    ),
                    "engine": "reasoning_engine",
                }
                engine_result["metadata"]["user_id"] = user_id
                logger.info(
                    "ReasoningEngine handled query: intent=%s evidence=%d grounded=%s",
                    engine_result.get("metadata", {}).get("intent", "?"),
                    engine_result.get("metadata", {}).get("evidence_count", 0),
                    engine_result.get("grounded"),
                )
                return engine_result
            else:
                logger.debug("ReasoningEngine returned no context, falling back to legacy pipeline")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ReasoningEngine failed, falling back to legacy pipeline: %s", exc)

    retriever = HybridRetriever(
        client=qdrant,
        embedder=embedder or get_embedding_model()[0],
        config=HybridRetrieverConfig(topk_dense=int(getattr(Config.Retrieval, "TOPK_DENSE", 50))),
    )

    if hasattr(qdrant, "get_collection"):
        try:
            QdrantVectorStore(client=qdrant).ensure_payload_indexes(
                collection_name,
                REQUIRED_PAYLOAD_INDEX_FIELDS,
                create_missing=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Qdrant payload index validation failed: %s", exc)
            return _build_retrieval_filter_failed_response(
                query=query,
                request_id=request_id,
                route_plan=route_plan,
                documents_searched=_documents_searched(
                    catalog,
                    target_doc_ids=route_plan.target_document_ids or None,
                ),
                details=str(exc),
                error_code="RETRIEVAL_FILTER_FAILED",
            )
    else:
        logger.debug("Skipping payload index validation for lightweight Qdrant client")

    filters = _build_filters(
        subscription_id=subscription_id,
        profile_id=profile_id,
        route_plan=route_plan,
        session_state=session_state,
    )

    # KG assist
    kg_result = kg_service.query(
        subscription_id=subscription_id,
        profile_id=profile_id,
        domain_hint=route_plan.domain_hint,
        entities=query_entities,
        limit=int(getattr(Config.Retrieval, "KG_PROBE_LIMIT", 20)),
    )
    explicit_hints: Dict[str, List[str]] = {}
    if kg_result.doc_ids:
        if len(kg_result.doc_ids) <= int(getattr(Config.Retrieval, "KG_DOC_FILTER_LIMIT", 8)):
            filters["document_ids"] = kg_result.doc_ids
        else:
            explicit_hints["document_ids"] = kg_result.doc_ids
    if kg_result.chunk_ids:
        explicit_hints["chunk_ids"] = kg_result.chunk_ids

    def _retrieve_once(active_filters: Dict[str, Any]) -> List[RetrievalCandidate]:
        return retriever.retrieve(
            collection_name=collection_name,
            query=query,
            profile_id=profile_id,
            top_k=int(getattr(Config.Retrieval, "TOPK_DENSE", 50)),
            filters=active_filters,
            explicit_hints=explicit_hints,
            subscription_id=subscription_id,
        )

    section_retrieval_info = {
        "enabled": bool(getattr(Config.Intelligence, "SECTION_RETRIEVAL_ENABLED", False)),
        "used": False,
        "summary_hits": 0,
        "section_ids": [],
        "latency_ms": None,
    }
    active_filters = dict(filters)
    if (
        getattr(Config.Intelligence, "SECTION_RETRIEVAL_ENABLED", False)
        and getattr(Config.Intelligence, "SECTION_SUMMARY_VECTORS_ENABLED", False)
    ):
        try:
            section_filters = dict(filters)
            section_filters["chunk_kinds"] = ["section_summary"]
            section_start = time.time()
            section_candidates = _with_retries(
                lambda: retriever.retrieve(
                    collection_name=collection_name,
                    query=query,
                    profile_id=profile_id,
                    top_k=int(getattr(Config.Intelligence, "SECTION_SUMMARY_TOPK", 6)),
                    filters=section_filters,
                    explicit_hints=explicit_hints,
                    subscription_id=subscription_id,
                )
            )
            section_retrieval_info["latency_ms"] = int((time.time() - section_start) * 1000)
            section_retrieval_info["summary_hits"] = len(section_candidates or [])
            section_ids = []
            for cand in section_candidates or []:
                payload = cand.metadata or {}
                sec_id = payload.get("section_id") or (payload.get("section") or {}).get("id")
                if sec_id:
                    section_ids.append(str(sec_id))
            section_ids = sorted(set(section_ids))
            if section_ids:
                section_retrieval_info["used"] = True
                section_retrieval_info["section_ids"] = section_ids
                active_filters["section_ids"] = section_ids
        except Exception as exc:  # noqa: BLE001
            logger.debug("Section summary retrieval skipped: %s", exc)

    try:
        candidates = _with_retries(lambda: _retrieve_once(active_filters))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retrieval failed: %s", exc)
        if _is_filter_failure(exc):
            return _build_retrieval_filter_failed_response(
                query=query,
                request_id=request_id,
                route_plan=route_plan,
                documents_searched=_documents_searched(
                    catalog,
                    target_doc_ids=route_plan.target_document_ids or None,
                ),
                details=str(getattr(exc, "details", str(exc))),
                error_code=getattr(exc, "code", "RETRIEVAL_FILTER_FAILED"),
            )
        return _build_not_found_response(
            query=query,
            request_id=request_id,
            route_plan=route_plan,
            documents_searched=_documents_searched(
                catalog,
                target_doc_ids=route_plan.target_document_ids or None,
            ),
            retrieval_attempted=True,
            retrieval_succeeded=False,
            details=str(exc),
        )
    candidates = _filter_candidates_by_profile(candidates, profile_id=profile_id)
    if not candidates and section_retrieval_info.get("used"):
        try:
            candidates = _with_retries(lambda: _retrieve_once(filters))
            active_filters = dict(filters)
        except Exception:
            pass
        candidates = _filter_candidates_by_profile(candidates, profile_id=profile_id)

    if route_plan.domain_hint in {"unknown", "mixed", "generic"} and candidates:
        inferred = _infer_domain_from_hits(candidates, catalog)
        if inferred and inferred not in {"unknown", "mixed"}:
            route_plan.domain_hint = inferred
            filters["doc_domains"] = [inferred]
            active_filters["doc_domains"] = [inferred]
            try:
                candidates = _with_retries(lambda: _retrieve_once(active_filters))
            except Exception:
                pass
            candidates = _filter_candidates_by_profile(candidates, profile_id=profile_id)

    if explicit_hints and not filters.get("document_ids"):
        candidates = _apply_kg_boost(
            candidates,
            doc_ids=explicit_hints.get("document_ids", []),
            chunk_ids=explicit_hints.get("chunk_ids", []),
        )

    chunks = _candidates_to_chunks(candidates)[: int(getattr(Config.Retrieval, "FINAL_CONTEXT_CHUNKS", 8))]
    if not chunks:
        return _build_not_found_response(
            query=query,
            request_id=request_id,
            route_plan=route_plan,
            documents_searched=_documents_searched(
                catalog,
                target_doc_ids=route_plan.target_document_ids or None,
            ),
            retrieval_attempted=True,
            retrieval_succeeded=False,
            details="no_candidates",
        )

    retrieval_succeeded = True

    if route_plan.task_type in {"summarize"}:
        answer = summarize_task(query=query, chunks=chunks, scope=route_plan.scope)
    elif route_plan.task_type in {"rank"}:
        answer = rank_task(query=query, chunks=chunks)
    elif route_plan.task_type in {"generate"}:
        answer = generate_document_task(query=query, chunks=chunks, target_person=route_plan.target_person)
    elif route_plan.task_type in {"compare"}:
        answer = summarize_task(query=query, chunks=chunks, scope="profile_all_docs")
        answer["response"] = f"Comparison:\n{answer.get('response')}"
    else:
        # Extract/list tasks: attempt to synthesize facts from retrieved chunks.
        facts: List[Dict[str, Any]] = []
        facts_store = FactsStore(redis_client=redis_client, db=db_client)
        grouped: Dict[str, List[IntelChunk]] = {}
        for chunk in chunks:
            doc_id = (chunk.metadata or {}).get("document_id")
            if not doc_id:
                continue
            grouped.setdefault(str(doc_id), []).append(chunk)

        for doc_id, doc_chunks in grouped.items():
            chunk_meta = [c.metadata or {} for c in doc_chunks]
            chunk_texts = [get_content_text(meta) or c.text for c, meta in zip(doc_chunks, chunk_meta)]
            document_text = " ".join([t for t in chunk_texts if t])
            try:
                builder = SectionIntelligenceBuilder()
                result = builder.build(
                    document_id=str(doc_id),
                    document_text=document_text,
                    chunk_texts=chunk_texts,
                    chunk_metadata=chunk_meta,
                    metadata={
                        "doc_type": (chunk_meta[0] or {}).get("doc_type"),
                        "source_name": (chunk_meta[0] or {}).get("source_name"),
                    },
                )
                sections_payload = [sec.__dict__ for sec in result.sections]
                facts_payload = result.section_facts
                facts.extend(facts_payload)
                if sections_payload and facts_payload:
                    facts_store.persist_document_sections(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        document_id=str(doc_id),
                        source_name=(chunk_meta[0] or {}).get("source_name") or str(doc_id),
                        doc_domain=result.doc_domain or (chunk_meta[0] or {}).get("doc_domain") or "generic",
                        sections=sections_payload,
                        section_facts=facts_payload,
                        section_summaries=result.section_summaries,
                    )
                    KGUpdater(redis_client=redis_client).update(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        document_id=str(doc_id),
                        source_name=(chunk_meta[0] or {}).get("source_name") or str(doc_id),
                        doc_domain=result.doc_domain or (chunk_meta[0] or {}).get("doc_domain") or "generic",
                        sections=sections_payload,
                        chunk_metadata=chunk_meta,
                        section_facts=facts_payload,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Facts backfill skipped for %s: %s", doc_id, exc)

        relevant_facts = _facts_for_section_focus(facts, route_plan.section_focus)
        response_text = format_facts_response(
            query=query,
            route=route_plan,
            facts=relevant_facts,
            catalog=catalog,
        )
        if response_text:
            sources = _collect_evidence_sources(relevant_facts, catalog)
            answer = {
                "response": response_text,
                "sources": sources,
                "grounded": bool(sources),
                "context_found": True,
                "metadata": {"task": route_plan.task_type},
            }
        else:
            return _build_not_found_response(
                query=query,
                request_id=request_id,
                route_plan=route_plan,
                documents_searched=_documents_searched(
                    catalog,
                    target_doc_ids=route_plan.target_document_ids or None,
                ),
                retrieval_attempted=True,
                retrieval_succeeded=retrieval_succeeded,
                details="facts_missing_after_retrieval",
            )

    answer.setdefault("metadata", {})
    if (
        isinstance(answer.get("response"), str)
        and not answer["metadata"].get("acknowledged")
        and route_plan.task_type != "greet"
    ):
        answer["response"] = compose_task_response(
            response_text=answer["response"],
            route_plan=route_plan,
            query=query,
        )
        answer["metadata"]["acknowledged"] = True
    answer["metadata"]["route_plan"] = route_plan.to_dict()
    answer["metadata"]["execution_trace"] = {
        "route_plan": route_plan.to_dict(),
        "facts_hit": False,
        "retrieval_attempted": True,
        "retrieval_succeeded": retrieval_succeeded,
        "documents_searched": _documents_searched(
            catalog,
            target_doc_ids=route_plan.target_document_ids or None,
        ),
    }
    answer["metadata"]["kg_hints"] = {
        "doc_ids": kg_result.doc_ids,
        "chunk_ids": kg_result.chunk_ids,
        "entities": kg_result.entities,
    }
    answer["metadata"]["section_retrieval"] = section_retrieval_info
    answer["metadata"]["user_id"] = user_id
    return answer

__all__ = ["run_intelligent_pipeline", "_apply_kg_boost", "IntelChunk"]
