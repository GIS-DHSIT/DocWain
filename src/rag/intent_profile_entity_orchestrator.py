from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.rag.doc_inventory import DocInventoryItem
from src.rag.entity_detector import EntityDetectionResult, detect_entities
from src.rag.intent_scope_orchestrator import EvidencePolicy, IntentScopeDecision, RetrievalPlan
from src.rag.scope_resolver import ScopeResolution, resolve_scope


_INTENT_TABLE_CUES = {"table", "tabular", "indexed"}
_INTENT_COMPARE_CUES = {"compare", "difference", "vs", "versus"}
_INTENT_RANK_CUES = {"rank", "ranking", "shortlist", "top", "best"}
_INTENT_SUMMARIZE_CUES = {"summarize", "summary", "overview", "recap"}
_INTENT_EXTRACT_CUES = {"extract", "list", "pull", "find", "show", "fields", "details"}
_INTENT_FILTER_CUES = {"filter", "only", "exclude", "without"}
_INTENT_CALCULATE_CUES = {"calculate", "compute", "sum", "total", "subtotal"}
_INTENT_ALL_DOCS_CUES = {"all", "across", "every", "all documents", "all files"}

_CANDIDATE_CUES = {
    "candidate",
    "candidates",
    "resume",
    "resumes",
    "cv",
    "curriculum vitae",
    "applicant",
    "talent",
}


@dataclass(frozen=True)
class OutputRequirements:
    wants_table: bool
    wants_rank_top_n: Optional[int]
    wants_all_docs: bool
    constraints: Dict[str, Any]


@dataclass(frozen=True)
class ProfileSignals:
    corpus_fingerprint: str
    content_type: str
    confidence: float


@dataclass(frozen=True)
class OrchestrationResult:
    intent_type: str
    output_requirements: OutputRequirements
    profile_signals: ProfileSignals
    entities: EntityDetectionResult
    scope: str
    target_docs: List[DocInventoryItem]
    retrieval_plan: RetrievalPlan
    assumption_line: Optional[str]
    candidate_mode: bool
    decision: IntentScopeDecision
    sequence: List[str]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_any(text: str, cues: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def _detect_top_n(text: str) -> Optional[int]:
    match = re.search(r"\btop\s+(\d{1,2})\b", text.lower())
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _detect_constraints(text: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    year_match = re.search(r"(at least|minimum|min)\s+(\d{1,2})\s+years?", text.lower())
    if year_match:
        constraints["min_years"] = int(year_match.group(2))
    currency_match = re.search(r"\b(usd|eur|gbp|inr|aud|cad)\b", text.lower())
    if currency_match:
        constraints["currency"] = currency_match.group(1).upper()
    return constraints


def classify_intent(query_text: str) -> tuple[str, OutputRequirements]:
    lowered = query_text.lower()
    wants_table = _contains_any(lowered, _INTENT_TABLE_CUES)
    wants_rank_top_n = _detect_top_n(lowered)

    if _contains_any(lowered, _INTENT_COMPARE_CUES):
        intent_type = "compare"
    elif _contains_any(lowered, _INTENT_RANK_CUES):
        intent_type = "rank"
    elif _contains_any(lowered, _INTENT_SUMMARIZE_CUES):
        intent_type = "summarize"
    elif _contains_any(lowered, _INTENT_EXTRACT_CUES):
        intent_type = "extract_fields"
    elif _contains_any(lowered, _INTENT_FILTER_CUES):
        intent_type = "filter"
    elif _contains_any(lowered, _INTENT_CALCULATE_CUES):
        intent_type = "calculate"
    elif wants_table:
        intent_type = "tabular_request"
    else:
        intent_type = "lookup_fact"

    wants_all_docs = _contains_any(lowered, _INTENT_ALL_DOCS_CUES) or intent_type in {"compare", "rank"}
    requirements = OutputRequirements(
        wants_table=wants_table,
        wants_rank_top_n=wants_rank_top_n,
        wants_all_docs=wants_all_docs,
        constraints=_detect_constraints(query_text),
    )
    return intent_type, requirements


def _corpus_fingerprint(doc_inventory: Sequence[DocInventoryItem]) -> str:
    parts = []
    for doc in sorted(doc_inventory, key=lambda d: (d.doc_id or d.source_file or d.document_name)):
        parts.append(
            "|".join([
                doc.doc_id or "",
                doc.source_file or "",
                doc.document_name or "",
                doc.doc_type or "",
            ])
        )
    parts.append(f"count={len(doc_inventory)}")
    return _hash("|".join(parts))


def _profile_content_type(doc_inventory: Sequence[DocInventoryItem]) -> tuple[str, float]:
    resume_tokens = {"resume", "cv", "curriculum"}
    invoice_tokens = {"invoice", "bill", "statement", "receipt"}
    resume_hits = 0
    invoice_hits = 0
    for doc in doc_inventory:
        haystack = " ".join([doc.source_file or "", doc.document_name or "", doc.doc_type or ""]).lower()
        if any(tok in haystack for tok in resume_tokens):
            resume_hits += 1
        if any(tok in haystack for tok in invoice_tokens):
            invoice_hits += 1
    total = max(1, resume_hits + invoice_hits)
    if resume_hits and invoice_hits:
        return "mixed", max(resume_hits, invoice_hits) / total
    if resume_hits:
        return "resume-heavy", resume_hits / total
    if invoice_hits:
        return "invoice-heavy", invoice_hits / total
    return "unknown", 0.0


def _detect_candidate_mode(query_text: str, doc_inventory: Sequence[DocInventoryItem]) -> bool:
    lowered = query_text.lower()
    if _contains_any(lowered, _CANDIDATE_CUES):
        return True
    for doc in doc_inventory:
        name = (doc.source_file or "").lower()
        if "resume" in name or "cv" in name:
            return True
    return False


def _legacy_candidate_intent(query_text: str, candidate_mode: bool, scope: str) -> str:
    lowered = query_text.lower()
    if "rank" in lowered or "shortlist" in lowered or "top" in lowered:
        return "ranking"
    if "compare" in lowered or "vs" in lowered:
        return "compare"
    if "summar" in lowered:
        return "summarize"
    if candidate_mode and scope == "multi_doc":
        return "multi_candidate_extract"
    if candidate_mode:
        return "candidate_profile_extract"
    return "lookup"


def _build_retrieval_plan(intent_type: str, scope: str, candidate_mode: bool) -> RetrievalPlan:
    if candidate_mode:
        return RetrievalPlan(
            retrieval_mode="coverage",
            per_doc_top_k=20,
            required_sections=["header/name", "experience", "skills", "education", "certifications", "awards"],
            adjacent_expand=True,
            rerank_policy="order_only",
            max_docs=25,
        )
    if intent_type in {"compare", "summarize", "rank", "filter", "extract_fields", "tabular_request"}:
        return RetrievalPlan(
            retrieval_mode="coverage",
            per_doc_top_k=12 if scope != "single_doc" else 16,
            required_sections=["summary", "key facts", "totals"],
            adjacent_expand=True,
            rerank_policy="order_only",
            max_docs=25,
        )
    if intent_type == "calculate":
        return RetrievalPlan(
            retrieval_mode="precision",
            per_doc_top_k=14,
            required_sections=["totals", "amounts"],
            adjacent_expand=True,
            rerank_policy="filter_and_order",
            max_docs=25,
        )
    return RetrievalPlan(
        retrieval_mode="precision",
        per_doc_top_k=12,
        required_sections=["key facts"],
        adjacent_expand=False,
        rerank_policy="filter_and_order",
        max_docs=25,
    )


def orchestrate_intent_profile_entity(
    *,
    subscription_id: str,
    profile_id: str,
    session_id: Optional[str],
    query_text: str,
    session_context: Optional[Dict[str, Any]],
    doc_inventory: Sequence[DocInventoryItem],
    doc_matcher: Optional[Any] = None,
    model_id: Optional[str] = None,
) -> OrchestrationResult:
    sequence: List[str] = []

    intent_type, requirements = classify_intent(query_text)
    sequence.append("intent")

    corpus_fingerprint = _corpus_fingerprint(doc_inventory)
    content_type, confidence = _profile_content_type(doc_inventory)
    profile_signals = ProfileSignals(
        corpus_fingerprint=corpus_fingerprint,
        content_type=content_type,
        confidence=confidence,
    )
    sequence.append("profile")

    entities = detect_entities(query_text, doc_inventory)
    sequence.append("entities")

    scope_resolution: ScopeResolution = resolve_scope(
        query_text=query_text,
        intent_type=intent_type,
        wants_all_docs=requirements.wants_all_docs,
        entities=entities,
        doc_inventory=doc_inventory,
        doc_matcher=doc_matcher,
    )
    sequence.append("scope")

    candidate_mode = _detect_candidate_mode(query_text, doc_inventory)
    retrieval_plan = _build_retrieval_plan(intent_type, scope_resolution.scope, candidate_mode)
    sequence.append("retrieval_plan")

    assumption_line = None
    if scope_resolution.scope == "multi_doc" and not requirements.wants_all_docs and len(doc_inventory) > 1:
        assumption_line = "I used all available documents because none was specified."
    elif scope_resolution.scope == "single_doc" and len(doc_inventory) == 1:
        assumption_line = "Only one document is available, so I used it by default."

    legacy_intent = _legacy_candidate_intent(query_text, candidate_mode, scope_resolution.scope)
    evidence_policy = EvidencePolicy(max_evidence_lines=3, only_high_signal=True, include_evidence=True)

    decision = IntentScopeDecision(
        intent=legacy_intent,
        scope=scope_resolution.scope,
        target_docs=scope_resolution.target_docs,
        retrieval_plan=retrieval_plan,
        evidence_policy=evidence_policy,
        assumption_line=assumption_line,
        candidate_mode=candidate_mode,
    )

    return OrchestrationResult(
        intent_type=intent_type,
        output_requirements=requirements,
        profile_signals=profile_signals,
        entities=entities,
        scope=scope_resolution.scope,
        target_docs=scope_resolution.target_docs,
        retrieval_plan=retrieval_plan,
        assumption_line=assumption_line,
        candidate_mode=candidate_mode,
        decision=decision,
        sequence=sequence,
    )


__all__ = [
    "OutputRequirements",
    "ProfileSignals",
    "OrchestrationResult",
    "classify_intent",
    "orchestrate_intent_profile_entity",
]
