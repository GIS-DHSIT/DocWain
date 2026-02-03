from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.rag.doc_inventory import DocInventoryItem


_MULTI_DOC_CUES = {
    "all documents",
    "all document",
    "all invoices",
    "all resumes",
    "all resume",
    "all candidates",
    "across documents",
    "across files",
    "compare all",
    "rank candidates",
    "shortlist",
    "top candidates",
    "top 5",
    "top 10",
    "top 3",
}

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

_RANK_CUES = {"rank", "ranking", "shortlist", "top", "best fit"}
_COMPARE_CUES = {"compare", "difference", "vs", "versus"}
_SUMMARIZE_CUES = {"summarize", "summary", "overview", "recap"}
_EXTRACT_CUES = {"extract", "list", "pull", "find", "show"}


@dataclass(frozen=True)
class RetrievalPlan:
    retrieval_mode: str
    per_doc_top_k: int
    required_sections: List[str]
    adjacent_expand: bool
    rerank_policy: str
    max_docs: int = 25


@dataclass(frozen=True)
class EvidencePolicy:
    max_evidence_lines: int
    only_high_signal: bool
    include_evidence: bool = True


@dataclass(frozen=True)
class IntentScopeDecision:
    intent: str
    scope: str
    target_docs: str | List[DocInventoryItem]
    retrieval_plan: RetrievalPlan
    evidence_policy: EvidencePolicy
    assumption_line: Optional[str] = None
    candidate_mode: bool = False


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s._-]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_extension(name: str) -> str:
    return re.sub(r"\.[a-z0-9]{1,5}$", "", name or "", flags=re.IGNORECASE).strip()


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    return any(cue in text for cue in cues)


def _match_explicit_docs(query: str, doc_inventory: List[DocInventoryItem]) -> List[DocInventoryItem]:
    if not query or not doc_inventory:
        return []
    q_norm = _normalize(query)
    matches: List[DocInventoryItem] = []
    for item in doc_inventory:
        candidates = [item.source_file, item.document_name, _strip_extension(item.source_file)]
        for candidate in candidates:
            cand_norm = _normalize(candidate)
            if not cand_norm:
                continue
            if cand_norm in q_norm:
                matches.append(item)
                break
    return matches


def _detect_intent(query: str, candidate_mode: bool, multi_doc: bool) -> str:
    lowered = (query or "").lower()
    if _contains_any(lowered, _RANK_CUES):
        return "ranking"
    if _contains_any(lowered, _COMPARE_CUES):
        return "compare"
    if _contains_any(lowered, _SUMMARIZE_CUES):
        return "summarize"
    if candidate_mode and (multi_doc or "candidates" in lowered or "resumes" in lowered):
        return "multi_candidate_extract"
    if candidate_mode and _contains_any(lowered, _EXTRACT_CUES):
        return "candidate_profile_extract"
    if candidate_mode and "candidate" in lowered:
        return "candidate_profile_extract"
    return "lookup"


def _build_retrieval_plan(intent: str) -> RetrievalPlan:
    if intent == "candidate_profile_extract":
        return RetrievalPlan(
            retrieval_mode="coverage",
            per_doc_top_k=20,
            required_sections=["header/name", "experience", "skills", "education", "certifications", "awards"],
            adjacent_expand=True,
            rerank_policy="order_only",
        )
    if intent == "multi_candidate_extract":
        return RetrievalPlan(
            retrieval_mode="coverage",
            per_doc_top_k=15,
            required_sections=["header/name", "experience", "skills", "education", "certifications", "awards"],
            adjacent_expand=True,
            rerank_policy="order_only",
        )
    if intent == "ranking":
        return RetrievalPlan(
            retrieval_mode="coverage",
            per_doc_top_k=15,
            required_sections=["header/name", "experience", "skills", "education", "certifications", "awards"],
            adjacent_expand=True,
            rerank_policy="order_only",
        )
    if intent in {"compare", "summarize"}:
        return RetrievalPlan(
            retrieval_mode="coverage",
            per_doc_top_k=12,
            required_sections=["summary", "key facts", "totals"],
            adjacent_expand=True,
            rerank_policy="order_only",
        )
    return RetrievalPlan(
        retrieval_mode="precision",
        per_doc_top_k=12,
        required_sections=["key facts"],
        adjacent_expand=False,
        rerank_policy="filter_and_order",
    )


def decide_intent_scope(
    *,
    query_text: str,
    doc_inventory: List[DocInventoryItem],
    session_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> IntentScopeDecision:
    _ = (session_id, subscription_id, profile_id)
    lowered = (query_text or "").lower()
    candidate_mode = _contains_any(lowered, _CANDIDATE_CUES)
    if not candidate_mode:
        candidate_mode = any(
            "resume" in (doc.source_file or "").lower() or "cv" in (doc.source_file or "").lower()
            for doc in doc_inventory
        )

    explicit_docs = _match_explicit_docs(query_text, doc_inventory)
    has_multi_doc_cues = _contains_any(lowered, _MULTI_DOC_CUES)
    multi_doc_default = len(doc_inventory) > 1

    scope = "single_doc"
    target_docs: str | List[DocInventoryItem] = []
    assumption_line = None

    if explicit_docs:
        scope = "single_doc"
        target_docs = explicit_docs
    elif has_multi_doc_cues or multi_doc_default:
        scope = "multi_doc"
        target_docs = "auto_all"
        if not has_multi_doc_cues and multi_doc_default:
            assumption_line = "I used all available documents because none was specified."
    else:
        scope = "single_doc"
        target_docs = doc_inventory[:1] if doc_inventory else []
        if len(doc_inventory) == 1:
            assumption_line = "Only one document is available, so I used it by default."

    intent = _detect_intent(query_text, candidate_mode=candidate_mode, multi_doc=scope == "multi_doc")
    retrieval_plan = _build_retrieval_plan(intent)
    evidence_policy = EvidencePolicy(max_evidence_lines=3, only_high_signal=True, include_evidence=True)

    return IntentScopeDecision(
        intent=intent,
        scope=scope,
        target_docs=target_docs,
        retrieval_plan=retrieval_plan,
        evidence_policy=evidence_policy,
        assumption_line=assumption_line,
        candidate_mode=candidate_mode,
    )


def retrieve_per_doc(
    *,
    doc_inventory: List[DocInventoryItem],
    retrieve_fn: Callable[[DocInventoryItem, int], List[Any]],
    per_doc_top_k: int,
    min_chunks_per_doc: int = 5,
    fallback_fn: Optional[Callable[[DocInventoryItem, int], List[Any]]] = None,
    max_docs: int = 25,
) -> Tuple[List[Any], Dict[str, List[Any]], List[str]]:
    """
    Generic per-document retrieval loop that prevents global collapse.
    """
    all_chunks: List[Any] = []
    chunks_by_doc: Dict[str, List[Any]] = {}
    docs_used: List[str] = []

    selected_docs = doc_inventory[: max(1, int(max_docs))]
    for doc in selected_docs:
        chunks = retrieve_fn(doc, per_doc_top_k) or []
        if len(chunks) < min_chunks_per_doc and fallback_fn:
            extra = fallback_fn(doc, min_chunks_per_doc - len(chunks))
            if extra:
                chunks = chunks + extra
        if chunks:
            doc_name = doc.source_file or doc.document_name or doc.doc_id or "Document"
            chunks_by_doc[doc_name] = chunks
            docs_used.append(doc_name)
            all_chunks.extend(chunks)

    return all_chunks, chunks_by_doc, docs_used


__all__ = [
    "RetrievalPlan",
    "EvidencePolicy",
    "IntentScopeDecision",
    "decide_intent_scope",
    "retrieve_per_doc",
]
