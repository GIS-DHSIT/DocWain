from __future__ import annotations

import concurrent.futures
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.agentic.memory import AgentMemory
from src.agentic.verification_agent import VerificationAgent

logger = get_logger(__name__)

@dataclass
class RetrievalAttempt:
    label: str
    query: str
    hits: int
    top_score: float
    strategy: str

@dataclass
class RetrievalPlan:
    chunks: List[Any]
    attempts: List[RetrievalAttempt]
    query_variants: List[str]
    metadata: Dict[str, Any]
    selected_strategy: str = "parallel_hybrid"
    profile_context: Dict[str, Any] = field(default_factory=dict)

class RetrieverManager:
    """Orchestrates parallel retrieval strategies and merges results."""

    def __init__(
        self,
        *,
        qdrant_retriever: Any,
        hybrid_retriever: Any,
        verification_agent: Optional[VerificationAgent] = None,
        chunk_factory: Optional[Callable[..., Any]] = None,
        max_workers: int = 4,
    ) -> None:
        self.qdrant_retriever = qdrant_retriever
        self.hybrid_retriever = hybrid_retriever
        self.verification_agent = verification_agent or VerificationAgent()
        self.chunk_factory = chunk_factory
        self.max_workers = max_workers

    def run(
        self,
        *,
        query: str,
        profile_id: str,
        collection_name: str,
        query_analysis: Dict[str, Any],
        query_understanding: Dict[str, Any],
        metadata_filters: Dict[str, Any],
        top_k: int,
        memory: Optional[AgentMemory] = None,
        evidence_plan: Optional[Dict[str, Any]] = None,
        profile_context: Optional[Dict[str, Any]] = None,
    ) -> RetrievalPlan:
        memory = memory or AgentMemory()
        profile_context = profile_context or {}

        query_variants = self._build_query_variants(query, query_analysis, query_understanding, evidence_plan)
        filters = self._build_filter_payload(metadata_filters, query_understanding)
        attempts: List[RetrievalAttempt] = []

        results = self._parallel_retrieve(
            collection_name=collection_name,
            profile_id=profile_id,
            query_variants=query_variants,
            filters=filters,
            top_k=top_k,
        )

        merged = self._merge_chunks(results, query_variants, memory)
        merged = self._rerank(query, merged, memory)

        for label, chunks in results:
            if chunks:
                attempts.append(
                    RetrievalAttempt(
                        label=label,
                        query=query,
                        hits=len(chunks),
                        top_score=round(float(chunks[0].score), 4),
                        strategy=label,
                    )
                )
            else:
                attempts.append(RetrievalAttempt(label=label, query=query, hits=0, top_score=0.0, strategy=label))

        verification = self.verification_agent.verify(
            query=query,
            intent=str(query_analysis.get("intent") or ""),
            chunks=merged,
            metadata_filters=metadata_filters,
        )

        if verification.should_refine and self._can_refine(metadata_filters):
            refined = self._parallel_retrieve(
                collection_name=collection_name,
                profile_id=profile_id,
                query_variants=query_variants,
                filters=metadata_filters,
                top_k=max(10, int(top_k * 0.6)),
                refined=True,
            )
            refined_chunks = self._merge_chunks(refined, query_variants, memory)
            if refined_chunks:
                merged = self._rerank(query, refined_chunks, memory)
                for label, chunks in refined:
                    attempts.append(
                        RetrievalAttempt(
                            label=f"{label}_refined",
                            query=query,
                            hits=len(chunks),
                            top_score=round(float(chunks[0].score), 4) if chunks else 0.0,
                            strategy=label,
                        )
                    )

        return RetrievalPlan(
            chunks=merged,
            attempts=attempts,
            query_variants=query_variants,
            metadata={
                "intent": query_analysis.get("intent"),
                "query_analysis": query_analysis,
                "query_understanding": query_understanding,
                "retrieval_alignment": verification.alignment_score,
                "retrieval_issues": verification.issues,
            },
            selected_strategy="parallel_hybrid",
            profile_context=profile_context,
        )

    def _parallel_retrieve(
        self,
        *,
        collection_name: str,
        profile_id: str,
        query_variants: List[str],
        filters: Dict[str, Any],
        top_k: int,
        refined: bool = False,
    ) -> List[Tuple[str, List[Any]]]:
        tasks: List[Tuple[str, Callable[[], List[Any]]]] = []
        for variant in query_variants:
            tasks.append(
                (
                    f"dense_{'refined' if refined else 'base'}",
                    lambda v=variant: self.qdrant_retriever.retrieve(
                        collection_name=collection_name,
                        query=v,
                        filter_profile=profile_id,
                        top_k=top_k,
                        document_ids=filters.get("document_ids"),
                        source_files=filters.get("source_files"),
                        doc_types=filters.get("doc_types"),
                        section_titles=filters.get("section_titles"),
                        section_paths=filters.get("section_paths"),
                        page_numbers=filters.get("page_numbers"),
                        min_confidence=filters.get("min_confidence"),
                    ),
                )
            )
            tasks.append(
                (
                    f"lexical_{'refined' if refined else 'base'}",
                    lambda v=variant: self.qdrant_retriever.sparse_retrieve(
                        collection_name=collection_name,
                        query=v,
                        profile_id=profile_id,
                        top_k=max(8, int(top_k * 0.6)),
                        filters=filters,
                    ),
                )
            )

        if self._has_metadata_filters(filters):
            tasks.append(
                (
                    "metadata_only",
                    lambda: self.qdrant_retriever.metadata_retrieve(
                        collection_name=collection_name,
                        profile_id=profile_id,
                        filters=filters,
                        limit=max(10, int(top_k * 0.5)),
                    ),
                )
            )

        results: List[Tuple[str, List[Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(fn): label for label, fn in tasks}
            for future in concurrent.futures.as_completed(future_map):
                label = future_map[future]
                try:
                    chunks = future.result() or []
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Parallel retrieval task failed: %s", exc)
                    chunks = []
                results.append((label, chunks))
        return results

    def _merge_chunks(
        self,
        results: Iterable[Tuple[str, List[Any]]],
        query_variants: List[str],
        memory: AgentMemory,
    ) -> List[Any]:
        merged: Dict[str, Any] = {}
        for label, chunks in results:
            for chunk in chunks:
                meta = getattr(chunk, "metadata", {}) or {}
                key = meta.get("chunk_id") or getattr(chunk, "id", None) or str(id(chunk))
                existing = merged.get(key)
                if existing:
                    existing_meta = getattr(existing, "metadata", {}) or {}
                    methods = set(existing_meta.get("methods") or [])
                    methods.add(label)
                    incoming_methods = meta.get("methods") or []
                    methods.update(incoming_methods)
                    if methods:
                        existing_meta["methods"] = list(methods)
                    variants = set(existing_meta.get("query_variants") or [])
                    variants.update(query_variants)
                    if variants:
                        existing_meta["query_variants"] = list(variants)
                    existing.metadata = existing_meta
                    if float(chunk.score) > float(existing.score):
                        existing.score = chunk.score
                        if getattr(chunk, "text", ""):
                            existing.text = chunk.text
                else:
                    meta = dict(meta)
                    meta["query_variants"] = list(dict.fromkeys((meta.get("query_variants") or []) + query_variants))
                    methods = set(meta.get("methods") or [])
                    methods.add(label)
                    meta["methods"] = list(methods)
                    chunk.metadata = meta
                    merged[key] = chunk
                memory.register_chunk(meta)
        return list(merged.values())

    def _rerank(self, query: str, chunks: List[Any], memory: AgentMemory) -> List[Any]:
        if not chunks:
            return []
        token_set = set(self._tokenize(query))
        scored: List[Tuple[float, Any]] = []
        for chunk in chunks:
            meta = getattr(chunk, "metadata", {}) or {}
            text_tokens = set(self._tokenize(getattr(chunk, "text", "")))
            overlap = len(token_set & text_tokens) / max(len(token_set), 1)
            score = float(getattr(chunk, "score", 0.0))
            doc_id = meta.get("document_id")
            if doc_id and str(doc_id) in memory.visited_documents:
                score -= 0.03
            combined = score + (0.15 * overlap)
            scored.append((combined, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]{3,}", (text or "").lower())

    @staticmethod
    def _has_metadata_filters(filters: Dict[str, Any]) -> bool:
        return any(filters.get(key) for key in ("document_ids", "source_files", "doc_types", "section_titles", "page_numbers"))

    @staticmethod
    def _build_query_variants(
        query: str,
        query_analysis: Dict[str, Any],
        query_understanding: Dict[str, Any],
        evidence_plan: Optional[Dict[str, Any]],
    ) -> List[str]:
        variants: List[str] = []
        for candidate in [query] + list(query_understanding.get("sub_queries") or []) + list(query_analysis.get("sub_queries") or []):
            cleaned = re.sub(r"\s+", " ", candidate or "").strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        expanded = query_analysis.get("expanded_query")
        if expanded and expanded not in variants:
            variants.append(expanded)

        if evidence_plan:
            for step in evidence_plan.get("steps") or []:
                claim = step.get("claim")
                if claim:
                    cleaned = re.sub(r"\s+", " ", str(claim)).strip()
                    if cleaned and cleaned not in variants:
                        variants.append(cleaned)
            missing = evidence_plan.get("missing_info")
            if missing:
                cleaned = re.sub(r"\s+", " ", str(missing)).strip()
                if cleaned and cleaned not in variants:
                    variants.append(cleaned)

        return variants[:6] if variants else [query]

    @staticmethod
    def _build_filter_payload(
        metadata_filters: Dict[str, Any],
        query_understanding: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "document_ids": metadata_filters.get("document_ids"),
            "source_files": metadata_filters.get("source_files"),
            "doc_types": metadata_filters.get("doc_types"),
            "section_titles": metadata_filters.get("section_titles"),
            "section_paths": metadata_filters.get("section_paths"),
            "page_numbers": metadata_filters.get("page_numbers"),
            "min_confidence": metadata_filters.get("min_confidence"),
            "document_hints": metadata_filters.get("document_hints"),
        }
        explicit = query_understanding.get("explicit_hints") or {}
        if explicit.get("source_files"):
            payload["source_files"] = explicit.get("source_files")
        if explicit.get("section_titles"):
            payload["section_titles"] = explicit.get("section_titles")
        return payload

    @staticmethod
    def _can_refine(metadata_filters: Dict[str, Any]) -> bool:
        if not metadata_filters:
            return False
        for key in ("source_files", "section_titles", "section_paths", "doc_types", "page_numbers", "document_ids"):
            if metadata_filters.get(key):
                return True
        return False
