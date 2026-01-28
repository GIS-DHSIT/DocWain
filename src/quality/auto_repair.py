from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class RepairConfig:
    max_attempts: int = 2
    repair_budget_ms: int = 500
    top_k_increase: int = 20
    rerank_increase: int = 10
    query_rewrite_once: bool = True


@dataclass
class RepairOutcome:
    updated: bool
    answer: str
    chunks: Sequence[Any]
    sources: List[Dict[str, Any]]
    context: str
    scoring: Dict[str, Any]
    eval_result: Any
    attempts: int
    elapsed_ms: float
    clarification: Optional[str] = None
    refusal: Optional[str] = None
    reason: Optional[str] = None


class AutoRepairEngine:
    def __init__(self, config: Optional[RepairConfig] = None, *, time_fn: Optional[Any] = None) -> None:
        self.config = config or RepairConfig()
        self._time_fn = time_fn or time.perf_counter

    def run(
        self,
        *,
        query: str,
        processed_query: str,
        intent_expected: str,
        evaluation: Any,
        retrieved_chunks: Sequence[Any],
        retrieval_confidence: Optional[float],
        metadata_filters: Dict[str, Any],
        top_k_retrieval: int,
        top_k_rerank: int,
        retrieve_fn: Callable[[str, Dict[str, Any], int], Dict[str, Any]],
        build_context_fn: Callable[[Sequence[Any], int, str], Tuple[Sequence[Any], Sequence[Any], str, List[Dict[str, Any]]]],
        build_verification_sources_fn: Callable[[List[Dict[str, Any]], Sequence[Any]], List[Dict[str, Any]]],
        generate_fn: Callable[[str, List[Dict[str, Any]], Sequence[Any], str, str, bool], Tuple[str, str, str, Dict[str, Any]]],
        rewrite_citations_fn: Callable[[str, List[Dict[str, Any]]], str],
        eval_fn: Callable[[str, Sequence[Any], Optional[float], Optional[str]], Any],
        retrieval_confidence_fn: Callable[[Sequence[Any], List[Dict[str, Any]]], float],
        template_for_intent_fn: Callable[[str], Tuple[str, str]],
        clarification_fn: Optional[Callable[[], str]] = None,
        refusal_fn: Optional[Callable[[], str]] = None,
    ) -> RepairOutcome:
        start = self._time_fn()
        if not evaluation or not getattr(evaluation, "bad", False):
            return RepairOutcome(
                updated=False,
                answer="",
                chunks=retrieved_chunks,
                sources=[],
                context="",
                scoring={},
                eval_result=evaluation,
                attempts=0,
                elapsed_ms=0.0,
                reason="not_bad",
            )

        metadata_boosts = _metadata_boosts(retrieved_chunks)
        boosted_filters = _merge_filters(metadata_filters, metadata_boosts)

        attempts = 0
        rewrite_used = False
        template_guidance, template_outline = template_for_intent_fn(intent_expected)

        while attempts < self.config.max_attempts:
            if _elapsed_ms(self._time_fn, start) >= self.config.repair_budget_ms:
                break

            attempts += 1
            repair_query = processed_query
            if self.config.query_rewrite_once and not rewrite_used:
                repair_query = _apply_query_boost(processed_query, metadata_boosts)
                rewrite_used = True

            retry_top_k = max(top_k_retrieval, top_k_retrieval + self.config.top_k_increase)
            rerank_k = max(top_k_rerank, top_k_rerank + self.config.rerank_increase)

            retrieval_plan = retrieve_fn(repair_query, boosted_filters, retry_top_k)
            chunks = retrieval_plan.get("chunks") or []
            attempts_payload = retrieval_plan.get("attempts") or []
            if not chunks:
                continue

            _, final_chunks, context, sources = build_context_fn(chunks, rerank_k, repair_query)
            verification_sources = build_verification_sources_fn(sources, final_chunks)
            answer, prompt, evidence_plan_text, scoring = generate_fn(
                context,
                verification_sources,
                final_chunks,
                template_guidance,
                template_outline,
                True,
            )
            answer = rewrite_citations_fn(answer, verification_sources)

            retrieval_conf = retrieval_confidence_fn(final_chunks or chunks, sources)
            eval_result = eval_fn(answer, final_chunks or chunks, retrieval_conf, intent_expected)

            if not getattr(eval_result, "bad", False):
                return RepairOutcome(
                    updated=True,
                    answer=answer,
                    chunks=final_chunks or chunks,
                    sources=sources,
                    context=context or "",
                    scoring={
                        "prompt": prompt,
                        "evidence_plan_text": evidence_plan_text,
                        "retrieval_attempts": attempts_payload,
                        **(scoring or {}),
                    },
                    eval_result=eval_result,
                    attempts=attempts,
                    elapsed_ms=_elapsed_ms(self._time_fn, start),
                )

        elapsed_ms = _elapsed_ms(self._time_fn, start)
        if clarification_fn:
            clarification = clarification_fn()
            if clarification:
                return RepairOutcome(
                    updated=False,
                    answer="",
                    chunks=retrieved_chunks,
                    sources=[],
                    context="",
                    scoring={},
                    eval_result=evaluation,
                    attempts=attempts,
                    elapsed_ms=elapsed_ms,
                    clarification=clarification,
                    reason="clarification",
                )
        if refusal_fn:
            refusal = refusal_fn()
            return RepairOutcome(
                updated=False,
                answer="",
                chunks=retrieved_chunks,
                sources=[],
                context="",
                scoring={},
                eval_result=evaluation,
                attempts=attempts,
                elapsed_ms=elapsed_ms,
                refusal=refusal,
                reason="refusal",
            )
        return RepairOutcome(
            updated=False,
            answer="",
            chunks=retrieved_chunks,
            sources=[],
            context="",
            scoring={},
            eval_result=evaluation,
            attempts=attempts,
            elapsed_ms=elapsed_ms,
            reason="budget_or_attempts",
        )


def _elapsed_ms(time_fn: Callable[[], float], start: float) -> float:
    return (time_fn() - start) * 1000


def _metadata_boosts(chunks: Sequence[Any]) -> Dict[str, List[Any]]:
    boosts: Dict[str, List[Any]] = {
        "source_files": [],
        "section_titles": [],
        "doc_types": [],
        "page_numbers": [],
    }
    for chunk in chunks[:3]:
        meta = getattr(chunk, "metadata", {}) or {}
        for key, meta_key in (
            ("source_files", "source_file"),
            ("section_titles", "section_title"),
            ("doc_types", "document_type"),
            ("page_numbers", "page"),
        ):
            value = meta.get(meta_key)
            if value and value not in boosts[key]:
                boosts[key].append(value)
    return boosts


def _merge_filters(base: Dict[str, Any], boosts: Dict[str, List[Any]]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, values in boosts.items():
        if values and not merged.get(key):
            merged[key] = values
    return merged


def _apply_query_boost(query: str, boosts: Dict[str, List[Any]]) -> str:
    hints: List[str] = []
    for key in ("source_files", "section_titles", "doc_types"):
        for value in boosts.get(key) or []:
            hints.append(str(value))
    if boosts.get("page_numbers"):
        hints.append(f"page {boosts['page_numbers'][0]}")
    if not hints:
        return query
    return f"{query} {' '.join(hints[:6])}"
