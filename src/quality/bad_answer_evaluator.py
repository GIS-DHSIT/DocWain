from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

from .fast_grounding import GroundingResult, evaluate_grounding
from .fast_intent import IntentResult, evaluate_intent_alignment
from .fast_relevance import RelevanceResult, evaluate_relevance_with_answer
from .structure_rules import StructureResult, evaluate_structure


_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
_SOURCE_TAG_RE = re.compile(r"\[SOURCE-\d+\]")

_LEAKAGE_TOKENS = [
    "collection_name",
    "point_id",
    "payload",
    "qdrant",
    "system",
    "developer",
    "tool",
    "hidden",
    "i will now",
    "/home/",
    "/mnt/",
    "src/",
    ".venv/",
    "c:\\",
]

_REFUSAL_PHRASES = [
    "unsupported",
    "can't find",
    "cannot find",
    "unable to find",
    "can't locate",
    "cannot locate",
    "no information",
    "not provided",
    "i don't have",
    "i do not have",
    "not available",
]


@dataclass
class EvalConfig:
    eval_budget_ms: int = 40
    high_confidence_threshold: float = 0.75
    lex_support_th: float = 0.18
    supported_ratio_th: float = 0.60
    critical_supported_ratio_th: float = 0.75
    low_relevance_threshold: float = 0.18
    min_coverage_ratio: float = 0.50
    overall_score_th: float = 0.72
    require_headings_over_words: int = 120


@dataclass
class EvalResult:
    overall_score: float
    bad: bool
    severity: Optional[str]
    reasons: List[str] = field(default_factory=list)
    hard_fail: bool = False
    intent_expected: str = ""
    intent_delivered: str = ""
    supported_ratio: float = 0.0
    critical_supported_ratio: float = 0.0
    coverage_ratio: float = 0.0
    structure: Optional[StructureResult] = None
    robot_opener: bool = False
    relevance_score: float = 0.0
    query_overlap: float = 0.0
    eval_latency_ms: float = 0.0
    budget_exceeded: bool = False


class BadAnswerEvaluator:
    def __init__(self, config: Optional[EvalConfig] = None, *, time_fn: Optional[Any] = None) -> None:
        self.config = config or EvalConfig()
        self._time_fn = time_fn or time.perf_counter

    def evaluate(
        self,
        *,
        query: str,
        answer: str,
        retrieved_chunks: Sequence[Any],
        retrieval_confidence: Optional[float] = None,
        query_intent: Optional[str] = None,
    ) -> EvalResult:
        start = self._time_fn()
        deadline = start + (self.config.eval_budget_ms / 1000.0)
        reasons: List[str] = []
        hard_fail = False

        def budget_exceeded() -> bool:
            return self._time_fn() > deadline

        normalized_answer = (answer or "").strip()
        lowered = normalized_answer.lower()

        structure = evaluate_structure(normalized_answer)
        if budget_exceeded():
            return EvalResult(
                overall_score=1.0,
                bad=False,
                severity=None,
                reasons=["eval_budget_exceeded"],
                eval_latency_ms=round((self._time_fn() - start) * 1000, 3),
                budget_exceeded=True,
            )

        leakage = False
        if _UUID_RE.search(normalized_answer) or _SOURCE_TAG_RE.search(normalized_answer):
            leakage = True
        else:
            leakage = any(token in lowered for token in _LEAKAGE_TOKENS)

        if leakage:
            reasons.append("leakage")
            hard_fail = True

        wrong_refusal = False
        if retrieval_confidence is not None and retrieval_confidence >= self.config.high_confidence_threshold:
            if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
                wrong_refusal = True
                reasons.append("wrong_domain_refusal")
                hard_fail = True

        robot_opener = bool(structure.robot_opener)
        if robot_opener:
            reasons.append("robot_opener")

        if budget_exceeded():
            return EvalResult(
                overall_score=1.0,
                bad=False,
                severity=None,
                reasons=reasons + ["eval_budget_exceeded"],
                structure=structure,
                robot_opener=robot_opener,
                eval_latency_ms=round((self._time_fn() - start) * 1000, 3),
                budget_exceeded=True,
            )

        intent_result = evaluate_intent_alignment(query, normalized_answer, query_intent=query_intent)
        if intent_result.mismatch:
            reasons.append("intent_mismatch")
            hard_fail = True

        chunk_texts = [getattr(chunk, "text", "") or "" for chunk in retrieved_chunks]
        relevance_result = evaluate_relevance_with_answer(
            query,
            normalized_answer,
            chunk_texts,
            retrieval_confidence=retrieval_confidence,
            low_relevance_threshold=self.config.low_relevance_threshold,
        )
        if relevance_result.low_relevance:
            reasons.append("low_relevance")

        fast_pass = (
            not hard_fail
            and retrieval_confidence is not None
            and retrieval_confidence >= self.config.high_confidence_threshold
            and not robot_opener
        )

        grounding: GroundingResult | None = None
        coverage_ratio = 1.0
        structure_fail = False

        if not fast_pass:
            grounding = evaluate_grounding(
                normalized_answer,
                chunk_texts,
                support_threshold=self.config.lex_support_th,
            )
            if grounding.supported_ratio < self.config.supported_ratio_th:
                reasons.append("low_supported_ratio")
                hard_fail = True
            if grounding.critical_supported_ratio < self.config.critical_supported_ratio_th:
                reasons.append("low_critical_support")
                hard_fail = True

            coverage_ratio = _coverage_ratio(query, grounding or GroundingResult(0.0, 0.0, [], [], []))
            if coverage_ratio < self.config.min_coverage_ratio:
                reasons.append("incomplete")
                hard_fail = True

            if intent_result and intent_result.expected == "comparison" and not structure.has_table:
                structure_fail = True
            if intent_result and intent_result.expected == "procedural" and not structure.has_steps:
                structure_fail = True
            if intent_result and intent_result.expected == "summary" and not structure.has_headings:
                structure_fail = True

        word_count = len(normalized_answer.split())
        if word_count > self.config.require_headings_over_words and not structure.has_headings:
            structure_fail = True

        if structure.long_paragraphs > 0:
            structure_fail = True

        if structure_fail:
            reasons.append("structure_fail")

        if budget_exceeded():
            return EvalResult(
                overall_score=1.0,
                bad=False,
                severity=None,
                reasons=reasons + ["eval_budget_exceeded"],
                structure=structure,
                robot_opener=robot_opener,
                eval_latency_ms=round((self._time_fn() - start) * 1000, 3),
                budget_exceeded=True,
            )

        overall_score = _score_overall(
            grounding=grounding,
            intent=intent_result,
            relevance=relevance_result,
            coverage_ratio=coverage_ratio,
            structure_fail=structure_fail,
            robot_opener=robot_opener,
            leakage=leakage,
        )
        bad = hard_fail or overall_score < self.config.overall_score_th

        severity = _severity(
            leakage=leakage,
            grounding_fail=bool(grounding and (
                grounding.supported_ratio < self.config.supported_ratio_th
                or grounding.critical_supported_ratio < self.config.critical_supported_ratio_th
            )),
            intent_mismatch=bool(intent_result and intent_result.mismatch),
            wrong_refusal=wrong_refusal,
            low_relevance=bool(relevance_result and relevance_result.low_relevance),
            incomplete=coverage_ratio < self.config.min_coverage_ratio,
            structure_fail=structure_fail,
            robot_opener=robot_opener,
        ) if bad else None

        elapsed_ms = (self._time_fn() - start) * 1000

        return EvalResult(
            overall_score=round(overall_score, 4),
            bad=bad,
            severity=severity,
            reasons=reasons,
            hard_fail=hard_fail,
            intent_expected=intent_result.expected if intent_result else "",
            intent_delivered=intent_result.delivered if intent_result else "",
            supported_ratio=grounding.supported_ratio if grounding else 1.0,
            critical_supported_ratio=grounding.critical_supported_ratio if grounding else 1.0,
            coverage_ratio=round(coverage_ratio, 4),
            structure=structure,
            robot_opener=robot_opener,
            relevance_score=relevance_result.relevance_score if relevance_result else 1.0,
            query_overlap=relevance_result.query_overlap if relevance_result else 1.0,
            eval_latency_ms=round(elapsed_ms, 3),
            budget_exceeded=False,
        )


def _coverage_ratio(query: str, grounding: GroundingResult) -> float:
    requirements = _extract_requirements(query)
    if not requirements:
        return 1.0
    sentences = grounding.supported_sentences or []
    if not sentences:
        sentences = grounding.unsupported_sentences or []
    if not sentences:
        return 0.0
    covered = 0
    for req in requirements:
        if _requirement_covered(req, sentences):
            covered += 1
    return covered / max(len(requirements), 1)


def _extract_requirements(query: str) -> List[str]:
    lowered = query.lower()
    compare_match = re.search(r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)", lowered)
    requirements: List[str] = []
    if compare_match:
        requirements.extend([compare_match.group(1).strip(), compare_match.group(2).strip()])
    parts = re.split(r"\band\b|,|;|\?|\n", query)
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip()
        if cleaned and len(cleaned) > 2:
            requirements.append(cleaned)
    deduped: List[str] = []
    for req in requirements:
        if req not in deduped:
            deduped.append(req)
    return deduped[:8]


def _requirement_covered(requirement: str, sentences: Iterable[str]) -> bool:
    req_tokens = set(re.findall(r"[a-z0-9]{3,}", requirement.lower()))
    if not req_tokens:
        return False
    for sentence in sentences:
        sent_tokens = set(re.findall(r"[a-z0-9]{3,}", sentence.lower()))
        if req_tokens & sent_tokens:
            return True
    return False


def _score_overall(
    *,
    grounding: Optional[GroundingResult],
    intent: Optional[IntentResult],
    relevance: Optional[RelevanceResult],
    coverage_ratio: float,
    structure_fail: bool,
    robot_opener: bool,
    leakage: bool,
) -> float:
    safety = 0.0 if leakage else 1.0
    grounding_score = grounding.supported_ratio if grounding else 1.0
    intent_score = 1.0 if not (intent and intent.mismatch) else 0.0
    relevance_score = relevance.relevance_score if relevance else 1.0
    completeness_score = coverage_ratio
    structure_score = 0.6 if structure_fail else 1.0
    conversational_score = 0.6 if robot_opener else 1.0

    weights = {
        "grounding": 0.3,
        "intent": 0.15,
        "relevance": 0.15,
        "completeness": 0.15,
        "structure": 0.1,
        "conversational": 0.1,
        "safety": 0.05,
    }
    score = (
        grounding_score * weights["grounding"]
        + intent_score * weights["intent"]
        + relevance_score * weights["relevance"]
        + completeness_score * weights["completeness"]
        + structure_score * weights["structure"]
        + conversational_score * weights["conversational"]
        + safety * weights["safety"]
    )
    return max(0.0, min(1.0, score))


def _severity(
    *,
    leakage: bool,
    grounding_fail: bool,
    intent_mismatch: bool,
    wrong_refusal: bool,
    low_relevance: bool,
    incomplete: bool,
    structure_fail: bool,
    robot_opener: bool,
) -> Optional[str]:
    if leakage or grounding_fail or intent_mismatch or wrong_refusal:
        return "Sev1"
    if low_relevance or incomplete:
        return "Sev2"
    if structure_fail or robot_opener:
        return "Sev3"
    return None
