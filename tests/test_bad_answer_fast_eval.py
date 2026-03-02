from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.quality.bad_answer_evaluator import BadAnswerEvaluator, EvalConfig
from src.quality.auto_repair import AutoRepairEngine, RepairConfig


@dataclass
class DummyChunk:
    text: str
    metadata: Dict[str, Any]
    score: float = 0.8


def test_leakage_detection_uuid_and_path():
    evaluator = BadAnswerEvaluator(EvalConfig(eval_budget_ms=50))
    answer = "Here is id 123e4567-e89b-12d3-a456-426614174000 and /home/user/path"
    result = evaluator.evaluate(
        query="What is the policy?",
        answer=answer,
        retrieved_chunks=[DummyChunk(text="policy details", metadata={})],
        retrieval_confidence=0.9,
        query_intent="qa",
    )
    assert result.bad
    assert result.severity == "Sev1"


def test_wrong_domain_refusal_high_confidence():
    evaluator = BadAnswerEvaluator(EvalConfig(eval_budget_ms=50))
    answer = "I can't find that in the documents."
    result = evaluator.evaluate(
        query="What is the refund policy?",
        answer=answer,
        retrieved_chunks=[DummyChunk(text="refund policy details", metadata={})],
        retrieval_confidence=0.8,
        query_intent="qa",
    )
    assert result.bad
    assert result.severity == "Sev1"


def test_intent_mismatch_comparison_without_table():
    evaluator = BadAnswerEvaluator(EvalConfig(eval_budget_ms=50))
    answer = "Policy A covers leave. Policy B covers travel."
    chunk_text = "Policy A covers leave. Policy B covers travel."
    result = evaluator.evaluate(
        query="Compare policy A vs policy B",
        answer=answer,
        retrieved_chunks=[DummyChunk(text=chunk_text, metadata={})],
        retrieval_confidence=0.4,
        query_intent="comparison",
    )
    assert result.bad
    assert result.severity == "Sev1"


def test_grounding_hard_fail():
    evaluator = BadAnswerEvaluator(EvalConfig(eval_budget_ms=50))
    answer = "The policy allows 10 days of leave."
    result = evaluator.evaluate(
        query="How many leave days?",
        answer=answer,
        retrieved_chunks=[DummyChunk(text="Travel policy details.", metadata={})],
        retrieval_confidence=0.3,
        query_intent="qa",
    )
    assert result.bad
    assert result.severity == "Sev1"


def test_completeness_missing_requirement():
    evaluator = BadAnswerEvaluator(EvalConfig(eval_budget_ms=50))
    answer = "Benefits include flexible hours."
    chunk_text = "Benefits include flexible hours. Risks include compliance gaps."
    result = evaluator.evaluate(
        query="List benefits, risks, and costs",
        answer=answer,
        retrieved_chunks=[DummyChunk(text=chunk_text, metadata={})],
        retrieval_confidence=0.3,
        query_intent="qa",
    )
    assert result.bad
    assert "incomplete" in result.reasons


def test_performance_budget_mock_clock():
    class FakeClock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            self.t += 0.001
            return self.t

    clock = FakeClock()
    evaluator = BadAnswerEvaluator(EvalConfig(eval_budget_ms=40), time_fn=clock)
    answer = "Policy covers travel."
    result = evaluator.evaluate(
        query="What does the policy cover?",
        answer=answer,
        retrieved_chunks=[DummyChunk(text="Policy covers travel.", metadata={})],
        retrieval_confidence=0.9,
        query_intent="qa",
    )
    assert result.eval_latency_ms <= 40


def test_repair_loop_caps_attempts_and_budget():
    class FakeClock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            self.t += 0.3
            return self.t

    clock = FakeClock()
    engine = AutoRepairEngine(RepairConfig(max_attempts=2, repair_budget_ms=500), time_fn=clock)

    class Eval:
        def __init__(self, bad: bool):
            self.bad = bad

    def retrieve_fn(query: str, filters: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        return {"chunks": [DummyChunk(text="context", metadata={})], "attempts": []}

    def build_context_fn(chunks: List[Any], rerank_k: int, rerank_query: str):
        return chunks, chunks, "context", [{"source_name": "doc"}]

    def build_verification_sources(sources: List[Dict[str, Any]], chunks: List[Any]):
        return sources

    def generate_fn(context: str, sources: List[Dict[str, Any]], chunks: List[Any], tg: str, to: str, strict: bool):
        return "answer", "prompt", "", {"verification": None, "confidence": 0.2, "confidence_breakdown": {}}

    def eval_fn(answer: str, chunks: List[Any], retrieval_conf: float, intent_expected: str):
        return Eval(bad=True)

    def retrieval_confidence_fn(chunks: List[Any], sources: List[Dict[str, Any]]):
        return 0.1

    def template_for_intent(intent: str):
        return "guidance", "outline"

    outcome = engine.run(
        query="q",
        processed_query="q",
        intent_expected="qa",
        evaluation=Eval(bad=True),
        retrieved_chunks=[DummyChunk(text="context", metadata={})],
        retrieval_confidence=0.1,
        metadata_filters={},
        top_k_retrieval=10,
        top_k_rerank=5,
        retrieve_fn=retrieve_fn,
        build_context_fn=build_context_fn,
        build_verification_sources_fn=build_verification_sources,
        generate_fn=generate_fn,
        rewrite_citations_fn=lambda a, s: a,
        eval_fn=eval_fn,
        retrieval_confidence_fn=retrieval_confidence_fn,
        template_for_intent_fn=template_for_intent,
        clarification_fn=None,
        refusal_fn=None,
    )
    assert outcome.attempts <= 2
    assert outcome.elapsed_ms >= 0.0
