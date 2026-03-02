from __future__ import annotations

from dataclasses import dataclass

from src.retrieval import (
    QueryAnalyzer,
    EvidenceRequirements,
    EvidenceConstraints,
    HybridRanker,
    RetrievalQualityScorer,
    RetrievalQualityResult,
    FallbackRepair,
    ContextAssembler,
)


@dataclass
class DummyChunk:
    text: str
    score: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def test_numeric_question_triggers_fallback_when_no_numbers():
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze("total years of experience for Ajay")
    requirements = EvidenceRequirements.from_required_evidence(analysis.required_evidence)
    chunks = [DummyChunk(text="Summary of project work", score=0.8, metadata={})]

    ranker = HybridRanker()
    ranked = ranker.rank("total years of experience for Ajay", chunks, requirements)

    scorer = RetrievalQualityScorer(threshold_low=0.45, threshold_high=0.75)
    quality = scorer.evaluate("total years of experience for Ajay", ranked, requirements)

    calls = {"count": 0}

    def retrieve_fn(query: str, top_k: int):
        calls["count"] += 1
        return [DummyChunk(text="Ajay has 5 years of experience.", score=0.9, metadata={})]

    repair = FallbackRepair(max_attempts=1, rewrite_enabled=True)
    result = repair.repair(
        query="total years of experience for Ajay",
        chunks=ranked,
        requirements=requirements,
        quality=quality,
        retrieve_fn=retrieve_fn,
        top_k=10,
    )

    assert calls["count"] == 1
    assert any("5 years" in chunk.text for chunk in result.chunks)


def test_comparison_requires_both_entities():
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze("Compare Alpha vs Beta revenue")
    requirements = EvidenceRequirements.from_required_evidence(analysis.required_evidence)
    chunks = [DummyChunk(text="Alpha revenue was strong.", score=0.9, metadata={})]

    evaluator = EvidenceConstraints()
    coverage = evaluator.evaluate(chunks, requirements, min_ratio=0.7)
    assert not coverage.satisfied
    assert "comparison" in coverage.missing


def test_deep_analysis_prefers_section_diversity():
    assembler = ContextAssembler(max_tokens=200, max_chunks=3)
    chunks = [
        {"text": "Section A detail 1.", "score": 0.9, "metadata": {"section_title": "A", "source_file": "doc.pdf"}},
        {"text": "Section A detail 2.", "score": 0.8, "metadata": {"section_title": "A", "source_file": "doc.pdf"}},
        {"text": "Section B insight.", "score": 0.85, "metadata": {"section_title": "B", "source_file": "doc.pdf"}},
    ]
    result = assembler.build(chunks, intent_type="deep_analysis")
    sections = {src.get("section") for src in result.sources if src.get("section")}
    assert len(sections) >= 2


def test_common_path_no_extra_retrieval_calls():
    quality = RetrievalQualityResult(
        score=0.9,
        breakdown={"overlap": 0.9, "evidence": 0.9, "gap": 0.2, "diversity": 0.2},
        is_low=False,
        is_high=True,
        elapsed_ms=1.0,
    )
    requirements = EvidenceRequirements()
    chunks = [DummyChunk(text="Relevant info", score=0.9, metadata={})]
    calls = {"count": 0}

    def retrieve_fn(query: str, top_k: int):
        calls["count"] += 1
        return []

    repair = FallbackRepair(max_attempts=1, rewrite_enabled=True)
    result = repair.repair(
        query="What is the policy?",
        chunks=chunks,
        requirements=requirements,
        quality=quality,
        retrieve_fn=retrieve_fn,
        top_k=10,
    )

    assert calls["count"] == 0
    assert result.used_fallback is False


def test_retrieval_quality_budget_mock_clock():
    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            self.t += 0.001
            return self.t

    clock = FakeClock()
    scorer = RetrievalQualityScorer(threshold_low=0.45, threshold_high=0.75, budget_ms=40, time_fn=clock)
    requirements = EvidenceRequirements()
    chunks = [DummyChunk(text="Policy covers travel.", score=0.9, metadata={})]
    result = scorer.evaluate("What does the policy cover?", chunks, requirements)
    assert result.elapsed_ms <= 40
