# tests/test_evolve_gate.py
import pytest


class TestQualityGate:
    def test_passes_when_all_criteria_met(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(composite=84.2, scores={"accuracy": 87, "groundedness": 83, "reasoning": 82, "formatting": 81, "tone": 79}, previous_composite=81.0)
        assert result.passed is True
        assert result.reason == ""

    def test_fails_composite_below_minimum(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(composite=75.0, scores={"accuracy": 80, "groundedness": 75, "reasoning": 70, "formatting": 70, "tone": 70}, previous_composite=70.0)
        assert result.passed is False
        assert "composite" in result.reason.lower()

    def test_fails_criterion_below_floor(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(composite=82.0, scores={"accuracy": 90, "groundedness": 85, "reasoning": 55, "formatting": 80, "tone": 80}, previous_composite=79.0)
        assert result.passed is False
        assert "reasoning" in result.reason.lower()

    def test_fails_not_beating_previous(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(composite=81.0, scores={"accuracy": 82, "groundedness": 80, "reasoning": 80, "formatting": 80, "tone": 80}, previous_composite=82.0)
        assert result.passed is False
        assert "previous" in result.reason.lower()

    def test_passes_without_previous_when_no_previous(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(composite=81.0, scores={"accuracy": 82, "groundedness": 80, "reasoning": 80, "formatting": 80, "tone": 80}, previous_composite=None)
        assert result.passed is True

    def test_multiple_failures_reported(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(composite=70.0, scores={"accuracy": 50, "groundedness": 55, "reasoning": 80, "formatting": 80, "tone": 80}, previous_composite=75.0)
        assert result.passed is False
        assert "accuracy" in result.reason.lower()
        assert "groundedness" in result.reason.lower()
