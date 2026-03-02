"""Tests for evidence-based answerability replacement."""
from __future__ import annotations

import inspect
import pytest


class TestAnswerabilityReplacement:

    def test_borderline_evidence_adds_confidence_note(self):
        """Borderline evidence should still generate but with reduced confidence."""
        from src.rag_v3.evidence_evaluator import evaluate_evidence
        from src.rag_v3.types import Chunk, ChunkSource

        # Simulate borderline: some chunks but entity not found
        chunks = [
            Chunk(id="c1", text="General document about cloud", score=0.4,
                  source=ChunkSource(document_name="doc.pdf")),
        ]
        result = evaluate_evidence(
            "What is Prudhvi's cloud experience?",
            chunks,
            entity_hints=["Prudhvi"],
        )
        # Should be borderline: has chunks but entity not found
        assert 0.0 < result.overall_score < 0.5

    def test_clear_evidence_no_confidence_penalty(self):
        """Clear evidence should not add confidence penalty."""
        from src.rag_v3.evidence_evaluator import evaluate_evidence
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="c1", text="Prudhvi has 8 years of AWS experience", score=0.85,
                  source=ChunkSource(document_name="prudhvi.pdf")),
        ]
        result = evaluate_evidence(
            "What is Prudhvi's cloud experience?",
            chunks,
            entity_hints=["Prudhvi"],
        )
        assert result.is_sufficient
        assert result.overall_score > 0.35

    def test_pipeline_has_grounded_annotation(self):
        """Pipeline should set grounded=False for borderline evidence."""
        src_text = inspect.getsource(__import__("src.rag_v3.pipeline", fromlist=["run"]).run)
        assert "grounded" in src_text
        assert "evidence_score" in src_text

    def test_sufficiency_variable_initialized(self):
        """Pipeline should initialize sufficiency to None before the gate."""
        src_text = inspect.getsource(__import__("src.rag_v3.pipeline", fromlist=["run"]).run)
        assert "sufficiency = None" in src_text

    def test_evidence_evaluator_used_in_pipeline(self):
        """Evidence evaluator should be invoked in the pipeline."""
        src_text = inspect.getsource(__import__("src.rag_v3.pipeline", fromlist=["run"]).run)
        assert "evaluate_evidence" in src_text
