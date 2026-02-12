"""Tests for evidence sufficiency evaluator."""
from __future__ import annotations

import pytest
from src.rag_v3.evidence_evaluator import evaluate_evidence, EvidenceSufficiency
from src.rag_v3.types import Chunk, ChunkSource


def _chunk(text: str, score: float = 0.7, doc: str = "d.pdf") -> Chunk:
    return Chunk(id=f"c_{hash(text) % 9999}", text=text, score=score,
                 source=ChunkSource(document_name=doc))


class TestEvidenceSufficiency:

    def test_sufficient_evidence_passes(self):
        chunks = [
            _chunk("Prudhvi has 8 years of AWS experience", 0.85, "prudhvi.pdf"),
            _chunk("Prudhvi holds AWS Solutions Architect certification", 0.78, "prudhvi.pdf"),
        ]
        result = evaluate_evidence(
            query="What is Prudhvi's cloud experience?",
            chunks=chunks,
            entity_hints=["Prudhvi"],
        )
        assert result.is_sufficient
        assert result.coverage_score > 0.5
        assert result.relevance_score > 0.5

    def test_empty_chunks_fails(self):
        result = evaluate_evidence("What is Prudhvi's experience?", [], ["Prudhvi"])
        assert not result.is_sufficient
        assert result.coverage_score == 0.0

    def test_irrelevant_chunks_low_score(self):
        chunks = [
            _chunk("Java developer with Spring Boot experience", 0.3),
            _chunk("React frontend specialist", 0.25),
        ]
        result = evaluate_evidence(
            query="What is Prudhvi's cloud experience?",
            chunks=chunks,
            entity_hints=["Prudhvi"],
        )
        assert result.coverage_score < 0.3

    def test_multi_entity_coverage(self):
        chunks = [
            _chunk("Prudhvi works at AWS", 0.8, "prudhvi.pdf"),
            _chunk("Ajay is an Azure architect", 0.75, "ajay.pdf"),
        ]
        result = evaluate_evidence(
            query="Compare Prudhvi and Ajay",
            chunks=chunks,
            entity_hints=["Prudhvi", "Ajay"],
        )
        assert result.coverage_score > 0.7

    def test_missing_entity_reduces_coverage(self):
        chunks = [
            _chunk("Prudhvi works at AWS", 0.8, "prudhvi.pdf"),
            _chunk("Prudhvi has 5 years experience", 0.7, "prudhvi.pdf"),
        ]
        result = evaluate_evidence(
            query="Compare Prudhvi and Ajay",
            chunks=chunks,
            entity_hints=["Prudhvi", "Ajay"],
        )
        assert result.coverage_score < 0.7
        assert "Ajay" in result.missing_entities

    def test_diversity_from_multiple_docs(self):
        chunks = [
            _chunk("Prudhvi skills", 0.8, "resume1.pdf"),
            _chunk("Prudhvi cert", 0.7, "cert.pdf"),
            _chunk("Prudhvi projects", 0.6, "projects.pdf"),
        ]
        result = evaluate_evidence("Prudhvi overview", chunks, ["Prudhvi"])
        assert result.diversity_score > 0.3
