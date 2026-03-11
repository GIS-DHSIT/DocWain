"""Tests for template-based response assembly."""
import pytest
from src.docwain_intel.response_assembler import assemble_response, AssembledResponse
from src.docwain_intel.query_router import QueryRoute


class TestAssembleResponse:
    def test_entity_lookup_response(self):
        """Direct entity facts should render as bullet list."""
        facts = [
            {"subject": "John Smith", "predicate": "EMAIL", "value": "john@example.com", "source": "resume.pdf", "page": 1},
            {"subject": "John Smith", "predicate": "PHONE", "value": "555-0123", "source": "resume.pdf", "page": 1},
            {"subject": "John Smith", "predicate": "ROLE", "value": "Senior Engineer", "source": "resume.pdf", "page": 1},
        ]
        result = assemble_response(
            query="What is John Smith's contact info?",
            route=QueryRoute.GRAPH_DIRECT,
            facts=facts,
        )
        assert isinstance(result, AssembledResponse)
        assert "john@example.com" in result.text
        assert "555-0123" in result.text
        assert len(result.sources) >= 1

    def test_comparison_response_as_table(self):
        """Comparison results should render as markdown table."""
        facts = [
            {"subject": "John Smith", "predicate": "SKILL", "value": "Python", "source": "john_resume.pdf", "page": 1},
            {"subject": "John Smith", "predicate": "EXPERIENCE", "value": "8 years", "source": "john_resume.pdf", "page": 1},
            {"subject": "Sarah Chen", "predicate": "SKILL", "value": "Java", "source": "sarah_resume.pdf", "page": 1},
            {"subject": "Sarah Chen", "predicate": "EXPERIENCE", "value": "5 years", "source": "sarah_resume.pdf", "page": 1},
        ]
        result = assemble_response(
            query="Compare John Smith and Sarah Chen",
            route=QueryRoute.HYBRID_SEARCH,
            facts=facts,
            is_comparison=True,
        )
        assert "|" in result.text  # markdown table
        assert "John Smith" in result.text
        assert "Sarah Chen" in result.text

    def test_aggregation_response(self):
        """Aggregation queries should include count."""
        facts = [
            {"subject": "John Smith", "predicate": "HAS_SKILL", "value": "Python", "source": "doc1.pdf", "page": 1},
            {"subject": "Sarah Chen", "predicate": "HAS_SKILL", "value": "Python", "source": "doc2.pdf", "page": 2},
            {"subject": "Mike Jones", "predicate": "HAS_SKILL", "value": "Python", "source": "doc3.pdf", "page": 1},
        ]
        result = assemble_response(
            query="How many people have Python skills?",
            route=QueryRoute.HYBRID_SEARCH,
            facts=facts,
            is_aggregation=True,
        )
        assert "3" in result.text

    def test_chunk_based_response(self):
        """When facts are empty but chunks provided, render chunk evidence."""
        chunks = [
            {"text": "The company reported $2M in revenue for Q3 2024.", "source": "report.pdf", "page": 5, "score": 0.92},
            {"text": "Operating costs were $1.5M during the same period.", "source": "report.pdf", "page": 6, "score": 0.85},
        ]
        result = assemble_response(
            query="What was the company's revenue?",
            route=QueryRoute.FULL_SEARCH,
            chunks=chunks,
        )
        assert "$2M" in result.text
        assert result.confidence > 0.0

    def test_no_results_response(self):
        """Empty results should produce a helpful no-results message."""
        result = assemble_response(
            query="What is the weather today?",
            route=QueryRoute.GRAPH_DIRECT,
        )
        assert result.text  # should not be empty
        assert result.confidence == 0.0

    def test_source_attribution(self):
        """Every response must include source attribution."""
        facts = [
            {"subject": "Contract", "predicate": "VALUE", "value": "$50,000", "source": "contract.pdf", "page": 3},
        ]
        result = assemble_response(
            query="What is the contract value?",
            route=QueryRoute.GRAPH_DIRECT,
            facts=facts,
        )
        assert len(result.sources) >= 1
        assert any("contract.pdf" in s.get("source", "") for s in result.sources)

    def test_confidence_signal(self):
        """High-quality matches should have high confidence."""
        facts = [
            {"subject": "John", "predicate": "EMAIL", "value": "john@test.com", "source": "doc.pdf", "page": 1, "confidence": 0.95},
        ]
        result = assemble_response(
            query="What is John's email?",
            route=QueryRoute.GRAPH_DIRECT,
            facts=facts,
        )
        assert result.confidence > 0.5

    def test_empty_query(self):
        result = assemble_response(query="", route=QueryRoute.GRAPH_DIRECT)
        assert isinstance(result, AssembledResponse)
