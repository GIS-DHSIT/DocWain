"""Tests for NLP-based query routing."""
import pytest
from src.docwain_intel.query_router import route_query, QueryRoute, QueryAnalysis


class TestRouteQuery:
    def test_entity_lookup_routes_to_graph(self):
        """Direct entity queries should use graph."""
        analysis = route_query("What is John Smith's email?")
        assert isinstance(analysis, QueryAnalysis)
        assert analysis.route in (QueryRoute.GRAPH_DIRECT, QueryRoute.HYBRID_SEARCH)
        assert len(analysis.entities) >= 1

    def test_factual_query_routes_to_graph(self):
        analysis = route_query("What is the salary mentioned in the contract?")
        assert analysis.route in (QueryRoute.GRAPH_DIRECT, QueryRoute.HYBRID_SEARCH)

    def test_comparison_routes_to_hybrid(self):
        analysis = route_query("Compare the skills of John Smith and Sarah Chen.")
        assert analysis.route in (QueryRoute.HYBRID_SEARCH, QueryRoute.FULL_SEARCH)
        assert len(analysis.entities) >= 2

    def test_open_ended_routes_to_full_search(self):
        analysis = route_query("Summarize all the key findings from the documents.")
        assert analysis.route in (QueryRoute.FULL_SEARCH, QueryRoute.LLM_GENERATION)

    def test_complex_reasoning_routes_to_llm(self):
        analysis = route_query("Based on the resumes, who would be the best candidate for a senior engineering role and why?")
        assert analysis.route == QueryRoute.LLM_GENERATION

    def test_simple_greeting_routes_minimally(self):
        analysis = route_query("Hello")
        assert analysis.route in (QueryRoute.GRAPH_DIRECT, QueryRoute.HYBRID_SEARCH)
        assert analysis.is_conversational

    def test_query_with_numbers(self):
        analysis = route_query("Show invoices over $10,000")
        assert analysis.route in (QueryRoute.GRAPH_DIRECT, QueryRoute.HYBRID_SEARCH)
        assert analysis.has_numeric_constraint

    def test_aggregation_query(self):
        analysis = route_query("How many employees have Python skills?")
        assert analysis.route in (QueryRoute.HYBRID_SEARCH, QueryRoute.FULL_SEARCH)
        assert analysis.is_aggregation

    def test_empty_query(self):
        analysis = route_query("")
        assert analysis.route == QueryRoute.GRAPH_DIRECT

    def test_entities_extracted(self):
        analysis = route_query("Tell me about Microsoft's revenue in 2024.")
        assert len(analysis.entities) >= 1
        assert any("microsoft" in e.lower() for e in analysis.entities)

    def test_query_specificity(self):
        """Specific queries should have higher specificity than vague ones."""
        specific = route_query("What is John Smith's phone number?")
        vague = route_query("Tell me about the documents.")
        assert specific.specificity >= vague.specificity
