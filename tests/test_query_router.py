"""Tests for NLP-based query routing."""
import threading

import pytest
from src.docwain_intel.query_router import route_query, QueryRoute, QueryAnalysis, _sanitize_query


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

    def test_none_query(self):
        """None input should be handled gracefully."""
        analysis = route_query(None)
        assert analysis.route == QueryRoute.GRAPH_DIRECT
        assert analysis.is_conversational

    def test_whitespace_only_query(self):
        """Whitespace-only input should be treated as empty."""
        analysis = route_query("   \t\n  ")
        assert analysis.route == QueryRoute.GRAPH_DIRECT
        assert analysis.is_conversational

    def test_null_bytes_stripped(self):
        """Null bytes and control characters should be removed."""
        analysis = route_query("What is\x00 John's email\x01?")
        assert len(analysis.entities) >= 1

    def test_control_characters_stripped(self):
        """C0/C1 control characters should be removed, whitespace preserved."""
        cleaned = _sanitize_query("hello\x00\x01\x02world")
        assert "\x00" not in cleaned
        assert "helloworld" in cleaned

    def test_query_truncation(self):
        """Queries exceeding 10000 chars should be truncated."""
        long_query = "a" * 20000
        cleaned = _sanitize_query(long_query)
        assert len(cleaned) == 10000

    def test_only_control_chars_query(self):
        """A query of only control characters should be treated as empty."""
        analysis = route_query("\x00\x01\x02")
        assert analysis.route == QueryRoute.GRAPH_DIRECT
        assert analysis.is_conversational

    def test_thread_safety(self):
        """route_query should be safe to call from multiple threads."""
        results = []
        errors = []

        def worker(q):
            try:
                r = route_query(q)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"What is item {i}?",))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert len(errors) == 0
        assert len(results) == 8
