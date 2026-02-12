"""Tests for query decomposition engine."""
from __future__ import annotations

import pytest
from src.rag_v3.query_decomposer import (
    SubQuery,
    DecomposedQuery,
    decompose_query,
)


class TestSubQueryModel:

    def test_subquery_defaults(self):
        sq = SubQuery(text="Python skills")
        assert sq.text == "Python skills"
        assert sq.entity_scope is None
        assert sq.section_focus is None
        assert sq.doc_type_filter is None
        assert sq.importance == 1.0

    def test_decomposed_query_defaults(self):
        dq = DecomposedQuery(
            original="test query",
            sub_queries=[SubQuery(text="test query")],
        )
        assert dq.fusion_strategy == "rrf"
        assert len(dq.sub_queries) == 1


class TestHeuristicDecomposition:

    def test_simple_query_no_decomposition(self):
        result = decompose_query("What are the technical skills?")
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].text == "What are the technical skills?"

    def test_comparison_two_entities(self):
        result = decompose_query("Compare the experience of Prudhvi and Ajay in cloud technologies")
        assert len(result.sub_queries) >= 2
        entities = {sq.entity_scope for sq in result.sub_queries if sq.entity_scope}
        assert "Prudhvi" in entities
        assert "Ajay" in entities

    def test_comparison_vs_syntax(self):
        result = decompose_query("Prudhvi vs Ajay for Python skills")
        assert len(result.sub_queries) >= 2
        entities = {sq.entity_scope for sq in result.sub_queries if sq.entity_scope}
        assert "Prudhvi" in entities
        assert "Ajay" in entities

    def test_multi_aspect_query(self):
        result = decompose_query("What are John's skills, experience, and education?")
        assert len(result.sub_queries) >= 2
        focuses = {sq.section_focus for sq in result.sub_queries if sq.section_focus}
        assert len(focuses) >= 2

    def test_entity_extraction_from_query(self):
        result = decompose_query("Tell me about Nandhagopal's certifications")
        assert len(result.sub_queries) >= 1
        assert any(sq.entity_scope == "Nandhagopal" for sq in result.sub_queries)

    def test_contact_query_sets_section_focus(self):
        result = decompose_query("What is Prudhvi's phone number?")
        assert any(
            sq.section_focus in ("identity_contact", "contact")
            for sq in result.sub_queries
        )

    def test_max_4_subqueries(self):
        result = decompose_query(
            "Compare skills, experience, education, certifications, "
            "projects, and achievements of all candidates"
        )
        assert len(result.sub_queries) <= 4
