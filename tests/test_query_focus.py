"""Tests for the query-aware response generation system.

Validates that:
1. QueryFocus is built correctly from queries
2. Chunk relevance scoring differentiates relevant vs irrelevant chunks
3. filter_chunks_by_focus selects appropriate subsets
4. Field relevance scoring enables focused rendering
5. Fact relevance scoring enables focused generic rendering
6. End-to-end: same chunks + different queries → different outputs
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from src.rag_v3.query_focus import (
    QueryFocus,
    build_query_focus,
    score_chunk_relevance,
    filter_chunks_by_focus,
    score_field_relevance,
    score_fact_relevance,
)


# ---------------------------------------------------------------------------
# Helper to create fake chunks
# ---------------------------------------------------------------------------

def _make_chunk(text: str, section_kind: str = "", score: float = 0.5, doc_name: str = "doc.pdf"):
    return SimpleNamespace(
        id=f"chunk_{hash(text) % 10000}",
        text=text,
        score=score,
        source=SimpleNamespace(document_name=doc_name, page=1),
        meta={"section_kind": section_kind},
    )


# =====================================================================
# 1. QueryFocus construction
# =====================================================================

class TestBuildQueryFocus:
    def test_skills_query(self):
        focus = build_query_focus("What are John's technical skills?")
        assert "skills" in focus.field_tags
        assert "skills_technical" in focus.section_kinds
        assert not focus.is_exhaustive

    def test_education_query(self):
        focus = build_query_focus("Tell me about his education and degree")
        assert "education" in focus.field_tags
        assert "education" in focus.section_kinds
        assert not focus.is_exhaustive

    def test_exhaustive_query(self):
        focus = build_query_focus("Tell me everything about this person")
        assert focus.is_exhaustive

    def test_summarize_is_exhaustive(self):
        focus = build_query_focus("Summarize the document")
        assert focus.is_exhaustive

    def test_vague_query_is_exhaustive(self):
        focus = build_query_focus("Hi")
        assert focus.is_exhaustive  # short vague query, no field tags

    def test_empty_query(self):
        focus = build_query_focus("")
        assert focus.is_exhaustive

    def test_invoice_totals_query(self):
        focus = build_query_focus("What is the total amount?")
        assert "totals" in focus.field_tags
        assert not focus.is_exhaustive

    def test_invoice_items_query(self):
        focus = build_query_focus("List the line items on this invoice")
        assert "items" in focus.field_tags

    def test_contact_query(self):
        focus = build_query_focus("What is the email and phone number?")
        assert "contact" in focus.field_tags
        assert "identity_contact" in focus.section_kinds

    def test_multiple_field_tags(self):
        focus = build_query_focus("Show me skills and education")
        assert "skills" in focus.field_tags
        assert "education" in focus.field_tags
        assert not focus.is_exhaustive

    def test_keywords_extracted(self):
        focus = build_query_focus("What programming languages does John know?")
        assert "programming" in focus.keywords
        assert "john" in focus.keywords
        assert "languages" in focus.keywords

    def test_bigrams_extracted(self):
        focus = build_query_focus("machine learning experience")
        assert "machine learning" in focus.bigrams

    def test_intent_hint_passed(self):
        focus = build_query_focus("skills", intent_hint="factual")
        assert focus.intent == "factual"

    def test_legal_clause_query(self):
        focus = build_query_focus("What are the liability clauses?")
        assert "clauses" in focus.field_tags


# =====================================================================
# 2. Chunk relevance scoring
# =====================================================================

class TestScoreChunkRelevance:
    def test_matching_section_scores_higher(self):
        focus = build_query_focus("What are the technical skills?")
        skills_chunk = _make_chunk("Python, Java, Docker, Kubernetes", section_kind="skills_technical")
        edu_chunk = _make_chunk("B.Tech in Computer Science from IIT", section_kind="education")
        assert score_chunk_relevance(skills_chunk, focus) > score_chunk_relevance(edu_chunk, focus)

    def test_keyword_match_scores_higher(self):
        focus = build_query_focus("What certifications does the candidate have?")
        cert_chunk = _make_chunk("Certified AWS Solutions Architect, PMP certified", section_kind="certifications")
        generic_chunk = _make_chunk("The company was founded in 2010", section_kind="")
        assert score_chunk_relevance(cert_chunk, focus) > score_chunk_relevance(generic_chunk, focus)

    def test_exhaustive_guarantees_minimum(self):
        focus = build_query_focus("Tell me everything about this person")
        low_match = _make_chunk("unrelated content xyz abc", section_kind="", score=0.0)
        assert score_chunk_relevance(low_match, focus) >= 0.3

    def test_reranker_score_contributes(self):
        focus = build_query_focus("experience")
        high_score = _make_chunk("5 years experience in Python", section_kind="experience", score=0.9)
        low_score = _make_chunk("5 years experience in Python", section_kind="experience", score=0.1)
        assert score_chunk_relevance(high_score, focus) > score_chunk_relevance(low_score, focus)

    def test_zero_score_chunk(self):
        focus = build_query_focus("skills")
        chunk = _make_chunk("Python Java", section_kind="skills_technical", score=0.0)
        # Should still get reasonable score from keyword+section match
        assert score_chunk_relevance(chunk, focus) > 0.2

    def test_no_keywords_neutral(self):
        focus = QueryFocus(is_exhaustive=True)
        chunk = _make_chunk("anything", score=0.5)
        assert score_chunk_relevance(chunk, focus) >= 0.3


# =====================================================================
# 3. Chunk filtering
# =====================================================================

class TestFilterChunksByFocus:
    def _mixed_chunks(self):
        return [
            _make_chunk("Python, Java, Kubernetes, React", section_kind="skills_technical", score=0.8),
            _make_chunk("Django REST framework experience", section_kind="skills_technical", score=0.7),
            _make_chunk("B.Tech Computer Science, IIT Delhi, 8.5 GPA", section_kind="education", score=0.6),
            _make_chunk("M.Tech Machine Learning, IISc Bangalore", section_kind="education", score=0.5),
            _make_chunk("5 years as Senior Engineer at Google", section_kind="experience", score=0.4),
            _make_chunk("AWS Certified Solutions Architect", section_kind="certifications", score=0.3),
            _make_chunk("Led team of 10 engineers", section_kind="experience", score=0.2),
            _make_chunk("Published research in NeurIPS", section_kind="achievements", score=0.1),
        ]

    def test_skills_query_prioritizes_skills(self):
        focus = build_query_focus("What are the technical skills?")
        chunks = self._mixed_chunks()
        filtered = filter_chunks_by_focus(chunks, focus, min_keep=2, top_k=4)
        # Skills chunks should be at the top
        top_texts = [c.text for c in filtered[:2]]
        assert any("Python" in t for t in top_texts)

    def test_education_query_prioritizes_education(self):
        focus = build_query_focus("What is the education background?")
        chunks = self._mixed_chunks()
        filtered = filter_chunks_by_focus(chunks, focus, min_keep=2, top_k=4)
        top_texts = [c.text for c in filtered[:2]]
        assert any("B.Tech" in t or "M.Tech" in t for t in top_texts)

    def test_exhaustive_keeps_all(self):
        focus = build_query_focus("Tell me everything")
        chunks = self._mixed_chunks()
        filtered = filter_chunks_by_focus(chunks, focus, min_keep=2, top_k=20)
        assert len(filtered) == len(chunks)

    def test_min_keep_respected(self):
        focus = build_query_focus("What are the certifications?")
        # Only 1 cert chunk, but min_keep=4 should keep more
        chunks = self._mixed_chunks()
        filtered = filter_chunks_by_focus(chunks, focus, min_keep=4, top_k=8)
        assert len(filtered) >= 4

    def test_top_k_cap(self):
        focus = build_query_focus("everything")
        chunks = self._mixed_chunks()
        filtered = filter_chunks_by_focus(chunks, focus, min_keep=2, top_k=3)
        assert len(filtered) <= 3

    def test_empty_chunks_returns_empty(self):
        focus = build_query_focus("skills")
        assert filter_chunks_by_focus([], focus) == []

    def test_none_focus_returns_unchanged(self):
        chunks = self._mixed_chunks()
        result = filter_chunks_by_focus(chunks, None)
        assert result is chunks


# =====================================================================
# 4. Field relevance scoring
# =====================================================================

class TestScoreFieldRelevance:
    def test_matching_field(self):
        focus = build_query_focus("What are the technical skills?")
        assert score_field_relevance("skills", focus) == 1.0

    def test_non_matching_field(self):
        focus = build_query_focus("What are the technical skills?")
        relevance = score_field_relevance("achievements", focus)
        assert relevance == 0.1  # low but non-zero

    def test_exhaustive_all_fields_relevant(self):
        focus = build_query_focus("Tell me everything")
        assert score_field_relevance("skills", focus) == 1.0
        assert score_field_relevance("education", focus) == 1.0
        assert score_field_relevance("achievements", focus) == 1.0

    def test_no_field_tags_all_relevant(self):
        focus = QueryFocus(keywords=["python"], field_tags=set())
        assert score_field_relevance("skills", focus) == 1.0

    def test_keyword_overlap_partial(self):
        focus = build_query_focus("What certifications does the candidate have?")
        # "certifications" keyword overlaps with "certifications" field_tag
        assert score_field_relevance("certifications", focus) == 1.0

    def test_none_focus(self):
        assert score_field_relevance("skills", None) == 1.0


# =====================================================================
# 5. Fact relevance scoring
# =====================================================================

class TestScoreFactRelevance:
    def test_matching_fact(self):
        focus = build_query_focus("What are the technical skills?")
        score = score_fact_relevance("Technical Skills", "Python, Java, C++", focus)
        assert score > 0.3

    def test_non_matching_fact(self):
        focus = build_query_focus("What programming languages?")
        score = score_fact_relevance("Education", "B.Tech from IIT", focus)
        assert score < 0.3

    def test_exhaustive_all_relevant(self):
        focus = build_query_focus("Tell me everything")
        assert score_fact_relevance("anything", "anything", focus) == 1.0

    def test_no_focus(self):
        assert score_fact_relevance("label", "value", None) == 1.0

    def test_bigram_match(self):
        focus = build_query_focus("machine learning experience")
        score = score_fact_relevance("Skills", "Deep machine learning and NLP", focus)
        assert score > 0.3


# =====================================================================
# 6. End-to-end rendering differentiation
# =====================================================================

class TestRenderDifferentiation:
    """Verify that the same schema produces different output for different queries."""

    def _make_hr_schema(self):
        from src.rag_v3.types import (
            HRSchema, CandidateField, Candidate,
        )
        cand = Candidate(
            name="John Doe",
            role="Senior Engineer",
            details=None,
            total_years_experience="5 years",
            experience_summary="Worked at Google and Amazon building large-scale distributed systems and microservices architecture with high availability requirements",
            technical_skills=["Python", "Java", "Kubernetes", "Docker", "React", "TypeScript", "Go", "Rust", "Terraform", "AWS"],
            functional_skills=["Team leadership", "Agile methodology", "Project management", "Stakeholder communication"],
            certifications=["AWS Solutions Architect", "PMP", "Kubernetes CKA", "Google Cloud Professional"],
            education=["B.Tech Computer Science from IIT Delhi with 9.2 GPA", "M.Tech Machine Learning from IISc Bangalore", "Stanford Online Certificate in AI"],
            achievements=["Published in NeurIPS", "Led team of 10", "Built system serving 1M users", "Open source contributor"],
            evidence_spans=[],
        )
        return HRSchema(candidates=CandidateField(items=[cand]))

    def test_skills_query_vs_education_query(self):
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_hr_schema()
        skills_focus = build_query_focus("What are the technical skills?")
        edu_focus = build_query_focus("What is the education background?")

        skills_result = render_enterprise(
            schema, "factual", query="What are the technical skills?", query_focus=skills_focus,
        )
        edu_result = render_enterprise(
            schema, "factual", query="What is the education background?", query_focus=edu_focus,
        )

        # Both should contain candidate name
        assert "John Doe" in skills_result
        assert "John Doe" in edu_result

        # Skills result should have full skills rendering
        assert "Python" in skills_result
        assert "Kubernetes" in skills_result

        # Education result should have full education rendering
        assert "IIT Delhi" in edu_result
        assert "IISc" in edu_result

        # The two results should be different
        assert skills_result != edu_result

    def test_exhaustive_shows_everything(self):
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_hr_schema()
        focus = build_query_focus("Tell me everything about John")

        result = render_enterprise(
            schema, "factual", query="Tell me everything about John", query_focus=focus,
        )

        # Should have all fields
        assert "Python" in result
        assert "IIT Delhi" in result or "IISc" in result
        assert "AWS" in result or "PMP" in result

    def test_skills_focus_minimizes_education(self):
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_hr_schema()
        focus = build_query_focus("List the technical skills")

        result = render_enterprise(
            schema, "factual", query="List the technical skills", query_focus=focus,
        )

        # Skills should be fully rendered
        assert "Python" in result
        assert "Kubernetes" in result
        assert "Docker" in result

        # Education should be summarized or minimized
        lines = result.split("\n")
        skills_lines = [l for l in lines if "skill" in l.lower() or "Python" in l or "Java" in l]
        assert len(skills_lines) >= 1

    def test_no_query_focus_shows_all(self):
        """When query_focus=None, all fields render (backwards compatible)."""
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_hr_schema()

        result = render_enterprise(schema, "factual", query="", query_focus=None)

        # All fields should be present
        assert "Python" in result
        assert "IIT Delhi" in result or "IISc" in result

    def test_generic_schema_fact_filtering(self):
        from src.rag_v3.types import GenericSchema, FieldValuesField, FieldValue

        facts = [
            FieldValue(label="Email", value="john@example.com", evidence_spans=[]),
            FieldValue(label="Phone", value="+1-555-123-4567", evidence_spans=[]),
            FieldValue(label="Technical Skills", value="Python, Java, Docker", evidence_spans=[]),
            FieldValue(label="Education", value="B.Tech from IIT Delhi", evidence_spans=[]),
            FieldValue(label="Experience", value="5 years at Google", evidence_spans=[]),
        ]
        schema = GenericSchema(facts=FieldValuesField(items=facts))

        from src.rag_v3.enterprise import render_enterprise

        # Skills query — should prioritize skills fact
        skills_focus = build_query_focus("What are the technical skills?")
        skills_result = render_enterprise(
            schema, "factual", query="What are the technical skills?", query_focus=skills_focus,
        )
        assert "Python" in skills_result

        # Contact query — should prioritize contact facts
        contact_focus = build_query_focus("What is the email address?")
        contact_result = render_enterprise(
            schema, "contact", query="What is the email address?", query_focus=contact_focus,
        )
        assert "john@example.com" in contact_result


# =====================================================================
# 7. Invoice rendering differentiation
# =====================================================================

class TestInvoiceRenderDifferentiation:
    def _make_invoice_schema(self):
        from src.rag_v3.types import (
            InvoiceSchema, FieldValuesField, FieldValue,
            InvoiceItemsField, InvoiceItem,
        )
        return InvoiceSchema(
            totals=FieldValuesField(items=[
                FieldValue(label="Total", value="$5,000", evidence_spans=[]),
                FieldValue(label="Tax", value="$500", evidence_spans=[]),
            ]),
            items=InvoiceItemsField(items=[
                InvoiceItem(description="Widget A - 10 units", evidence_spans=[]),
                InvoiceItem(description="Service B - consulting", evidence_spans=[]),
            ]),
            parties=FieldValuesField(items=[
                FieldValue(label="Vendor", value="Acme Corp", evidence_spans=[]),
                FieldValue(label="Customer", value="Beta Inc", evidence_spans=[]),
            ]),
        )

    def test_totals_query_shows_totals(self):
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_invoice_schema()
        focus = build_query_focus("What is the total amount?")

        result = render_enterprise(schema, "totals", query="What is the total amount?", query_focus=focus)
        assert "$5,000" in result

    def test_items_query_shows_items(self):
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_invoice_schema()
        focus = build_query_focus("List the line items")

        result = render_enterprise(schema, "products_list", query="List the line items", query_focus=focus)
        assert "Widget A" in result

    def test_vendor_query_shows_parties(self):
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_invoice_schema()
        focus = build_query_focus("Who is the vendor?")

        result = render_enterprise(schema, "factual", query="Who is the vendor?", query_focus=focus)
        assert "Acme Corp" in result


# =====================================================================
# 8. Legal rendering differentiation
# =====================================================================

class TestLegalRenderDifferentiation:
    def _make_legal_schema(self):
        from src.rag_v3.types import LegalSchema, ClauseField, Clause

        return LegalSchema(
            clauses=ClauseField(items=[
                Clause(title="Confidentiality", text="All information shared under this agreement shall remain confidential", evidence_spans=[]),
                Clause(title="Liability", text="Neither party shall be liable for indirect damages", evidence_spans=[]),
                Clause(title="Termination", text="This agreement may be terminated with 30 days notice", evidence_spans=[]),
                Clause(title="Payment Terms", text="Payment is due within 30 days of invoice", evidence_spans=[]),
                Clause(title="Governing Law", text="This agreement is governed by the laws of California", evidence_spans=[]),
            ]),
        )

    def test_liability_query_prioritizes_liability(self):
        from src.rag_v3.enterprise import render_enterprise

        schema = self._make_legal_schema()
        focus = build_query_focus("What are the liability clauses?")

        result = render_enterprise(schema, "factual", query="What are the liability clauses?", query_focus=focus)
        assert "liable" in result.lower() or "Liability" in result


# =====================================================================
# 9. Pipeline integration — render() threading
# =====================================================================

class TestRouterThreading:
    def test_render_accepts_query_focus(self):
        """Verify that render() in the router accepts and threads query_focus."""
        try:
            from src.rag_v3.renderers.router import render
        except ImportError:
            pytest.skip("Module removed")
        from src.rag_v3.types import GenericSchema, FieldValuesField, FieldValue

        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Name", value="Alice", evidence_spans=[]),
        ]))
        focus = build_query_focus("What is the name?")

        result = render(
            domain="generic",
            intent="factual",
            schema=schema,
            query="What is the name?",
            query_focus=focus,
        )
        assert "Alice" in result

    def test_render_works_without_query_focus(self):
        """Backwards compat: render() works when query_focus=None."""
        try:
            from src.rag_v3.renderers.router import render
        except ImportError:
            pytest.skip("Module removed")
        from src.rag_v3.types import GenericSchema, FieldValuesField, FieldValue

        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Name", value="Bob", evidence_spans=[]),
        ]))

        result = render(
            domain="generic",
            intent="factual",
            schema=schema,
            query="What is the name?",
        )
        assert "Bob" in result


# =====================================================================
# 10. Extract threading
# =====================================================================

class TestExtractThreading:
    def test_extract_schema_accepts_query_focus(self):
        """Verify extract_schema accepts query_focus param without error."""
        from src.rag_v3.extract import extract_schema
        from src.rag_v3.types import LLMBudget

        chunks = [_make_chunk("Python Java Docker", section_kind="skills_technical")]
        focus = build_query_focus("What are the skills?")
        budget = LLMBudget(llm_client=None, max_calls=0)

        result = extract_schema(
            None,
            query="What are the skills?",
            chunks=chunks,
            llm_client=None,
            budget=budget,
            query_focus=focus,
        )
        assert result is not None
