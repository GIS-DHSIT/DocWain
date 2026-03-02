"""Tests for corpus analytics engine (Task 3)."""
from __future__ import annotations

import pytest
from src.rag_v3.corpus_analytics import (
    CorpusStats,
    answer_analytics_query,
    compute_corpus_stats,
    is_analytics_query,
)
from src.rag_v3.document_context import DocumentContext
from src.rag_v3.types import Chunk, ChunkSource


def _make_context(
    doc_id: str,
    doc_name: str,
    doc_domain: str = "resume",
    fields: dict | None = None,
    chunk_texts: list | None = None,
) -> DocumentContext:
    chunks = []
    for i, text in enumerate(chunk_texts or []):
        chunks.append(Chunk(
            id=f"{doc_id}-c{i}",
            text=text,
            score=0.9,
            source=ChunkSource(document_name=doc_name, page=1),
            meta={"document_id": doc_id, "doc_domain": doc_domain},
        ))
    return DocumentContext(
        document_id=doc_id,
        document_name=doc_name,
        doc_domain=doc_domain,
        chunks=chunks,
        fields=fields or {},
        section_kinds=[],
    )


class TestIsAnalyticsQuery:
    def test_is_analytics_query_how_many(self):
        assert is_analytics_query("How many resumes are there?") is True

    def test_is_analytics_query_total_amount(self):
        assert is_analytics_query("What is the total amount across all invoices?") is True

    def test_is_analytics_query_normal(self):
        assert is_analytics_query("What are Abinaya's skills?") is False

    def test_is_analytics_combined_patterns(self):
        assert is_analytics_query("How many invoices are there and what are the total amounts?") is True


class TestComputeCorpusStats:
    def test_compute_stats_counts(self):
        contexts = [
            _make_context("d1", "Resume1.pdf", "resume"),
            _make_context("d2", "Resume2.pdf", "resume"),
            _make_context("d3", "Invoice1.pdf", "invoice"),
            _make_context("d4", "Invoice2.pdf", "invoice"),
            _make_context("d5", "Invoice3.pdf", "invoice"),
        ]
        stats = compute_corpus_stats(contexts)
        assert stats.total_documents == 5
        assert stats.documents_by_domain["resume"] == 2
        assert stats.documents_by_domain["invoice"] == 3

    def test_compute_stats_numeric_aggregation(self):
        contexts = [
            _make_context("d1", "R1.pdf", "resume", chunk_texts=["Total Experience: 5 years"]),
            _make_context("d2", "R2.pdf", "resume", chunk_texts=["Total Experience: 8 years"]),
        ]
        stats = compute_corpus_stats(contexts)
        assert "Total Experience" in stats.numeric_aggregations
        agg = stats.numeric_aggregations["Total Experience"]
        assert agg["sum"] == 13.0
        assert agg["avg"] == 6.5
        assert agg["min"] == 5.0
        assert agg["max"] == 8.0


class TestAnswerAnalyticsQuery:
    def test_answer_count_query(self):
        contexts = [
            _make_context("d1", "Resume1.pdf", "resume"),
            _make_context("d2", "Resume2.pdf", "resume"),
        ]
        stats = compute_corpus_stats(contexts)
        answer = answer_analytics_query("How many resumes?", stats, contexts)
        assert "2" in answer
        assert "resume" in answer.lower()

    def test_answer_total_query(self):
        contexts = [
            _make_context("d1", "Inv1.pdf", "invoice", chunk_texts=["Amount: $1500.00"]),
            _make_context("d2", "Inv2.pdf", "invoice", chunk_texts=["Amount: $2500.00"]),
        ]
        stats = compute_corpus_stats(contexts)
        answer = answer_analytics_query("What is the total amount?", stats, contexts)
        assert "4000" in answer

    def test_answer_list_query(self):
        contexts = [
            _make_context("d1", "Resume_Alice.pdf", "resume"),
            _make_context("d2", "Resume_Bob.pdf", "resume"),
        ]
        stats = compute_corpus_stats(contexts)
        answer = answer_analytics_query("List all candidates", stats, contexts)
        assert "Resume_Alice.pdf" in answer
        assert "Resume_Bob.pdf" in answer

    def test_answer_analytics_empty_corpus(self):
        stats = CorpusStats()
        answer = answer_analytics_query("How many resumes?", stats, [])
        assert "No documents found" in answer
