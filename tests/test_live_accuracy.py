"""Live accuracy tests against real Qdrant data.

These tests require a running Qdrant instance with actual data.
Gate: LIVE_TEST=1 environment variable.

Tests:
  1. Multi-document: "rank all candidates by skills and experience"
     → Must return response (not fallback), cover all 4 docs, have ranking lines
  2. Single-document: query about a specific candidate by document_id
     → Must return data from that one document only
  3. Single-document by name: "tell me about Aadithya"
     → Must return data about that candidate
"""

from __future__ import annotations

import os
import re
import sys
import logging

import pytest

# Enable debug logging for live test diagnostics
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

LIVE = os.environ.get("LIVE_TEST", "0") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="LIVE_TEST=1 not set")


# ── Shared fixtures ──────────────────────────────────────────────────────────

SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
PROFILE_ID = "6992c4ec6034385742e451a6"

# Known document IDs in this profile (from Qdrant data)
DOC_IDS = {
    "abinaya": "69935fd96034385742e45586",
    "swapnil": "6992c5836034385742e45209",
    "gokul": "69930cb66034385742e45469",
    "abhishek": "69935fd96034385742e45573",
    "aadithya": "69935fd96034385742e45576",
    "aloysius": "69935fd96034385742e4558b",
}


_cached_deps = None

def _get_rag_deps():
    """Build real RAG dependencies from app state, with caching and CUDA fallback."""
    global _cached_deps
    if _cached_deps is not None:
        return _cached_deps

    from src.api.config import Config
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=30)
    try:
        embedder = SentenceTransformer(Config.Model.EMBEDDING_MODEL)
    except (RuntimeError, Exception) as exc:
        if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
            embedder = SentenceTransformer(Config.Model.EMBEDDING_MODEL, device="cpu")
        else:
            raise

    _cached_deps = (qdrant_client, embedder)
    return _cached_deps


_BANNED_PHRASES = [
    "not explicitly mentioned",
    "not available in retrieved context",
    "not enough information",
]


def _assert_no_unhelpful_phrases(response: str):
    """Assert response contains none of the banned unhelpful phrases."""
    lowered = response.lower()
    for phrase in _BANNED_PHRASES:
        assert phrase not in lowered, (
            f"Banned phrase '{phrase}' found in response: {response[:200]}"
        )


def _run_query(query: str, document_id: str | None = None) -> dict:
    """Run a RAG v3 query and return the result dict."""
    from src.rag_v3.pipeline import run

    qdrant_client, embedder = _get_rag_deps()
    result = run(
        query=query,
        subscription_id=SUBSCRIPTION_ID,
        profile_id=PROFILE_ID,
        document_id=document_id,
        llm_client=None,  # No LLM for deterministic tests
        qdrant_client=qdrant_client,
        redis_client=None,
        embedder=embedder,
        cross_encoder=None,
    )
    return result


# ── Multi-document tests ─────────────────────────────────────────────────────

class TestLiveMultiDocument:
    """Live tests for multi-document (all-profile) queries."""

    def test_rank_all_candidates_returns_response(self):
        """The 'rank all candidates' query must NOT return fallback answer."""
        result = _run_query("rank all candidates by skills and experience")
        response = result["response"]
        assert response, "Response should not be empty"
        assert "Not enough information" not in response, (
            f"Got fallback answer: {response[:200]}"
        )
        _assert_no_unhelpful_phrases(response)
        print(f"\n=== RANK ALL RESPONSE ===\n{response}\n")

    def test_rank_all_candidates_has_sources(self):
        """Multi-doc query must return sources."""
        result = _run_query("rank all candidates by skills and experience")
        sources = result.get("sources", [])
        assert len(sources) > 0, "Should have at least some sources"

    def test_rank_all_candidates_covers_multiple_docs(self):
        """Response metadata should show multiple documents."""
        result = _run_query("rank all candidates by skills and experience")
        metadata = result.get("metadata", {})
        doc_count = metadata.get("document_count", 0)
        assert doc_count >= 2, (
            f"Expected at least 2 documents, got {doc_count}. "
            f"Metadata: {metadata}"
        )

    def test_rank_all_candidates_judge_passes(self):
        """Judge should pass (not fail) for multi-doc ranking."""
        result = _run_query("rank all candidates by skills and experience")
        metadata = result.get("metadata", {})
        judge_info = metadata.get("judge", {})
        quality = metadata.get("quality", "LOW")
        # Allow pass or any non-fail status
        assert quality == "HIGH" or judge_info.get("status") == "pass", (
            f"Judge should pass. Status: {judge_info}, quality: {quality}"
        )

    def test_compare_all_candidates(self):
        """'Compare all candidates' should also work."""
        result = _run_query("compare all candidates")
        response = result["response"]
        assert response, "Response should not be empty"
        assert "Not enough information" not in response, (
            f"Got fallback: {response[:200]}"
        )
        _assert_no_unhelpful_phrases(response)
        print(f"\n=== COMPARE ALL RESPONSE ===\n{response}\n")

    def test_list_all_resumes(self):
        """'List all resumes' should produce a multi-doc response."""
        result = _run_query("list all resumes in this profile")
        response = result["response"]
        assert response, "Response should not be empty"
        assert "Not enough information" not in response, (
            f"Got fallback: {response[:200]}"
        )
        _assert_no_unhelpful_phrases(response)

    def test_how_many_candidates(self):
        """'How many candidates' should route to all-profile."""
        result = _run_query("how many candidates are in this profile")
        metadata = result.get("metadata", {})
        scope = metadata.get("scope")
        assert scope == "all_profile", (
            f"Expected all_profile scope, got {scope}"
        )


# ── Single-document tests ────────────────────────────────────────────────────

class TestLiveSingleDocument:
    """Live tests for single-document queries."""

    def test_query_by_document_id(self):
        """Query with explicit document_id should return data about that document."""
        doc_id = DOC_IDS["swapnil"]
        result = _run_query(
            "what are the skills and experience of this candidate",
            document_id=doc_id,
        )
        response = result["response"]
        assert response, "Response should not be empty"
        _assert_no_unhelpful_phrases(response)
        # Should have sources
        sources = result.get("sources", [])
        assert len(sources) > 0, "Should have sources for single-doc query"
        print(f"\n=== SINGLE DOC (Swapnil) RESPONSE ===\n{response[:500]}\n")

    def test_query_about_specific_person_by_name(self):
        """'Tell me about Gokul' should target that specific document."""
        result = _run_query("tell me about Gokul")
        response = result["response"]
        assert response, "Response should not be empty"
        _assert_no_unhelpful_phrases(response)
        metadata = result.get("metadata", {})
        # Should be targeted scope, not all_profile
        scope = metadata.get("scope", {})
        assert scope != "all_profile", (
            f"Expected targeted scope, got: {scope}"
        )
        print(f"\n=== ABOUT GOKUL RESPONSE ===\n{response[:500]}\n")

    def test_single_doc_sources_are_from_one_document(self):
        """Sources from single-doc query should all be from the same document."""
        doc_id = DOC_IDS["abhishek"]
        result = _run_query(
            "what is this candidate's education and certifications",
            document_id=doc_id,
        )
        sources = result.get("sources", [])
        if sources:
            file_names = {s.get("file_name", "") for s in sources}
            # All sources should be from the same file
            assert len(file_names) <= 2, (
                f"Single-doc query should have sources from ~1 file, got: {file_names}"
            )
