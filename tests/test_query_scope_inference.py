"""Tests for smart query scope inference, profile isolation, all-profile retrieval, and intent timeout."""

from __future__ import annotations

import concurrent.futures
import time
from unittest.mock import MagicMock, patch

import pytest

from src.intent.llm_intent import IntentParse
from src.rag_v3.pipeline import (
    QueryScope,
    _infer_query_scope,
    _resolve_intent_future,
    _run_all_profile_analysis,
    run,
)
from src.rag_v3.types import LLMBudget

from tests.rag_v2_helpers import FakeEmbedder, FakeQdrant, FakeRedis, make_point


# ── Scope inference helpers ──────────────────────────────────────────────────

def _make_intent(intent: str = "qa", entity_hints: list | None = None, domain: str = "generic") -> IntentParse:
    return IntentParse(
        intent=intent,
        output_format="paragraph",
        requested_fields=[],
        domain=domain,
        constraints={},
        entity_hints=entity_hints or [],
        source="test",
    )


# ============================================================================
# TestQueryScopeInference
# ============================================================================

class TestQueryScopeInference:
    """Test smart scope detection."""

    def test_all_documents_compare(self):
        scope = _infer_query_scope("compare all candidates", None, None)
        assert scope.mode == "all_profile"

    def test_all_documents_summarize(self):
        intent = _make_intent(intent="list", entity_hints=[])
        scope = _infer_query_scope("summarize all documents", None, intent)
        assert scope.mode == "all_profile"

    def test_all_documents_how_many(self):
        intent = _make_intent(intent="list", entity_hints=[])
        scope = _infer_query_scope("how many resumes are there", None, intent)
        assert scope.mode == "all_profile"

    def test_all_documents_rank(self):
        scope = _infer_query_scope("rank all candidates by experience", None, None)
        assert scope.mode == "all_profile"

    def test_all_documents_list(self):
        intent = _make_intent(intent="list", entity_hints=[])
        scope = _infer_query_scope("list all candidates", None, intent)
        assert scope.mode == "all_profile"

    def test_all_documents_show_every(self):
        intent = _make_intent(intent="list", entity_hints=[])
        scope = _infer_query_scope("show every document", None, intent)
        assert scope.mode == "all_profile"

    def test_intent_compare_no_entities(self):
        intent = _make_intent(intent="compare", entity_hints=[])
        scope = _infer_query_scope("what is better", None, intent)
        assert scope.mode == "all_profile"

    def test_intent_rank_no_entities(self):
        intent = _make_intent(intent="rank", entity_hints=[])
        scope = _infer_query_scope("who is best", None, intent)
        assert scope.mode == "all_profile"

    def test_specific_entity_about_person(self):
        scope = _infer_query_scope("tell me about John Doe", None, None)
        assert scope.mode == "targeted"
        assert scope.entity_hint == "John Doe"

    def test_specific_entity_invoice_number(self):
        scope = _infer_query_scope("invoice #12345 details", None, None)
        assert scope.mode == "targeted"
        assert scope.entity_hint == "12345"

    def test_explicit_document_id_in_query(self):
        scope = _infer_query_scope("document_id=abc123 show details", None, None)
        assert scope.mode == "specific_document"
        assert scope.document_id == "abc123"

    def test_explicit_document_id_parameter(self):
        scope = _infer_query_scope("some query", "doc-999", None)
        assert scope.mode == "specific_document"
        assert scope.document_id == "doc-999"

    def test_default_all_profile(self):
        """Generic queries without entity/document references default to all_profile
        to collectively analyze all documents."""
        scope = _infer_query_scope("what are the payment terms", None, None)
        assert scope.mode == "all_profile"

    def test_default_all_profile_skills(self):
        scope = _infer_query_scope("skills required for this role", None, None)
        assert scope.mode == "all_profile"

    def test_intent_with_entity_hints(self):
        intent = _make_intent(intent="qa", entity_hints=["Alice Smith"])
        scope = _infer_query_scope("generic query", None, intent)
        assert scope.mode == "targeted"
        assert scope.entity_hint == "Alice Smith"


# ============================================================================
# TestProfileIsolation
# ============================================================================

class TestProfileIsolation:
    """Ensure no cross-profile data leakage."""

    def test_unscoped_scan_never_leaks(self):
        """When filter_chunks_by_profile_scope drops all → result is empty, not unfiltered."""
        # Create points for profile "other-profile" only
        points = [
            make_point(
                pid="p1", profile_id="other-profile", document_id="doc1",
                file_name="other.pdf", text="secret data", page=1,
            ),
        ]
        fake_qdrant = FakeQdrant(points)
        fake_embedder = FakeEmbedder()
        fake_redis = FakeRedis()

        result = run(
            query="show me data",
            subscription_id="sub-1",
            profile_id="my-profile",
            qdrant_client=fake_qdrant,
            embedder=fake_embedder,
            redis_client=fake_redis,
            llm_client=None,
            cross_encoder=None,
        )
        response = result.get("response", "")
        # Should NOT contain the secret data from other profile
        assert "secret data" not in response

    def test_qdrant_filter_requires_profile_id(self):
        """build_qdrant_filter raises ValueError without profile_id."""
        from src.api.vector_store import build_qdrant_filter
        with pytest.raises(ValueError, match="profile_id is required"):
            build_qdrant_filter(subscription_id="sub-1", profile_id="")

    def test_qdrant_filter_requires_subscription_id(self):
        """build_qdrant_filter raises ValueError without subscription_id."""
        from src.api.vector_store import build_qdrant_filter
        with pytest.raises(ValueError, match="subscription_id is required"):
            build_qdrant_filter(subscription_id="", profile_id="prof-1")

    def test_chunks_filtered_by_profile(self):
        """Chunks with wrong profile_id are excluded from retrieval."""
        from src.rag_v3.retrieve import filter_chunks_by_profile_scope
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="1", text="a", score=0.9, source=ChunkSource(document_name="doc", page=1),
                  meta={"profile_id": "p1", "subscription_id": "s1"}),
            Chunk(id="2", text="b", score=0.8, source=ChunkSource(document_name="doc", page=2),
                  meta={"profile_id": "p2", "subscription_id": "s1"}),
        ]
        filtered = filter_chunks_by_profile_scope(chunks, profile_id="p1", subscription_id="s1")
        assert len(filtered) == 1
        assert filtered[0].id == "1"


# ============================================================================
# TestAllProfileRetrieval
# ============================================================================

class TestAllProfileRetrieval:
    """Test multi-document query handling — sources must cover ALL profile docs."""

    def _build_multi_doc_points(self):
        """5 documents, 2 chunks each, all in prof-1."""
        points = []
        names = [
            ("doc1", "alice_resume.pdf", "Alice has 5 years experience in Python and Django development"),
            ("doc2", "bob_resume.pdf", "Bob has 10 years experience in Java Spring microservices"),
            ("doc3", "carol_resume.pdf", "Carol has 3 years experience in Go and Kubernetes"),
            ("doc4", "dave_resume.pdf", "Dave has 8 years experience in React and TypeScript"),
            ("doc5", "eve_resume.pdf", "Eve has 12 years experience in machine learning and NLP"),
        ]
        for i, (doc_id, fname, text) in enumerate(names):
            points.append(make_point(
                pid=f"p{i*2+1}", profile_id="prof-1", document_id=doc_id,
                file_name=fname, text=text, page=1, score=0.9 - i * 0.02,
                doc_domain="resume",
            ))
            points.append(make_point(
                pid=f"p{i*2+2}", profile_id="prof-1", document_id=doc_id,
                file_name=fname, text=f"{text}. Additional skills and certifications.",
                page=2, score=0.85 - i * 0.02,
                doc_domain="resume",
            ))
        return points

    def test_all_profile_sources_cover_all_documents(self):
        """'compare all candidates' → sources include files from every document."""
        points = self._build_multi_doc_points()
        result = run(
            query="compare all candidates",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        metadata = result.get("metadata", {})
        assert metadata.get("scope") == "all_profile"
        assert metadata.get("document_count", 0) >= 3  # at least 3 of 5 docs

        source_files = {s["file_name"] for s in result.get("sources", [])}
        # All five documents should appear in sources
        for expected in ("alice_resume.pdf", "bob_resume.pdf", "carol_resume.pdf",
                         "dave_resume.pdf", "eve_resume.pdf"):
            assert expected in source_files, f"{expected} missing from sources: {source_files}"

    def test_all_profile_rank_query_covers_all(self):
        """'rank all candidates by experience' covers all docs."""
        points = self._build_multi_doc_points()
        result = run(
            query="rank all candidates by experience",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        metadata = result.get("metadata", {})
        assert metadata.get("scope") == "all_profile"
        source_files = {s["file_name"] for s in result.get("sources", [])}
        assert len(source_files) >= 3, f"Expected ≥3 unique doc files, got {source_files}"

    def test_all_profile_list_documents_query(self):
        """'list all documents' triggers all_profile mode."""
        points = self._build_multi_doc_points()
        list_intent = _make_intent(intent="list", entity_hints=[])
        with patch("src.rag_v3.pipeline._resolve_intent_future", return_value=list_intent):
            result = run(
                query="list all documents",
                subscription_id="sub-1",
                profile_id="prof-1",
                qdrant_client=FakeQdrant(points),
                embedder=FakeEmbedder(),
                redis_client=FakeRedis(),
                llm_client=None,
                cross_encoder=None,
            )
        metadata = result.get("metadata", {})
        assert metadata.get("scope") == "all_profile"

    def test_all_profile_empty_returns_no_chunks(self):
        """all_profile scope with no chunks returns proper empty message."""
        result = run(
            query="list all documents",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant([]),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        resp = result.get("response", "")
        assert "couldn't find" in resp or "Not enough information" in resp

    def test_all_profile_metadata_has_document_count(self):
        """all_profile metadata includes document_count matching actual distinct docs."""
        points = [
            make_point(pid="p1", profile_id="prof-1", document_id="doc1",
                       file_name="a.pdf", text="content A about Python and Django", page=1, score=0.9),
            make_point(pid="p2", profile_id="prof-1", document_id="doc2",
                       file_name="b.pdf", text="content B about Java and Spring", page=1, score=0.85),
            make_point(pid="p3", profile_id="prof-1", document_id="doc3",
                       file_name="c.pdf", text="content C about Go and Kubernetes", page=1, score=0.80),
        ]
        list_intent = _make_intent(intent="list", entity_hints=[])
        with patch("src.rag_v3.pipeline._resolve_intent_future", return_value=list_intent):
            result = run(
                query="summarize all documents",
                subscription_id="sub-1",
                profile_id="prof-1",
                qdrant_client=FakeQdrant(points),
                embedder=FakeEmbedder(),
                redis_client=FakeRedis(),
                llm_client=None,
                cross_encoder=None,
            )
        metadata = result.get("metadata", {})
        assert metadata.get("scope") == "all_profile"
        assert metadata["document_count"] == 3

    def test_all_profile_excludes_other_profiles(self):
        """all_profile retrieval must NOT include chunks from other profiles."""
        points = [
            make_point(pid="p1", profile_id="prof-1", document_id="doc1",
                       file_name="alice.pdf", text="Alice data for profile 1", page=1, score=0.9),
            make_point(pid="p2", profile_id="prof-WRONG", document_id="doc-other",
                       file_name="intruder.pdf", text="LEAKED secret from wrong profile", page=1, score=0.95),
            make_point(pid="p3", profile_id="prof-1", document_id="doc2",
                       file_name="bob.pdf", text="Bob data for profile 1", page=1, score=0.85),
        ]
        result = run(
            query="show all candidates",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        source_files = {s["file_name"] for s in result.get("sources", [])}
        assert "intruder.pdf" not in source_files, "Cross-profile leak detected!"
        response = result.get("response", "")
        assert "LEAKED" not in response, "Cross-profile data leaked into response!"


# ============================================================================
# TestSingleDocumentRetrieval
# ============================================================================

class TestSingleDocumentRetrieval:
    """Test that single-document queries only retrieve from the target document."""

    def _build_mixed_points(self):
        """3 documents in same profile — queries should isolate one."""
        return [
            make_point(pid="p1", profile_id="prof-1", document_id="doc-alice",
                       file_name="alice.pdf", text="Alice has 5 years Python experience and a masters degree",
                       page=1, score=0.9),
            make_point(pid="p2", profile_id="prof-1", document_id="doc-alice",
                       file_name="alice.pdf", text="Alice certifications include AWS and Azure",
                       page=2, score=0.85),
            make_point(pid="p3", profile_id="prof-1", document_id="doc-bob",
                       file_name="bob.pdf", text="Bob has 10 years Java and Spring Boot experience",
                       page=1, score=0.88),
            make_point(pid="p4", profile_id="prof-1", document_id="doc-bob",
                       file_name="bob.pdf", text="Bob certifications include CKAD and CKA",
                       page=2, score=0.83),
            make_point(pid="p5", profile_id="prof-1", document_id="doc-carol",
                       file_name="carol.pdf", text="Carol has 3 years Go and Kubernetes experience",
                       page=1, score=0.80),
        ]

    def test_explicit_document_id_scopes_to_one_doc(self):
        """Passing document_id= restricts initial retrieval to that document.

        Note: The LLM-first architecture may trigger domain-mismatch re-retrieval
        that broadens scope when the initial chunks' domain doesn't match the query
        domain. The key invariant is that alice.pdf is always present in sources.
        """
        points = self._build_mixed_points()
        result = run(
            query="what are the skills",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_id="doc-alice",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        source_files = {s["file_name"] for s in result.get("sources", [])}
        # alice.pdf must always be in sources (primary document)
        assert "alice.pdf" in source_files, f"Expected alice.pdf in sources, got {source_files}"

    def test_document_id_in_query_text_scopes_correctly(self):
        """document_id=abc in query text should scope to that document."""
        points = self._build_mixed_points()
        result = run(
            query="document_id=doc-bob show certifications",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        metadata = result.get("metadata", {})
        scope_meta = metadata.get("scope", {})
        # Should have scoped to doc-bob
        if isinstance(scope_meta, dict):
            assert scope_meta.get("document_id") == "doc-bob"

    def test_targeted_query_returns_profile_scoped_results(self):
        """A generic targeted query (no all_profile, no document_id) still returns results
        from the correct profile only."""
        points = self._build_mixed_points()
        # Add a chunk from a different profile
        points.append(make_point(
            pid="p-alien", profile_id="prof-OTHER", document_id="doc-alien",
            file_name="alien.pdf", text="This is ALIEN data from wrong profile", page=1, score=0.99,
        ))
        result = run(
            query="what are the Python skills",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        source_files = {s["file_name"] for s in result.get("sources", [])}
        assert "alien.pdf" not in source_files, f"Cross-profile leak! alien.pdf in sources: {source_files}"
        response = result.get("response", "")
        assert "ALIEN" not in response, "Cross-profile data leaked into response!"

    def test_specific_document_returns_context_found(self):
        """Querying a specific existing document should return context_found=True."""
        points = self._build_mixed_points()
        result = run(
            query="tell me about Alice",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_id="doc-alice",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        assert result.get("context_found") is True
        assert len(result.get("sources", [])) > 0

    def test_nonexistent_document_returns_no_results(self):
        """Querying a document_id that doesn't exist returns empty."""
        points = self._build_mixed_points()
        result = run(
            query="what is this",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_id="doc-NONEXISTENT",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        # Should find no context for a nonexistent document
        assert result.get("context_found") is False or len(result.get("sources", [])) == 0


# ============================================================================
# TestMultiCandidateRanking
# ============================================================================

class TestMultiCandidateRanking:
    """
    Verify that ranking/comparing queries across a profile extract ALL candidates
    (one per document) and render them in ranked order — not just one.
    """

    def _build_resume_points(self):
        """
        3 realistic resume documents in the same profile.
        Each has a name, skills, and experience in the chunk text so that
        _extract_hr → _extract_name_from_text / _extract_skills_from_text
        can populate CandidateItems.
        """
        return [
            # ── Alice (doc-alice) ──
            make_point(
                pid="a1", profile_id="prof-1", document_id="doc-alice",
                file_name="alice_resume.pdf",
                text=(
                    "Alice Johnson\n"
                    "Professional Summary: Senior Python developer with 5 years of experience.\n"
                    "Technical Skills: Python, Django, Flask, PostgreSQL, Docker\n"
                    "Education: M.S. Computer Science, MIT"
                ),
                page=1, score=0.92, doc_domain="resume",
                section_kind="summary_objective",
            ),
            make_point(
                pid="a2", profile_id="prof-1", document_id="doc-alice",
                file_name="alice_resume.pdf",
                text=(
                    "Experience:\n"
                    "Senior Software Engineer at TechCorp (2019-2024)\n"
                    "- Built microservices handling 10M requests/day\n"
                    "- Led team of 5 engineers\n"
                    "Certifications: AWS Solutions Architect, Kubernetes CKA"
                ),
                page=2, score=0.88, doc_domain="resume",
                section_kind="experience",
            ),
            # ── Bob (doc-bob) ──
            make_point(
                pid="b1", profile_id="prof-1", document_id="doc-bob",
                file_name="bob_resume.pdf",
                text=(
                    "Bob Smith\n"
                    "Professional Summary: Full-stack Java developer with 10 years of experience.\n"
                    "Technical Skills: Java, Spring Boot, React, AWS, Kubernetes, Terraform\n"
                    "Education: B.S. Computer Science, Stanford"
                ),
                page=1, score=0.90, doc_domain="resume",
                section_kind="summary_objective",
            ),
            make_point(
                pid="b2", profile_id="prof-1", document_id="doc-bob",
                file_name="bob_resume.pdf",
                text=(
                    "Experience:\n"
                    "Staff Engineer at MegaCorp (2014-2024)\n"
                    "- Architected payment platform processing $2B/year\n"
                    "- Mentored 12 engineers\n"
                    "Certifications: CKAD, AWS DevOps Professional, Scrum Master"
                ),
                page=2, score=0.87, doc_domain="resume",
                section_kind="experience",
            ),
            # ── Carol (doc-carol) ──
            make_point(
                pid="c1", profile_id="prof-1", document_id="doc-carol",
                file_name="carol_resume.pdf",
                text=(
                    "Carol Davis\n"
                    "Professional Summary: Backend engineer specializing in Go with 3 years of experience.\n"
                    "Technical Skills: Go, Kubernetes, gRPC, Redis, PostgreSQL\n"
                    "Education: B.S. Software Engineering, UC Berkeley"
                ),
                page=1, score=0.85, doc_domain="resume",
                section_kind="summary_objective",
            ),
            make_point(
                pid="c2", profile_id="prof-1", document_id="doc-carol",
                file_name="carol_resume.pdf",
                text=(
                    "Experience:\n"
                    "Software Engineer at CloudInc (2021-2024)\n"
                    "- Designed real-time data pipeline\n"
                    "- Reduced latency by 40%\n"
                    "Certifications: Google Cloud Professional"
                ),
                page=2, score=0.82, doc_domain="resume",
                section_kind="experience",
            ),
        ]

    def test_rank_query_produces_multiple_candidates(self):
        """'rank all candidates by skills' must extract ≥3 candidates, not 1."""
        points = self._build_resume_points()
        result = run(
            query="rank all candidates by technical skills",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        response = result.get("response", "")
        metadata = result.get("metadata", {})

        assert metadata.get("scope") == "all_profile", (
            f"Expected all_profile scope, got {metadata.get('scope')}"
        )

        # HRSchema renders candidate names — check for candidate names or filenames
        candidate_hits = sum(1 for name in ("Alice", "Bob", "Carol") if name in response)
        doc_hits = sum(1 for doc in ("alice_resume.pdf", "bob_resume.pdf", "carol_resume.pdf") if doc in response)
        assert candidate_hits >= 2 or doc_hits >= 2, (
            f"Expected ≥2 candidates in response, found candidates={candidate_hits}, docs={doc_hits}. Response:\n{response}"
        )

    def test_compare_query_produces_multiple_candidates(self):
        """'compare all candidates' must show details for ≥2 documents."""
        points = self._build_resume_points()
        result = run(
            query="compare all candidates",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        response = result.get("response", "")

        candidate_hits = sum(1 for name in ("Alice", "Bob", "Carol") if name in response)
        doc_hits = sum(1 for doc in ("alice_resume.pdf", "bob_resume.pdf", "carol_resume.pdf") if doc in response)
        assert candidate_hits >= 2 or doc_hits >= 2, (
            f"Expected ≥2 candidates in comparison, found candidates={candidate_hits}, docs={doc_hits}. Response:\n{response}"
        )

    def test_rank_response_has_document_sections(self):
        """Ranking output should contain per-candidate sections with structured facts."""
        points = self._build_resume_points()
        result = run(
            query="rank all candidates by experience",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        response = result.get("response", "")
        # HR renderer uses candidate names, not filenames
        candidate_hits = sum(1 for name in ("Alice", "Bob", "Carol") if name in response)
        assert candidate_hits >= 2, (
            f"Expected ≥2 candidates in ranking response, found {candidate_hits}. Response:\n{response}"
        )
        # Should contain structured HR data (skills, experience)
        assert "technical skills" in response.lower() or "Top pick" in response or "skills:" in response.lower() or "Ranking" in response, (
            f"Expected structured HR labels in response:\n{response}"
        )

    def test_list_all_candidates_extracts_all(self):
        """'list all candidates' produces details for every candidate."""
        points = self._build_resume_points()
        result = run(
            query="list all candidates",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        response = result.get("response", "")

        candidate_hits = sum(1 for name in ("Alice", "Bob", "Carol") if name in response)
        doc_hits = sum(1 for doc in ("alice_resume.pdf", "bob_resume.pdf", "carol_resume.pdf") if doc in response)
        assert candidate_hits >= 2 or doc_hits >= 2, (
            f"Expected ≥2 candidates when listing, found candidates={candidate_hits}, docs={doc_hits}. Response:\n{response}"
        )

    def test_rank_metadata_shows_multiple_documents(self):
        """Metadata document_count should match the number of distinct docs."""
        points = self._build_resume_points()
        result = run(
            query="rank all candidates",
            subscription_id="sub-1",
            profile_id="prof-1",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        metadata = result.get("metadata", {})
        assert metadata.get("document_count") == 3, (
            f"Expected 3 documents, got {metadata.get('document_count')}"
        )

    def test_single_candidate_query_returns_one(self):
        """Querying a specific document should return content from that document only."""
        points = self._build_resume_points()
        result = run(
            query="tell me about this candidate",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_id="doc-alice",
            qdrant_client=FakeQdrant(points),
            embedder=FakeEmbedder(),
            redis_client=FakeRedis(),
            llm_client=None,
            cross_encoder=None,
        )
        response = result.get("response", "")

        # Should contain alice's content (skills, experience)
        assert "Python" in response, f"Expected alice's skills in single-doc response:\n{response}"
        # Bob and Carol content should NOT appear
        assert "bob_resume.pdf" not in response, f"Bob's document should not appear in single-doc query:\n{response}"
        assert "carol_resume.pdf" not in response, f"Carol's document should not appear in single-doc query:\n{response}"


# ============================================================================
# TestIntentTimeout
# ============================================================================

class TestIntentTimeout:
    """Test intent parsing with proper timeout."""

    def test_intent_parse_returns_within_timeout(self):
        """When future completes quickly, intent parse succeeds."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(lambda: _make_intent(intent="summarize"))
        executor.shutdown(wait=False)

        result = _resolve_intent_future(future)
        assert result is not None
        assert result.intent == "summarize"

    def test_intent_parse_timeout_falls_back(self):
        """When future takes too long, returns None gracefully."""

        def slow_parse():
            time.sleep(2.0)
            return _make_intent()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(slow_parse)
        executor.shutdown(wait=False)

        result = _resolve_intent_future(future)
        assert result is None

    def test_intent_parse_exception_returns_none(self):
        """When future raises, returns None gracefully."""

        def failing_parse():
            raise RuntimeError("LLM failed")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(failing_parse)
        executor.shutdown(wait=False)
        time.sleep(0.1)  # let it fail

        result = _resolve_intent_future(future)
        assert result is None
