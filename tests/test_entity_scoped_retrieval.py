"""Tests for entity-scoped and section-filtered retrieval."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


class TestEntityScopedRetrieval:

    def test_retrieve_entity_scoped_exists(self):
        from src.rag_v3.retrieve import retrieve_entity_scoped
        import inspect
        sig = inspect.signature(retrieve_entity_scoped)
        assert "entity_name" in sig.parameters
        assert "query" in sig.parameters
        assert "collection" in sig.parameters

    def test_retrieve_entity_scoped_returns_chunks(self):
        from src.rag_v3.retrieve import retrieve_entity_scoped
        from src.rag_v3.types import Chunk

        mock_qdrant = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.85
        mock_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "source_name": "prudhvi_resume.pdf",
            "canonical_text": "Prudhvi has 8 years of cloud experience in AWS and Azure.",
        }
        # scroll returns entity-matching points
        mock_qdrant.scroll.return_value = ([mock_point], None)
        # query_points returns search results
        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_entity_scoped(
            query="cloud experience",
            entity_name="Prudhvi",
            collection="test_col",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        # Should have called scroll to find entity docs
        assert mock_qdrant.scroll.called
        # Should have called query_points for vector search
        assert mock_qdrant.query_points.called

    def test_entity_scoped_fallback_when_no_entity_match(self):
        """When no documents match the entity, falls back to unscoped profile search."""
        from src.rag_v3.retrieve import retrieve_entity_scoped
        from src.rag_v3.types import Chunk

        mock_qdrant = MagicMock()
        # Scroll returns points that do NOT match the entity
        unrelated_point = MagicMock()
        unrelated_point.id = "p2"
        unrelated_point.score = 0.0
        unrelated_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc2",
            "source_name": "invoice_123.pdf",
            "canonical_text": "Invoice for office supplies totaling $500.",
        }
        mock_qdrant.scroll.return_value = ([unrelated_point], None)

        # query_points still returns results (unscoped fallback)
        result_point = MagicMock()
        result_point.id = "p3"
        result_point.score = 0.70
        result_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc3",
            "source_name": "general_doc.pdf",
            "canonical_text": "General information about cloud computing.",
        }
        mock_qdrant.query_points.return_value = MagicMock(points=[result_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_entity_scoped(
            query="cloud experience",
            entity_name="NonExistentPerson",
            collection="test_col",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        # Should still return chunks from the fallback unscoped search
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_entity_scoped_filters_empty_text_chunks(self):
        """Chunks with empty text after conversion are filtered out."""
        from src.rag_v3.retrieve import retrieve_entity_scoped

        mock_qdrant = MagicMock()
        # Entity match on scroll
        scroll_point = MagicMock()
        scroll_point.id = "p1"
        scroll_point.score = 0.0
        scroll_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "source_name": "alice_resume.pdf",
            "canonical_text": "Alice is a software engineer.",
        }
        mock_qdrant.scroll.return_value = ([scroll_point], None)

        # Search returns a point with empty canonical_text
        empty_point = MagicMock()
        empty_point.id = "p_empty"
        empty_point.score = 0.50
        empty_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "canonical_text": "",
        }
        mock_qdrant.query_points.return_value = MagicMock(points=[empty_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_entity_scoped(
            query="skills",
            entity_name="Alice",
            collection="test_col",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        # Empty text chunks should be filtered out
        assert len(chunks) == 0

    def test_entity_scoped_handles_scroll_exception(self):
        """Gracefully handles scroll failures."""
        from src.rag_v3.retrieve import retrieve_entity_scoped

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.side_effect = Exception("Connection refused")

        # query_points still works (falls back to base filter)
        result_point = MagicMock()
        result_point.id = "p1"
        result_point.score = 0.80
        result_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "canonical_text": "Some relevant text content here.",
        }
        mock_qdrant.query_points.return_value = MagicMock(points=[result_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_entity_scoped(
            query="experience",
            entity_name="Bob",
            collection="test_col",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        # Should still return results via fallback
        assert len(chunks) >= 1

    def test_entity_scoped_handles_embed_exception(self):
        """Returns empty list when embedding fails."""
        from src.rag_v3.retrieve import retrieve_entity_scoped

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)

        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = RuntimeError("Model not loaded")

        chunks = retrieve_entity_scoped(
            query="skills",
            entity_name="Carol",
            collection="test_col",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert chunks == []

    def test_entity_scoped_matches_source_name(self):
        """Entity matching works on source_name field too."""
        from src.rag_v3.retrieve import retrieve_entity_scoped
        from qdrant_client.models import MatchAny

        mock_qdrant = MagicMock()
        scroll_point = MagicMock()
        scroll_point.id = "p1"
        scroll_point.score = 0.0
        scroll_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc_alice",
            "source_name": "alice_johnson_resume.pdf",
            "canonical_text": "Experience in project management.",
        }
        mock_qdrant.scroll.return_value = ([scroll_point], None)

        result_point = MagicMock()
        result_point.id = "r1"
        result_point.score = 0.90
        result_point.payload = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc_alice",
            "canonical_text": "Project management with Agile and Scrum.",
        }
        mock_qdrant.query_points.return_value = MagicMock(points=[result_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_entity_scoped(
            query="project management",
            entity_name="alice",
            collection="test_col",
            subscription_id="sub1",
            profile_id="prof1",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert len(chunks) >= 1
        # Verify that query_points was called with a document_id filter (entity matched via source_name)
        call_kwargs = mock_qdrant.query_points.call_args
        query_filter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")
        # The filter should contain a document_id condition with MatchAny
        filter_must = query_filter.must
        doc_conditions = [c for c in filter_must if hasattr(c, 'key') and c.key == "document_id"]
        assert len(doc_conditions) == 1
        assert isinstance(doc_conditions[0].match, MatchAny)
        assert "doc_alice" in doc_conditions[0].match.any


class TestSectionFilteredRetrieval:
    """Tests for retrieve_section_filtered() in src/rag_v3/retrieve.py."""

    def test_retrieve_section_filtered_signature(self):
        """Function has the expected parameters."""
        from src.rag_v3.retrieve import retrieve_section_filtered
        import inspect

        sig = inspect.signature(retrieve_section_filtered)
        assert "section_kind" in sig.parameters
        assert "doc_domain" in sig.parameters
        assert "query" in sig.parameters
        assert "collection" in sig.parameters
        assert "subscription_id" in sig.parameters
        assert "profile_id" in sig.parameters
        assert "top_k" in sig.parameters
        assert "embedder" in sig.parameters
        assert "qdrant_client" in sig.parameters
        assert "correlation_id" in sig.parameters

    def test_section_filter_builds_correct_qdrant_filter(self):
        """When section_kind is provided, the Qdrant filter includes it."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        retrieve_section_filtered(
            query="education background",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            section_kind="education",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert mock_qdrant.query_points.called
        call_kwargs = mock_qdrant.query_points.call_args
        used_filter = call_kwargs.kwargs.get("query_filter") if call_kwargs.kwargs else None
        assert used_filter is not None

    def test_section_filter_fallback_when_empty(self):
        """When section filter returns no results, should fallback to base filter."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_qdrant = MagicMock()
        # First call (with section filter) returns empty
        # Second call (fallback) returns a point
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.7
        mock_point.payload = {
            "subscription_id": "s1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "Some education content here for testing purposes.",
        }
        mock_qdrant.query_points.side_effect = [
            MagicMock(points=[]),  # First: filtered, empty
            MagicMock(points=[mock_point]),  # Second: fallback
        ]

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="education",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            section_kind="education",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        # Should have called query_points twice (filtered + fallback)
        assert mock_qdrant.query_points.call_count == 2
        assert len(chunks) >= 1

    def test_no_fallback_when_filter_returns_results(self):
        """When section filter returns results, no fallback should happen."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.8
        mock_point.payload = {
            "subscription_id": "s1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "Bachelor of Science in Computer Science from MIT.",
        }

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="education",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            section_kind="education",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        # Should have called query_points only once (filtered returned results)
        assert mock_qdrant.query_points.call_count == 1
        assert len(chunks) == 1

    def test_no_fallback_without_section_or_domain(self):
        """When neither section_kind nor doc_domain is set, no fallback even if empty."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="something",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        # Only one call -- no fallback because no section/domain filter was applied
        assert mock_qdrant.query_points.call_count == 1
        assert len(chunks) == 0

    def test_doc_domain_filter(self):
        """When doc_domain is provided, Qdrant filter includes it."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.9
        mock_point.payload = {
            "subscription_id": "s1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "Invoice total: $500.00 due by end of month.",
        }

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="invoice total",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            doc_domain="invoice",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert mock_qdrant.query_points.call_count == 1
        assert len(chunks) == 1

    def test_empty_text_chunks_filtered_out(self):
        """Chunks with empty text after _to_chunk should be excluded."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_point_empty = MagicMock()
        mock_point_empty.id = "p1"
        mock_point_empty.score = 0.9
        mock_point_empty.payload = {
            "subscription_id": "s1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "",  # Empty text
        }
        mock_point_good = MagicMock()
        mock_point_good.id = "p2"
        mock_point_good.score = 0.8
        mock_point_good.payload = {
            "subscription_id": "s1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "Valid education content with enough text to pass.",
        }

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(
            points=[mock_point_empty, mock_point_good]
        )

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="education",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            section_kind="education",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        # Only the non-empty chunk should be returned
        assert len(chunks) == 1
        assert "Valid education content" in chunks[0].text

    def test_embed_failure_returns_empty(self):
        """If embedding fails, return empty list (no crash)."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_qdrant = MagicMock()

        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = RuntimeError("GPU OOM")

        chunks = retrieve_section_filtered(
            query="education",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            section_kind="education",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert chunks == []
        # Qdrant should never have been called
        assert mock_qdrant.query_points.call_count == 0

    def test_qdrant_exception_returns_empty(self):
        """If Qdrant throws, return empty list (no crash)."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.side_effect = Exception("Connection refused")

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="education",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            section_kind="education",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert chunks == []

    def test_both_section_and_domain_filters(self):
        """When both section_kind and doc_domain are provided, both are applied."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.85
        mock_point.payload = {
            "subscription_id": "s1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "Education section from a resume document with details.",
        }

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="education",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            section_kind="education",
            doc_domain="resume",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert len(chunks) == 1
        assert mock_qdrant.query_points.call_count == 1

    def test_missing_embedder_raises(self):
        """If embedder is not provided, should raise ValueError."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_qdrant = MagicMock()

        with pytest.raises(ValueError, match="embedder is required"):
            retrieve_section_filtered(
                query="test",
                collection="test",
                subscription_id="s1",
                profile_id="p1",
                qdrant_client=mock_qdrant,
            )

    def test_missing_qdrant_client_raises(self):
        """If qdrant_client is not provided, should raise ValueError."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_embedder = MagicMock()

        with pytest.raises(ValueError, match="qdrant_client is required"):
            retrieve_section_filtered(
                query="test",
                collection="test",
                subscription_id="s1",
                profile_id="p1",
                embedder=mock_embedder,
            )

    def test_top_k_respected(self):
        """The top_k parameter is forwarded to Qdrant."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[])

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        retrieve_section_filtered(
            query="education",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            top_k=5,
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        call_kwargs = mock_qdrant.query_points.call_args
        assert call_kwargs.kwargs.get("limit") == 5

    def test_fallback_also_uses_doc_domain_trigger(self):
        """Fallback is triggered when doc_domain is set and results are empty."""
        from src.rag_v3.retrieve import retrieve_section_filtered

        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.6
        mock_point.payload = {
            "subscription_id": "s1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "Some invoice content with amount and totals listed.",
        }

        mock_qdrant = MagicMock()
        mock_qdrant.query_points.side_effect = [
            MagicMock(points=[]),  # Filtered: empty
            MagicMock(points=[mock_point]),  # Fallback
        ]

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * 1024]

        chunks = retrieve_section_filtered(
            query="invoice total",
            collection="test",
            subscription_id="s1",
            profile_id="p1",
            doc_domain="invoice",
            embedder=mock_embedder,
            qdrant_client=mock_qdrant,
        )

        assert mock_qdrant.query_points.call_count == 2
        assert len(chunks) == 1
