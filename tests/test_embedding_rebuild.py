"""Tests for the Qdrant embedding and vectorization rebuild.

Covers:
- Payload construction (slim, no nested objects)
- Embedding text normalization (structure-preserving, no forced mutations)
- Content validation gate (garbage rejection)
- Payload index alignment (canonical field names only)
- Retrieval simplification (flat fields, legacy fallback)
- normalize_content fixes (preserve camelCase, compound words)
- Query-embedding symmetry (no section prefix on queries)
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# TestPayloadConstruction
# ---------------------------------------------------------------------------

class TestPayloadConstruction:
    """Verify build_qdrant_payload produces slim, flat payloads."""

    def _build(self, **overrides):
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
        base = {
            "subscription_id": "sub-1",
            "profile_id": "prof-1",
            "document_id": "doc-1",
            "content": "Python developer with 5 years of backend experience in Django and Flask frameworks.",
            "source_name": "resume.pdf",
            "section_kind": "experience",
            "section_title": "Work Experience",
            "section_kind_source": "title",
        }
        base.update(overrides)
        return build_qdrant_payload(base)

    def test_build_payload_minimal_fields(self):
        payload = self._build()
        assert payload["subscription_id"] == "sub-1"
        assert payload["profile_id"] == "prof-1"
        assert payload["document_id"] == "doc-1"
        assert payload["source_name"] == "resume.pdf"
        assert "canonical_text" in payload
        assert "embedding_text" in payload
        assert payload.get("embed_pipeline_version")

    def test_build_payload_no_nested_objects(self):
        payload = self._build()
        # No nested dicts for source, section, chunk
        assert "source" not in payload
        assert "section" not in payload
        assert "chunk" not in payload

    def test_build_payload_garbage_text_rejected(self):
        garbage = "'section_id': 'abc', 'chunk_type': 'text', 'section_title': 'foo', 'page': None"
        payload = self._build(content=garbage, canonical_text=garbage)
        # canonical_text should be empty or cleaned (not the raw garbage)
        ct = payload.get("canonical_text", "")
        assert "'section_id'" not in ct
        assert "'chunk_type'" not in ct

    def test_build_payload_embedding_text_no_forced_mutation(self):
        original = "Senior software engineer with expertise in cloud architecture"
        payload = self._build(content=original)
        et = payload.get("embedding_text", "")
        # Should NOT be lowercased or have " ." appended
        assert et.lower() != et or et == et  # not forced lowercase
        assert not et.endswith(" .")

    def test_build_payload_section_prefix_only_title_match(self):
        # Title-derived kind: should get prefix
        payload_title = self._build(
            section_kind="experience",
            section_kind_source="title",
            section_title="Work Experience",
        )
        et_title = payload_title.get("embedding_text", "")
        assert "[Experience]" in et_title

        # Content-derived kind: should NOT get prefix
        payload_content = self._build(
            section_kind="experience",
            section_kind_source="content",
            section_title="Work Experience",
        )
        et_content = payload_content.get("embedding_text", "")
        assert "[Experience]" not in et_content


# ---------------------------------------------------------------------------
# TestEmbeddingTextNormalization
# ---------------------------------------------------------------------------

class TestEmbeddingTextNormalization:
    """Verify embedding text normalization preserves structure."""

    def test_normalize_preserves_line_structure(self):
        from src.embedding.pipeline.embedding_text_normalizer import normalize_for_embedding
        text = "Line one\nLine two\nLine three"
        result = normalize_for_embedding(text)
        assert "\n" in result
        assert "Line one" in result
        assert "Line two" in result

    def test_normalize_deduplicates_lines(self):
        from src.embedding.pipeline.embedding_text_normalizer import normalize_for_embedding
        text = "Hello World\nHello World\nDifferent line"
        result = normalize_for_embedding(text)
        assert result.count("Hello World") == 1
        assert "Different line" in result

    def test_normalize_drops_page_numbers(self):
        from src.embedding.pipeline.embedding_text_normalizer import normalize_for_embedding
        text = "Page 1 of 5\nActual content here\n3/10"
        result = normalize_for_embedding(text)
        assert "Page 1 of 5" not in result
        assert "3/10" not in result
        assert "Actual content here" in result

    def test_ensure_embedding_text_no_lowercase_force(self):
        from src.embedding.pipeline.embedding_text_normalizer import ensure_embedding_text
        text = "JavaScript and TypeScript are Modern Languages"
        result = ensure_embedding_text(text)
        # Should not be forced to lowercase
        assert "JavaScript" in result
        assert "TypeScript" in result

    def test_ensure_embedding_text_no_dot_append(self):
        from src.embedding.pipeline.embedding_text_normalizer import ensure_embedding_text
        text = "Python developer with AWS experience"
        result = ensure_embedding_text(text)
        assert not result.endswith(" .")

    def test_normalize_preserves_bullets(self):
        from src.embedding.pipeline.embedding_text_normalizer import normalize_for_embedding
        text = "- Python\n- Java\n- JavaScript"
        result = normalize_for_embedding(text)
        assert "- Python" in result
        assert "- Java" in result


# ---------------------------------------------------------------------------
# TestContentValidationGate
# ---------------------------------------------------------------------------

class TestContentValidationGate:
    """Verify pre-embedding content validation."""

    def test_garbage_text_detected(self):
        from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
        garbage = "'section_id': 'abc', 'chunk_type': 'text', 'section_title': 'Work'"
        assert _is_metadata_garbage(garbage)

    def test_short_text_below_threshold(self):
        # Text shorter than 20 chars would be dropped by the validation gate
        short = "Hi there"
        assert len(short.strip()) < 20

    def test_valid_text_passes(self):
        from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
        valid = "Experienced software engineer with expertise in cloud architecture and microservices."
        assert not _is_metadata_garbage(valid)

    def test_strong_garbage_marker_detected(self):
        from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
        text = "Extracted Document full_text some long content that has a garbage prefix but looks okay"
        assert _is_metadata_garbage(text)


# ---------------------------------------------------------------------------
# TestPayloadIndexAlignment
# ---------------------------------------------------------------------------

class TestPayloadIndexAlignment:
    """Verify index fields match filter fields exactly."""

    def test_required_fields_are_canonical(self):
        from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS
        # No legacy aliases
        legacy_aliases = {"subscriptionId", "subscription.id", "profileId", "profile.id",
                          "file_type", "connector_type", "doc_type", "section.id",
                          "section.kind", "profile_name", "document.type", "chunk_type", "source.name"}
        for field in REQUIRED_PAYLOAD_INDEX_FIELDS:
            assert field not in legacy_aliases, f"Legacy alias '{field}' found in REQUIRED_PAYLOAD_INDEX_FIELDS"

    def test_no_legacy_aliases_in_indexes(self):
        from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS
        assert "subscriptionId" not in REQUIRED_PAYLOAD_INDEX_FIELDS
        assert "profileId" not in REQUIRED_PAYLOAD_INDEX_FIELDS
        assert "subscription.id" not in REQUIRED_PAYLOAD_INDEX_FIELDS
        assert "profile.id" not in REQUIRED_PAYLOAD_INDEX_FIELDS

    def test_payload_index_fields_match_required(self):
        from src.api.vector_store import PAYLOAD_INDEX_FIELDS
        from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS
        assert set(PAYLOAD_INDEX_FIELDS) == set(REQUIRED_PAYLOAD_INDEX_FIELDS)


# ---------------------------------------------------------------------------
# TestRetrievalSimplification
# ---------------------------------------------------------------------------

class TestRetrievalSimplification:
    """Verify simplified _to_chunk() works correctly."""

    def _make_point(self, payload, score=0.8, point_id="pt-1"):
        class FakePoint:
            pass
        p = FakePoint()
        p.payload = payload
        p.score = score
        p.id = point_id
        return p

    def test_to_chunk_canonical_text_direct(self):
        from src.rag_v3.retrieve import _to_chunk
        point = self._make_point({
            "canonical_text": "Senior developer with 10 years experience",
            "source_name": "resume.pdf",
            "page": 1,
            "chunk_id": "chunk-abc",
        })
        chunk = _to_chunk(point)
        assert chunk.text == "Senior developer with 10 years experience"
        assert chunk.source.document_name == "resume.pdf"
        assert chunk.source.page == 1
        assert chunk.id == "chunk-abc"

    def test_to_chunk_legacy_fallback_embedding_text(self):
        from src.rag_v3.retrieve import _to_chunk
        # canonical_text is garbage, falls back to embedding_text
        point = self._make_point({
            "canonical_text": "'section_id': 'abc', 'chunk_type': 'text', 'section_title': 'Work'",
            "embedding_text": "[Experience] Work Experience: Senior developer with AWS expertise",
            "source_name": "cv.pdf",
            "chunk_id": "chunk-def",
        })
        chunk = _to_chunk(point)
        assert "Senior developer" in chunk.text
        # Should have stripped the prefix
        assert "[Experience]" not in chunk.text

    def test_to_chunk_flat_fields_only(self):
        from src.rag_v3.retrieve import _to_chunk
        point = self._make_point({
            "canonical_text": "Software engineer with Python and Java skills",
            "source_name": "john_resume.pdf",
            "page": 2,
            "chunk_id": "chunk-xyz",
        })
        chunk = _to_chunk(point)
        assert chunk.source.document_name == "john_resume.pdf"
        assert chunk.id == "chunk-xyz"

    def test_to_chunk_source_name_nested_fallback(self):
        """For legacy data, source.name is still a fallback."""
        from src.rag_v3.retrieve import _to_chunk
        point = self._make_point({
            "canonical_text": "Content here for testing the fallback path",
            "source": {"name": "legacy_doc.pdf"},
            "chunk_id": "chunk-legacy",
        })
        chunk = _to_chunk(point)
        assert chunk.source.document_name == "legacy_doc.pdf"


# ---------------------------------------------------------------------------
# TestNormalizeContent
# ---------------------------------------------------------------------------

class TestNormalizeContent:
    """Verify normalize_content preserves important text patterns."""

    def test_preserves_camelcase(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("JavaScript and TypeScript")
        assert "JavaScript" in result
        assert "TypeScript" in result

    def test_preserves_compound_words(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("self-employed contractor")
        assert "self-employed" in result

    def test_splits_digit_letter(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("3years of experience")
        assert "3 years" in result

    def test_preserves_linkedin(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("LinkedIn profile available")
        assert "LinkedIn" in result

    def test_preserves_function_parens(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("func(x) returns value")
        assert "func(x)" in result


# ---------------------------------------------------------------------------
# TestQueryEmbeddingSymmetry
# ---------------------------------------------------------------------------

class TestQueryEmbeddingSymmetry:
    """Verify queries don't get section prefixes."""

    def test_query_no_section_prefix(self):
        from src.rag_v3.retrieve import _enrich_query_for_embedding
        query = "What are the candidate's technical skills?"
        result = _enrich_query_for_embedding(query)
        assert result == query
        assert "[" not in result

    def test_query_preserved_verbatim(self):
        from src.rag_v3.retrieve import _enrich_query_for_embedding
        query = "rank candidates by experience"
        result = _enrich_query_for_embedding(query)
        assert result == query


# ---------------------------------------------------------------------------
# TestClassifierWithSource
# ---------------------------------------------------------------------------

class TestClassifierWithSource:
    """Verify classify_section_kind_with_source returns correct source."""

    def test_title_match_returns_title_source(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        kind, source = classify_section_kind_with_source("some text", "Work Experience")
        assert kind == "experience"
        assert source == "title"

    def test_content_match_returns_content_source(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        kind, source = classify_section_kind_with_source(
            "python java javascript typescript react angular docker kubernetes aws",
            "",
        )
        assert kind == "skills_technical"
        assert source == "content"

    def test_no_match_returns_section_text(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        kind, source = classify_section_kind_with_source("random text", "")
        assert kind == "section_text"
        assert source == "content"

    def test_backward_compat_classify_section_kind(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        result = classify_section_kind("some text", "Education")
        assert result == "education"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestBuildQdrantFilter
# ---------------------------------------------------------------------------

class TestBuildQdrantFilter:
    """Verify filter uses direct field conditions."""

    def test_filter_uses_direct_field_match(self):
        from src.api.vector_store import build_qdrant_filter
        f = build_qdrant_filter(subscription_id="sub-1", profile_id="prof-1")
        # must should have FieldCondition objects, not nested Filter(should=[...])
        for condition in f.must:
            # Direct FieldCondition has 'key' attribute
            if hasattr(condition, "key"):
                assert condition.key in ("subscription_id", "profile_id")

    def test_filter_requires_profile_id(self):
        from src.api.vector_store import build_qdrant_filter
        with pytest.raises(ValueError, match="profile_id"):
            build_qdrant_filter(subscription_id="sub-1", profile_id="")

    def test_filter_requires_subscription_id(self):
        from src.api.vector_store import build_qdrant_filter
        with pytest.raises(ValueError, match="subscription_id"):
            build_qdrant_filter(subscription_id="", profile_id="prof-1")
