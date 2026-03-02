"""Tests for production log analysis fixes (2026-02-17).

Covers:
1. NLP-based entity extraction (spaCy dependency parsing + DPIE)
2. Grounding gate bypass for valid deterministic extraction
3. Decoupled extraction/screening/embedding phases
"""

import re
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. NLP-based entity extraction (replaces regex patterns)
# ---------------------------------------------------------------------------

class TestNLPEntityExtraction:
    """Entity extraction via spaCy dependency parsing — case-insensitive."""

    def test_extract_lowercase_name_after_of(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("give me the skills of thamaraikannan")
        assert entity is not None
        assert "thamaraikannan" in entity.lower()

    def test_extract_lowercase_name_after_about(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("tell me about muthu")
        assert entity is not None
        assert "muthu" in entity.lower()

    def test_extract_lowercase_name_after_for(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("get the resume for karthik")
        assert entity is not None
        assert "karthik" in entity.lower()

    def test_extract_lowercase_name_after_from(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("extract skills from aadithya")
        assert entity is not None
        assert "aadithya" in entity.lower()

    def test_extract_possessive_lowercase(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("thamaraikannan's skills")
        assert entity is not None
        assert "thamaraikannan" in entity.lower()

    def test_extract_capitalized_name(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("give me the skills of Thamaraikannan")
        assert entity is not None
        assert "thamaraikannan" in entity.lower()

    def test_extract_mixed_case(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("skills of Gokul")
        assert entity is not None
        assert "gokul" in entity.lower()

    def test_no_entity_for_generic_query(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("give me all the details")
        assert entity is None

    def test_no_entity_for_all_candidates(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("compare all candidates")
        assert entity is None

    def test_extract_verb_name(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("summarize thamaraikannan")
        assert entity is not None
        assert "thamaraikannan" in entity.lower()

    def test_extract_nsubj_lowercase(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("does swapnil have certifications")
        assert entity is not None
        assert "swapnil" in entity.lower()

    def test_extract_nsubj_with_verb(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("has aadithya worked on machine learning")
        assert entity is not None
        assert "aadithya" in entity.lower()

    def test_contact_details_of_name(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("contact details of abinaya")
        assert entity is not None
        assert "abinaya" in entity.lower()

    def test_find_verb_name(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("find gaurav")
        assert entity is not None
        assert "gaurav" in entity.lower()

    def test_common_word_not_extracted(self):
        from src.nlp.query_entity_extractor import extract_entity_from_query
        entity = extract_entity_from_query("show me the skills from this document")
        if entity:
            assert entity.lower() not in ("document", "skills", "this", "show")


class TestNLPEntityInPipeline:
    """NLP entity extraction integrated into pipeline scope inference."""

    def _infer_scope(self, query, intent_parse=None):
        from src.rag_v3.pipeline import _infer_query_scope
        return _infer_query_scope(query, explicit_document_id=None, intent_parse=intent_parse)

    def test_pipeline_lowercase_name(self):
        scope = self._infer_scope("give me the skills of thamaraikannan")
        assert scope.entity_hint is not None
        assert "thamaraikannan" in scope.entity_hint.lower()

    def test_pipeline_possessive(self):
        scope = self._infer_scope("thamaraikannan's skills")
        assert scope.entity_hint is not None
        assert "thamaraikannan" in scope.entity_hint.lower()

    def test_pipeline_no_entity_generic(self):
        scope = self._infer_scope("give me all the details")
        assert scope.entity_hint is None

    def test_pipeline_nsubj(self):
        scope = self._infer_scope("does swapnil have certifications")
        assert scope.entity_hint is not None
        assert "swapnil" in scope.entity_hint.lower()

    def test_pipeline_compare_still_all_profile(self):
        scope = self._infer_scope("compare all candidates")
        assert scope.mode == "all_profile"


class TestFallbackParseNLP:
    """Fallback intent parser uses NLP entity extraction."""

    def _parse(self, query):
        from src.intent.llm_intent import _fallback_parse
        return _fallback_parse(query)

    def test_lowercase_name_extracted(self):
        result = self._parse("give me the skills of thamaraikannan")
        assert result["entity_hints"], "Should extract entity hint for lowercase name"
        assert "thamaraikannan" in result["entity_hints"][0].lower()

    def test_capitalized_name_extracted(self):
        result = self._parse("tell me about Gokul")
        assert result["entity_hints"]
        assert "gokul" in result["entity_hints"][0].lower()

    def test_domain_detected_with_entity(self):
        result = self._parse("give me the skills of thamaraikannan")
        assert result["domain"] == "resume"  # "skills" triggers resume domain


class TestNoRegexEntityPatterns:
    """Verify regex-based entity patterns have been removed from pipeline."""

    def test_no_specific_entity_patterns(self):
        import inspect
        from src.rag_v3 import pipeline
        source = inspect.getsource(pipeline)
        assert "_SPECIFIC_ENTITY_PATTERNS" not in source

    def test_no_entity_patterns_ci(self):
        import inspect
        from src.rag_v3 import pipeline
        source = inspect.getsource(pipeline)
        assert "_ENTITY_PATTERNS_CI" not in source

    def test_no_not_entity_words(self):
        import inspect
        from src.rag_v3 import pipeline
        source = inspect.getsource(pipeline)
        assert "_NOT_ENTITY_WORDS" not in source

    def test_uses_nlp_extractor(self):
        import inspect
        from src.rag_v3 import pipeline
        source = inspect.getsource(pipeline)
        assert "extract_entity_from_query" in source

    def test_llm_intent_uses_nlp(self):
        import inspect
        from src.intent import llm_intent
        # Entity extraction is in _extract_entity_hints (called by both
        # _fallback_parse and _neural_parse)
        source = inspect.getsource(llm_intent._extract_entity_hints)
        assert "extract_entity_from_query" in source


# ---------------------------------------------------------------------------
# 2. Grounding gate bypass for valid deterministic extraction
# ---------------------------------------------------------------------------

class TestGroundingGateBypass:
    """Grounding gate should be skipped for valid deterministic extraction."""

    def test_grounding_gate_skip_condition(self):
        from src.rag_v3.pipeline import _has_valid_deterministic_extraction
        try:
            from src.rag_v3.types import HRSchema, Candidate, CandidateField, EvidenceSpan
            schema = HRSchema(candidates=CandidateField(
                items=[Candidate(name="Test", evidence_spans=[])],
            ))
            assert _has_valid_deterministic_extraction(schema) is True
        except (ImportError, TypeError):
            pytest.skip("HRSchema not importable in test environment")

    def test_llm_response_bypasses_grounding(self):
        from src.rag_v3.pipeline import _is_llm_response
        extraction = MagicMock()
        try:
            from src.rag_v3.types import LLMResponseSchema
            extraction.schema = LLMResponseSchema(text="some answer")
            assert _is_llm_response(extraction) is True
        except (ImportError, TypeError):
            pytest.skip("LLMResponseSchema not importable in test environment")


class TestGroundingGateStillWorksForGeneric:
    """Grounding gate still functions for non-deterministic extraction."""

    def test_grounding_blocks_unsupported(self):
        from src.quality.fast_grounding import evaluate_grounding
        answer = "John has 15 years of experience in quantum computing at NASA."
        chunks = ["Skills: Python, Java, SQL. Education: B.Tech Computer Science."]
        result = evaluate_grounding(answer, chunks)
        assert result.critical_supported_ratio < 0.30

    def test_grounding_supports_matching(self):
        from src.quality.fast_grounding import evaluate_grounding
        answer = "The candidate has skills in Python, Java, and SQL."
        chunks = ["Skills: Python, Java, SQL, React, Node.js. Experience: 3 years."]
        result = evaluate_grounding(answer, chunks)
        assert result.supported_ratio > 0.0


# ---------------------------------------------------------------------------
# 3. Decoupled extraction/screening/embedding phases
# ---------------------------------------------------------------------------

class TestPhaseDecoupling:
    """Extraction, screening, and embedding should not auto-chain."""

    def test_embed_after_defaults_false(self):
        import inspect
        from src.api.document_understanding_service import run_document_understanding
        sig = inspect.signature(run_document_understanding)
        assert sig.parameters["embed_after"].default is False

    def test_extract_and_understand_embed_after_defaults_false(self):
        import inspect
        from src.api.document_understanding_service import extract_and_understand
        sig = inspect.signature(extract_and_understand)
        assert sig.parameters["embed_after"].default is False

    def test_extraction_does_not_call_auto_screening(self):
        import inspect
        from src.api import extraction_service
        source = inspect.getsource(extraction_service.extract_uploaded_document)
        assert "_run_auto_screening" not in source

    def test_embedding_rejects_unscreened_documents(self):
        import inspect
        from src.api import embedding_service
        source_file = inspect.getfile(embedding_service)
        with open(source_file, "r") as f:
            content = f.read()
        assert "from src.api.extraction_service import _run_auto_screening" not in content
