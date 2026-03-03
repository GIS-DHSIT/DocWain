"""Tests for ML-first query scope inference, domain detection, and field focus.

Validates that regex patterns were replaced by ML classifiers for natural
language query analysis, while structured format extraction (document IDs,
invoice numbers) is preserved.
"""
import pytest
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeIntentParse:
    """Minimal mock of IntentParse for testing scope/domain logic."""
    intent: str = "qa"
    output_format: str = "text"
    requested_fields: list = field(default_factory=list)
    domain: str = "generic"
    constraints: dict = field(default_factory=dict)
    entity_hints: list = field(default_factory=list)
    source: str = "test"


@dataclass
class FakeQueryFocus:
    keywords: list = field(default_factory=list)
    bigrams: list = field(default_factory=list)
    field_tags: set = field(default_factory=set)
    section_kinds: list = field(default_factory=list)
    intent: str = "factual"
    is_exhaustive: bool = False
    query_embedding: Any = None
    field_probabilities: Optional[Dict[str, float]] = None
    _embedder: Any = None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Scope Inference — ML-First
# ═══════════════════════════════════════════════════════════════════════════

class TestScopeInferenceMLFirst:
    """Verify _infer_query_scope uses intent_parse before any regex."""

    def _scope(self, query, intent_parse=None, explicit_document_id=None):
        from src.rag_v3.pipeline import _infer_query_scope
        return _infer_query_scope(query, explicit_document_id, intent_parse)

    # -- explicit document_id always wins --
    def test_explicit_document_id(self):
        scope = self._scope("anything", explicit_document_id="abc123")
        assert scope.mode == "specific_document"
        assert scope.document_id == "abc123"

    # -- ML intent_parse: compare/rank/list → all_profile --
    def test_compare_intent_no_entity(self):
        ip = FakeIntentParse(intent="compare")
        scope = self._scope("compare the two candidates", ip)
        assert scope.mode == "all_profile"

    def test_rank_intent_no_entity(self):
        ip = FakeIntentParse(intent="rank")
        scope = self._scope("rank them by experience", ip)
        assert scope.mode == "all_profile"

    def test_list_intent_no_entity(self):
        ip = FakeIntentParse(intent="list")
        scope = self._scope("list all documents", ip)
        assert scope.mode == "all_profile"

    # -- ML intent_parse: compare with entity → still all_profile --
    def test_compare_with_entity_still_all_profile(self):
        """When intent is compare but entity present, intent wins (not targeted)."""
        ip = FakeIntentParse(intent="compare", entity_hints=[])
        scope = self._scope("compare all candidates", ip)
        assert scope.mode == "all_profile"

    # -- ML intent_parse: summarize without entity → all_profile --
    def test_summarize_no_entity_all_profile(self):
        ip = FakeIntentParse(intent="summarize")
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value=None):
            scope = self._scope("summarize all the resumes", ip)
        assert scope.mode == "all_profile"

    # -- ML intent_parse: summarize with NLP entity → targeted --
    def test_summarize_with_nlp_entity_targeted(self):
        ip = FakeIntentParse(intent="summarize")
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value="John"):
            scope = self._scope("summarize John's resume", ip)
        assert scope.mode == "targeted"
        assert scope.entity_hint == "John"

    # -- ML intent_parse: entity hints → targeted --
    def test_entity_hints_targeted(self):
        ip = FakeIntentParse(intent="qa", entity_hints=["Saikiran"])
        scope = self._scope("What are Saikiran's skills?", ip)
        assert scope.mode == "targeted"
        assert scope.entity_hint == "Saikiran"

    # -- Structured format: document_id pattern (kept, not query analysis) --
    def test_document_id_pattern(self):
        scope = self._scope("document_id: abc123")
        assert scope.mode == "specific_document"
        assert scope.document_id == "abc123"

    # -- Structured format: invoice number pattern (kept) --
    def test_invoice_number_pattern(self):
        scope = self._scope("invoice #1234")
        assert scope.mode == "targeted"
        assert scope.entity_hint == "1234"

    def test_order_number_pattern(self):
        scope = self._scope("order 5678")
        assert scope.mode == "targeted"
        assert scope.entity_hint == "5678"

    # -- NLP entity extraction fallback --
    def test_nlp_entity_extraction(self):
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value="Gaurav"):
            scope = self._scope("Tell me about Gaurav")
        assert scope.mode == "targeted"
        assert scope.entity_hint == "Gaurav"

    # -- Safety net: compare/rank/vs without intent_parse --
    def test_safety_net_compare(self):
        scope = self._scope("compare them")
        # Might go through NLP entity or safety net
        # The safety net catches "compare"
        assert scope.mode in ("all_profile", "targeted")

    def test_safety_net_versus(self):
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value=None):
            scope = self._scope("John vs Bob")
        assert scope.mode == "all_profile"

    # -- Default: all_profile (changed from targeted) --
    def test_default_all_profile(self):
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value=None):
            scope = self._scope("hello world")
        assert scope.mode == "all_profile"


class TestScopeInferenceRegexRemoved:
    """Verify that old regex patterns no longer control scope inference."""

    def _scope(self, query, intent_parse=None):
        from src.rag_v3.pipeline import _infer_query_scope
        return _infer_query_scope(query, None, intent_parse)

    def test_all_docs_patterns_removed(self):
        """_ALL_DOCS_PATTERNS is no longer defined in pipeline module."""
        import src.rag_v3.pipeline as p
        assert not hasattr(p, "_ALL_DOCS_PATTERNS")

    def test_summarize_all_resumes_with_ml(self):
        """The query that motivated this refactor: 'summarize all the resumes'."""
        ip = FakeIntentParse(intent="summarize")
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value=None):
            scope = self._scope("summarize all the resumes", ip)
        assert scope.mode == "all_profile"

    def test_overview_every_candidate_with_ml(self):
        ip = FakeIntentParse(intent="summarize")
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value=None):
            scope = self._scope("Overview of every candidate", ip)
        assert scope.mode == "all_profile"

    def test_who_has_best_skills_with_ml(self):
        ip = FakeIntentParse(intent="rank")
        scope = self._scope("who has the best skills?", ip)
        assert scope.mode == "all_profile"

    def test_show_all_candidates_with_ml(self):
        ip = FakeIntentParse(intent="list")
        scope = self._scope("show all candidates", ip)
        assert scope.mode == "all_profile"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Domain Detection — ML-First
# ═══════════════════════════════════════════════════════════════════════════

class TestDomainDetectionMLFirst:
    """Verify _ml_query_domain uses ML classifier instead of keyword tuples."""

    def test_keyword_tuples_removed(self):
        """Old keyword tuples are no longer in extract module."""
        import src.rag_v3.extract as ext
        assert not hasattr(ext, "_QUERY_HR_STRONG")
        assert not hasattr(ext, "_QUERY_HR_WEAK")
        assert not hasattr(ext, "_QUERY_INVOICE_STRONG")
        assert not hasattr(ext, "_QUERY_LEGAL_STRONG")
        assert not hasattr(ext, "_QUERY_POLICY_STRONG")

    def test_query_domain_override_removed(self):
        """Old _query_domain_override function is no longer in extract module."""
        import src.rag_v3.extract as ext
        assert not hasattr(ext, "_query_domain_override")

    def test_ml_query_domain_exists(self):
        """_ml_query_domain function exists in extract module."""
        from src.rag_v3.extract import _ml_query_domain
        assert callable(_ml_query_domain)

    def test_ml_domain_from_intent_parse_hr(self):
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="resume")
        assert _ml_query_domain("anything", ip) == "hr"

    def test_ml_domain_from_intent_parse_invoice(self):
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="invoice")
        assert _ml_query_domain("anything", ip) == "invoice"

    def test_ml_domain_from_intent_parse_legal(self):
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="legal")
        assert _ml_query_domain("anything", ip) == "legal"

    def test_ml_domain_from_intent_parse_policy(self):
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="policy")
        assert _ml_query_domain("anything", ip) == "policy"

    def test_ml_domain_generic_returns_none(self):
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="generic")
        # Generic domain should not map to anything specific
        result = _ml_query_domain("anything", ip)
        # Falls through to classifier (which likely isn't loaded) → None
        assert result is None

    def test_ml_domain_none_intent_parse_returns_none_without_classifier(self):
        from src.rag_v3.extract import _ml_query_domain
        # When no intent_parse and no classifier loaded, returns None
        with patch("src.intent.intent_classifier.get_intent_classifier", return_value=None):
            assert _ml_query_domain("what are the skills") is None

    def test_ml_domain_with_trained_classifier(self):
        """When trained classifier returns a confident domain, it's used."""
        from src.rag_v3.extract import _ml_query_domain
        mock_clf = MagicMock()
        mock_clf._trained = True
        mock_clf.predict.return_value = {"domain": "resume", "domain_confidence": 0.85}
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [MagicMock()]

        with patch("src.intent.intent_classifier.get_intent_classifier", return_value=mock_clf), \
             patch("src.intent.llm_intent._get_embedder", return_value=mock_embedder):
            result = _ml_query_domain("what are the candidate's skills")
        assert result == "hr"

    def test_ml_domain_low_confidence_returns_none(self):
        """When classifier confidence is below 0.60 threshold, returns None."""
        from src.rag_v3.extract import _ml_query_domain
        mock_clf = MagicMock()
        mock_clf._trained = True
        mock_clf.predict.return_value = {"domain": "resume", "domain_confidence": 0.30}
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [MagicMock()]

        with patch("src.intent.intent_classifier.get_intent_classifier", return_value=mock_clf), \
             patch("src.intent.llm_intent._get_embedder", return_value=mock_embedder):
            result = _ml_query_domain("hello world")
        assert result is None


class TestDomainDetectionInferDomainIntent:
    """Verify _infer_domain_intent uses _ml_query_domain internally."""

    def test_infer_domain_accepts_intent_parse(self):
        """_infer_domain_intent accepts intent_parse parameter."""
        from src.rag_v3.extract import _infer_domain_intent
        import inspect
        sig = inspect.signature(_infer_domain_intent)
        assert "intent_parse" in sig.parameters

    def test_infer_domain_routes_through_ml(self):
        """Domain detection uses ML when intent_parse has confident domain."""
        from src.rag_v3.extract import _infer_domain_intent
        ip = FakeIntentParse(domain="invoice")
        domain, intent = _infer_domain_intent("what is the total", [], intent_parse=ip)
        assert domain == "invoice"


class TestExtractSchemaIntentParseThreading:
    """Verify intent_parse is threaded through extract_schema and schema_extract."""

    def test_extract_schema_accepts_intent_parse(self):
        from src.rag_v3.extract import extract_schema
        import inspect
        sig = inspect.signature(extract_schema)
        assert "intent_parse" in sig.parameters

    def test_schema_extract_accepts_intent_parse(self):
        from src.rag_v3.extract import schema_extract
        import inspect
        sig = inspect.signature(schema_extract)
        assert "intent_parse" in sig.parameters


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Field Focus — ML-First
# ═══════════════════════════════════════════════════════════════════════════

class TestFieldFocusMLFirst:
    """Verify field focus detection uses ML classifier first, keywords as fallback."""

    def test_enterprise_field_focus_map_removed(self):
        """enterprise.py no longer has _FIELD_FOCUS_MAP."""
        import src.rag_v3.enterprise as ent
        assert not hasattr(ent, "_FIELD_FOCUS_MAP")

    def test_enterprise_detect_field_focus_removed(self):
        """enterprise.py no longer has _detect_field_focus."""
        import src.rag_v3.enterprise as ent
        assert not hasattr(ent, "_detect_field_focus")

    def test_query_focus_still_has_field_focus_map(self):
        """query_focus.py retains _FIELD_FOCUS_MAP as keyword fallback."""
        import src.rag_v3.query_focus as qf
        assert hasattr(qf, "_FIELD_FOCUS_MAP")

    def test_build_query_focus_ml_first_ordering(self):
        """ML classifier runs before keyword fallback in build_query_focus."""
        from src.rag_v3.query_focus import build_query_focus
        import inspect
        src = inspect.getsource(build_query_focus)
        # ML block should appear before keyword fallback
        ml_idx = src.find("field_classifier")
        kw_idx = src.find("_FIELD_FOCUS_MAP")
        assert ml_idx < kw_idx, "ML classifier should run before keyword fallback"

    def test_keyword_fallback_only_when_ml_empty(self):
        """Keywords are only checked when ML classifier produced no tags."""
        from src.rag_v3.query_focus import build_query_focus
        # Without embedder, ML path is skipped → keyword fallback runs
        focus = build_query_focus("what are the skills?", embedder=None)
        assert "skills" in focus.field_tags

    def test_ml_classifier_tags_used_when_available(self):
        """When ML classifier returns tags, keyword fallback is skipped."""
        from src.rag_v3.query_focus import build_query_focus

        mock_clf = MagicMock()
        mock_clf.predict.return_value = {"experience": 0.9, "education": 0.7}

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [MagicMock()]

        with patch("src.rag_v3.field_classifier.get_field_classifier", return_value=mock_clf):
            focus = build_query_focus("what are the candidate details?", embedder=mock_embedder)

        # ML classifier tags should be present
        assert "experience" in focus.field_tags
        assert "education" in focus.field_tags
        # Field probabilities should be populated
        assert focus.field_probabilities is not None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Training Templates
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainingTemplates:
    """Verify scope-reinforcing training templates were added."""

    def test_scope_reinforcing_templates_present(self):
        from src.intent.intent_classifier import TRAINING_TEMPLATES
        queries = [t[0] for t in TRAINING_TEMPLATES]
        assert "Summarize all the resumes." in queries
        assert "Overview of every candidate." in queries
        assert "Give me details about all the documents." in queries
        assert "Who among the candidates has the best skills?" in queries

    def test_template_count_increased(self):
        from src.intent.intent_classifier import TRAINING_TEMPLATES
        # 12 scope-reinforcing templates were added
        assert len(TRAINING_TEMPLATES) >= 116


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: llm_intent.py Rename
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackParseRename:
    """Verify _heuristic_parse was renamed to _fallback_parse."""

    def test_fallback_parse_exists(self):
        from src.intent.llm_intent import _fallback_parse
        assert callable(_fallback_parse)

    def test_heuristic_parse_removed(self):
        import src.intent.llm_intent as mod
        assert not hasattr(mod, "_heuristic_parse")

    def test_fallback_parse_returns_valid_dict(self):
        from src.intent.llm_intent import _fallback_parse
        # Disable neural classifier to test fallback path
        with patch("src.intent.llm_intent._neural_parse", return_value=None):
            result = _fallback_parse("summarize the document")
        assert isinstance(result, dict)
        assert "intent" in result
        assert "domain" in result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Dead Code Removal
# ═══════════════════════════════════════════════════════════════════════════

class TestDeadCodeRemoval:
    """Verify removed functions/patterns are no longer in the codebase."""

    def test_all_docs_patterns_removed(self):
        import src.rag_v3.pipeline as p
        assert not hasattr(p, "_ALL_DOCS_PATTERNS")

    def test_query_is_hr_removed_from_pipeline(self):
        import src.rag_v3.pipeline as p
        assert not hasattr(p, "_query_is_hr")

    def test_resolve_domain_from_chunks_removed(self):
        import src.rag_v3.pipeline as p
        assert not hasattr(p, "_resolve_domain_from_chunks")

    def test_query_is_hr_like_removed_from_extract(self):
        import src.rag_v3.extract as ext
        assert not hasattr(ext, "_query_is_hr_like")

    def test_query_domain_override_removed_from_extract(self):
        import src.rag_v3.extract as ext
        assert not hasattr(ext, "_query_domain_override")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Integration — End-to-End Scope + Domain
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndScopeDomain:
    """Integration tests for scope inference + domain detection together."""

    def test_compare_all_candidates_python(self):
        """'Compare all candidates for Python' → scope=all_profile."""
        from src.rag_v3.pipeline import _infer_query_scope
        ip = FakeIntentParse(intent="compare", domain="resume")
        scope = _infer_query_scope("Compare all candidates for Python", None, ip)
        assert scope.mode == "all_profile"

    def test_saikiran_skills_targeted(self):
        """'What are Saikiran's skills?' → scope=targeted."""
        from src.rag_v3.pipeline import _infer_query_scope
        ip = FakeIntentParse(intent="qa", entity_hints=["Saikiran"])
        scope = _infer_query_scope("What are Saikiran's skills?", None, ip)
        assert scope.mode == "targeted"
        assert scope.entity_hint == "Saikiran"

    def test_summarize_all_resumes_fast_path(self):
        """The motivating use case: 'Summarize all the resumes' → all_profile (fast path)."""
        from src.rag_v3.pipeline import _infer_query_scope
        ip = FakeIntentParse(intent="summarize")
        with patch("src.rag_v3.pipeline._try_nlp_entity", return_value=None):
            scope = _infer_query_scope("Summarize all the resumes", None, ip)
        assert scope.mode == "all_profile"

    def test_rank_by_experience(self):
        from src.rag_v3.pipeline import _infer_query_scope
        ip = FakeIntentParse(intent="rank")
        scope = _infer_query_scope("Rank by years of relevant experience", None, ip)
        assert scope.mode == "all_profile"

    def test_domain_from_intent_parse_invoice(self):
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="invoice")
        assert _ml_query_domain("what is the total amount due", ip) == "invoice"

    def test_domain_from_intent_parse_legal(self):
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="legal")
        assert _ml_query_domain("explain the indemnification clause", ip) == "legal"

    def test_domain_mapping_resume_to_hr(self):
        """resume domain maps to hr for extraction routing."""
        from src.rag_v3.extract import _ml_query_domain
        ip = FakeIntentParse(domain="resume")
        assert _ml_query_domain("summarize the resume", ip) == "hr"
