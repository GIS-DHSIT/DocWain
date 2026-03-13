"""Tests for GLiNER zero-shot entity extraction integration."""
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.docwain_intel.models import SemanticUnit, UnitType, StructuredDocument
from src.docwain_intel.entity_engine import (
    ExtractionResult,
    MAX_GLINER_TEXT_CHARS,
    _GLINER_LABELS,
    _extract_gliner_entities,
    _get_gliner,
    extract_entities_and_facts,
)


def _make_doc(units):
    return StructuredDocument(document_id="doc_001", units=units, unit_count=len(units))


def _reset_gliner_state():
    """Reset GLiNER module-level singleton state for test isolation."""
    import src.docwain_intel.entity_engine as mod

    mod._gliner_model = None
    mod._gliner_available = True


class TestGlinerGracefulFallback:
    """GLiNER not installed -- must degrade gracefully."""

    def test_import_error_returns_none(self):
        _reset_gliner_state()
        with patch.dict("sys.modules", {"gliner": None}):
            # Force re-import to hit ImportError
            import src.docwain_intel.entity_engine as mod

            mod._gliner_model = None
            mod._gliner_available = True
            with patch("builtins.__import__", side_effect=ImportError("no gliner")):
                result = mod._get_gliner()
            assert result is None
            assert mod._gliner_available is False
        _reset_gliner_state()

    def test_extract_returns_empty_when_unavailable(self):
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mod._gliner_available = False
        entities = _extract_gliner_entities("Python is a skill", "u1")
        assert entities == []
        _reset_gliner_state()

    def test_main_extraction_works_without_gliner(self):
        """extract_entities_and_facts should succeed even when GLiNER is absent."""
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mod._gliner_available = False
        unit = SemanticUnit(
            unit_id="u1",
            unit_type=UnitType.PARAGRAPH,
            text="John Smith is a software engineer at Google.",
            page_start=1,
            page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))
        assert isinstance(result, ExtractionResult)
        # spaCy should still find PERSON/ORG
        labels = {e.label for e in result.entities}
        assert "PERSON" in labels or "ORG" in labels
        _reset_gliner_state()


class TestGlinerEntityExtraction:
    """GLiNER extraction with a mock model."""

    def _mock_model(self):
        model = MagicMock()
        model.predict_entities.return_value = [
            {"text": "Python", "label": "skill", "start": 0, "end": 6},
            {"text": "AWS Solutions Architect", "label": "certification", "start": 20, "end": 43},
            {"text": "Senior Developer", "label": "job_title", "start": 50, "end": 66},
        ]
        return model

    def test_entity_extraction_with_mock(self):
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mock_model = self._mock_model()
        mod._gliner_model = mock_model
        mod._gliner_available = True

        entities = _extract_gliner_entities(
            "Python programming, AWS Solutions Architect, Senior Developer role",
            "u1",
        )
        assert len(entities) == 3
        assert all(e.source == "gliner" for e in entities)
        assert all(e.confidence == 0.80 for e in entities)

        # Verify model was called with correct labels
        mock_model.predict_entities.assert_called_once()
        call_args = mock_model.predict_entities.call_args
        assert call_args[0][1] == _GLINER_LABELS
        _reset_gliner_state()

    def test_label_mapping_to_uppercase(self):
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mock_model = self._mock_model()
        mod._gliner_model = mock_model
        mod._gliner_available = True

        entities = _extract_gliner_entities("some text", "u1")
        labels = [e.label for e in entities]
        assert "SKILL" in labels
        assert "CERTIFICATION" in labels
        assert "JOB_TITLE" in labels
        # No lowercase labels
        assert all(e.label == e.label.upper() for e in entities)
        _reset_gliner_state()

    def test_text_truncation(self):
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        mod._gliner_model = mock_model
        mod._gliner_available = True

        long_text = "x" * (MAX_GLINER_TEXT_CHARS + 5000)
        _extract_gliner_entities(long_text, "u1")

        call_args = mock_model.predict_entities.call_args
        passed_text = call_args[0][0]
        assert len(passed_text) == MAX_GLINER_TEXT_CHARS
        _reset_gliner_state()

    def test_empty_text_skipped(self):
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mock_model = MagicMock()
        mod._gliner_model = mock_model
        mod._gliner_available = True

        entities = _extract_gliner_entities("   ", "u1")
        assert entities == []
        mock_model.predict_entities.assert_not_called()
        _reset_gliner_state()


class TestGlinerIntegrationWithMainExtraction:
    """GLiNER entities integrated into extract_entities_and_facts."""

    def test_gliner_entities_appear_in_result(self):
        """GLiNER entities for PARAGRAPH units should be in final result."""
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Python", "label": "skill", "start": 30, "end": 36},
        ]
        mod._gliner_model = mock_model
        mod._gliner_available = True

        unit = SemanticUnit(
            unit_id="u1",
            unit_type=UnitType.PARAGRAPH,
            text="The candidate is proficient in Python and Java programming.",
            page_start=1,
            page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))

        # GLiNER SKILL entity should survive dedup and validation
        skill_ents = [e for e in result.entities if e.label == "SKILL"]
        assert len(skill_ents) >= 1
        assert any(e.source == "gliner" for e in skill_ents)
        _reset_gliner_state()

    def test_gliner_not_called_for_table_units(self):
        """GLiNER should NOT run on TABLE units."""
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        mod._gliner_model = mock_model
        mod._gliner_available = True

        unit = SemanticUnit(
            unit_id="u1",
            unit_type=UnitType.TABLE,
            text="Name | Age\nJohn | 30",
            page_start=1,
            page_end=1,
            table_rows=[{"Name": "John", "Age": "30"}],
            table_headers=["Name", "Age"],
        )
        extract_entities_and_facts(_make_doc([unit]))
        mock_model.predict_entities.assert_not_called()
        _reset_gliner_state()

    def test_deduplication_with_spacy_entities(self):
        """When spaCy and GLiNER find the same entity, dedup keeps higher confidence."""
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        # GLiNER finds "Google" as PRODUCT (0.80)
        # spaCy finds "Google" as ORG (0.85) -- different label, both should survive
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Google", "label": "product", "start": 35, "end": 41},
        ]
        mod._gliner_model = mock_model
        mod._gliner_available = True

        unit = SemanticUnit(
            unit_id="u1",
            unit_type=UnitType.PARAGRAPH,
            text="She is a software engineer at Google using Python daily.",
            page_start=1,
            page_end=1,
        )
        result = extract_entities_and_facts(_make_doc([unit]))

        # "Google" should appear with ORG label (spaCy) and PRODUCT label (GLiNER)
        google_ents = [e for e in result.entities if "google" in e.normalized]
        labels = {e.label for e in google_ents}
        # At minimum spaCy's ORG should be present
        assert "ORG" in labels
        # GLiNER's PRODUCT for Google should also be present (different label = no dedup)
        assert "PRODUCT" in labels
        _reset_gliner_state()

    def test_gliner_exception_handled(self):
        """If GLiNER model.predict_entities raises, extraction continues."""
        _reset_gliner_state()
        import src.docwain_intel.entity_engine as mod

        mock_model = MagicMock()
        mock_model.predict_entities.side_effect = RuntimeError("CUDA OOM")
        mod._gliner_model = mock_model
        mod._gliner_available = True

        entities = _extract_gliner_entities("some text here", "u1")
        assert entities == []
        _reset_gliner_state()
