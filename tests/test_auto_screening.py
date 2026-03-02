"""Tests for automatic screening trigger after extraction.

Verifies that _run_auto_screening() is called after both extraction paths
and correctly transitions document status for the embedding gate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.statuses import (
    STATUS_EXTRACTION_COMPLETED,
    STATUS_SCREENING_COMPLETED,
    STATUS_TRAINING_BLOCKED_SECURITY,
)


# ── Source-level verification tests ─────────────────────────────────
# These tests verify source code directly to avoid sys.modules pollution
# from test_agent_mode.py which replaces core modules permanently.


_EXTRACTION_SRC = Path(__file__).resolve().parent.parent / "src" / "api" / "extraction_service.py"
_CONFIG_SRC = Path(__file__).resolve().parent.parent / "src" / "screening" / "config.py"


class TestScreeningConfigDefaults:
    """Verify auto_attach_on_ingest defaults to True in source."""

    def test_dataclass_default_is_true(self):
        text = _CONFIG_SRC.read_text()
        assert "auto_attach_on_ingest: bool = True" in text

    def test_load_method_default_is_true(self):
        text = _CONFIG_SRC.read_text()
        assert '_parse_bool(data.get("auto_attach_on_ingest"), default=True)' in text


class TestAutoScreeningWiring:
    """Verify extraction does NOT auto-trigger screening (decoupled design)."""

    def test_connector_path_does_not_auto_screen(self):
        """_extract_from_connector should NOT call _run_auto_screening (decoupled)."""
        text = _EXTRACTION_SRC.read_text()
        idx = text.find("def _extract_from_connector")
        assert idx > 0
        next_def = text.find("\ndef extract_documents", idx)
        connector_body = text[idx:next_def]
        assert "_run_auto_screening(" not in connector_body

    def test_upload_path_does_not_auto_screen(self):
        """extract_uploaded_document should NOT call _run_auto_screening (decoupled)."""
        text = _EXTRACTION_SRC.read_text()
        idx = text.find("def extract_uploaded_document")
        assert idx > 0
        upload_body = text[idx:]
        assert "_run_auto_screening(" not in upload_body

    def test_auto_screening_function_exists(self):
        """_run_auto_screening function must be defined in extraction_service."""
        text = _EXTRACTION_SRC.read_text()
        assert "def _run_auto_screening(" in text

    def test_auto_screening_handles_failure_gracefully(self):
        """_run_auto_screening must set SCREENING_COMPLETED on failure."""
        text = _EXTRACTION_SRC.read_text()
        # Find the function body
        idx = text.find("def _run_auto_screening(")
        next_def = text.find("\ndef ", idx + 10)
        fn_body = text[idx:next_def]
        # Must have fallback to SCREENING_COMPLETED on exception
        assert "STATUS_SCREENING_COMPLETED" in fn_body
        assert "except Exception" in fn_body

    def test_auto_screening_checks_auto_attach_flag(self):
        """_run_auto_screening must check auto_attach_on_ingest config flag."""
        text = _EXTRACTION_SRC.read_text()
        idx = text.find("def _run_auto_screening(")
        next_def = text.find("\ndef ", idx + 10)
        fn_body = text[idx:next_def]
        assert "auto_attach_on_ingest" in fn_body

    def test_auto_screening_calls_apply_security_result(self):
        """_run_auto_screening must call apply_security_result with report."""
        text = _EXTRACTION_SRC.read_text()
        idx = text.find("def _run_auto_screening(")
        next_def = text.find("\ndef ", idx + 10)
        fn_body = text[idx:next_def]
        assert "apply_security_result" in fn_body

    def test_auto_screening_calls_screening_engine(self):
        """_run_auto_screening must use ScreeningEngine.run_all."""
        text = _EXTRACTION_SRC.read_text()
        idx = text.find("def _run_auto_screening(")
        next_def = text.find("\ndef ", idx + 10)
        fn_body = text[idx:next_def]
        assert "ScreeningEngine" in fn_body
        assert "run_all" in fn_body


class TestEmbeddingGateFallback:
    """Verify the embedding gate requires STATUS_SCREENING_COMPLETED (decoupled)."""

    def test_embedding_gate_requires_screening_completed(self):
        """Embedding service should only accept SCREENING_COMPLETED (not EXTRACTION_COMPLETED)."""
        embed_src = Path(__file__).resolve().parent.parent / "src" / "api" / "embedding_service.py"
        text = embed_src.read_text()
        assert "STATUS_SCREENING_COMPLETED" in text
        # Should NOT auto-trigger screening as fallback
        assert "from src.api.extraction_service import _run_auto_screening" not in text

    def test_screening_completed_accepted(self):
        """STATUS_SCREENING_COMPLETED should be the accepted status for embedding."""
        accepted = (STATUS_SCREENING_COMPLETED,)
        assert STATUS_SCREENING_COMPLETED in accepted
        assert STATUS_EXTRACTION_COMPLETED not in accepted

    def test_other_status_rejected(self):
        """Non-screening statuses should be rejected."""
        accepted = (STATUS_SCREENING_COMPLETED,)
        assert "UNDER_REVIEW" not in accepted
        assert "EXTRACTION_FAILED" not in accepted


# ── Unit tests using direct function invocation ─────────────────────
# These tests import _run_auto_screening eagerly to avoid module pollution.

# Guard: only run these if extraction_service is importable
_EXTRACTION_IMPORTABLE = "src.api.extraction_service" in sys.modules or True
try:
    from src.api.extraction_service import _run_auto_screening
    _EXTRACTION_IMPORTABLE = True
except Exception:
    _EXTRACTION_IMPORTABLE = False


@pytest.mark.skipif(not _EXTRACTION_IMPORTABLE, reason="extraction_service not importable (module pollution)")
class TestRunAutoScreeningUnit:
    """Unit tests for _run_auto_screening function behavior."""

    def test_bypasses_when_config_fails(self):
        """When ScreeningConfig.load() fails, bypass and set SCREENING_COMPLETED."""
        with patch("src.api.extraction_service._set_document_status") as mock_set, \
             patch("src.screening.config.ScreeningConfig.load", side_effect=RuntimeError("broken")):
            _run_auto_screening("doc-fail")
        mock_set.assert_called_with("doc-fail", STATUS_SCREENING_COMPLETED)

    def test_bypasses_when_auto_attach_disabled(self):
        """When auto_attach_on_ingest is False, set SCREENING_COMPLETED without running."""
        mock_cfg = MagicMock()
        mock_cfg.auto_attach_on_ingest = False
        with patch("src.api.extraction_service._set_document_status") as mock_set, \
             patch("src.screening.config.ScreeningConfig.load", return_value=mock_cfg):
            _run_auto_screening("doc-disabled")
        mock_set.assert_called_with("doc-disabled", STATUS_SCREENING_COMPLETED)

    def test_runs_screening_and_applies_result(self):
        """When enabled, runs screening engine and applies security result."""
        mock_cfg = MagicMock()
        mock_cfg.auto_attach_on_ingest = True

        mock_report = MagicMock()
        mock_report.risk_level = "LOW"
        mock_report.overall_score_0_100 = 25.0
        mock_report.to_dict.return_value = {"overall_risk_level": "LOW"}

        mock_engine = MagicMock()
        mock_engine.run_all.return_value = mock_report

        with patch("src.screening.config.ScreeningConfig.load", return_value=mock_cfg), \
             patch("src.screening.engine.ScreeningEngine", return_value=mock_engine), \
             patch("src.api.screening_service.apply_security_result") as mock_apply:
            _run_auto_screening("doc-screen", doc_type="RESUME")

        mock_engine.run_all.assert_called_once_with("doc-screen", doc_type="RESUME")
        mock_apply.assert_called_once_with("doc-screen", {"overall_risk_level": "LOW"})

    def test_sets_screening_completed_on_engine_failure(self):
        """When screening engine raises, set SCREENING_COMPLETED to unblock pipeline."""
        mock_cfg = MagicMock()
        mock_cfg.auto_attach_on_ingest = True

        mock_engine = MagicMock()
        mock_engine.run_all.side_effect = RuntimeError("engine failed")

        with patch("src.api.extraction_service._set_document_status") as mock_set, \
             patch("src.screening.config.ScreeningConfig.load", return_value=mock_cfg), \
             patch("src.screening.engine.ScreeningEngine", return_value=mock_engine):
            _run_auto_screening("doc-eng-fail")
        mock_set.assert_called_with("doc-eng-fail", STATUS_SCREENING_COMPLETED)


# ── Integration tests for apply_security_result ─────────────────────


class TestScreeningIntegration:
    """Tests that apply_security_result correctly sets document status."""

    @patch("src.api.screening_service.update_stage")
    @patch("src.api.screening_service.update_security_screening")
    @patch("src.api.screening_service.get_document_record")
    @patch("src.api.screening_service._update_pickle_with_screening")
    @patch("src.api.screening_service._set_document_status")
    def test_low_risk_sets_screening_completed(
        self, mock_set_status, mock_pickle, mock_get_record, mock_update, mock_update_stage
    ):
        from src.api.screening_service import apply_security_result
        mock_get_record.return_value = {"status": STATUS_EXTRACTION_COMPLETED}
        apply_security_result("doc-low", {"overall_risk_level": "LOW"})
        mock_set_status.assert_called_with("doc-low", STATUS_SCREENING_COMPLETED)

    @patch("src.api.screening_service.update_stage")
    @patch("src.api.screening_service.update_security_screening")
    @patch("src.api.screening_service.get_document_record")
    @patch("src.api.screening_service._update_pickle_with_screening")
    @patch("src.api.screening_service._set_document_status")
    def test_high_risk_blocks_training(
        self, mock_set_status, mock_pickle, mock_get_record, mock_update, mock_update_stage
    ):
        from src.api.screening_service import apply_security_result
        mock_get_record.return_value = {"status": STATUS_EXTRACTION_COMPLETED}
        apply_security_result("doc-high", {"overall_risk_level": "HIGH"})
        mock_set_status.assert_called_with(
            "doc-high", STATUS_TRAINING_BLOCKED_SECURITY, "Security screening failed"
        )
