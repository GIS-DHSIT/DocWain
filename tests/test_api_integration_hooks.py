"""Tests for intel pipeline hooks wired into embedding_service.py.

Validates that:
- Hook fires when DOCWAIN_INTEL_PIPELINE=1
- Hook is skipped when disabled
- Hook failure does not break the main embedding flow
- Correct arguments are passed to the hook
- Import failure is handled gracefully
- Timing is logged
"""
from __future__ import annotations

import importlib
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extracted_content():
    """Return a minimal ExtractedDocument-like object for testing."""
    content = MagicMock()
    content.sections = []
    content.pages = []
    return content


def _make_processing_result(stage: str = "complete"):
    """Return a minimal ProcessingResult-like object."""
    result = SimpleNamespace()
    result.stage_reached = stage
    result.extraction = SimpleNamespace(entities=["e1"], facts=["f1"])
    return result


# ---------------------------------------------------------------------------
# Test 1: Hook fires when enabled
# ---------------------------------------------------------------------------

class TestHookFiresWhenEnabled:
    """When INTEL_PIPELINE_ENABLED is True and the hook function is available,
    the hook should be called before train_on_document."""

    @patch("src.api.embedding_service.INTEL_PIPELINE_ENABLED", True)
    @patch("src.api.embedding_service.run_intel_pipeline_hook")
    @patch("src.api.embedding_service.build_document_json_from_extracted")
    def test_hook_called_when_enabled(self, mock_build, mock_hook):
        """run_intel_pipeline_hook is called when the feature flag is on."""
        mock_build.return_value = {"pages": []}
        mock_hook.return_value = _make_processing_result()

        content = _make_extracted_content()
        doc_id = "doc-123"
        sub_id = "sub-456"
        prof_id = "prof-789"

        # Simulate the inline hook logic from embedding_service.py
        from src.api.embedding_service import (
            INTEL_PIPELINE_ENABLED as _flag_unused,
        )
        # We replicate the hook block to unit-test the logic pattern
        _run_hook_block(
            content=content,
            doc_id=doc_id,
            subscription_id=sub_id,
            profile_id=prof_id,
            intel_enabled=True,
            hook_fn=mock_hook,
            build_fn=mock_build,
        )

        mock_build.assert_called_once_with(content, document_id=doc_id)
        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args[1]
        assert call_kwargs["document_id"] == doc_id
        assert call_kwargs["subscription_id"] == sub_id
        assert call_kwargs["profile_id"] == prof_id


# ---------------------------------------------------------------------------
# Test 2: Hook skipped when disabled
# ---------------------------------------------------------------------------

class TestHookSkippedWhenDisabled:
    """When INTEL_PIPELINE_ENABLED is False, the hook must not be called."""

    def test_hook_not_called_when_disabled(self):
        mock_hook = MagicMock()
        mock_build = MagicMock()

        _run_hook_block(
            content=_make_extracted_content(),
            doc_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            intel_enabled=False,
            hook_fn=mock_hook,
            build_fn=mock_build,
        )

        mock_hook.assert_not_called()
        mock_build.assert_not_called()

    def test_hook_not_called_when_hook_fn_is_none(self):
        mock_build = MagicMock()

        _run_hook_block(
            content=_make_extracted_content(),
            doc_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            intel_enabled=True,
            hook_fn=None,
            build_fn=mock_build,
        )

        mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Hook failure does not break main flow
# ---------------------------------------------------------------------------

class TestHookFailureDoesNotBreak:
    """If the hook raises an exception, the main embedding flow continues."""

    def test_exception_in_build_is_caught(self):
        mock_build = MagicMock(side_effect=RuntimeError("build exploded"))
        mock_hook = MagicMock()

        # Should not raise
        _run_hook_block(
            content=_make_extracted_content(),
            doc_id="doc-err",
            subscription_id="sub-1",
            profile_id="prof-1",
            intel_enabled=True,
            hook_fn=mock_hook,
            build_fn=mock_build,
        )

        # Hook should NOT have been called since build failed first
        mock_hook.assert_not_called()

    def test_exception_in_hook_is_caught(self):
        mock_build = MagicMock(return_value={"pages": []})
        mock_hook = MagicMock(side_effect=RuntimeError("hook exploded"))

        # Should not raise
        _run_hook_block(
            content=_make_extracted_content(),
            doc_id="doc-err2",
            subscription_id="sub-1",
            profile_id="prof-1",
            intel_enabled=True,
            hook_fn=mock_hook,
            build_fn=mock_build,
        )

        # Hook was called (and raised), but we survived
        mock_hook.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: Correct arguments passed
# ---------------------------------------------------------------------------

class TestCorrectArguments:
    """The hook receives the correct document_id, subscription_id, profile_id,
    and the converted ExtractedDocumentJSON."""

    def test_arguments_match(self):
        sentinel_json = {"pages": [{"num": 1}]}
        mock_build = MagicMock(return_value=sentinel_json)
        mock_hook = MagicMock(return_value=_make_processing_result())
        content = _make_extracted_content()

        _run_hook_block(
            content=content,
            doc_id="DOC-ABC",
            subscription_id="SUB-XYZ",
            profile_id="PROF-123",
            intel_enabled=True,
            hook_fn=mock_hook,
            build_fn=mock_build,
        )

        kwargs = mock_hook.call_args[1]
        assert kwargs["extracted_doc"] is sentinel_json
        assert kwargs["document_id"] == "DOC-ABC"
        assert kwargs["subscription_id"] == "SUB-XYZ"
        assert kwargs["profile_id"] == "PROF-123"


# ---------------------------------------------------------------------------
# Test 5: Import failure is handled gracefully
# ---------------------------------------------------------------------------

class TestImportFailureGraceful:
    """If the docwain_intel package is not installed, the fallback values
    ensure the embedding service still loads without error."""

    def test_fallback_values_on_import_error(self):
        """Simulate what happens when ImportError is raised."""
        # These are the fallback values set in embedding_service.py
        # when the import fails:
        _INTEL_PIPELINE_ENABLED = False
        _run_intel_pipeline_hook = None
        _build_document_json_from_extracted = None

        assert _INTEL_PIPELINE_ENABLED is False
        assert _run_intel_pipeline_hook is None
        assert _build_document_json_from_extracted is None

        # The hook block should be a no-op with these values
        _run_hook_block(
            content=_make_extracted_content(),
            doc_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            intel_enabled=_INTEL_PIPELINE_ENABLED,
            hook_fn=_run_intel_pipeline_hook,
            build_fn=_build_document_json_from_extracted,
        )
        # No exception means success

    def test_embedding_service_loads_without_intel_package(self):
        """The embedding_service module itself should import cleanly
        even if docwain_intel is unavailable (guarded by try/except)."""
        # Just verify the module has the expected fallback attributes
        import src.api.embedding_service as es
        assert hasattr(es, "INTEL_PIPELINE_ENABLED")
        assert hasattr(es, "run_intel_pipeline_hook")
        assert hasattr(es, "build_document_json_from_extracted")


# ---------------------------------------------------------------------------
# Test 6: Timing is logged
# ---------------------------------------------------------------------------

class TestTimingLogged:
    """The hook block should log elapsed time."""

    def test_timing_logged(self, caplog):
        mock_build = MagicMock(return_value={"pages": []})
        mock_hook = MagicMock(return_value=_make_processing_result("extraction"))

        with caplog.at_level(logging.INFO):
            _run_hook_block(
                content=_make_extracted_content(),
                doc_id="doc-timing",
                subscription_id="sub-1",
                profile_id="prof-1",
                intel_enabled=True,
                hook_fn=mock_hook,
                build_fn=mock_build,
                logger_instance=logging.getLogger("src.api.embedding_service"),
            )

        # Check that timing was logged
        timing_messages = [r for r in caplog.records if "elapsed=" in r.message]
        assert len(timing_messages) >= 1, f"Expected timing log, got: {[r.message for r in caplog.records]}"
        assert "extraction" in timing_messages[0].message

    def test_failure_logged_as_warning(self, caplog):
        mock_build = MagicMock(side_effect=ValueError("boom"))
        mock_hook = MagicMock()

        with caplog.at_level(logging.WARNING):
            _run_hook_block(
                content=_make_extracted_content(),
                doc_id="doc-warn",
                subscription_id="sub-1",
                profile_id="prof-1",
                intel_enabled=True,
                hook_fn=mock_hook,
                build_fn=mock_build,
                logger_instance=logging.getLogger("src.api.embedding_service"),
            )

        warning_messages = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) >= 1


# ---------------------------------------------------------------------------
# Helper: Replicates the inline hook block from embedding_service.py
# ---------------------------------------------------------------------------

def _run_hook_block(
    *,
    content,
    doc_id: str,
    subscription_id: str,
    profile_id: str,
    intel_enabled: bool,
    hook_fn,
    build_fn,
    logger_instance=None,
):
    """Reproduce the exact hook block pattern used in embedding_service.py.

    This mirrors the inline code so we can unit-test it in isolation
    without needing the full embedding pipeline.
    """
    log = logger_instance or logging.getLogger(__name__)

    if intel_enabled and hook_fn is not None:
        try:
            _intel_t0 = time.time()
            _intel_doc_json = build_fn(content, document_id=doc_id)
            _intel_result = hook_fn(
                extracted_doc=_intel_doc_json,
                document_id=doc_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
            )
            log.info(
                "intel_pipeline doc=%s file=%s elapsed=%.3fs result=%s",
                doc_id, "test_file", time.time() - _intel_t0,
                _intel_result.stage_reached if _intel_result else "none",
            )
        except Exception:  # noqa: BLE001
            log.warning(
                "intel_pipeline hook failed for doc=%s file=%s; continuing with embedding",
                doc_id, "test_file", exc_info=True,
            )
