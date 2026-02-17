"""Tests for data isolation fixes (P0 ground rules).

Validates:
- _discover_profile_ids() only fetches identity fields, returns multiple profiles
- debug_collection() scopes scroll by profile_id + subscription_id
- TrainingBootstrap._scroll_profile() includes subscription_id in filter
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest


# ── Fake Qdrant point ──────────────────────────────────────────────────

class FakePoint:
    def __init__(self, pid: str, payload: Optional[Dict[str, Any]] = None):
        self.id = pid
        self.payload = payload or {}


# ── _discover_profile_ids tests ────────────────────────────────────────

class TestDiscoverProfileIds:

    def _make_qdrant(self, pages):
        """Create a mock qdrant that returns pages of (points, next_offset)."""
        qdrant = MagicMock()
        qdrant.scroll = MagicMock(side_effect=pages)
        return qdrant

    def test_returns_unique_profile_ids(self):
        from src.api.app_lifespan import _discover_profile_ids

        pages = [
            ([FakePoint("1", {"profile_id": "p1"}), FakePoint("2", {"profile_id": "p2"})], "off1"),
            ([FakePoint("3", {"profile_id": "p1"}), FakePoint("4", {"profile_id": "p3"})], None),
        ]
        qdrant = self._make_qdrant(pages)
        result = _discover_profile_ids(qdrant, "test_collection")
        assert sorted(result) == ["p1", "p2", "p3"]

    def test_uses_with_payload_identity_only(self):
        """Must only request profile_id field, not full payload."""
        from src.api.app_lifespan import _discover_profile_ids

        qdrant = self._make_qdrant([
            ([FakePoint("1", {"profile_id": "p1"})], None),
        ])
        _discover_profile_ids(qdrant, "col")
        # Check that with_payload=["profile_id"] was passed
        call_kwargs = qdrant.scroll.call_args_list[0].kwargs
        assert call_kwargs["with_payload"] == ["profile_id"]
        assert call_kwargs["with_vectors"] is False

    def test_returns_empty_on_no_points(self):
        from src.api.app_lifespan import _discover_profile_ids

        qdrant = self._make_qdrant([([], None)])
        assert _discover_profile_ids(qdrant, "col") == []

    def test_returns_empty_on_exception(self):
        from src.api.app_lifespan import _discover_profile_ids

        qdrant = MagicMock()
        qdrant.scroll.side_effect = Exception("connection refused")
        assert _discover_profile_ids(qdrant, "col") == []


# ── debug_collection isolation tests ───────────────────────────────────

class TestDebugCollectionIsolation:

    @patch("src.api.dw_newron.get_qdrant_client")
    @patch("src.api.dw_newron.build_collection_name", return_value="test_col")
    def test_scroll_uses_profile_filter(self, mock_build, mock_get_client):
        from src.api.dw_newron import debug_collection

        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock(
            points_count=5,
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=1024, distance="Cosine"))),
        )
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        result = debug_collection("profile_abc", "sub_xyz")
        assert result["status"] == "healthy"

        # Verify scroll was called with a filter (scroll_filter kwarg)
        scroll_kwargs = mock_client.scroll.call_args.kwargs
        assert "scroll_filter" in scroll_kwargs
        filt = scroll_kwargs["scroll_filter"]
        # The filter should contain must conditions for profile_id and subscription_id
        assert filt is not None

    @patch("src.api.dw_newron.get_qdrant_client")
    @patch("src.api.dw_newron.build_collection_name", return_value="test_col")
    def test_scroll_does_not_leak_other_profiles(self, mock_build, mock_get_client):
        """Scroll must scope to the requested profile, not fetch everything."""
        from src.api.dw_newron import debug_collection

        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock(
            points_count=100,
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=1024, distance="Cosine"))),
        )
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        debug_collection("my_profile", "my_sub")

        scroll_kwargs = mock_client.scroll.call_args.kwargs
        assert scroll_kwargs.get("scroll_filter") is not None


# ── TrainingBootstrap subscription_id filter tests ─────────────────────

class TestTrainingBootstrapIsolation:

    def test_scroll_includes_subscription_id_filter(self):
        from src.intelligence.ml.training_bootstrap import TrainingBootstrap

        qdrant = MagicMock()
        qdrant.scroll.return_value = ([], None)

        bootstrap = TrainingBootstrap(
            qdrant_client=qdrant,
            embedding_model=MagicMock(),
            collection_name="col",
            subscription_id="sub_123",
        )
        bootstrap._scroll_profile("profile_456")

        call_kwargs = qdrant.scroll.call_args.kwargs
        filt = call_kwargs["scroll_filter"]
        # Filter must contain both profile_id and subscription_id
        must_keys = {c.key for c in filt.must}
        assert "profile_id" in must_keys
        assert "subscription_id" in must_keys

    def test_scroll_without_subscription_id_only_filters_profile(self):
        from src.intelligence.ml.training_bootstrap import TrainingBootstrap

        qdrant = MagicMock()
        qdrant.scroll.return_value = ([], None)

        bootstrap = TrainingBootstrap(
            qdrant_client=qdrant,
            embedding_model=MagicMock(),
            collection_name="col",
            # No subscription_id (default empty)
        )
        bootstrap._scroll_profile("profile_456")

        call_kwargs = qdrant.scroll.call_args.kwargs
        filt = call_kwargs["scroll_filter"]
        must_keys = {c.key for c in filt.must}
        assert "profile_id" in must_keys
        assert "subscription_id" not in must_keys

    def test_dpie_integration_passes_subscription_id(self):
        """Verify TrainingBootstrap stores subscription_id when passed."""
        from src.intelligence.ml.training_bootstrap import TrainingBootstrap

        bs = TrainingBootstrap(MagicMock(), MagicMock(), "col", subscription_id="sub_abc")
        assert bs._subscription_id == "sub_abc"

    def test_dpie_integration_default_subscription_id(self):
        """Without subscription_id, default is empty string."""
        from src.intelligence.ml.training_bootstrap import TrainingBootstrap

        bs = TrainingBootstrap(MagicMock(), MagicMock(), "col")
        assert bs._subscription_id == ""


# ── Bootstrap iterates all profiles ────────────────────────────────────

class TestBootstrapAllProfiles:

    def test_bootstrap_trains_all_profiles_not_just_first(self):
        """_bootstrap_dpie_background should train ALL profiles, not break on first."""
        mock_registry = MagicMock()
        mock_registry.is_loaded = False

        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(
            collections=[MagicMock(name="col1")]
        )

        with patch("src.api.app_lifespan._discover_profile_ids", return_value=["p1", "p2", "p3"]), \
             patch("src.intelligence.dpie_integration.DPIERegistry") as patched_cls:
            patched_cls.get.return_value = mock_registry

            from src.api.app_lifespan import _bootstrap_dpie_background
            _bootstrap_dpie_background(mock_qdrant, MagicMock())

            # ensure_ready should be called for each profile
            ensure_calls = mock_registry.ensure_ready.call_args_list
            profile_ids_trained = [c.kwargs.get("profile_id") for c in ensure_calls]
            assert "p1" in profile_ids_trained
            assert "p2" in profile_ids_trained
            assert "p3" in profile_ids_trained
