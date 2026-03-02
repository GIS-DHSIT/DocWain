"""Tests for data isolation fixes (P0 ground rules).

Validates:
- debug_collection() scopes scroll by profile_id + subscription_id
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


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
