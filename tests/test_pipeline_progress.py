"""Verify pipeline sends clear stage progress indicators with in-place updates."""
import pytest


def test_stage_progress_card_renders():
    from src.teams.cards import build_card

    card = build_card(
        "stage_progress_card",
        step_indicator="2/3",
        stage_title="Screening Document",
        stage_detail="Checking for sensitive content...",
        progress_bar="[██████░░░░░░░░] 50%",
    )
    assert card["type"] == "AdaptiveCard"
    body_text = str(card)
    assert "2/3" in body_text
    assert "Screening Document" in body_text


def test_send_card_returns_activity_id():
    """_send_card must return the activity ID for in-place updates."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from src.teams.pipeline import _send_card

    mock_turn_context = MagicMock()
    mock_response = MagicMock()
    mock_response.id = "activity-123"
    mock_turn_context.send_activity = AsyncMock(return_value=mock_response)

    log = MagicMock()
    card = {"type": "AdaptiveCard", "body": []}

    result = asyncio.get_event_loop().run_until_complete(
        _send_card(mock_turn_context, card, "test", log)
    )
    assert result == "activity-123", "Must return activity ID for subsequent updates"


def test_update_card_updates_in_place():
    """_update_card must call update_activity with the activity ID."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from src.teams.pipeline import _update_card

    mock_turn_context = MagicMock()
    mock_turn_context.update_activity = AsyncMock(return_value=None)

    log = MagicMock()
    card = {"type": "AdaptiveCard", "body": []}

    result = asyncio.get_event_loop().run_until_complete(
        _update_card(mock_turn_context, "activity-123", card, "test", log)
    )
    assert result == "activity-123"
    mock_turn_context.update_activity.assert_called_once()
    # Verify the activity dict had the ID set
    call_args = mock_turn_context.update_activity.call_args
    activity = call_args[0][0]
    assert activity.get("id") == "activity-123" or getattr(activity, "id", None) == "activity-123"


def test_update_card_fallback_on_failure():
    """_update_card falls back to send if update_activity fails."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from src.teams.pipeline import _update_card

    mock_turn_context = MagicMock()
    mock_turn_context.update_activity = AsyncMock(side_effect=Exception("Not supported"))
    mock_turn_context.delete_activity = AsyncMock(return_value=None)

    mock_response = MagicMock()
    mock_response.id = "new-activity-456"
    mock_turn_context.send_activity = AsyncMock(return_value=mock_response)

    log = MagicMock()
    card = {"type": "AdaptiveCard", "body": []}

    result = asyncio.get_event_loop().run_until_complete(
        _update_card(mock_turn_context, "old-activity-123", card, "test", log)
    )
    # Should return the new activity ID from the fallback send
    assert result == "new-activity-456"
    mock_turn_context.send_activity.assert_called_once()


def test_progress_bar_all_stages():
    """Verify progress card renders for all 3 pipeline stages."""
    from src.teams.cards import build_card

    stages = [
        ("1/3", "Analyzing: test.pdf", "[██░░░░░░░░░░░░] 33%"),
        ("2/3", "Screening: test.pdf", "[██████░░░░░░░░] 50%"),
        ("3/3", "Embedding: test.pdf", "[██████████░░░░] 80%"),
    ]
    for step, title, bar in stages:
        card = build_card(
            "stage_progress_card",
            step_indicator=step,
            stage_title=title,
            stage_detail="Processing...",
            progress_bar=bar,
        )
        assert card["type"] == "AdaptiveCard"
        body_str = str(card)
        assert step in body_str
        assert title in body_str
