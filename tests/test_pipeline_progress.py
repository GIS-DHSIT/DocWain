"""Verify pipeline sends clear stage progress indicators."""
import pytest


def test_stage_progress_card_renders():
    from src.teams.cards import build_card

    card = build_card(
        "stage_progress_card",
        step_indicator="2/4",
        stage_title="Screening Document",
        stage_detail="Checking for sensitive content...",
        progress_bar="[===>      ] 50%",
    )
    assert card["type"] == "AdaptiveCard"
    body_text = str(card)
    assert "2/4" in body_text
    assert "Screening Document" in body_text
