"""Verify Teams Q&A answers are plain text, not Adaptive Cards."""
import pytest


def test_answer_is_plain_text_not_card():
    """Q&A answers must be plain text messages, not Adaptive Card attachments."""
    from src.teams.tools import format_text_answer

    result = format_text_answer(
        response_text="The invoice total is $5,000.",
        sources=[
            {"source_name": "invoice.pdf", "excerpt": "Total: $5,000.00"},
        ],
        domain="invoice",
        grounded=True,
    )
    assert result["type"] == "message"
    assert "attachments" not in result or not result.get("attachments")
    assert "The invoice total is $5,000." in result["text"]
    assert "invoice.pdf" in result["text"]


def test_answer_includes_confidence_indicator():
    from src.teams.tools import format_text_answer
    result = format_text_answer(
        response_text="Answer here.",
        sources=[{"source_name": "doc.pdf"}, {"source_name": "doc2.pdf"}, {"source_name": "doc3.pdf"}],
        domain="contract",
        grounded=True,
    )
    assert "3 sources" in result["text"].lower() or "high confidence" in result["text"].lower()


def test_answer_no_sources_graceful():
    from src.teams.tools import format_text_answer
    result = format_text_answer(
        response_text="I could not find information.",
        sources=[],
        domain="",
        grounded=False,
    )
    assert result["type"] == "message"
    assert "I could not find information." in result["text"]


def test_processing_card_still_uses_card():
    """Status/progress indicators should remain as Adaptive Cards."""
    from src.teams.cards import build_card
    from src.teams.tools import _card_activity

    card = build_card("processing_card", status_message="Analyzing...")
    activity = _card_activity(card)
    assert activity.get("attachments")
    assert activity["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"
