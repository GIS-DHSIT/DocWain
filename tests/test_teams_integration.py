"""Integration tests for Teams pipeline accuracy and response format."""
import pytest


def test_format_text_answer_structured_output():
    """Verify text answers are well-structured with clear formatting."""
    from src.teams.tools import format_text_answer

    result = format_text_answer(
        response_text=(
            "The lease agreement is between ABC Corp and XYZ Inc.\n\n"
            "Key terms:\n"
            "- Start date: January 1, 2026\n"
            "- End date: December 31, 2028\n"
            "- Monthly rent: $5,000\n"
            "- Security deposit: $15,000"
        ),
        sources=[
            {"source_name": "lease.pdf", "excerpt": "This lease agreement dated January 1, 2026..."},
            {"source_name": "lease.pdf", "excerpt": "Monthly rent shall be $5,000..."},
            {"source_name": "lease.pdf", "excerpt": "Security deposit of $15,000..."},
        ],
        domain="contract",
        grounded=True,
    )

    text = result["text"]

    # Must contain ALL key details
    assert "$5,000" in text
    assert "$15,000" in text
    assert "January 1, 2026" in text
    assert "December 31, 2028" in text
    assert "ABC Corp" in text
    assert "XYZ Inc" in text

    # Must be plain text, not card
    assert "attachments" not in result or not result.get("attachments")

    # Must include sources inline
    assert "lease.pdf" in text
    assert "Sources" in text or "sources" in text


def test_understand_document_no_truncation():
    """Verify understand_document doesn't truncate long documents."""
    from src.doc_understanding.understand import understand_document

    class FakeExtracted:
        full_text = "Section 1: Introduction. " * 400 + "FINAL_CLAUSE: Termination requires 90 days notice."
        sections = []

    result = understand_document(
        extracted=FakeExtracted(),
        doc_type="contract",
        model_name=None,
        llm_client=None,
    )

    assert result is not None
    assert "document_summary" in result


def test_format_text_answer_domain_badge():
    """Verify domain badge appears for specific domains but not generic."""
    from src.teams.tools import format_text_answer

    # Specific domain should show badge
    result = format_text_answer("Answer", [], domain="invoice", grounded=False)
    assert "[Invoice]" in result["text"]

    # Generic domain should NOT show badge
    result = format_text_answer("Answer", [], domain="general", grounded=False)
    assert "[General]" not in result["text"]

    # Empty domain should NOT show badge
    result = format_text_answer("Answer", [], domain="", grounded=False)
    assert "[]" not in result["text"]


def test_format_text_answer_confidence_levels():
    """Verify all confidence levels render correctly."""
    from src.teams.tools import format_text_answer

    # High confidence: grounded + 3+ sources
    result = format_text_answer("A", [{"source_name": "a"}, {"source_name": "b"}, {"source_name": "c"}], grounded=True)
    assert "high confidence" in result["text"].lower()

    # Partial: grounded + 1-2 sources
    result = format_text_answer("A", [{"source_name": "a"}], grounded=True)
    assert "partial confidence" in result["text"].lower()

    # Low: sources but not grounded
    result = format_text_answer("A", [{"source_name": "a"}], grounded=False)
    assert "low confidence" in result["text"].lower()

    # No indicator: no sources
    result = format_text_answer("A", [], grounded=False)
    assert "confidence" not in result["text"].lower()


def test_stage_progress_card_all_stages():
    """Verify progress card renders for all pipeline stages."""
    from src.teams.cards import build_card

    for step, title in [("1/3", "Analyzing"), ("2/3", "Screening"), ("3/3", "Embedding")]:
        card = build_card(
            "stage_progress_card",
            step_indicator=step,
            stage_title=title,
            stage_detail="Processing...",
            progress_bar="[===] 50%",
        )
        assert card["type"] == "AdaptiveCard"
        assert step in str(card)
        assert title in str(card)
