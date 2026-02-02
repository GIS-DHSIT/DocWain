import pytest

from src.response.formatter import format_response_text, sanitize_leading_tokens, strip_internal_meta_blocks


def test_removes_leaked_routing_token_B_prefix():
    text = "B) DOCUMENT / INFORMATION Diagnosis: Stable Angina"
    cleaned = format_response_text(text)
    assert cleaned.startswith("DOCUMENT / INFORMATION"), cleaned
    assert "B) DOCUMENT / INFORMATION" not in cleaned


def test_does_not_remove_mid_text_B_parenthesis():
    text = "The lab value is Vitamin B)12 within range."
    cleaned = format_response_text(text)
    assert cleaned == text


def test_strips_intent_meta_block():
    text = (
        "Intent: A) META / PERSONA\n"
        "I'm DocWain and I will not introduce myself.\n\n"
        "DOCUMENT / INFORMATION\n"
        "- Diagnosis: Stable Angina"
    )
    cleaned = format_response_text(text)
    assert "Intent:" not in cleaned
    assert "META / PERSONA" not in cleaned
    assert "DOCUMENT / INFORMATION" in cleaned
