from src.rag.format_enforcer import enforce_response_formatting


def test_format_enforcer_normalizes_bullets():
    text = "\u2022 Item one\n\u2013 Item two\n1) Item three\n"
    formatted = enforce_response_formatting(text=text)
    assert "\u2022" not in formatted
    assert "\u2013" not in formatted
    assert "1. Item three" in formatted
    assert formatted.splitlines()[0].startswith("- ")
    assert formatted.splitlines()[1].startswith("- ")
