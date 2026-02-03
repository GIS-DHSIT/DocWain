from src.docwain_intel.sanitizer import sanitize_output


def test_sanitizer_removes_meta_and_ids():
    text = """
I analyzed the documents.
Citations:
- doc 1
- doc 2
UUID: 123e4567-e89b-12d3-a456-426614174000
subscription_id=abc123def456
internal url: http://internal.local/secret
/path/to/file.pdf
Answer line.
Answer line.
"""
    sanitized = sanitize_output(text)
    assert "I analyzed" not in sanitized
    assert "Citations" not in sanitized
    assert "123e4567" not in sanitized
    assert "subscription_id" in sanitized
    assert "abc123def456" not in sanitized
    assert "internal" not in sanitized
    assert "/path/to/file.pdf" not in sanitized
    assert sanitized.count("Answer line.") == 1
