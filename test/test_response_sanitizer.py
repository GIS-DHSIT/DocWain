from src.security.response_sanitizer import sanitize_user_payload, sanitize_user_text


def test_sanitize_user_text_masks_ids_and_drops_internal_lines():
    text = (
        "Public UUID 123e4567-e89b-12d3-a456-426614174000 should be masked.\n"
        "document_id: 507f1f77bcf86cd799439011 should be removed.\n"
        "Normal sentence remains."
    )

    sanitized = sanitize_user_text(text)

    assert "123e4567-e89b-12d3-a456-426614174000" not in sanitized
    assert "507f1f77bcf86cd799439011" not in sanitized
    assert "document_id" not in sanitized.lower()
    assert "Normal sentence remains." in sanitized
    assert "[REDACTED]" in sanitized


def test_sanitize_user_payload_drops_internal_keys_recursively():
    payload = {
        "document_id": "507f1f77bcf86cd799439011",
        "message": "Point 123e4567-e89b-12d3-a456-426614174000 referenced.",
        "nested": {
            "chunk_id": "abc123",
            "note": "All good here.",
        },
    }

    sanitized = sanitize_user_payload(payload)

    assert "document_id" not in sanitized
    assert "chunk_id" not in sanitized.get("nested", {})
    assert "123e4567-e89b-12d3-a456-426614174000" not in sanitized["message"]
    assert "[REDACTED]" in sanitized["message"]
    assert sanitized["nested"]["note"] == "All good here."
