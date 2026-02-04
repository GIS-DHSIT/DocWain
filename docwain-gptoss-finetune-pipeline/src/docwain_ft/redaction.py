import re

UUID_PATTERN = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
KEY_PATTERN = re.compile(r"\b(subscription_id|profile_id|chunk_id|embedding|payload_key|internal_id)\b", re.IGNORECASE)


def redact(text: str) -> str:
    text = UUID_PATTERN.sub("[REDACTED_ID]", text)
    text = KEY_PATTERN.sub("[REDACTED_ID]", text)
    return text
