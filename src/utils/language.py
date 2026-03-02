from __future__ import annotations

from typing import Tuple


def detect_language(text: str) -> Tuple[str, float]:
    if not text:
        return "unknown", 0.0
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return "unknown", 0.0
    ascii_letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    ratio = ascii_letters / max(letters, 1)
    if ratio >= 0.75:
        return "en", round(ratio, 3)
    if ratio <= 0.35:
        return "non_en", round(1 - ratio, 3)
    return "mixed", round(0.5 + abs(0.5 - ratio) / 2, 3)


__all__ = ["detect_language"]
