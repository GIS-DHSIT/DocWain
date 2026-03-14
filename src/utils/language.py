from __future__ import annotations

from typing import Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def detect_language(text: str) -> Tuple[str, float]:
    """Detect the language of *text* using ``langdetect``.

    Returns (iso_639_1_code, confidence) — e.g. ``("tr", 0.95)``.
    Falls back to a simple heuristic when ``langdetect`` is unavailable.
    """
    if not text or not text.strip():
        return "unknown", 0.0

    # Take a meaningful sample — langdetect works best with 50+ chars
    sample = text.strip()[:3000]
    if len(sample) < 10:
        return "unknown", 0.0

    # Primary: langdetect (Google's language detection port)
    try:
        from langdetect import detect_langs  # type: ignore[import-untyped]
        from langdetect.detector_factory import LangDetectException  # type: ignore[import-untyped]

        try:
            results = detect_langs(sample)
            if results:
                best = results[0]
                return str(best.lang), round(float(best.prob), 3)
        except LangDetectException:
            pass
    except ImportError:
        logger.debug("langdetect not installed; falling back to heuristic")

    # Fallback: ASCII ratio heuristic (catches CJK, Arabic, Cyrillic, etc.)
    letters = sum(1 for ch in sample if ch.isalpha())
    if letters == 0:
        return "unknown", 0.0
    ascii_letters = sum(1 for ch in sample if "a" <= ch.lower() <= "z")
    ratio = ascii_letters / max(letters, 1)
    if ratio >= 0.90:
        return "en", round(ratio, 3)
    if ratio <= 0.35:
        return "non_en", round(1 - ratio, 3)
    return "mixed", round(0.5 + abs(0.5 - ratio) / 2, 3)


__all__ = ["detect_language"]
