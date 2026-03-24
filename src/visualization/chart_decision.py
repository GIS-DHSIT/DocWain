"""Chart decision engine — determines whether a chart should be generated for a response."""

import re
from typing import Tuple

from src.utils.logging_utils import get_logger

logger = get_logger("chart_decision")

# ---------------------------------------------------------------------------
# Keyword detection
# ---------------------------------------------------------------------------

_CHART_KEYWORDS: list[str] = [
    "chart", "graph", "visualize", "visualise", "plot",
    "diagram", "bar chart", "pie chart", "line graph",
    "show me a", "create a chart", "draw a", "generate a chart",
    "visual", "breakdown chart", "trend graph", "comparison chart",
    "donut", "histogram", "scatter plot", "heatmap",
]


def is_user_triggered(query: str) -> bool:
    """Return True when the query explicitly asks for a visual."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _CHART_KEYWORDS)


# ---------------------------------------------------------------------------
# Suppression patterns (checked first — fast reject)
# ---------------------------------------------------------------------------

_GAP_PHRASES: list[str] = [
    "not found",
    "don't address",
    "no documents",
    "no relevant",
    "unable to find",
]

_GREETING_PATTERNS: re.Pattern[str] = re.compile(
    r"^(hello|hi |hey |i'?m docwain|greetings|welcome)",
    re.IGNORECASE,
)

_ANY_NUMBER: re.Pattern[str] = re.compile(r"\d")

_SINGLE_NUMBER_ONLY: re.Pattern[str] = re.compile(
    r"(?<!\d[\.,])\b\d[\d,.]*\b(?![\.,]\d)"
)


def _is_suppressed(response_text: str) -> Tuple[bool, str]:
    """Return (True, reason) when chart generation should be suppressed."""
    if len(response_text) < 100:
        return True, "response too short"

    if not _ANY_NUMBER.search(response_text):
        return True, "no numeric values found"

    text_lower = response_text.lower()
    for phrase in _GAP_PHRASES:
        if phrase in text_lower:
            return True, f"gap response detected ('{phrase}')"

    if _GREETING_PATTERNS.search(response_text):
        return True, "greeting / meta response"

    numbers = _SINGLE_NUMBER_ONLY.findall(response_text)
    if len(numbers) <= 1:
        return True, "only a single numeric value"

    return False, ""


# ---------------------------------------------------------------------------
# Trigger patterns
# ---------------------------------------------------------------------------

# Markdown table row: starts with |, contains at least one digit/currency
_TABLE_ROW: re.Pattern[str] = re.compile(
    r"^\|(?:[^|]*\|){2,}",  # at least 2 cells
    re.MULTILINE,
)
_TABLE_ROW_NUMERIC: re.Pattern[str] = re.compile(
    r"^\|.*[\d$€£¥].*\|",
    re.MULTILINE,
)

_PERCENTAGE: re.Pattern[str] = re.compile(r"\b\d+(?:\.\d+)?%")

_CURRENCY: re.Pattern[str] = re.compile(
    r"[$€£¥]\s?\d[\d,]*(?:\.\d{1,2})?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP)\b",
    re.IGNORECASE,
)
_BULLET_OR_LIST: re.Pattern[str] = re.compile(r"(?:^[\-\*•]|\d+\.)\s", re.MULTILINE)

_TEMPORAL_LABEL: re.Pattern[str] = re.compile(
    r"\b(?:Q[1-4]|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    r"|January|February|March|April|June|July|August|September"
    r"|October|November|December)\b",
    re.IGNORECASE,
)

_RANKED_SCORE: re.Pattern[str] = re.compile(
    r"(?:^[\-\*•]|\d+\.)\s.*\b\d+(?:\.\d+)?\s*/?\s*\d*",
    re.MULTILINE,
)


def _check_triggers(response_text: str) -> Tuple[bool, str]:
    """Return (True, reason) when a chart-worthy pattern is detected."""

    # 1. Markdown table with 3+ numeric rows
    numeric_rows = _TABLE_ROW_NUMERIC.findall(response_text)
    # Exclude separator rows (e.g. |---|---|)
    data_rows = [r for r in numeric_rows if not re.fullmatch(r"\|[\s\-:|]+\|", r.strip())]
    if len(data_rows) >= 3:
        return True, f"markdown table with {len(data_rows)} numeric rows"

    # 2. 2+ percentages
    pct_matches = _PERCENTAGE.findall(response_text)
    if len(pct_matches) >= 2:
        return True, f"{len(pct_matches)} percentage values detected"

    # 3. 2+ currency values in list/bullet context
    currency_matches = _CURRENCY.findall(response_text)
    if len(currency_matches) >= 2 and _BULLET_OR_LIST.search(response_text):
        return True, f"{len(currency_matches)} currency values in list context"

    # 4. Temporal data with numeric values
    temporal_labels = _TEMPORAL_LABEL.findall(response_text)
    if len(temporal_labels) >= 2 and len(_SINGLE_NUMBER_ONLY.findall(response_text)) >= 2:
        return True, f"temporal pattern ({len(temporal_labels)} period labels with values)"

    # 5. Ranked list with numeric scores
    ranked = _RANKED_SCORE.findall(response_text)
    if len(ranked) >= 3:
        return True, f"ranked list with {len(ranked)} scored items"

    return False, ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def should_generate_chart(response_text: str, query: str) -> Tuple[bool, str]:
    """Decide whether a chart should accompany the response.

    Args:
        response_text: The generated answer text.
        query: The original user query.

    Returns:
        (should_chart, reason) — a boolean decision and a human-readable
        explanation suitable for debug logging.
    """
    # Explicit user request always wins
    if is_user_triggered(query):
        logger.info("chart decision: user-triggered via query keywords")
        return True, "user explicitly requested a chart"

    # Fast-reject path
    suppressed, sup_reason = _is_suppressed(response_text)
    if suppressed:
        logger.debug("chart decision: suppressed — %s", sup_reason)
        return False, sup_reason

    # Trigger evaluation
    triggered, trig_reason = _check_triggers(response_text)
    if triggered:
        logger.info("chart decision: triggered — %s", trig_reason)
        return True, trig_reason

    logger.debug("chart decision: no trigger matched")
    return False, "no chart-worthy pattern detected"
