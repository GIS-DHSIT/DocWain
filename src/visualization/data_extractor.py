"""Extract chartable structured data from DocWain response text.

Layered strategy: markdown tables → bullet lists → inline bold numbers.
Pure regex, no LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChartData:
    labels: List[str]
    values: List[float]
    series_name: str = ""
    secondary_values: List[float] = field(default_factory=list)
    secondary_name: str = ""
    data_type: str = "nominal"     # nominal | temporal | ordinal
    unit: str = ""                 # "$", "%", "years", etc.


# ---------------------------------------------------------------------------
# Number / unit helpers
# ---------------------------------------------------------------------------

_SUFFIX_MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

_NUMBER_RE = re.compile(
    r"[$(€£]?\s*"              # optional currency symbol
    r"(-?\s*\d[\d,]*\.?\d*)"   # digits with optional negative sign and commas
    r"\s*([kmbt])?"            # optional suffix
    r"\s*%?",                  # optional percent
    re.IGNORECASE,
)


def _parse_number(text: str) -> Optional[float]:
    """Parse a human-formatted number string into a float.

    Handles currency symbols, commas, K/M/B suffixes, and percentages.
    """
    text = text.strip()
    if not text:
        return None

    # Strip markdown bold markers
    text = text.replace("**", "")

    match = _NUMBER_RE.search(text)
    if not match:
        return None

    digits_str = match.group(1).replace(",", "").replace(" ", "")
    try:
        value = float(digits_str)
    except ValueError:
        return None

    suffix = (match.group(2) or "").lower()
    if suffix in _SUFFIX_MULTIPLIERS:
        value *= _SUFFIX_MULTIPLIERS[suffix]

    return value


def _detect_unit(values_text: List[str]) -> str:
    """Detect the most common unit from a list of raw value strings."""
    if not values_text:
        return ""

    dollar_count = sum(1 for v in values_text if "$" in v)
    pct_count = sum(1 for v in values_text if "%" in v)
    euro_count = sum(1 for v in values_text if "€" in v)
    pound_count = sum(1 for v in values_text if "£" in v)

    threshold = len(values_text) / 2
    if dollar_count > threshold:
        return "$"
    if pct_count > threshold:
        return "%"
    if euro_count > threshold:
        return "€"
    if pound_count > threshold:
        return "£"
    return ""


_TEMPORAL_PATTERNS = re.compile(
    r"^("
    r"Q[1-4]|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)|"
    r"\d{4}|"                           # bare year
    r"\d{4}[-/]\d{2}(?:[-/]\d{2})?|"   # 2024-01 or 2024-01-15
    r"(?:FY|H[12])\s*\d{2,4}"          # FY2024, H1 2025
    r")",
    re.IGNORECASE,
)


def _is_temporal(labels: List[str]) -> bool:
    """Return True when the majority of labels look like time periods."""
    if not labels:
        return False
    matches = sum(1 for lb in labels if _TEMPORAL_PATTERNS.search(lb.strip()))
    return matches > len(labels) / 2


# ---------------------------------------------------------------------------
# Layer 1 — Markdown table parsing
# ---------------------------------------------------------------------------

def _try_markdown_table(text: str) -> Optional[ChartData]:
    """Extract chart data from the first markdown table found."""

    # Find table blocks: at least a header row, a separator, and one data row.
    table_re = re.compile(
        r"^(\|.+\|)\s*\n"       # header row
        r"(\|[\s:|-]+\|)\s*\n"  # separator row
        r"((?:\|.+\|\s*\n?)+)", # data rows
        re.MULTILINE,
    )
    match = table_re.search(text)
    if not match:
        return None

    header_line = match.group(1)
    data_block = match.group(3)

    headers = [h.strip() for h in header_line.strip("|").split("|")]
    rows_raw = [
        line for line in data_block.strip().splitlines() if line.strip()
    ]
    rows: List[List[str]] = []
    for row_line in rows_raw:
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(cells)

    if not rows or len(headers) < 2:
        return None

    # Identify numeric columns by checking if >50% of cells are purely numeric.
    # A cell is "purely numeric" when stripping currency/percent/bold leaves
    # only digits, commas, dots, minus, and whitespace.
    _NUMERIC_CELL_RE = re.compile(
        r"^[$(€£)%*\s]*-?\s*\d[\d,]*\.?\d*\s*[kmbtKMBT]?\s*[%$(€£)*]*$"
    )
    numeric_cols: List[int] = []
    for col_idx in range(len(headers)):
        numeric_count = sum(
            1 for r in rows if _NUMERIC_CELL_RE.match(r[col_idx].strip())
        )
        if numeric_count > len(rows) / 2:
            numeric_cols.append(col_idx)

    if not numeric_cols:
        logger.debug("Markdown table found but no numeric columns detected.")
        return None

    # Label column = first non-numeric column (prefer index 0).
    label_col = 0
    if label_col in numeric_cols:
        non_numeric = [i for i in range(len(headers)) if i not in numeric_cols]
        label_col = non_numeric[0] if non_numeric else 0

    # Filter out summary/total rows that would distort the chart
    _SUMMARY_LABELS = {"total", "subtotal", "sub-total", "grand total", "sum", "net", "overall"}
    filtered_rows = [
        r for r in rows
        if r[label_col].replace("**", "").strip().lower() not in _SUMMARY_LABELS
    ]
    if not filtered_rows:
        filtered_rows = rows  # fall back if ALL rows are "total"-like

    labels = [r[label_col].replace("**", "").strip() for r in filtered_rows]

    val_col = numeric_cols[0]
    raw_vals = [r[val_col] for r in filtered_rows]
    values = [_parse_number(v) or 0.0 for v in raw_vals]

    series_name = headers[val_col].replace("**", "").strip()
    unit = _detect_unit(raw_vals)

    secondary_values: List[float] = []
    secondary_name = ""
    if len(numeric_cols) >= 2:
        sec_col = numeric_cols[1]
        secondary_values = [_parse_number(r[sec_col]) or 0.0 for r in filtered_rows]
        secondary_name = headers[sec_col].replace("**", "").strip()

    data_type = "temporal" if _is_temporal(labels) else "nominal"

    logger.info(
        "Extracted chart data from markdown table: %d rows, series=%r",
        len(labels),
        series_name,
    )
    return ChartData(
        labels=labels,
        values=values,
        series_name=series_name,
        secondary_values=secondary_values,
        secondary_name=secondary_name,
        data_type=data_type,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# Layer 2 — Bullet / list extraction
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(
    r"^[\s]*[-*•]\s+"                  # bullet marker
    r"\*{0,2}([^*:]+?)\*{0,2}"        # label (possibly bold)
    r"[:\s]+\s*"                       # separator
    r"(.+)$",                          # value portion
    re.MULTILINE,
)


def _try_bullet_list(text: str) -> Optional[ChartData]:
    """Extract data from bullet-point lists like ``- **Label:** $value``."""

    matches = _BULLET_RE.findall(text)
    if len(matches) < 2:
        return None

    labels: List[str] = []
    values: List[float] = []
    raw_vals: List[str] = []

    for label_raw, value_raw in matches:
        num = _parse_number(value_raw)
        if num is None:
            continue
        labels.append(label_raw.strip())
        values.append(num)
        raw_vals.append(value_raw.strip())

    if len(labels) < 2:
        return None

    unit = _detect_unit(raw_vals)
    data_type = "temporal" if _is_temporal(labels) else "nominal"

    logger.info("Extracted chart data from bullet list: %d items", len(labels))
    return ChartData(
        labels=labels,
        values=values,
        data_type=data_type,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# Layer 3 — Inline bold number extraction
# ---------------------------------------------------------------------------

_INLINE_BOLD_RE = re.compile(
    r"\*\*([^*]+?)\*\*"   # bold span
)


def _try_inline_numbers(text: str) -> Optional[ChartData]:
    """Extract sequences of bold numeric values with surrounding context."""

    bold_spans = _INLINE_BOLD_RE.findall(text)
    if not bold_spans:
        return None

    entries: List[Tuple[str, float, str]] = []
    for span in bold_spans:
        num = _parse_number(span)
        if num is None:
            continue

        # Grab surrounding context as label (up to 40 chars after the bold).
        esc = re.escape(f"**{span}**")
        ctx_match = re.search(
            rf"{esc}\s*(?:in|for|during|—|–|-)?\s*([A-Za-z0-9 /,]+)",
            text,
        )
        label = ctx_match.group(1).strip() if ctx_match else f"Value {len(entries) + 1}"
        entries.append((label, num, span))

    if len(entries) < 2:
        # Fallback: try **Label** ... number pattern (e.g., "**Alice** has 7 years")
        fallback_entries = []
        for m in re.finditer(
            r"\*\*([A-Za-z][^*]{1,30})\*\*[^*\n]{0,30}?(\d+(?:\.\d+)?)",
            text,
        ):
            label = m.group(1).strip()
            num = _parse_number(m.group(2))
            if num is not None and num > 0:
                fallback_entries.append((label, num, m.group(2)))
        if len(fallback_entries) >= 2:
            entries = fallback_entries

    if len(entries) < 2:
        return None

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    raw_vals = [e[2] for e in entries]

    unit = _detect_unit(raw_vals)
    data_type = "temporal" if _is_temporal(labels) else "nominal"

    logger.info("Extracted chart data from inline bold numbers: %d items", len(labels))
    return ChartData(
        labels=labels,
        values=values,
        data_type=data_type,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_chart_data(response_text: str) -> Optional[ChartData]:
    """Extract chartable data from a DocWain response.

    Tries three extraction layers in order and returns the first success:
    1. Markdown table parsing
    2. Bullet / list extraction
    3. Inline bold number extraction

    Returns ``None`` when no chartable data is found.
    """
    if not response_text or not response_text.strip():
        logger.debug("Empty response text — skipping chart extraction.")
        return None

    for strategy_name, strategy_fn in (
        ("markdown_table", _try_markdown_table),
        ("bullet_list", _try_bullet_list),
        ("inline_numbers", _try_inline_numbers),
    ):
        try:
            result = strategy_fn(response_text)
            if result is not None:
                logger.info("Chart data extracted via %s strategy.", strategy_name)
                return result
        except Exception:
            logger.exception("Error in %s extraction strategy.", strategy_name)

    logger.debug("No chartable data found in response text.")
    return None
