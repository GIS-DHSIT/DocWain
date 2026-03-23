"""Chart type selection engine — picks the optimal chart for extracted data.

Pure rule-based logic (no LLM). Covers ~80% of common cases via heuristics
on data shape, type, and optional user hints from the query string.
"""

from __future__ import annotations

import logging
from typing import Dict

from src.visualization.data_extractor import ChartData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported chart types
# ---------------------------------------------------------------------------

CHART_TYPES: list[str] = [
    "bar",
    "horizontal_bar",
    "grouped_bar",
    "stacked_bar",
    "donut",
    "line",
    "multi_line",
    "area",
    "scatter",
    "heatmap",
    "waterfall",
    "treemap",
    "radar",
    "wordcloud",
    "gauge",
    "sankey",
]

_DEFAULT_CHART_TYPE = "bar"

# ---------------------------------------------------------------------------
# Chart configuration presets
# ---------------------------------------------------------------------------

_CHART_CONFIGS: Dict[str, Dict] = {
    "bar": {
        "orientation": "v",
        "show_values": True,
        "show_legend": False,
        "sort_values": True,
    },
    "horizontal_bar": {
        "orientation": "h",
        "show_values": True,
        "show_legend": False,
        "sort_values": True,
    },
    "grouped_bar": {
        "orientation": "v",
        "show_values": True,
        "show_legend": True,
        "sort_values": False,
    },
    "stacked_bar": {
        "orientation": "v",
        "show_values": False,
        "show_legend": True,
        "sort_values": False,
    },
    "donut": {
        "orientation": "v",
        "show_values": True,
        "show_legend": True,
        "sort_values": True,
    },
    "line": {
        "orientation": "v",
        "show_values": False,
        "show_legend": False,
        "sort_values": False,
    },
    "multi_line": {
        "orientation": "v",
        "show_values": False,
        "show_legend": True,
        "sort_values": False,
    },
    "area": {
        "orientation": "v",
        "show_values": False,
        "show_legend": False,
        "sort_values": False,
    },
    "scatter": {
        "orientation": "v",
        "show_values": False,
        "show_legend": False,
        "sort_values": False,
    },
    "heatmap": {
        "orientation": "v",
        "show_values": True,
        "show_legend": True,
        "sort_values": False,
    },
    "waterfall": {
        "orientation": "v",
        "show_values": True,
        "show_legend": False,
        "sort_values": False,
    },
    "treemap": {
        "orientation": "v",
        "show_values": True,
        "show_legend": False,
        "sort_values": True,
    },
    "radar": {
        "orientation": "v",
        "show_values": False,
        "show_legend": True,
        "sort_values": False,
    },
    "wordcloud": {
        "orientation": "v",
        "show_values": False,
        "show_legend": False,
        "sort_values": False,
    },
    "gauge": {
        "orientation": "v",
        "show_values": True,
        "show_legend": False,
        "sort_values": False,
    },
    "sankey": {
        "orientation": "v",
        "show_values": False,
        "show_legend": True,
        "sort_values": False,
    },
}

# ---------------------------------------------------------------------------
# Query hint keywords → chart type
# ---------------------------------------------------------------------------

import re as _re

# Each entry: (compiled regex, chart_type_or_None)
# None means "resolve dynamically" — handled in code below.
# Patterns require chart-type words to appear in a chart/visualization context
# (e.g., "line chart", "as a line", "show line graph") to avoid false positives
# on common English phrases like "line items" or "bar association".
_QUERY_HINTS: list[tuple[_re.Pattern, str | None]] = [
    (_re.compile(r"\b(?:pie|donut)\s*(?:chart|graph|diagram|plot)?\b", _re.IGNORECASE), "donut"),
    (_re.compile(r"\bline\s+(?:chart|graph|diagram|plot)\b", _re.IGNORECASE), "line"),
    (_re.compile(r"\b(?:as\s+a\s+)?(?:trend|time\s*series)\b", _re.IGNORECASE), "line"),
    (_re.compile(r"\bbar\s+(?:chart|graph|diagram|plot)\b", _re.IGNORECASE), None),
    (_re.compile(r"\bscatter\s+(?:chart|graph|diagram|plot)\b", _re.IGNORECASE), "scatter"),
    (_re.compile(r"\b(?:radar|spider)\s*(?:chart|graph|diagram|plot)?\b", _re.IGNORECASE), "radar"),
    (_re.compile(r"\bheatmap\b", _re.IGNORECASE), "heatmap"),
    (_re.compile(r"\bwaterfall\b", _re.IGNORECASE), "waterfall"),
    (_re.compile(r"\btreemap\b", _re.IGNORECASE), "treemap"),
    (_re.compile(r"\bgauge\b", _re.IGNORECASE), "gauge"),
    (_re.compile(r"\bsankey\b", _re.IGNORECASE), "sankey"),
    (_re.compile(r"\barea\s+(?:chart|graph|diagram|plot)\b", _re.IGNORECASE), "area"),
]


def _match_query_hint(query: str, has_secondary: bool) -> str | None:
    """Return a chart type if the user's query contains an explicit hint."""
    query_lower = query.lower()

    # Special two-word check
    if "word" in query_lower and "cloud" in query_lower:
        return "wordcloud"

    for pattern, chart_type in _QUERY_HINTS:
        if pattern.search(query):
            if chart_type is not None:
                return chart_type
            # Dynamic resolution for "bar"
            return "grouped_bar" if has_secondary else "bar"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_chart_type(data: ChartData, query: str = "") -> str:
    """Select the optimal chart type for *data*, optionally guided by *query*.

    Parameters
    ----------
    data:
        Extracted chart-ready data (labels, values, metadata).
    query:
        Original user query — may contain hints like "show as pie chart".

    Returns
    -------
    str
        One of :data:`CHART_TYPES`.
    """
    n = len(data.labels)
    has_secondary = len(data.secondary_values) > 0
    is_temporal = data.data_type == "temporal"
    all_positive = all(v >= 0 for v in data.values)
    is_percentage = data.unit == "%"

    logger.debug(
        "select_chart_type: n=%d has_secondary=%s is_temporal=%s "
        "all_positive=%s is_percentage=%s query=%r",
        n, has_secondary, is_temporal, all_positive, is_percentage, query,
    )

    # 1. Honour explicit user hints -----------------------------------------
    if query:
        hint = _match_query_hint(query, has_secondary)
        if hint is not None:
            logger.info("Chart type resolved via query hint: %s", hint)
            return hint

    # 2. Temporal data → line charts ----------------------------------------
    if is_temporal:
        chart = "multi_line" if has_secondary else "line"
        logger.info("Temporal data detected — selected %s", chart)
        return chart

    # 3. Multi-series → grouped bar -----------------------------------------
    if has_secondary:
        logger.info("Secondary series present — selected grouped_bar")
        return "grouped_bar"

    # 4. Proportions / percentages ------------------------------------------
    if all_positive and is_percentage:
        chart = "donut" if n <= 6 else "bar"
        logger.info("Percentage data (n=%d) — selected %s", n, chart)
        return chart

    if all_positive and n <= 6:
        logger.info("Small positive dataset (n=%d) — selected donut", n)
        return "donut"

    # 5. Size-based fallback ------------------------------------------------
    if n <= 12:
        logger.info("Medium dataset (n=%d) — selected bar", n)
        return "bar"

    logger.info("Large dataset (n=%d) — selected horizontal_bar", n)
    return "horizontal_bar"


def get_chart_config(chart_type: str) -> Dict:
    """Return rendering hints for *chart_type*.

    Parameters
    ----------
    chart_type:
        Must be one of :data:`CHART_TYPES`.

    Returns
    -------
    dict
        Keys: ``orientation`` (``"v"`` | ``"h"``), ``show_values``,
        ``show_legend``, ``sort_values``.
    """
    if chart_type not in _CHART_CONFIGS:
        logger.warning(
            "Unknown chart type %r — falling back to %r config",
            chart_type, _DEFAULT_CHART_TYPE,
        )
        return dict(_CHART_CONFIGS[_DEFAULT_CHART_TYPE])

    return dict(_CHART_CONFIGS[chart_type])
