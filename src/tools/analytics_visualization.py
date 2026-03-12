"""Analytics visualization tool — generates charts from document data using matplotlib.

Provides functions to extract structured data from text and render various chart
types as base64-encoded PNG images for embedding in responses.
"""
from __future__ import annotations

import base64
import io
import json
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy matplotlib import (avoid import-time side effects)
# ---------------------------------------------------------------------------

_MPL_BACKEND_SET = False

def _ensure_mpl():
    """Set matplotlib to non-interactive backend (once)."""
    global _MPL_BACKEND_SET
    if not _MPL_BACKEND_SET:
        import matplotlib
        matplotlib.use("Agg")
        _MPL_BACKEND_SET = True

# ---------------------------------------------------------------------------
# Data extraction (LLM-based)
# ---------------------------------------------------------------------------

def extract_chart_data(
    text: str,
    query: str,
    llm_client: Any,
) -> List[Dict[str, Any]]:
    """Use LLM to extract structured {label, value} data from chunk text.

    Returns a list of dicts like [{"label": "Python", "value": 5}, ...].
    Falls back to regex extraction if LLM fails.
    """
    if not text:
        return []

    prompt = (
        f"Extract structured data from the following text that is relevant to "
        f"this query: \"{query}\"\n\n"
        f"Return ONLY a JSON array of objects with \"label\" and \"value\" keys.\n"
        f"The \"value\" must be a number. Example:\n"
        f'[{{"label": "Python", "value": 5}}, {{"label": "Java", "value": 3}}]\n\n'
        f"If the data has categories/names with counts/amounts/scores, use those.\n"
        f"If data is temporal, use dates/periods as labels and measurements as values.\n"
        f"Return at most 20 data points. Return [] if no chartable data exists.\n\n"
        f"Text:\n{text[:3000]}"
    )

    try:
        if hasattr(llm_client, "generate_with_metadata"):
            raw, _ = llm_client.generate_with_metadata(
                prompt,
                options={"temperature": 0.1, "num_predict": 1024, "num_ctx": 4096},
            )
        else:
            raw = llm_client.generate(prompt)

        return _parse_json_array(raw)
    except Exception as exc:
        logger.debug("LLM chart data extraction failed: %s", exc)

    # Fallback: regex extraction of number-label pairs
    return _regex_extract(text)

def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Extract and parse a JSON array from LLM output."""
    # Find JSON array in the response
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        if not isinstance(data, list):
            return []
        # Validate and sanitize entries
        result = []
        for item in data:
            if isinstance(item, dict) and "label" in item and "value" in item:
                try:
                    val = float(item["value"])
                    result.append({"label": str(item["label"]), "value": val})
                except (ValueError, TypeError):
                    continue
        return result
    except json.JSONDecodeError:
        return []

def _regex_extract(text: str) -> List[Dict[str, Any]]:
    """Simple regex fallback to find label-number pairs in text."""
    results: List[Dict[str, Any]] = []
    # Pattern: "Label: 123" or "Label - 123" or "Label (123)"
    for m in re.finditer(r'([A-Za-z][\w\s]{1,30})\s*[:–\-]\s*(\d+(?:\.\d+)?)', text):
        label = m.group(1).strip()
        value = float(m.group(2))
        results.append({"label": label, "value": value})
    return results[:20]

# ---------------------------------------------------------------------------
# Chart type selection
# ---------------------------------------------------------------------------

_CHART_KEYWORDS = {
    "pie": {"pie", "proportion", "percentage", "share", "breakdown", "composition"},
    "line": {"trend", "timeline", "over time", "progression", "temporal", "growth",
             "decline", "history", "chronological", "monthly", "yearly", "quarterly"},
    "histogram": {"distribution", "frequency", "histogram", "spread", "density"},
    "grouped_bar": {"compare", "comparison", "versus", "vs", "side by side", "grouped"},
    "bar": {"bar", "count", "total", "amount", "chart", "graph"},
}

def select_chart_type(query: str, data: List[Dict[str, Any]]) -> str:
    """Select the best chart type based on query keywords and data shape."""
    q = query.lower()

    # Check for explicit chart type mentions
    for chart_type, keywords in _CHART_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return chart_type

    # Heuristic: few categories (<=6) with all positive values → pie
    if len(data) <= 6 and all(d.get("value", 0) > 0 for d in data):
        return "pie"

    # Default
    return "bar"

# ---------------------------------------------------------------------------
# Chart generation functions
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return b64

def generate_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str = "",
    ylabel: str = "Value",
) -> str:
    """Generate a bar chart and return as base64-encoded PNG."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3([i / max(len(labels), 1) for i in range(len(labels))])
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:g}", ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    return _fig_to_base64(fig)

def generate_pie_chart(
    labels: List[str],
    values: List[float],
    title: str,
) -> str:
    """Generate a pie chart and return as base64-encoded PNG."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    # Filter out zero/negative values
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not filtered:
        filtered = list(zip(labels, values))
    pie_labels, pie_values = zip(*filtered) if filtered else (labels, values)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3([i / max(len(pie_labels), 1) for i in range(len(pie_labels))])
    wedges, texts, autotexts = ax.pie(
        pie_values, labels=pie_labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.85,
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return _fig_to_base64(fig)

def generate_line_chart(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str = "",
    ylabel: str = "Value",
) -> str:
    """Generate a line chart and return as base64-encoded PNG."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(labels)), values, marker="o", linewidth=2, color="#2196F3",
            markersize=6, markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(range(len(labels)), values, alpha=0.1, color="#2196F3")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)

def generate_grouped_bar_chart(
    data: List[Dict[str, Any]],
    title: str,
    xlabel: str = "",
    ylabel: str = "Value",
) -> str:
    """Generate a grouped bar chart for comparison data.

    Expects data with 'label' and 'value' keys. If data items have a 'group'
    key, uses it for grouping; otherwise treats as single group.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    # Group data
    groups: Dict[str, List] = {}
    for d in data:
        group = d.get("group", "Value")
        groups.setdefault(group, []).append(d)

    if len(groups) <= 1:
        # Single group — fall back to regular bar chart
        labels = [d.get("label", "") for d in data]
        values = [d.get("value", 0) for d in data]
        return generate_bar_chart(labels, values, title, xlabel, ylabel)

    # Multi-group
    group_names = list(groups.keys())
    all_labels = sorted({d.get("label", "") for d in data})
    n_groups = len(group_names)
    x = np.arange(len(all_labels))
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_groups))

    for i, gname in enumerate(group_names):
        gdata = {d.get("label", ""): d.get("value", 0) for d in groups[gname]}
        vals = [gdata.get(label, 0) for label in all_labels]
        ax.bar(x + i * width - (n_groups - 1) * width / 2, vals,
               width, label=gname, color=colors[i], edgecolor="gray", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)

def generate_histogram(
    values: List[float],
    title: str,
    xlabel: str = "Value",
    ylabel: str = "Frequency",
) -> str:
    """Generate a histogram and return as base64-encoded PNG."""
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = min(max(int(len(values) ** 0.5), 5), 30)
    ax.hist(values, bins=n_bins, color="#4CAF50", edgecolor="white", linewidth=1, alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    # Add statistics annotations
    if values:
        mean_val = np.mean(values)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.legend()

    fig.tight_layout()
    return _fig_to_base64(fig)
