"""Response enhancement orchestrator — wires visualization into DocWain's response pipeline.

This module provides `enhance_with_visualization()` which is called after
`normalize_answer()` in the response pipeline. It examines the text response,
determines if a chart adds value, extracts data, selects chart type, renders,
and appends to the response's media field.

Integration:
    answer = normalize_answer(answer)
    answer = enhance_with_visualization(answer, query, channel="web")
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Timeout for the entire visualization pipeline
VIZ_TIMEOUT_SECONDS = 5.0


def enhance_with_visualization(
    answer: Dict[str, Any],
    query: str,
    channel: str = "web",
) -> Dict[str, Any]:
    """Enhance a DocWain response with auto-generated charts if appropriate.

    This function NEVER modifies the text response. It only appends to
    the media[] field. Any failure returns the original answer unchanged.

    Args:
        answer: Normalized answer dict with at minimum {"response": str}
        query: Original user query
        channel: "web" for Plotly interactive, "teams" for matplotlib PNG

    Returns:
        The same answer dict, possibly with chart data appended to media[].
    """
    start = time.time()

    try:
        response_text = ""
        if isinstance(answer, dict):
            # Handle nested answer structure
            inner = answer.get("answer", answer)
            if isinstance(inner, dict):
                response_text = inner.get("response", "")
            elif isinstance(inner, str):
                response_text = inner
            else:
                response_text = str(answer.get("response", ""))
        else:
            return answer

        if not response_text:
            return answer

        # Step 1: Check if user explicitly requested a chart
        from src.visualization.chart_decision import is_user_triggered, should_generate_chart

        force_chart = is_user_triggered(query)

        # Step 2: Run decision engine (skip if user-triggered)
        if not force_chart:
            should_chart, reason = should_generate_chart(response_text, query)
            if not should_chart:
                logger.debug("Chart skipped: %s", reason)
                return answer
        else:
            logger.info("Chart forced by user query")

        # Timeout check
        if time.time() - start > VIZ_TIMEOUT_SECONDS:
            logger.warning("Visualization timeout before extraction")
            return answer

        # Step 3: Extract chartable data from response
        from src.visualization.data_extractor import extract_chart_data

        chart_data = extract_chart_data(response_text)

        if not chart_data or len(chart_data.labels) < 2:
            # Try word cloud for text-heavy responses
            if force_chart:
                return _try_wordcloud(answer, response_text, channel)
            logger.debug("Chart skipped: insufficient data points")
            return answer

        # Timeout check
        if time.time() - start > VIZ_TIMEOUT_SECONDS:
            logger.warning("Visualization timeout before rendering")
            return answer

        # Step 4: Select chart type
        from src.visualization.chart_selector import select_chart_type

        chart_type = select_chart_type(chart_data, query)
        logger.info("Selected chart type: %s for %d data points", chart_type, len(chart_data.labels))

        # Step 5: Handle word cloud separately
        if chart_type == "wordcloud":
            return _try_wordcloud(answer, response_text, channel)

        # Step 6: Render chart
        from src.visualization.chart_renderer import render_chart

        # Generate smart title
        title = _generate_title(chart_data, chart_type, query)

        rendered = render_chart(
            chart_type=chart_type,
            labels=chart_data.labels,
            values=chart_data.values,
            title=title,
            secondary_values=chart_data.secondary_values or None,
            secondary_name=chart_data.secondary_name,
            series_name=chart_data.series_name or "Value",
            unit=chart_data.unit,
            channel=channel,
        )

        if not rendered.matplotlib_png_base64 and not rendered.plotly_html:
            logger.warning("Chart rendering produced no output")
            return answer

        # Step 7: Append to media[]
        media_entry = {
            "type": "chart",
            "chart_type": rendered.chart_type,
            "title": rendered.title,
            "png_base64": rendered.matplotlib_png_base64,
            "data_summary": rendered.data_summary,
        }

        if channel == "web" and rendered.plotly_html:
            media_entry["plotly_html"] = rendered.plotly_html
            media_entry["plotly_json"] = rendered.plotly_json

        _append_media(answer, media_entry)

        elapsed = time.time() - start
        logger.info("Chart generated: %s (%s, %.1fs)", title, chart_type, elapsed)

        return answer

    except Exception as exc:
        logger.warning("Visualization enhancement failed (returning original): %s", exc)
        return answer


def _try_wordcloud(
    answer: Dict[str, Any],
    response_text: str,
    channel: str,
) -> Dict[str, Any]:
    """Attempt to generate a word cloud from the response text."""
    try:
        from src.visualization.wordcloud_renderer import extract_word_frequencies, render_wordcloud

        frequencies = extract_word_frequencies(response_text)
        if len(frequencies) < 5:
            return answer

        rendered = render_wordcloud(frequencies, title="Key Terms", channel=channel)
        if not rendered:
            return answer

        media_entry = {
            "type": "chart",
            "chart_type": "wordcloud",
            "title": rendered.title,
            "png_base64": rendered.matplotlib_png_base64,
            "data_summary": rendered.data_summary,
        }
        if channel == "web" and rendered.plotly_html:
            media_entry["plotly_html"] = rendered.plotly_html

        _append_media(answer, media_entry)
        logger.info("Word cloud generated: %d terms", len(frequencies))
        return answer

    except Exception as exc:
        logger.debug("Word cloud failed: %s", exc)
        return answer


def _append_media(answer: Dict[str, Any], media_entry: Dict[str, Any]) -> None:
    """Append a media entry to the answer's media list."""
    # Handle nested answer structure
    inner = answer.get("answer", answer)
    if isinstance(inner, dict):
        if "media" not in inner:
            inner["media"] = []
        inner["media"].append(media_entry)
    else:
        if "media" not in answer:
            answer["media"] = []
        answer["media"].append(media_entry)

    # Also set at top level for normalize_answer compatibility
    if "media" not in answer:
        answer["media"] = []
    if media_entry not in answer["media"]:
        answer["media"].append(media_entry)


def _generate_title(chart_data, chart_type: str, query: str) -> str:
    """Generate a smart, context-aware chart title."""
    labels = chart_data.labels
    unit = chart_data.unit
    n = len(labels)

    # Try to infer context from query
    query_lower = query.lower()

    if "expense" in query_lower or "cost" in query_lower:
        return f"Expense Breakdown ({n} Categories)"
    if "revenue" in query_lower or "income" in query_lower:
        return f"Revenue Analysis"
    if "candidate" in query_lower or "resume" in query_lower:
        return f"Candidate Comparison ({n} Candidates)"
    if "invoice" in query_lower:
        return f"Invoice Line Items ({n} Items)"
    if "compare" in query_lower or "comparison" in query_lower:
        return f"Comparison ({n} Items)"
    if "trend" in query_lower or "over time" in query_lower:
        return f"Trend Analysis"
    if "budget" in query_lower:
        return f"Budget Analysis"

    # Infer from data characteristics
    if chart_data.data_type == "temporal":
        return f"Trend Over Time ({n} Periods)"
    if unit == "%":
        return f"Distribution ({n} Categories)"
    if unit == "$":
        return f"Financial Breakdown ({n} Items)"
    if chart_type in ("donut", "pie"):
        return f"Distribution ({n} Segments)"
    if chart_type == "radar":
        return f"Profile Comparison ({n} Attributes)"

    # Fallback
    return f"Data Analysis ({n} Items)"
