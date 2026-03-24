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
# First render is slower due to matplotlib font cache initialization (~6s)
VIZ_TIMEOUT_SECONDS = 15.0


def enhance_with_visualization(
    answer: Dict[str, Any],
    query: str,
    channel: str = "web",
) -> Dict[str, Any]:
    """Enhance a DocWain response with auto-generated charts if appropriate.

    When the user explicitly requests a chart and one cannot be generated,
    a validation note explaining why is appended to the response.
    For auto-detected opportunities, charts are silently skipped when not viable.

    Args:
        answer: Normalized answer dict with at minimum {"response": str}
        query: Original user query
        channel: "web" for Plotly interactive, "teams" for matplotlib PNG

    Returns:
        The same answer dict, possibly with chart data appended to media[].
    """
    start = time.time()
    from src.visualization.chart_decision import is_user_triggered
    force_chart = is_user_triggered(query) if query else False

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

        # Step 0: Render relationship graphs — triggered by either:
        #   (a) user query asks for a graph/diagram/relationship, OR
        #   (b) LLM response contains Mermaid code or a relationship table
        try:
            from src.visualization.graph_renderer import (
                try_graph_rendering, is_graph_query,
                parse_mermaid, extract_relationship_table, extract_relationships,
                render_graph,
            )

            # Determine if we should attempt graph rendering
            user_wants_graph = query and is_graph_query(query)
            response_has_mermaid = "```mermaid" in response_text or "graph TD" in response_text or "graph LR" in response_text
            response_has_rel_table = _has_relationship_table(response_text)

            if user_wants_graph or response_has_mermaid or response_has_rel_table:
                # Try parsing graph data from the response
                graph_data = parse_mermaid(response_text)
                if graph_data is None and response_has_rel_table:
                    graph_data = extract_relationship_table(response_text)
                if graph_data is None:
                    graph_data = extract_relationships(response_text)

                if graph_data and graph_data.edges:
                    title = graph_data.title or (query.strip().rstrip("?.").title() if query else "Relationship Diagram")
                    rendered = render_graph(graph_data, title=title, channel=channel)
                    if rendered and rendered.matplotlib_png_base64:
                        media_entry = {
                            "type": "chart",
                            "chart_type": "graph",
                            "title": rendered.title,
                            "png_base64": rendered.matplotlib_png_base64,
                            "data_summary": rendered.data_summary,
                        }
                        _append_media(answer, media_entry)
                        _strip_mermaid_from_response(answer)
                        elapsed = time.time() - start
                        logger.info("Relationship graph rendered (%.1fs)", elapsed)
                        return answer
        except Exception as exc:
            logger.debug("Graph rendering skipped: %s", exc)

        # Step 0.5: Parse model-driven VIZ directive (high confidence)
        viz_spec = parse_viz_directive(response_text)
        if viz_spec:
            logger.info("Model-driven VIZ directive found: %s", viz_spec.get("chart_type"))
            from src.visualization.chart_renderer import render_chart

            rendered = render_chart(
                chart_type=viz_spec["chart_type"],
                labels=viz_spec["labels"],
                values=viz_spec["values"],
                title=viz_spec.get("title", "Data Analysis"),
                secondary_values=viz_spec.get("secondary_values"),
                secondary_name=viz_spec.get("secondary_name", ""),
                series_name=viz_spec.get("series_name", "Value"),
                unit=viz_spec.get("unit", ""),
                channel=channel,
            )

            if rendered.matplotlib_png_base64 or rendered.plotly_html:
                media_entry = {
                    "type": "chart",
                    "chart_type": rendered.chart_type,
                    "title": rendered.title,
                    "png_base64": rendered.matplotlib_png_base64,
                    "data_summary": rendered.data_summary,
                }
                _append_media(answer, media_entry)
                _strip_viz_directive(answer)
                elapsed = time.time() - start
                logger.info("Model-driven chart rendered: %s (%.1fs)", rendered.title, elapsed)
                return answer
            else:
                logger.warning("Model-driven VIZ directive failed to render; falling back to auto-detect")

        # Step 1: Check if user explicitly requested a chart
        from src.visualization.chart_decision import should_generate_chart

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
            if force_chart:
                _append_chart_validation(answer, "Chart generation timed out. The response content is available in text form above.")
            return answer

        # Step 3: Extract chartable data from response
        from src.visualization.data_extractor import extract_chart_data

        chart_data = extract_chart_data(response_text)

        if not chart_data or len(chart_data.labels) < 2:
            if force_chart:
                # Try word cloud first; if that also fails, explain why
                wc_result = _try_wordcloud(answer, response_text, channel)
                if not _has_media(wc_result):
                    _append_chart_validation(
                        answer,
                        _build_chart_validation_reason(response_text),
                    )
                return wc_result
            logger.debug("Chart skipped: insufficient data points")
            return answer

        # Timeout check
        if time.time() - start > VIZ_TIMEOUT_SECONDS:
            logger.warning("Visualization timeout before rendering")
            if force_chart:
                _append_chart_validation(answer, "Chart generation timed out during rendering. The data is available in the text response.")
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
            if force_chart:
                _append_chart_validation(answer, "Chart rendering failed. The data could not be visualized in the requested format, but is available in the text response above.")
            return answer

        # Step 7: Append to media[] — only the rendered image, no raw JSON/code
        media_entry = {
            "type": "chart",
            "chart_type": rendered.chart_type,
            "title": rendered.title,
            "png_base64": rendered.matplotlib_png_base64,
            "data_summary": rendered.data_summary,
        }

        _append_media(answer, media_entry)

        elapsed = time.time() - start
        logger.info("Chart generated: %s (%s, %.1fs)", title, chart_type, elapsed)

        return answer

    except Exception as exc:
        logger.warning("Visualization enhancement failed (returning original): %s", exc)
        if force_chart:
            _append_chart_validation(answer, f"Chart generation encountered an error: {exc}. The response data is available in text form above.")
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


# ---------------------------------------------------------------------------
# Relationship table detection
# ---------------------------------------------------------------------------

import re as _re

_VIZ_DIRECTIVE_RE = _re.compile(
    r"<!--DOCWAIN_VIZ\s*(\{.*?\})\s*-->",
    _re.DOTALL,
)

_REL_TABLE_HEADER = _re.compile(
    r"\|\s*(?:source|from|entity|party|node)\s*\|"
    r"\s*(?:relationship|relation|type|action|link|role|connects?)\s*\|"
    r"\s*(?:target|to|destination|object|party)\s*\|",
    _re.IGNORECASE,
)


def _has_relationship_table(text: str) -> bool:
    """Detect if the response contains a Source|Relationship|Target table."""
    return bool(_REL_TABLE_HEADER.search(text))


# ---------------------------------------------------------------------------
# Chart validation helpers
# ---------------------------------------------------------------------------

_ANY_NUMBER_RE = _re.compile(r"\d")


def _has_media(answer: Dict[str, Any]) -> bool:
    """Check if the answer already has media entries."""
    inner = answer.get("answer", answer) if isinstance(answer, dict) else answer
    if isinstance(inner, dict) and inner.get("media"):
        return True
    return bool(answer.get("media")) if isinstance(answer, dict) else False


def _build_chart_validation_reason(response_text: str) -> str:
    """Build a specific explanation for why a chart cannot be generated."""
    has_numbers = bool(_ANY_NUMBER_RE.search(response_text))
    has_table = "|" in response_text and response_text.count("|") >= 4
    text_len = len(response_text.strip())

    if text_len < 100:
        return (
            "A chart could not be generated because the response is too brief. "
            "Charts require structured data such as tables, lists with numeric values, "
            "or multiple comparable data points."
        )
    if not has_numbers:
        return (
            "A chart could not be generated because the response contains no numeric data. "
            "Charts require quantitative values (numbers, percentages, currency amounts) "
            "to create a meaningful visualization."
        )
    if has_table:
        return (
            "A chart could not be generated from the table data. The table may contain "
            "non-numeric or mixed content that cannot be plotted. Charts require at least "
            "2 rows with comparable numeric values."
        )
    return (
        "A chart could not be generated because the response data is not in a chartable format. "
        "For best results, ask a question that produces structured comparisons, ranked lists, "
        "breakdowns with percentages, or time-based trends — e.g., 'Compare the costs across "
        "all line items' or 'Show a breakdown of expenses by category'."
    )


def _append_chart_validation(answer: Dict[str, Any], reason: str) -> None:
    """Append a chart validation note to the answer's media list."""
    validation_entry = {
        "type": "chart_validation",
        "status": "not_generated",
        "reason": reason,
    }
    _append_media(answer, validation_entry)

    # Also append a visible note to the response text so the user sees it
    _set_response_text(answer, reason)


def _strip_mermaid_from_response(answer: Dict[str, Any]) -> None:
    """Remove Mermaid code blocks from the response text since we rendered an image."""
    import re as _mermaid_re
    _MERMAID_BLOCK = _mermaid_re.compile(
        r"```(?:mermaid)?\s*\n\s*(?:graph|flowchart)\s+(?:TD|LR|TB|RL|BT).*?```",
        _mermaid_re.DOTALL,
    )
    inner = answer.get("answer", answer) if isinstance(answer, dict) else answer
    if isinstance(inner, dict) and "response" in inner:
        current = inner.get("response", "")
        if isinstance(current, str) and _MERMAID_BLOCK.search(current):
            cleaned = _MERMAID_BLOCK.sub("", current).strip()
            if cleaned:
                inner["response"] = cleaned
    elif isinstance(answer, dict) and "response" in answer:
        current = answer.get("response", "")
        if isinstance(current, str) and _MERMAID_BLOCK.search(current):
            cleaned = _MERMAID_BLOCK.sub("", current).strip()
            if cleaned:
                answer["response"] = cleaned


def parse_viz_directive(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse a <!--DOCWAIN_VIZ {...} --> directive from the response.
    Returns the parsed JSON dict if valid, None otherwise.
    """
    import json as _json
    match = _VIZ_DIRECTIVE_RE.search(response_text)
    if not match:
        return None
    try:
        spec = _json.loads(match.group(1))
    except _json.JSONDecodeError:
        logger.debug("VIZ directive has invalid JSON")
        return None
    # Normalize field aliases — model may emit "type" or "chart" instead of "chart_type"
    if "chart_type" not in spec:
        for alias in ("type", "chart"):
            if alias in spec:
                spec["chart_type"] = spec.pop(alias)
                break
    if not all(k in spec for k in ("chart_type", "labels", "values")):
        logger.debug("VIZ directive missing required fields")
        return None
    if not isinstance(spec["labels"], list) or not isinstance(spec["values"], list):
        return None
    if len(spec["labels"]) < 1 or len(spec["values"]) < 1:
        return None
    return spec


def _strip_viz_directive(answer: Dict[str, Any]) -> None:
    """Remove <!--DOCWAIN_VIZ ... --> blocks from response text."""
    inner = answer.get("answer", answer) if isinstance(answer, dict) else answer
    if isinstance(inner, dict) and "response" in inner:
        current = inner.get("response", "")
        if isinstance(current, str) and _VIZ_DIRECTIVE_RE.search(current):
            cleaned = _VIZ_DIRECTIVE_RE.sub("", current).strip()
            if cleaned:
                inner["response"] = cleaned
    elif isinstance(answer, dict) and "response" in answer:
        current = answer.get("response", "")
        if isinstance(current, str) and _VIZ_DIRECTIVE_RE.search(current):
            cleaned = _VIZ_DIRECTIVE_RE.sub("", current).strip()
            if cleaned:
                answer["response"] = cleaned


def _set_response_text(answer: Dict[str, Any], note: str) -> None:
    """Append a visualization note to the response text."""
    separator = "\n\n---\n**Visualization Note:** "
    inner = answer.get("answer", answer) if isinstance(answer, dict) else answer
    if isinstance(inner, dict) and "response" in inner:
        current = inner["response"] or ""
        if isinstance(current, str) and note not in current:
            inner["response"] = current + separator + note
    elif isinstance(answer, dict) and "response" in answer:
        current = answer["response"] or ""
        if isinstance(current, str) and note not in current:
            answer["response"] = current + separator + note
