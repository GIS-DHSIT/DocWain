"""Dual chart renderer — Plotly (interactive/web) + Matplotlib (static/Teams).

Renders charts using DocWain brand theme. Each chart is produced in both
formats simultaneously so the response pipeline can serve the right one
based on the delivery channel.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── DocWain Visual Theme ─────────────────────────────────────────────────────

DOCWAIN_THEME = {
    "primary": "#1933D2",
    "secondary": "#4A90D9",
    "accent": "#2ECC71",
    "warning": "#E74C3C",
    "neutral": "#95A5A6",
    "background": "#FFFFFF",
    "text": "#2C3E50",
    "font": "Inter, Helvetica, sans-serif",
    "palette": [
        "#1933D2", "#4A90D9", "#2ECC71", "#F39C12",
        "#E74C3C", "#9B59B6", "#1ABC9C", "#34495E",
        "#E67E22", "#3498DB", "#27AE60", "#8E44AD",
    ],
}


@dataclass
class RenderedChart:
    """Output of a chart rendering operation."""
    chart_type: str
    title: str
    plotly_html: str = ""
    plotly_json: dict = field(default_factory=dict)
    matplotlib_png_base64: str = ""
    width: int = 700
    height: int = 450
    data_summary: str = ""


# ── Plotly Renderer ──────────────────────────────────────────────────────────

def _plotly_layout(title: str, width: int = 700, height: int = 450) -> Dict[str, Any]:
    """Standard Plotly layout with DocWain theme."""
    return {
        "title": {"text": title, "font": {"size": 16, "color": DOCWAIN_THEME["text"]}},
        "font": {"family": DOCWAIN_THEME["font"], "color": DOCWAIN_THEME["text"]},
        "plot_bgcolor": DOCWAIN_THEME["background"],
        "paper_bgcolor": DOCWAIN_THEME["background"],
        "width": width,
        "height": height,
        "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
    }


def render_plotly(
    chart_type: str,
    labels: List[str],
    values: List[float],
    title: str,
    *,
    secondary_values: Optional[List[float]] = None,
    secondary_name: str = "",
    series_name: str = "Value",
    unit: str = "",
    width: int = 700,
    height: int = 450,
) -> Dict[str, Any]:
    """Render a chart using Plotly. Returns {html, json}."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed — skipping interactive chart render")
        return {"html": "", "json": {}}

    palette = DOCWAIN_THEME["palette"]
    layout = _plotly_layout(title, width, height)
    fig = go.Figure()

    try:
        if chart_type == "bar":
            fig.add_trace(go.Bar(
                x=labels, y=values, name=series_name,
                marker_color=palette[0],
                text=[f"{unit}{v:,.0f}" if unit == "$" else f"{v:,.1f}{unit}" for v in values],
                textposition="outside",
            ))

        elif chart_type == "horizontal_bar":
            fig.add_trace(go.Bar(
                y=labels, x=values, name=series_name,
                marker_color=palette[0], orientation="h",
                text=[f"{unit}{v:,.0f}" if unit == "$" else f"{v:,.1f}{unit}" for v in values],
                textposition="outside",
            ))
            layout["margin"]["l"] = 120

        elif chart_type == "grouped_bar":
            fig.add_trace(go.Bar(
                x=labels, y=values, name=series_name,
                marker_color=palette[0],
            ))
            if secondary_values:
                fig.add_trace(go.Bar(
                    x=labels, y=secondary_values,
                    name=secondary_name or "Series 2",
                    marker_color=palette[1],
                ))
            layout["barmode"] = "group"

        elif chart_type == "stacked_bar":
            fig.add_trace(go.Bar(x=labels, y=values, name=series_name, marker_color=palette[0]))
            if secondary_values:
                fig.add_trace(go.Bar(
                    x=labels, y=secondary_values,
                    name=secondary_name or "Series 2",
                    marker_color=palette[1],
                ))
            layout["barmode"] = "stack"

        elif chart_type in ("donut", "pie"):
            colors = palette[:len(labels)]
            fig.add_trace(go.Pie(
                labels=labels, values=values,
                hole=0.4 if chart_type == "donut" else 0,
                marker={"colors": colors},
                textinfo="label+percent",
                textposition="outside",
            ))

        elif chart_type == "line":
            fig.add_trace(go.Scatter(
                x=labels, y=values, mode="lines+markers",
                name=series_name, line={"color": palette[0], "width": 3},
                marker={"size": 8},
            ))

        elif chart_type == "multi_line":
            fig.add_trace(go.Scatter(
                x=labels, y=values, mode="lines+markers",
                name=series_name, line={"color": palette[0], "width": 3},
            ))
            if secondary_values:
                fig.add_trace(go.Scatter(
                    x=labels, y=secondary_values, mode="lines+markers",
                    name=secondary_name or "Series 2",
                    line={"color": palette[1], "width": 3},
                ))

        elif chart_type == "area":
            fig.add_trace(go.Scatter(
                x=labels, y=values, fill="tozeroy",
                name=series_name, line={"color": palette[0]},
                fillcolor=f"rgba(25,51,210,0.2)",
            ))

        elif chart_type == "scatter":
            fig.add_trace(go.Scatter(
                x=values, y=secondary_values or values,
                mode="markers+text", text=labels,
                textposition="top center",
                marker={"size": 12, "color": palette[0]},
            ))
            layout["xaxis"] = {"title": series_name}
            layout["yaxis"] = {"title": secondary_name or "Value"}

        elif chart_type == "waterfall":
            measures = []
            for i, v in enumerate(values):
                if i == 0:
                    measures.append("absolute")
                elif i == len(values) - 1:
                    measures.append("total")
                else:
                    measures.append("relative")
            fig.add_trace(go.Waterfall(
                x=labels, y=values, measure=measures,
                increasing={"marker": {"color": DOCWAIN_THEME["accent"]}},
                decreasing={"marker": {"color": DOCWAIN_THEME["warning"]}},
                totals={"marker": {"color": palette[0]}},
            ))

        elif chart_type == "gauge":
            value = values[0] if values else 0
            max_val = values[1] if len(values) > 1 else 100
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": labels[0] if labels else "Value"},
                gauge={
                    "axis": {"range": [0, max_val]},
                    "bar": {"color": palette[0]},
                    "steps": [
                        {"range": [0, max_val * 0.5], "color": "#f0f0f0"},
                        {"range": [max_val * 0.5, max_val * 0.8], "color": "#e0e0e0"},
                        {"range": [max_val * 0.8, max_val], "color": "#d0d0d0"},
                    ],
                },
            ))

        elif chart_type == "radar":
            fig.add_trace(go.Scatterpolar(
                r=values, theta=labels, fill="toself",
                name=series_name, line={"color": palette[0]},
            ))
            if secondary_values:
                fig.add_trace(go.Scatterpolar(
                    r=secondary_values, theta=labels, fill="toself",
                    name=secondary_name or "Series 2",
                    line={"color": palette[1]},
                ))
            layout["polar"] = {"radialaxis": {"visible": True}}

        elif chart_type == "heatmap":
            # Treat values as flattened matrix, labels as axes
            n = len(labels)
            size = int(n ** 0.5) or 1
            z = [values[i:i+size] for i in range(0, len(values), size)]
            fig.add_trace(go.Heatmap(
                z=z, colorscale="Blues",
            ))

        elif chart_type == "treemap":
            parents = [""] * len(labels)
            fig.add_trace(go.Treemap(
                labels=labels, values=values, parents=parents,
                marker={"colors": palette[:len(labels)]},
            ))

        elif chart_type == "sankey":
            n = len(labels) // 2
            sources = list(range(n))
            targets = list(range(n, len(labels)))
            fig.add_trace(go.Sankey(
                node={"label": labels, "color": palette[:len(labels)]},
                link={"source": sources, "target": targets, "value": values[:n]},
            ))

        else:
            # Fallback to bar
            fig.add_trace(go.Bar(
                x=labels, y=values, marker_color=palette[0],
            ))

        fig.update_layout(**layout)

        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        json_spec = fig.to_dict()

        return {"html": html, "json": json_spec}

    except Exception as exc:
        logger.warning("Plotly render failed for %s: %s", chart_type, exc)
        return {"html": "", "json": {}}


# ── Matplotlib Renderer ──────────────────────────────────────────────────────

def render_matplotlib(
    chart_type: str,
    labels: List[str],
    values: List[float],
    title: str,
    *,
    secondary_values: Optional[List[float]] = None,
    secondary_name: str = "",
    series_name: str = "Value",
    unit: str = "",
    width: int = 700,
    height: int = 450,
) -> str:
    """Render a chart using matplotlib. Returns base64 PNG string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping static PNG render")
        return ""

    palette = DOCWAIN_THEME["palette"]
    fig_w, fig_h = width / 100, height / 100

    try:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(DOCWAIN_THEME["background"])
        ax.set_facecolor(DOCWAIN_THEME["background"])

        if chart_type == "bar":
            bars = ax.bar(labels, values, color=palette[0])
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{unit}{v:,.0f}" if unit == "$" else f"{v:,.1f}{unit}",
                        ha="center", va="bottom", fontsize=8)

        elif chart_type == "horizontal_bar":
            ax.barh(labels, values, color=palette[0])

        elif chart_type == "grouped_bar":
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w/2, values, w, label=series_name, color=palette[0])
            if secondary_values:
                ax.bar(x + w/2, secondary_values, w,
                       label=secondary_name or "Series 2", color=palette[1])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()

        elif chart_type == "stacked_bar":
            ax.bar(labels, values, label=series_name, color=palette[0])
            if secondary_values:
                ax.bar(labels, secondary_values, bottom=values,
                       label=secondary_name or "Series 2", color=palette[1])
            ax.legend()

        elif chart_type in ("donut", "pie"):
            # Guard against all-zero values which cause NaN in pie chart
            plot_values = [v if v > 0 else 0.001 for v in values]
            if sum(plot_values) == 0:
                plot_values = [1] * len(labels)  # Equal slices fallback
            colors = palette[:len(labels)]
            wedges, texts, autotexts = ax.pie(
                plot_values, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90,
            )
            if chart_type == "donut":
                centre = plt.Circle((0, 0), 0.5, fc=DOCWAIN_THEME["background"])
                ax.add_artist(centre)

        elif chart_type == "line":
            ax.plot(labels, values, marker="o", color=palette[0], linewidth=2, markersize=6)

        elif chart_type == "multi_line":
            ax.plot(labels, values, marker="o", color=palette[0], linewidth=2, label=series_name)
            if secondary_values:
                ax.plot(labels, secondary_values, marker="s", color=palette[1],
                        linewidth=2, label=secondary_name or "Series 2")
            ax.legend()

        elif chart_type == "area":
            ax.fill_between(labels, values, alpha=0.3, color=palette[0])
            ax.plot(labels, values, color=palette[0], linewidth=2)

        elif chart_type == "scatter":
            y_vals = secondary_values if secondary_values else values
            ax.scatter(values, y_vals, c=palette[0], s=80)
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (values[i], y_vals[i]), fontsize=7,
                            textcoords="offset points", xytext=(5, 5))

        elif chart_type == "radar":
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            vals = values + [values[0]]
            angles += [angles[0]]
            lbls = labels + [labels[0]]
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, vals, color=palette[0], linewidth=2)
            ax.fill(angles, vals, alpha=0.25, color=palette[0])
            if secondary_values:
                sv = secondary_values + [secondary_values[0]]
                ax.plot(angles, sv, color=palette[1], linewidth=2)
                ax.fill(angles, sv, alpha=0.15, color=palette[1])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=8)

        elif chart_type == "gauge":
            val = values[0] if values else 0
            max_val = values[1] if len(values) > 1 else 100
            pct = val / max_val if max_val else 0
            ax.barh([0], [pct], color=palette[0], height=0.3)
            ax.barh([0], [1 - pct], left=[pct], color="#e0e0e0", height=0.3)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.text(0.5, -0.15, f"{val:,.0f} / {max_val:,.0f}",
                    ha="center", transform=ax.transAxes, fontsize=12)

        else:
            # Fallback bar chart
            ax.bar(labels, values, color=palette[0])

        ax.set_title(title, fontsize=13, color=DOCWAIN_THEME["text"], pad=10)
        plt.xticks(rotation=45 if len(labels) > 5 and chart_type not in ("donut", "pie", "radar", "gauge") else 0,
                   ha="right" if len(labels) > 5 else "center", fontsize=9)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=DOCWAIN_THEME["background"])
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as exc:
        logger.warning("Matplotlib render failed for %s: %s", chart_type, exc)
        plt.close("all")
        return ""


# ── Unified Render Function ──────────────────────────────────────────────────

def render_chart(
    chart_type: str,
    labels: List[str],
    values: List[float],
    title: str,
    *,
    secondary_values: Optional[List[float]] = None,
    secondary_name: str = "",
    series_name: str = "Value",
    unit: str = "",
    channel: str = "web",
    width: int = 700,
    height: int = 450,
) -> RenderedChart:
    """Render a chart in the appropriate format for the delivery channel.

    Args:
        channel: "web" for Plotly interactive + PNG fallback,
                 "teams" for matplotlib PNG only.
    """
    result = RenderedChart(
        chart_type=chart_type,
        title=title,
        width=width,
        height=height,
    )

    kwargs = {
        "chart_type": chart_type,
        "labels": labels,
        "values": values,
        "title": title,
        "secondary_values": secondary_values,
        "secondary_name": secondary_name,
        "series_name": series_name,
        "unit": unit,
        "width": width,
        "height": height,
    }

    # Always attempt matplotlib PNG (works for both channels)
    try:
        result.matplotlib_png_base64 = render_matplotlib(**kwargs)
    except Exception as exc:
        logger.warning("matplotlib render failed: %s", exc)
        result.matplotlib_png_base64 = ""

    # Render Plotly for web channel
    if channel == "web":
        try:
            plotly_result = render_plotly(**kwargs)
            result.plotly_html = plotly_result.get("html", "")
            result.plotly_json = plotly_result.get("json", {})
        except Exception as exc:
            logger.warning("Plotly render failed: %s", exc)
            result.plotly_html = ""
            result.plotly_json = {}

    # Generate data summary
    if values:
        total = sum(values)
        top_label = labels[values.index(max(values))] if labels else ""
        result.data_summary = f"{title}: {len(values)} items, total {unit}{total:,.0f}, largest: {top_label}"

    return result
