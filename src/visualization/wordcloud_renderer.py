"""Word cloud renderer for text-heavy analysis responses.

Generates word clouds from entity lists, skill distributions,
term frequencies, and topic summaries found in DocWain responses.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Import theme
from src.visualization.chart_renderer import DOCWAIN_THEME, RenderedChart


def extract_word_frequencies(response_text: str) -> Dict[str, float]:
    """Extract word/entity frequencies from response text.

    Looks for:
    - Bold entities: **Entity Name** → higher weight
    - Bullet items: - **Label:** value → label gets weight from value if numeric
    - Repeated terms across the response
    - Skill lists, entity lists, topic mentions
    """
    frequencies: Dict[str, float] = {}

    # Extract bold entities (highest weight)
    bold_pattern = re.compile(r"\*\*([^*]{2,40})\*\*")
    for match in bold_pattern.finditer(response_text):
        term = match.group(1).strip()
        # Skip pure numbers/dates
        if re.match(r"^[\$€£]?[\d,.\s]+%?$", term):
            continue
        if term.lower() in ("total", "subtotal", "grand total", "n/a", "not found"):
            continue
        frequencies[term] = frequencies.get(term, 0) + 3.0

    # Extract bullet labels with values
    bullet_pattern = re.compile(r"[-•]\s*\*\*([^*]+)\*\*[:\s]+(.+?)(?:\n|$)")
    for match in bullet_pattern.finditer(response_text):
        label = match.group(1).strip()
        value_text = match.group(2).strip()
        # Try to extract numeric weight from value
        num_match = re.search(r"[\d,]+(?:\.\d+)?", value_text)
        if num_match:
            try:
                weight = float(num_match.group().replace(",", ""))
                # Normalize large numbers
                if weight > 1000:
                    weight = weight / 1000
                if weight > 100:
                    weight = weight / 10
                frequencies[label] = max(frequencies.get(label, 0), weight)
            except ValueError:
                frequencies[label] = frequencies.get(label, 0) + 2.0
        else:
            frequencies[label] = frequencies.get(label, 0) + 2.0

    # Extract comma-separated lists (skills, technologies, etc.)
    list_pattern = re.compile(r"(?:Skills?|Technologies|Languages?|Tools?|Certifications?)[:\s]+([^\n]+)", re.IGNORECASE)
    for match in list_pattern.finditer(response_text):
        items = re.split(r"[,;·•]", match.group(1))
        for item in items:
            clean = item.strip().strip("*").strip()
            if clean and len(clean) > 1 and len(clean) < 30:
                frequencies[clean] = frequencies.get(clean, 0) + 1.5

    # Filter out very short or stop-word-like entries
    stop_words = {"the", "and", "for", "with", "from", "that", "this", "are", "was",
                  "not", "but", "has", "have", "had", "been", "will", "can", "may",
                  "yes", "no", "n/a", "na", "none", "all", "any", "each", "per"}
    frequencies = {k: v for k, v in frequencies.items()
                   if k.lower() not in stop_words and len(k) > 1}

    return frequencies


def render_wordcloud(
    frequencies: Dict[str, float],
    title: str = "Key Terms",
    width: int = 700,
    height: int = 450,
    channel: str = "web",
) -> Optional[RenderedChart]:
    """Render a word cloud from term frequencies.

    Returns RenderedChart with matplotlib PNG and optional Plotly HTML.
    """
    if not frequencies or len(frequencies) < 3:
        logger.debug("Not enough terms for word cloud: %d", len(frequencies))
        return None

    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Create color function using DocWain palette
        palette = DOCWAIN_THEME["palette"]

        def color_func(word, font_size, position, orientation, **kwargs):
            idx = hash(word) % len(palette)
            return palette[idx]

        wc = WordCloud(
            width=width * 2,  # Higher resolution
            height=height * 2,
            background_color=DOCWAIN_THEME["background"],
            color_func=color_func,
            max_words=80,
            min_font_size=10,
            max_font_size=120,
            prefer_horizontal=0.8,
            relative_scaling=0.5,
        ).generate_from_frequencies(frequencies)

        # Matplotlib PNG
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=13, color=DOCWAIN_THEME["text"], pad=10)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=DOCWAIN_THEME["background"])
        plt.close(fig)
        buf.seek(0)
        png_b64 = base64.b64encode(buf.read()).decode("utf-8")

        result = RenderedChart(
            chart_type="wordcloud",
            title=title,
            matplotlib_png_base64=png_b64,
            width=width,
            height=height,
            data_summary=f"{title}: {len(frequencies)} terms, top: {', '.join(list(frequencies.keys())[:5])}",
        )

        # For web channel, embed as image in simple HTML
        if channel == "web":
            result.plotly_html = (
                f'<div style="text-align:center">'
                f'<h3 style="color:{DOCWAIN_THEME["text"]};font-family:{DOCWAIN_THEME["font"]}">{title}</h3>'
                f'<img src="data:image/png;base64,{png_b64}" '
                f'style="max-width:100%;height:auto" alt="{title}"/>'
                f'</div>'
            )

        return result

    except ImportError:
        logger.warning("wordcloud package not installed — skipping word cloud")
        return None
    except Exception as exc:
        logger.warning("Word cloud render failed: %s", exc)
        return None
