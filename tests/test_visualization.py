"""Tests for the DocWain dynamic visualization system."""
import pytest


# ── Chart Decision Tests ─────────────────────────────────────────────────────

class TestChartDecision:
    def test_user_triggered_chart(self):
        from src.visualization.chart_decision import is_user_triggered
        assert is_user_triggered("Show me a chart of expenses")
        assert is_user_triggered("Visualize the data")
        assert is_user_triggered("graph the revenue")
        assert is_user_triggered("plot the trend")
        assert not is_user_triggered("What is the total?")
        assert not is_user_triggered("Summarize this document")

    def test_suppress_short_response(self):
        from src.visualization.chart_decision import should_generate_chart
        result, reason = should_generate_chart("Total: $47,250.", "query")
        assert not result
        assert "short" in reason.lower() or "too" in reason.lower()

    def test_suppress_gap_response(self):
        from src.visualization.chart_decision import should_generate_chart
        result, _ = should_generate_chart(
            "The documents don't address warranty terms. Not found in provided documents. " * 3,
            "What is the warranty?"
        )
        assert not result

    def test_trigger_on_table(self):
        from src.visualization.chart_decision import should_generate_chart
        table = (
            "## Expenses\n\n"
            "| Category | Amount |\n|----------|--------|\n"
            "| Travel | $5,000 |\n| Software | $12,000 |\n"
            "| Hardware | $3,000 |\n| Meals | $800 |\n"
        )
        result, _ = should_generate_chart(table, "show expenses")
        assert result

    def test_trigger_on_percentages(self):
        from src.visualization.chart_decision import should_generate_chart
        text = (
            "Software accounts for 54% of expenses. "
            "Travel is 24%. Hardware is 14%. Meals is 4%. "
            "Other is 4%. Total: $20,800."
        )
        result, _ = should_generate_chart(text, "breakdown")
        assert result


# ── Data Extractor Tests ─────────────────────────────────────────────────────

class TestDataExtractor:
    def test_extract_from_markdown_table(self):
        from src.visualization.data_extractor import extract_chart_data
        table = (
            "| Item | Amount |\n|------|--------|\n"
            "| Widget A | **$500** |\n| Widget B | **$300** |\n| Service C | **$200** |"
        )
        data = extract_chart_data(table)
        assert data is not None
        assert len(data.labels) == 3
        assert data.values == [500.0, 300.0, 200.0]

    def test_extract_from_bullets(self):
        from src.visualization.data_extractor import extract_chart_data
        bullets = (
            "- **Software:** $21,700\n"
            "- **Travel:** $9,400\n"
            "- **Hardware:** $6,500\n"
        )
        data = extract_chart_data(bullets)
        assert data is not None
        assert len(data.labels) >= 3

    def test_no_extraction_from_empty(self):
        from src.visualization.data_extractor import extract_chart_data
        assert extract_chart_data("") is None
        assert extract_chart_data("No data here.") is None

    def test_temporal_detection(self):
        from src.visualization.data_extractor import extract_chart_data
        table = (
            "| Quarter | Revenue |\n|---------|--------|\n"
            "| Q1 | $2.3M |\n| Q2 | $2.7M |\n| Q3 | $3.1M |"
        )
        data = extract_chart_data(table)
        assert data is not None
        assert data.data_type == "temporal"


# ── Chart Selector Tests ─────────────────────────────────────────────────────

class TestChartSelector:
    def test_donut_for_small_positive(self):
        from src.visualization.data_extractor import ChartData
        from src.visualization.chart_selector import select_chart_type
        data = ChartData(labels=["A", "B", "C"], values=[50, 30, 20])
        assert select_chart_type(data) == "donut"

    def test_line_for_temporal(self):
        from src.visualization.data_extractor import ChartData
        from src.visualization.chart_selector import select_chart_type
        data = ChartData(labels=["Q1", "Q2", "Q3"], values=[2.3, 2.7, 3.1], data_type="temporal")
        assert select_chart_type(data) == "line"

    def test_bar_for_many_items(self):
        from src.visualization.data_extractor import ChartData
        from src.visualization.chart_selector import select_chart_type
        data = ChartData(labels=[f"Item{i}" for i in range(10)], values=list(range(10)))
        result = select_chart_type(data)
        assert result in ("bar", "horizontal_bar")

    def test_user_hint_overrides(self):
        from src.visualization.data_extractor import ChartData
        from src.visualization.chart_selector import select_chart_type
        data = ChartData(labels=["A", "B", "C"], values=[50, 30, 20])
        assert select_chart_type(data, "show me a pie chart") == "donut"
        assert select_chart_type(data, "line trend") == "line"


# ── Chart Renderer Tests ─────────────────────────────────────────────────────

class TestChartRenderer:
    def test_render_bar_web(self):
        from src.visualization.chart_renderer import render_chart
        result = render_chart(
            "bar", ["A", "B", "C"], [10, 20, 30], "Test Bar",
            channel="web",
        )
        assert result.matplotlib_png_base64
        assert result.plotly_html
        assert result.chart_type == "bar"
        assert result.title == "Test Bar"

    def test_render_donut_teams(self):
        from src.visualization.chart_renderer import render_chart
        result = render_chart(
            "donut", ["X", "Y", "Z"], [50, 30, 20], "Test Donut",
            channel="teams",
        )
        assert result.matplotlib_png_base64
        assert not result.plotly_html  # Teams = no Plotly

    def test_render_line(self):
        from src.visualization.chart_renderer import render_chart
        result = render_chart(
            "line", ["Q1", "Q2", "Q3", "Q4"], [10, 15, 13, 20], "Trend",
            channel="web",
        )
        assert result.matplotlib_png_base64
        assert result.plotly_html

    def test_render_with_unit(self):
        from src.visualization.chart_renderer import render_chart
        result = render_chart(
            "bar", ["Travel", "Software"], [5000, 12000], "Expenses",
            unit="$", channel="web",
        )
        assert result.data_summary
        assert "$" in result.data_summary or "17" in result.data_summary


# ── Word Cloud Tests ─────────────────────────────────────────────────────────

class TestWordCloud:
    def test_extract_frequencies(self):
        from src.visualization.wordcloud_renderer import extract_word_frequencies
        text = "**Python** is used by **Alice**, **Bob**, and **Carol**. **Python** and **Java** are popular."
        freqs = extract_word_frequencies(text)
        assert "Python" in freqs
        assert "Alice" in freqs
        assert freqs["Python"] > freqs["Alice"]  # Python mentioned twice

    def test_render_wordcloud(self):
        from src.visualization.wordcloud_renderer import render_wordcloud
        freqs = {"Python": 10, "Java": 8, "SQL": 6, "AWS": 5, "Docker": 4, "React": 3}
        result = render_wordcloud(freqs, title="Skills Cloud", channel="web")
        assert result is not None
        assert result.matplotlib_png_base64
        assert result.chart_type == "wordcloud"

    def test_skip_too_few_terms(self):
        from src.visualization.wordcloud_renderer import render_wordcloud
        result = render_wordcloud({"A": 1, "B": 2}, title="Too Few")
        assert result is None


# ── Enhancer Integration Tests ───────────────────────────────────────────────

class TestEnhancer:
    def test_auto_chart_on_table(self):
        from src.visualization.enhancer import enhance_with_visualization
        answer = {
            "response": (
                "## Expenses\n\n"
                "| Category | Amount |\n|----------|--------|\n"
                "| Travel | **$5,000** |\n| Software | **$12,000** |\n"
                "| Hardware | **$3,000** |\n| Meals | **$800** |"
            ),
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {},
        }
        result = enhance_with_visualization(answer, "show expenses", channel="web")
        assert "media" in result
        assert len(result["media"]) > 0
        assert result["media"][0]["chart_type"] in ("donut", "bar", "pie")

    def test_no_chart_on_simple_lookup(self):
        from src.visualization.enhancer import enhance_with_visualization
        answer = {
            "response": "The total is **$47,250.00**.",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {},
        }
        result = enhance_with_visualization(answer, "What is the total?", channel="web")
        assert not result.get("media")

    def test_no_chart_on_gap(self):
        from src.visualization.enhancer import enhance_with_visualization
        answer = {
            "response": "Not found in provided documents. The documents don't address this topic.",
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {},
        }
        result = enhance_with_visualization(answer, "What is the warranty?", channel="web")
        assert not result.get("media")

    def test_user_triggered_forces_chart(self):
        from src.visualization.enhancer import enhance_with_visualization
        answer = {
            "response": (
                "Candidates:\n"
                "1. **Alice** — **7 years**\n"
                "2. **Bob** — **4 years**\n"
                "3. **Carol** — **6 years**\n"
                "4. **Dave** — **3 years**\n"
                "5. **Eve** — **5 years**"
            ),
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {},
        }
        result = enhance_with_visualization(answer, "Visualize experience", channel="web")
        media = result.get("media", [])
        assert len(media) > 0

    def test_teams_channel_no_plotly(self):
        from src.visualization.enhancer import enhance_with_visualization
        answer = {
            "response": (
                "| Item | Cost |\n|------|------|\n"
                "| A | $100 |\n| B | $200 |\n| C | $300 |"
            ),
            "sources": [],
            "grounded": True,
            "context_found": True,
            "metadata": {},
        }
        result = enhance_with_visualization(answer, "chart this", channel="teams")
        media = result.get("media", [])
        if media:
            assert media[0].get("png_base64")  # Has PNG
            assert not media[0].get("plotly_html")  # No Plotly for Teams

    def test_original_answer_unchanged_on_failure(self):
        from src.visualization.enhancer import enhance_with_visualization
        answer = {"response": "Test", "sources": [], "metadata": {}}
        result = enhance_with_visualization(answer, "query", channel="web")
        assert result["response"] == "Test"  # Unchanged
