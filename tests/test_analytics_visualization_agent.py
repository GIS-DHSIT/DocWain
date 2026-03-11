"""Tests for the Analytics Visualization Agent and chart generation tool.

Tests chart generation (bar, pie, line, grouped bar, histogram), data extraction,
chart type selection, base64 output validation, and agent execution.
"""
import base64
import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Chart generation tool tests
# ---------------------------------------------------------------------------

class TestChartGeneration:
    """Test the low-level chart generation functions in analytics_visualization.py."""

    def test_bar_chart_returns_base64_png(self):
        from src.tools.analytics_visualization import generate_bar_chart
        labels = ["Python", "Java", "Go"]
        values = [10, 7, 5]
        result = generate_bar_chart(labels, values, "Test Bar Chart")
        assert isinstance(result, str)
        assert len(result) > 100  # non-trivial base64 string
        # Verify it's valid base64 that decodes to PNG
        raw = base64.b64decode(result)
        assert raw[:4] == b'\x89PNG'

    def test_pie_chart_returns_base64_png(self):
        from src.tools.analytics_visualization import generate_pie_chart
        labels = ["A", "B", "C"]
        values = [40, 35, 25]
        result = generate_pie_chart(labels, values, "Test Pie Chart")
        raw = base64.b64decode(result)
        assert raw[:4] == b'\x89PNG'

    def test_line_chart_returns_base64_png(self):
        from src.tools.analytics_visualization import generate_line_chart
        labels = ["Jan", "Feb", "Mar", "Apr"]
        values = [100, 120, 115, 140]
        result = generate_line_chart(labels, values, "Test Line Chart")
        raw = base64.b64decode(result)
        assert raw[:4] == b'\x89PNG'

    def test_grouped_bar_chart_single_group(self):
        from src.tools.analytics_visualization import generate_grouped_bar_chart
        data = [
            {"label": "A", "value": 10},
            {"label": "B", "value": 20},
        ]
        result = generate_grouped_bar_chart(data, "Test Grouped")
        raw = base64.b64decode(result)
        assert raw[:4] == b'\x89PNG'

    def test_grouped_bar_chart_multi_group(self):
        from src.tools.analytics_visualization import generate_grouped_bar_chart
        data = [
            {"label": "A", "value": 10, "group": "2024"},
            {"label": "B", "value": 20, "group": "2024"},
            {"label": "A", "value": 15, "group": "2025"},
            {"label": "B", "value": 25, "group": "2025"},
        ]
        result = generate_grouped_bar_chart(data, "Multi-Group Test")
        raw = base64.b64decode(result)
        assert raw[:4] == b'\x89PNG'

    def test_histogram_returns_base64_png(self):
        from src.tools.analytics_visualization import generate_histogram
        values = [1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10]
        result = generate_histogram(values, "Test Histogram")
        raw = base64.b64decode(result)
        assert raw[:4] == b'\x89PNG'

    def test_empty_bar_chart(self):
        from src.tools.analytics_visualization import generate_bar_chart
        result = generate_bar_chart([], [], "Empty Chart")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Chart type selection tests
# ---------------------------------------------------------------------------

class TestChartTypeSelection:
    def test_pie_keyword(self):
        from src.tools.analytics_visualization import select_chart_type
        data = [{"label": "A", "value": 1}]
        assert select_chart_type("show me a pie chart of expenses", data) == "pie"

    def test_line_keyword(self):
        from src.tools.analytics_visualization import select_chart_type
        data = [{"label": "A", "value": 1}]
        assert select_chart_type("show trend over time", data) == "line"

    def test_histogram_keyword(self):
        from src.tools.analytics_visualization import select_chart_type
        data = [{"label": "A", "value": 1}]
        assert select_chart_type("show the distribution of scores", data) == "histogram"

    def test_comparison_keyword(self):
        from src.tools.analytics_visualization import select_chart_type
        data = [{"label": "A", "value": 1}]
        assert select_chart_type("compare candidates side by side", data) == "grouped_bar"

    def test_default_bar(self):
        from src.tools.analytics_visualization import select_chart_type
        data = [{"label": str(i), "value": i} for i in range(10)]
        assert select_chart_type("show the data", data) == "bar"

    def test_few_items_default_pie(self):
        from src.tools.analytics_visualization import select_chart_type
        data = [{"label": "A", "value": 30}, {"label": "B", "value": 70}]
        assert select_chart_type("show the data", data) == "pie"


# ---------------------------------------------------------------------------
# Data extraction tests
# ---------------------------------------------------------------------------

class TestDataExtraction:
    def test_parse_json_array(self):
        from src.tools.analytics_visualization import _parse_json_array
        text = 'Here is the data: [{"label": "Python", "value": 5}, {"label": "Java", "value": 3}]'
        result = _parse_json_array(text)
        assert len(result) == 2
        assert result[0]["label"] == "Python"
        assert result[0]["value"] == 5.0

    def test_parse_json_array_invalid(self):
        from src.tools.analytics_visualization import _parse_json_array
        assert _parse_json_array("no json here") == []

    def test_parse_json_array_bad_entries(self):
        from src.tools.analytics_visualization import _parse_json_array
        text = '[{"label": "A", "value": "not_a_number"}, {"label": "B", "value": 5}]'
        result = _parse_json_array(text)
        assert len(result) == 1
        assert result[0]["label"] == "B"

    def test_regex_extract(self):
        from src.tools.analytics_visualization import _regex_extract
        text = "Python: 10\nJava: 7\nGo: 5"
        result = _regex_extract(text)
        assert len(result) == 3
        assert result[0]["label"] == "Python"
        assert result[0]["value"] == 10.0

    def test_extract_chart_data_with_mock_llm(self):
        from src.tools.analytics_visualization import extract_chart_data
        mock_llm = MagicMock()
        mock_llm.generate_with_metadata = MagicMock(
            return_value=('[{"label": "Skills", "value": 8}]', {})
        )
        result = extract_chart_data("text about skills", "show skills chart", mock_llm)
        assert len(result) == 1
        assert result[0]["label"] == "Skills"

    def test_extract_chart_data_empty_text(self):
        from src.tools.analytics_visualization import extract_chart_data
        mock_llm = MagicMock()
        result = extract_chart_data("", "query", mock_llm)
        assert result == []


# ---------------------------------------------------------------------------
# Agent registration and capabilities
# ---------------------------------------------------------------------------

class TestAnalyticsVizRegistration:
    def test_agent_exists(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("analytics_viz")
        assert agent is not None
        assert agent.domain == "analytics_viz"

    def test_chart_alias(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("chart")
        assert agent is not None
        assert agent.domain == "analytics_viz"

    def test_visualization_alias(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("visualization")
        assert agent is not None
        assert agent.domain == "analytics_viz"

    def test_has_six_capabilities(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("analytics_viz")
        caps = agent.get_capabilities()
        assert len(caps) == 6
        expected = {"generate_chart", "generate_distribution", "generate_comparison_chart",
                    "generate_timeline_chart", "generate_summary_dashboard", "compute_statistics"}
        assert set(caps) == expected

    def test_does_not_use_thinking_model(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("analytics_viz")
        assert agent.use_thinking_model is False


# ---------------------------------------------------------------------------
# NLU task detection
# ---------------------------------------------------------------------------

class TestAnalyticsVizDetection:
    def test_generate_chart_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("create a bar chart graph visualization from the data")
        assert r is not None
        assert r["domain"] == "analytics_viz"
        assert r["task_type"] == "generate_chart"

    def test_pie_chart_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("generate a pie chart showing data proportions visually")
        assert r is not None
        assert r["domain"] == "analytics_viz"
        # May match generate_chart (which auto-selects pie) or specific type
        assert "generate" in r["task_type"]

    def test_timeline_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("visualize chronological temporal progression over time as a timeline chart")
        assert r is not None
        assert r["domain"] == "analytics_viz"
        assert r["task_type"] == "generate_timeline_chart"

    def test_comparison_chart_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("create a comparison chart to visualize differences side by side")
        assert r is not None
        assert r["domain"] == "analytics_viz"

    def test_dashboard_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("create a dashboard with multiple charts summarizing key metrics overview")
        assert r is not None
        assert r["domain"] == "analytics_viz"
        assert r["task_type"] == "generate_summary_dashboard"

    def test_statistics_detected(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("calculate statistics averages and totals with visualization chart showing results")
        assert r is not None
        assert r["domain"] == "analytics_viz"
        assert r["task_type"] == "compute_statistics"


# ---------------------------------------------------------------------------
# Agent execution (with mocked LLM)
# ---------------------------------------------------------------------------

class TestAnalyticsVizExecution:
    def _mock_agent(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("analytics_viz")
        agent._llm = MagicMock()
        agent._llm.generate_with_metadata = MagicMock(
            return_value=('[{"label": "Python", "value": 8}, {"label": "Java", "value": 5}]', {})
        )
        return agent

    def test_generate_chart_with_data(self):
        agent = self._mock_agent()
        # Mock the LLM to return both chart data and summary
        call_count = [0]
        def side_effect(prompt, options=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return '[{"label": "Python", "value": 8}, {"label": "Java", "value": 5}]', {}
            return "Chart shows Python leads with 8 vs Java with 5.", {}
        agent._llm.generate_with_metadata = MagicMock(side_effect=side_effect)

        result = agent.execute("generate_chart", {
            "text": "Python: 8 developers, Java: 5 developers",
            "query": "Create a chart of developer skills",
        })
        assert result.success
        assert result.output
        assert result.task_type == "generate_chart"

    def test_generate_chart_no_data_fallback(self):
        """When data extraction returns empty, agent should provide text summary."""
        agent = self._mock_agent()
        agent._llm.generate_with_metadata = MagicMock(
            return_value=("No chartable data found. Summary: total of 3 items.", {})
        )
        result = agent.execute("generate_chart", {
            "text": "Some unstructured text without numbers",
            "query": "Create a chart",
        })
        assert result.success
        assert result.output

    def test_compute_statistics(self):
        agent = self._mock_agent()
        call_count = [0]
        def side_effect(prompt, options=None):
            call_count[0] += 1
            if "Extract structured" in prompt:
                return '[{"label": "Q1", "value": 100}, {"label": "Q2", "value": 150}]', {}
            return "Statistics: Mean=125, Min=100, Max=150.", {}
        agent._llm.generate_with_metadata = MagicMock(side_effect=side_effect)

        result = agent.execute("compute_statistics", {
            "text": "Q1 revenue: $100K, Q2 revenue: $150K",
            "query": "Calculate statistics for quarterly revenue",
        })
        assert result.success
        assert result.output

    def test_unknown_task_returns_error(self):
        agent = self._mock_agent()
        result = agent.execute("nonexistent_task", {"text": "test"})
        assert not result.success
        assert result.error


# ---------------------------------------------------------------------------
# Media field propagation
# ---------------------------------------------------------------------------

class TestMediaPropagation:
    def test_normalize_answer_preserves_media(self):
        from src.execution.common import normalize_answer
        answer = {
            "response": "Here is a chart.",
            "sources": [],
            "grounded": True,
            "context_found": True,
            "media": [{"type": "image/png", "title": "Test", "data": "base64data"}],
        }
        normalized = normalize_answer(answer)
        assert "media" in normalized
        assert len(normalized["media"]) == 1
        assert normalized["media"][0]["title"] == "Test"

    def test_normalize_answer_without_media(self):
        from src.execution.common import normalize_answer
        answer = {
            "response": "No media here.",
            "sources": [],
        }
        normalized = normalize_answer(answer)
        assert "media" not in normalized


# ---------------------------------------------------------------------------
# NLU agent matcher integration
# ---------------------------------------------------------------------------

class TestAnalyticsVizNLUMatcher:
    def test_analytics_viz_registered_in_nlu(self):
        from src.agentic.nlu_agent_matcher import _ensure_registry
        from src.nlp.nlu_engine import get_registry
        _ensure_registry()
        reg = get_registry("agent")
        assert "analytics_viz" in reg.entries
