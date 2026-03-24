"""Tests for the visualization evaluation harness.

All tests work offline -- no running model required.
"""

import json
import pytest

from src.finetune.viz_evaluator import (
    score_viz_response,
    build_viz_eval_set,
    _are_compatible_types,
)


# ---------------------------------------------------------------------------
# Helper: build a model response with a VIZ directive
# ---------------------------------------------------------------------------

def _make_chart_response(
    chart_type: str = "bar",
    title: str = "Test Chart",
    labels: list = None,
    values: list = None,
    include_table: bool = True,
    extra_text: str = "Here is the analysis.",
) -> str:
    """Build a synthetic model response containing a VIZ directive."""
    labels = labels or ["A", "B", "C"]
    values = values or [10, 20, 30]

    parts = [extra_text, ""]

    if include_table:
        parts.append("| Label | Value |")
        parts.append("|-------|-------|")
        for l, v in zip(labels, values):
            parts.append(f"| {l} | **{v}** |")
        parts.append("")

    payload = json.dumps({
        "chart_type": chart_type,
        "title": title,
        "labels": labels,
        "values": values,
    })
    parts.append(f"<!--DOCWAIN_VIZ {payload} -->")
    return "\n".join(parts)


def _make_text_response(text: str = "The contract expires January 1, 2027.") -> str:
    """Build a plain text response with no directive."""
    return text


def _make_flow_response() -> str:
    """Build a flow analysis response with arrows but no VIZ directive."""
    return (
        "## Invoice Approval Process\n\n"
        "1. Vendor submits invoice\n"
        "2. Finance validates \u2192 Manager reviews\n"
        "3. Under $10K \u2192 Manager approval\n"
        "4. Over $10K \u2192 Director approval\n"
        "5. Payment issued within Net 30"
    )


# ---------------------------------------------------------------------------
# score_viz_response tests
# ---------------------------------------------------------------------------


class TestScoreVizResponseChartGood:
    """Good chart response: directive + correct type + table."""

    def test_composite_above_80(self):
        response = _make_chart_response(
            chart_type="bar",
            labels=["Q1", "Q2", "Q3", "Q4"],
            values=[1.2, 1.5, 1.8, 2.1],
        )
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
            expected_labels=["Q1", "Q2", "Q3", "Q4"],
            expected_values=[1.2, 1.5, 1.8, 2.1],
        )
        assert result["composite"] >= 80
        assert result["has_directive"] is True
        assert result["chart_type_correct"] is True
        assert result["has_table"] is True
        assert result["mode_correct"] is True

    def test_parsed_spec_present(self):
        response = _make_chart_response()
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
        )
        assert result["parsed_spec"] is not None
        assert result["parsed_spec"]["chart_type"] == "bar"


class TestScoreVizResponseMissingDirective:
    """Chart expected but directive missing -> low score."""

    def test_composite_below_50(self):
        response = "Here is the revenue data.\n\n| Q | Revenue |\n|---|---|\n| Q1 | $1.2M |"
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
            expected_labels=["Q1"],
            expected_values=[1.2],
        )
        assert result["composite"] < 50
        assert result["has_directive"] is False
        assert result["mode_correct"] is False


class TestScoreVizResponseTextOnly:
    """Text-only response when no chart expected -> high score."""

    def test_composite_above_80(self):
        response = _make_text_response()
        result = score_viz_response(
            response=response,
            expected_chart_type=None,
            expects_chart=False,
        )
        assert result["composite"] >= 80
        assert result["has_directive"] is False
        assert result["mode_correct"] is True


class TestScoreVizResponseFlowAnalysis:
    """Flow analysis response (arrows) when no chart expected."""

    def test_composite_above_70(self):
        response = _make_flow_response()
        result = score_viz_response(
            response=response,
            expected_chart_type="flow",
            expects_chart=False,
        )
        assert result["composite"] >= 70
        assert result["has_flow_analysis"] is True
        assert result["has_directive"] is False
        assert result["mode_correct"] is True


class TestScoreVizResponseWrongChartType:
    """Directive present but wrong chart type."""

    def test_chart_type_correct_false(self):
        response = _make_chart_response(
            chart_type="donut",
            labels=["Q1", "Q2", "Q3", "Q4"],
            values=[1.2, 1.5, 1.8, 2.1],
        )
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
            expected_labels=["Q1", "Q2", "Q3", "Q4"],
            expected_values=[1.2, 1.5, 1.8, 2.1],
        )
        assert result["chart_type_correct"] is False


class TestScoreVizResponseCompatibleTypes:
    """Compatible types (e.g. bar/horizontal_bar) get partial credit."""

    def test_bar_horizontal_bar_partial(self):
        response = _make_chart_response(
            chart_type="horizontal_bar",
            labels=["Q1", "Q2"],
            values=[10, 20],
        )
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
            expected_labels=["Q1", "Q2"],
            expected_values=[10, 20],
        )
        # Not exact match, but compatible
        assert result["chart_type_correct"] is False
        # Composite should be reasonable but less than perfect
        assert result["composite"] > 50


# ---------------------------------------------------------------------------
# build_viz_eval_set tests
# ---------------------------------------------------------------------------


class TestBuildVizEvalSet:
    """Eval set structure and coverage."""

    def test_at_least_20_items(self):
        items = build_viz_eval_set()
        assert len(items) >= 20

    def test_required_fields_present(self):
        items = build_viz_eval_set()
        required_keys = {"query", "expects_chart", "expected_chart_type", "expected_labels", "expected_values"}
        for item in items:
            assert required_keys.issubset(item.keys()), f"Missing keys in item: {item.get('query', '')[:40]}"

    def test_chart_expected_count(self):
        items = build_viz_eval_set()
        chart_items = [i for i in items if i["expects_chart"]]
        assert len(chart_items) >= 12

    def test_text_only_count(self):
        items = build_viz_eval_set()
        text_items = [i for i in items if not i["expects_chart"] and i["expected_chart_type"] is None]
        assert len(text_items) >= 6

    def test_flow_count(self):
        items = build_viz_eval_set()
        flow_items = [i for i in items if i["expected_chart_type"] == "flow"]
        assert len(flow_items) >= 2


# ---------------------------------------------------------------------------
# _are_compatible_types tests
# ---------------------------------------------------------------------------


class TestCompatibleTypes:

    def test_same_type(self):
        assert _are_compatible_types("bar", "bar") is True

    def test_bar_horizontal_bar(self):
        assert _are_compatible_types("bar", "horizontal_bar") is True
        assert _are_compatible_types("horizontal_bar", "bar") is True

    def test_donut_pie(self):
        assert _are_compatible_types("donut", "pie") is True

    def test_line_area(self):
        assert _are_compatible_types("line", "area") is True

    def test_grouped_stacked(self):
        assert _are_compatible_types("grouped_bar", "stacked_bar") is True

    def test_incompatible(self):
        assert _are_compatible_types("bar", "donut") is False
        assert _are_compatible_types("line", "bar") is False


# ---------------------------------------------------------------------------
# Offline scoring of known-good response
# ---------------------------------------------------------------------------


class TestOfflineKnownGood:
    """Verify that a known-good training example scores well."""

    def test_known_good_bar_response(self):
        """A response modeled after viz_training_data bar example should score >= 80."""
        response = (
            "## Quarterly Revenue\n\n"
            "| Quarter | Revenue |\n"
            "|---------|---------|\n"
            "| Q1 | **$1.2M** |\n"
            "| Q2 | **$1.5M** |\n"
            "| Q3 | **$1.8M** |\n"
            "| Q4 | **$2.1M** |\n\n"
            "Revenue grew steadily each quarter.\n\n"
            '<!--DOCWAIN_VIZ {"chart_type": "bar", "title": "Quarterly Revenue", '
            '"labels": ["Q1", "Q2", "Q3", "Q4"], "values": [1.2, 1.5, 1.8, 2.1], "unit": "$M"} -->'
        )
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
            expected_labels=["Q1", "Q2", "Q3", "Q4"],
            expected_values=[1.2, 1.5, 1.8, 2.1],
        )
        assert result["composite"] >= 80

    def test_known_good_text_response(self):
        """A plain text answer for a factual question should score >= 80."""
        response = "The contract expires **January 1, 2027** (24-month term from January 1, 2025)."
        result = score_viz_response(
            response=response,
            expected_chart_type=None,
            expects_chart=False,
        )
        assert result["composite"] >= 80
