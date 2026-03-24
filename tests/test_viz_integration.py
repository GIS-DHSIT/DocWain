"""End-to-end integration tests for the DocWain visualization pipeline.

Verifies the complete flow: model-driven VIZ directives -> enhancer -> renderer -> media output.
All tests run without a model/GPU by providing pre-formed response text.
"""

from __future__ import annotations

import json
import pytest

from src.visualization.enhancer import enhance_with_visualization, parse_viz_directive
from src.finetune.viz_training_data import generate_viz_sft_examples
from src.finetune.dpo_data_generator import generate_viz_preference_pairs
from src.finetune.viz_evaluator import score_viz_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_answer(response_text: str) -> dict:
    """Build an answer dict in the standard DocWain format."""
    return {
        "answer": {
            "response": response_text,
            "sources": [],
            "grounded": True,
        }
    }


def _viz_directive(spec: dict) -> str:
    """Render a DOCWAIN_VIZ directive from a spec dict."""
    return f"<!--DOCWAIN_VIZ {json.dumps(spec)} -->"


def _get_media(result: dict) -> list:
    """Extract the media list from an answer dict (handles nested structure)."""
    inner = result.get("answer", result)
    if isinstance(inner, dict):
        return inner.get("media", [])
    return result.get("media", [])


def _get_response(result: dict) -> str:
    """Extract the response text from an answer dict."""
    inner = result.get("answer", result)
    if isinstance(inner, dict):
        return inner.get("response", "")
    return result.get("response", "")


# ---------------------------------------------------------------------------
# Test 1: Model-driven bar chart
# ---------------------------------------------------------------------------

class TestModelDrivenBarChart:
    def test_bar_directive_produces_media(self):
        directive = _viz_directive({
            "chart_type": "bar",
            "title": "Quarterly Revenue",
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "values": [1.2, 1.5, 1.8, 2.1],
            "unit": "$M",
        })
        response_text = (
            "## Quarterly Revenue\n\n"
            "| Quarter | Revenue |\n|---------|--------|\n"
            "| Q1 | **$1.2M** |\n| Q2 | **$1.5M** |\n"
            "| Q3 | **$1.8M** |\n| Q4 | **$2.1M** |\n\n"
            f"Revenue grew steadily each quarter.\n\n{directive}"
        )
        answer = _build_answer(response_text)
        result = enhance_with_visualization(answer, "Show revenue by quarter", channel="web")

        media = _get_media(result)
        assert len(media) >= 1, "Expected at least one media entry"
        chart = media[0]
        assert chart["chart_type"] == "bar"
        assert chart["type"] == "chart"
        assert "DOCWAIN_VIZ" not in _get_response(result)


# ---------------------------------------------------------------------------
# Test 2: Model-driven donut
# ---------------------------------------------------------------------------

class TestModelDrivenDonut:
    def test_donut_directive_produces_media(self):
        directive = _viz_directive({
            "chart_type": "donut",
            "title": "Expense Breakdown",
            "labels": ["Salaries", "Cloud", "Marketing", "Office", "Travel"],
            "values": [55, 20, 12, 8, 5],
            "unit": "%",
        })
        response_text = (
            "## Expense Breakdown\n\n"
            "| Category | Share |\n|----------|------|\n"
            "| Salaries | **55%** |\n| Cloud | **20%** |\n"
            "| Marketing | **12%** |\n| Office | **8%** |\n"
            "| Travel | **5%** |\n\n"
            f"Salaries account for more than half of total expenses.\n\n{directive}"
        )
        answer = _build_answer(response_text)
        result = enhance_with_visualization(answer, "What is the expense breakdown?", channel="web")

        media = _get_media(result)
        assert len(media) >= 1
        assert media[0]["chart_type"] == "donut"
        assert "DOCWAIN_VIZ" not in _get_response(result)


# ---------------------------------------------------------------------------
# Test 3: Model-driven line
# ---------------------------------------------------------------------------

class TestModelDrivenLine:
    def test_line_directive_produces_media(self):
        directive = _viz_directive({
            "chart_type": "line",
            "title": "Monthly Revenue Trend",
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "values": [400, 420, 390, 450, 480, 510],
            "unit": "$K",
        })
        response_text = (
            "## Monthly Revenue Trend\n\n"
            "| Month | Revenue |\n|-------|--------|\n"
            "| Jan | **$400K** |\n| Feb | **$420K** |\n"
            "| Mar | **$390K** |\n| Apr | **$450K** |\n"
            "| May | **$480K** |\n| Jun | **$510K** |\n\n"
            f"Revenue trended upward, reaching $510K in June.\n\n{directive}"
        )
        answer = _build_answer(response_text)
        result = enhance_with_visualization(answer, "Show the monthly revenue trend", channel="web")

        media = _get_media(result)
        assert len(media) >= 1
        assert media[0]["chart_type"] == "line"
        assert "DOCWAIN_VIZ" not in _get_response(result)


# ---------------------------------------------------------------------------
# Test 4: Model-driven radar with secondary values
# ---------------------------------------------------------------------------

class TestModelDrivenRadar:
    def test_radar_with_secondary_values(self):
        directive = _viz_directive({
            "chart_type": "radar",
            "title": "Candidate Comparison",
            "labels": ["Communication", "Technical", "Leadership", "Teamwork", "Problem Solving"],
            "values": [8, 9, 7, 9, 8],
            "secondary_values": [7, 6, 9, 8, 7],
            "secondary_name": "Candidate B",
            "series_name": "Candidate A",
        })
        response_text = (
            "## Candidate Comparison\n\n"
            "| Competency | Candidate A | Candidate B |\n"
            "|------------|------------|------------|\n"
            "| Communication | **8** | 7 |\n"
            "| Technical | **9** | 6 |\n"
            "| Leadership | 7 | **9** |\n"
            "| Teamwork | **9** | 8 |\n"
            "| Problem Solving | **8** | 7 |\n\n"
            f"Candidate A excels technically; Candidate B leads on leadership.\n\n{directive}"
        )
        answer = _build_answer(response_text)
        result = enhance_with_visualization(answer, "Compare the two candidates", channel="web")

        media = _get_media(result)
        assert len(media) >= 1
        assert media[0]["chart_type"] == "radar"
        assert "DOCWAIN_VIZ" not in _get_response(result)


# ---------------------------------------------------------------------------
# Test 5: Fallback to auto-detect (no VIZ directive, table with chart keyword)
# ---------------------------------------------------------------------------

class TestFallbackAutoDetect:
    def test_auto_detect_from_table_and_keyword(self):
        response_text = (
            "## Invoice Line Items\n\n"
            "| Vendor | Amount |\n|--------|--------|\n"
            "| Acme Corp | $45,000 |\n"
            "| GlobalTech | $32,000 |\n"
            "| NovaSys | $28,000 |\n"
            "| PrimeLine | $51,000 |\n"
            "| DataFlow | $19,000 |\n\n"
            "PrimeLine has the highest invoice total at $51,000."
        )
        answer = _build_answer(response_text)
        # Query contains a chart keyword to trigger is_user_triggered
        result = enhance_with_visualization(answer, "Show me a chart of invoice totals by vendor", channel="web")

        media = _get_media(result)
        assert len(media) >= 1, "Auto-detect should produce a chart for table data with chart keyword"
        assert media[0]["type"] == "chart"


# ---------------------------------------------------------------------------
# Test 6: No chart for text-only response
# ---------------------------------------------------------------------------

class TestNoChartForTextOnly:
    def test_text_only_no_media(self):
        response_text = 'The contract expires **January 1, 2027**.'
        answer = _build_answer(response_text)
        result = enhance_with_visualization(answer, "When does the contract expire?", channel="web")

        media = _get_media(result)
        # Should have no chart media (chart_validation entries may be present
        # but there should be no type=="chart" entry)
        chart_media = [m for m in media if m.get("type") == "chart"]
        assert len(chart_media) == 0, "Text-only response should not generate a chart"


# ---------------------------------------------------------------------------
# Test 7: VIZ directive with secondary values parsing
# ---------------------------------------------------------------------------

class TestParseVizDirective:
    def test_parse_secondary_values(self):
        directive_json = {
            "chart_type": "grouped_bar",
            "title": "Q1 vs Q2 Expenses",
            "labels": ["Salaries", "Cloud", "Travel"],
            "values": [800, 120, 45],
            "secondary_values": [820, 150, 60],
            "secondary_name": "Q2",
            "unit": "$K",
        }
        text = f"Some analysis text.\n\n{_viz_directive(directive_json)}"
        spec = parse_viz_directive(text)

        assert spec is not None, "Should parse a valid VIZ directive"
        assert spec["chart_type"] == "grouped_bar"
        assert spec["labels"] == ["Salaries", "Cloud", "Travel"]
        assert spec["values"] == [800, 120, 45]
        assert spec["secondary_values"] == [820, 150, 60]
        assert spec["secondary_name"] == "Q2"

    def test_parse_returns_none_for_missing_fields(self):
        text = '<!--DOCWAIN_VIZ {"chart_type": "bar"} -->'
        spec = parse_viz_directive(text)
        assert spec is None, "Missing labels/values should return None"

    def test_parse_returns_none_for_no_directive(self):
        text = "Just some regular text without any directives."
        spec = parse_viz_directive(text)
        assert spec is None


# ---------------------------------------------------------------------------
# Test 8: Full SFT + DPO data pipeline
# ---------------------------------------------------------------------------

class TestTrainingDataPipeline:
    def test_sft_examples_count_and_structure(self):
        examples = generate_viz_sft_examples()
        assert len(examples) >= 50, f"Expected >= 50 SFT examples, got {len(examples)}"

        for ex in examples:
            assert "messages" in ex
            msgs = ex["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"
            assert isinstance(msgs[1]["content"], str)
            assert isinstance(msgs[2]["content"], str)
            assert len(msgs[1]["content"]) > 0
            assert len(msgs[2]["content"]) > 0

    def test_dpo_pairs_count_and_structure(self):
        pairs = generate_viz_preference_pairs()
        assert len(pairs) >= 10, f"Expected >= 10 DPO pairs, got {len(pairs)}"

        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair
            assert isinstance(pair["prompt"], str)
            assert isinstance(pair["chosen"], str)
            assert isinstance(pair["rejected"], str)
            assert len(pair["chosen"]) > 0
            assert len(pair["rejected"]) > 0


# ---------------------------------------------------------------------------
# Test 9: Viz evaluator scores correctly
# ---------------------------------------------------------------------------

class TestVizEvaluator:
    def test_good_chart_response_high_score(self):
        """Score a known-good chart response (directive + correct type + table)."""
        directive_json = {
            "chart_type": "bar",
            "title": "Quarterly Revenue",
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "values": [1.2, 1.5, 1.8, 2.1],
        }
        response = (
            "## Quarterly Revenue\n\n"
            "| Quarter | Revenue |\n|---------|--------|\n"
            "| Q1 | $1.2M |\n| Q2 | $1.5M |\n| Q3 | $1.8M |\n| Q4 | $2.1M |\n\n"
            f"{_viz_directive(directive_json)}"
        )
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
            expected_labels=["Q1", "Q2", "Q3", "Q4"],
            expected_values=[1.2, 1.5, 1.8, 2.1],
        )
        assert result["composite"] >= 80, f"Good response should score >= 80, got {result['composite']}"
        assert result["has_directive"] is True
        assert result["has_table"] is True

    def test_bad_response_low_score(self):
        """Score a bad response (no directive when expected)."""
        response = (
            "The quarterly revenue was Q1: $1.2M, Q2: $1.5M, Q3: $1.8M, Q4: $2.1M. "
            "Revenue grew steadily throughout the year."
        )
        result = score_viz_response(
            response=response,
            expected_chart_type="bar",
            expects_chart=True,
            expected_labels=["Q1", "Q2", "Q3", "Q4"],
            expected_values=[1.2, 1.5, 1.8, 2.1],
        )
        assert result["composite"] < 50, f"Bad response should score < 50, got {result['composite']}"
        assert result["has_directive"] is False

    def test_text_only_when_no_chart_expected(self):
        """Score text-only when no chart expected."""
        response = 'The contract expires **January 1, 2027** (24-month term from January 1, 2025).'
        result = score_viz_response(
            response=response,
            expected_chart_type=None,
            expects_chart=False,
        )
        assert result["composite"] >= 80, f"Text-only (no chart expected) should score >= 80, got {result['composite']}"
        assert result["has_directive"] is False
        assert result["mode_correct"] is True


# ---------------------------------------------------------------------------
# Test 10: Model-driven grouped_bar
# ---------------------------------------------------------------------------

class TestModelDrivenGroupedBar:
    def test_grouped_bar_with_secondary_values(self):
        directive = _viz_directive({
            "chart_type": "grouped_bar",
            "title": "Planned vs Actual Budget",
            "labels": ["IT", "HR", "Ops"],
            "values": [500, 200, 350],
            "secondary_values": [480, 210, 360],
            "secondary_name": "Actual",
            "series_name": "Planned",
            "unit": "$K",
        })
        response_text = (
            "## Planned vs Actual Budget\n\n"
            "| Dept | Planned | Actual |\n|------|---------|--------|\n"
            "| IT | **$500K** | **$480K** |\n"
            "| HR | **$200K** | **$210K** |\n"
            "| Ops | **$350K** | **$360K** |\n\n"
            f"IT came in under budget while HR and Ops slightly exceeded targets.\n\n{directive}"
        )
        answer = _build_answer(response_text)
        result = enhance_with_visualization(answer, "Show planned vs actual budget", channel="web")

        media = _get_media(result)
        assert len(media) >= 1
        assert media[0]["chart_type"] == "grouped_bar"
        assert "DOCWAIN_VIZ" not in _get_response(result)


# ---------------------------------------------------------------------------
# Test 11: Model-driven waterfall
# ---------------------------------------------------------------------------

class TestModelDrivenWaterfall:
    def test_waterfall_directive_produces_media(self):
        directive = _viz_directive({
            "chart_type": "waterfall",
            "title": "Profit & Loss Waterfall",
            "labels": ["Revenue", "COGS", "Gross Profit", "Opex", "Tax", "Net Profit"],
            "values": [5.0, -2.0, 3.0, -1.5, -0.4, 1.1],
            "unit": "$M",
        })
        response_text = (
            "## Profit & Loss Waterfall\n\n"
            "| Item | Amount |\n|------|-------|\n"
            "| Revenue | **$5.0M** |\n"
            "| COGS | -$2.0M |\n"
            "| Gross Profit | **$3.0M** |\n"
            "| Opex | -$1.5M |\n"
            "| Tax | -$0.4M |\n"
            "| **Net Profit** | **$1.1M** |\n\n"
            f"{directive}"
        )
        answer = _build_answer(response_text)
        result = enhance_with_visualization(answer, "Show the profit and loss waterfall", channel="web")

        media = _get_media(result)
        assert len(media) >= 1
        assert media[0]["chart_type"] == "waterfall"
        assert "DOCWAIN_VIZ" not in _get_response(result)
