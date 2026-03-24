"""Tests for the DOCWAIN_VIZ directive parser and enhancer integration."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field

from src.visualization.enhancer import parse_viz_directive, _strip_viz_directive


# ---------------------------------------------------------------------------
# parse_viz_directive() unit tests
# ---------------------------------------------------------------------------


def test_parse_valid_directive():
    """parse_viz_directive returns valid dict for response with VIZ directive."""
    text = (
        'Here is the data.\n'
        '<!--DOCWAIN_VIZ {"chart_type": "bar", "labels": ["A", "B"], '
        '"values": [10, 20], "title": "Test"} -->\n'
        'Some trailing text.'
    )
    result = parse_viz_directive(text)
    assert result is not None
    assert result["chart_type"] == "bar"
    assert result["labels"] == ["A", "B"]
    assert result["values"] == [10, 20]
    assert result["title"] == "Test"


def test_parse_returns_none_without_directive():
    """parse_viz_directive returns None for text without directive."""
    text = "This is a normal response with no chart directive at all."
    assert parse_viz_directive(text) is None


def test_parse_returns_none_for_invalid_json():
    """parse_viz_directive returns None for invalid JSON in directive."""
    text = '<!--DOCWAIN_VIZ {not valid json} -->'
    assert parse_viz_directive(text) is None


def test_parse_returns_none_missing_required_fields():
    """parse_viz_directive returns None when required fields missing."""
    # Missing "values"
    text = '<!--DOCWAIN_VIZ {"chart_type": "bar", "labels": ["A", "B"]} -->'
    assert parse_viz_directive(text) is None


def test_parse_returns_none_empty_labels():
    """parse_viz_directive returns None when labels list is empty."""
    text = '<!--DOCWAIN_VIZ {"chart_type": "bar", "labels": [], "values": [10]} -->'
    assert parse_viz_directive(text) is None


def test_parse_returns_none_empty_values():
    """parse_viz_directive returns None when values list is empty."""
    text = '<!--DOCWAIN_VIZ {"chart_type": "bar", "labels": ["A"], "values": []} -->'
    assert parse_viz_directive(text) is None


def test_parse_secondary_values():
    """Secondary values are parsed correctly."""
    text = (
        '<!--DOCWAIN_VIZ {"chart_type": "grouped_bar", '
        '"labels": ["Q1", "Q2", "Q3"], '
        '"values": [100, 200, 150], '
        '"secondary_values": [80, 180, 130], '
        '"secondary_name": "Projected", '
        '"series_name": "Actual"} -->'
    )
    result = parse_viz_directive(text)
    assert result is not None
    assert result["secondary_values"] == [80, 180, 130]
    assert result["secondary_name"] == "Projected"
    assert result["series_name"] == "Actual"


# ---------------------------------------------------------------------------
# _strip_viz_directive() tests
# ---------------------------------------------------------------------------


def test_strip_viz_directive_nested():
    """VIZ directive is stripped from nested response text."""
    answer = {
        "answer": {
            "response": (
                'Here is the data.\n'
                '<!--DOCWAIN_VIZ {"chart_type": "bar", "labels": ["A"], "values": [1]} -->\n'
                'More text.'
            )
        }
    }
    _strip_viz_directive(answer)
    response = answer["answer"]["response"]
    assert "DOCWAIN_VIZ" not in response
    assert "Here is the data." in response
    assert "More text." in response


def test_strip_viz_directive_flat():
    """VIZ directive is stripped from flat response text."""
    answer = {
        "response": (
            'Result: <!--DOCWAIN_VIZ {"chart_type": "pie", "labels": ["X"], "values": [5]} --> done.'
        )
    }
    _strip_viz_directive(answer)
    assert "DOCWAIN_VIZ" not in answer["response"]


# ---------------------------------------------------------------------------
# enhance_with_visualization() integration with directive
# ---------------------------------------------------------------------------


@dataclass
class _FakeRendered:
    chart_type: str = "bar"
    title: str = "Test Chart"
    matplotlib_png_base64: str = "fakepng"
    plotly_html: str = ""
    data_summary: str = "2 items"


def test_enhance_uses_model_directive():
    """enhance_with_visualization uses model directive when present and produces correct media."""
    from src.visualization.enhancer import enhance_with_visualization

    answer = {
        "answer": {
            "response": (
                'Expenses breakdown:\n'
                '<!--DOCWAIN_VIZ {"chart_type": "bar", "labels": ["Travel", "Food"], '
                '"values": [500, 300], "title": "Expenses"} -->'
            )
        }
    }

    fake_rendered = _FakeRendered(chart_type="bar", title="Expenses")

    with patch("src.visualization.chart_decision.is_user_triggered", return_value=False), \
         patch("src.visualization.chart_renderer.render_chart", return_value=fake_rendered) as mock_render:
        result = enhance_with_visualization(answer, "show expenses", channel="web")

    # render_chart should have been called with the directive data
    mock_render.assert_called_once()
    call_kwargs = mock_render.call_args
    assert call_kwargs[1]["chart_type"] == "bar" or call_kwargs[0][0] == "bar"

    # Media should be present with correct chart_type
    inner = result.get("answer", result)
    assert "media" in inner
    media = inner["media"]
    assert any(m.get("chart_type") == "bar" for m in media)

    # Directive should be stripped from response
    response_text = inner.get("response", "")
    assert "DOCWAIN_VIZ" not in response_text


def test_enhance_strips_directive_after_rendering():
    """VIZ directive is stripped from response text after rendering."""
    from src.visualization.enhancer import enhance_with_visualization

    directive_text = (
        'Data analysis complete.\n'
        '<!--DOCWAIN_VIZ {"chart_type": "pie", "labels": ["A", "B", "C"], '
        '"values": [30, 50, 20], "title": "Distribution"} -->\n'
        'Please review.'
    )
    answer = {"answer": {"response": directive_text}}
    fake_rendered = _FakeRendered(chart_type="pie", title="Distribution")

    with patch("src.visualization.chart_decision.is_user_triggered", return_value=False), \
         patch("src.visualization.chart_renderer.render_chart", return_value=fake_rendered):
        result = enhance_with_visualization(answer, "show distribution", channel="web")

    inner = result.get("answer", result)
    assert "DOCWAIN_VIZ" not in inner.get("response", "")
    assert "Data analysis complete." in inner.get("response", "")
    assert "Please review." in inner.get("response", "")
