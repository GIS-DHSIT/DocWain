"""Tests that visualization directive instructions are present in prompts."""

import pytest
from src.generation.prompts import build_system_prompt, TASK_FORMATS


def test_system_prompt_contains_docwain_viz():
    prompt = build_system_prompt()
    assert "DOCWAIN_VIZ" in prompt


def test_task_format_aggregate_contains_docwain_viz():
    assert "DOCWAIN_VIZ" in TASK_FORMATS["aggregate"]


def test_task_format_compare_contains_docwain_viz():
    assert "DOCWAIN_VIZ" in TASK_FORMATS["compare"]


def test_task_format_extract_contains_docwain_viz():
    assert "DOCWAIN_VIZ" in TASK_FORMATS["extract"]


def test_task_format_summarize_contains_docwain_viz():
    assert "DOCWAIN_VIZ" in TASK_FORMATS["summarize"]
