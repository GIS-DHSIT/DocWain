from src.docwain_intel.fact_cache import FactCache
from src.docwain_intel.hr_renderers import render_task
from src.docwain_intel.intent_router import TASK_1, TASK_2, TASK_3, TASK_4, TASK_5, TASK_6


def _index_of(text: str, token: str) -> int:
    idx = text.find(token)
    assert idx >= 0
    return idx


def test_task1_headings_order():
    cache = FactCache()
    output = render_task(TASK_1, cache, [])
    headings = [
        "Name",
        "Experience Summary",
        "Technical",
        "Functional",
        "Certifications",
        "Education",
        "Achievements/Awards",
        "Source -> Resume",
        "Source -> LinkedIn",
    ]
    last = -1
    for heading in headings:
        idx = _index_of(output, heading)
        assert idx > last
        last = idx


def test_task2_header():
    cache = FactCache()
    output = render_task(TASK_2, cache, [])
    assert output.startswith("Top 5 of with experience between 5 to 10 years")


def test_task3_header():
    cache = FactCache()
    output = render_task(TASK_3, cache, [])
    assert output.startswith("Top 5 of SCM peoples")


def test_task4_header():
    cache = FactCache()
    output = render_task(TASK_4, cache, [])
    assert output.startswith("Top 5 of with experience between 5 to 10 years with certifications")


def test_task5_format():
    cache = FactCache()
    output = render_task(TASK_5, cache, [])
    assert output.startswith("Candidate X:")
    assert "Matching Details:" in output
    assert "Missing Information:" in output
    assert "Inconsistencies:" in output
    assert "Validation Summary:" in output


def test_task6_lists():
    cache = FactCache()
    output = render_task(TASK_6, cache, [])
    assert "List 1: Top 5 Candidates with 5–10 Years Experience" in output
    assert "List 2: Top 5 SCM Professionals" in output
    assert "List 3: Top 5 Candidates with 5–10 Years Experience + Certifications" in output
