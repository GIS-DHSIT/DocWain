from src.docwain_intel.fact_cache import FactCache
from src.docwain_intel.hr_renderers import render_task
from src.docwain_intel.intent_router import TASK_1, TASK_6


def test_not_mentioned_only_in_task6():
    cache = FactCache()
    task1 = render_task(TASK_1, cache, [])
    task6 = render_task(TASK_6, cache, [])
    assert "Not Mentioned" not in task1
    assert "Not Mentioned" in task6
