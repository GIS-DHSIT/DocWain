import tempfile
from pathlib import Path

from docwain_ft.eval_harness import run_eval


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: requires ollama runtime")


def load_eval_outputs():
    out_path = Path(tempfile.gettempdir()) / "docwain_eval_outputs.jsonl"
    run_eval("data/docwain_eval.jsonl", str(out_path), mock=True)
    rows = []
    with open(out_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(__import__("json").loads(line))
    return rows
