import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docwain_ft.eval_harness import run_eval  # noqa: E402


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
