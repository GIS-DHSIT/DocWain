import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DOCWAIN_FT_SRC = ROOT_DIR / "docwain-gptoss-finetune-pipeline" / "src"
if DOCWAIN_FT_SRC.exists() and str(DOCWAIN_FT_SRC) not in sys.path:
    sys.path.insert(0, str(DOCWAIN_FT_SRC))


def load_eval_outputs():
    from docwain_ft.eval_harness import run_eval

    out_path = Path(tempfile.gettempdir()) / "docwain_eval_outputs.jsonl"
    run_eval("data/docwain_eval.jsonl", str(out_path), mock=True)
    rows = []
    with out_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(__import__("json").loads(line))
    return rows
