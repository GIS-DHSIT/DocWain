import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from docwain_ft.infer import run_one
from docwain_ft.utils import read_jsonl, setup_logging, write_jsonl


def ollama_available() -> bool:
    return shutil.which("ollama") is not None


def run_eval(eval_path: str, out_path: str, mock: bool = False) -> None:
    rows = read_jsonl(eval_path)
    outputs = []
    for row in rows:
        prompt = row["prompt"]
        if mock or not ollama_available():
            answer = row.get("expected_output", "")
        else:
            answer = run_one(prompt)
        outputs.append({"prompt": prompt, "answer": answer, "context": row.get("context", ""), "output_rules": row.get("output_rules", "")})
    write_jsonl(out_path, outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DocWain eval harness.")
    parser.add_argument("--data", default="data/docwain_eval.jsonl")
    parser.add_argument("--out", default="out/eval_outputs.jsonl")
    parser.add_argument("--mock", action="store_true", help="Use expected outputs instead of Ollama.")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("eval_harness")
    run_eval(args.data, args.out, mock=args.mock)
    logger.info("Eval outputs written to %s", args.out)


if __name__ == "__main__":
    main()
