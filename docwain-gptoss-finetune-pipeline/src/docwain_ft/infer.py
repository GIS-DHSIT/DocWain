import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import requests

from docwain_ft.config import CONFIG
from docwain_ft.redaction import redact
from docwain_ft.utils import read_jsonl, setup_logging, write_jsonl


OLLAMA_URL = "http://localhost:11434/api/generate"


def _ollama_http(prompt: str) -> str | None:
    payload = {
        "model": CONFIG.ollama_model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": CONFIG.temperature,
            "top_p": CONFIG.top_p,
            "repeat_penalty": CONFIG.repeat_penalty,
            "num_ctx": CONFIG.num_ctx,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception:
        return None


def _ollama_cli(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", CONFIG.ollama_model_name, prompt],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def ollama_ready() -> bool:
    if shutil.which("ollama") is None:
        return False
    result = subprocess.run(
        ["ollama", "show", CONFIG.ollama_model_name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_one(prompt: str) -> str:
    response = _ollama_http(prompt)
    if response is None:
        response = _ollama_cli(prompt)
    return redact(response)


def run_batch(jsonl_file: str, out_file: str) -> None:
    rows = read_jsonl(jsonl_file)
    outputs = []
    for row in rows:
        prompt = row["prompt"]
        answer = run_one(prompt)
        outputs.append({"prompt": prompt, "answer": answer})
    write_jsonl(out_file, outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DocWain inference via Ollama.")
    parser.add_argument("--prompt", help="Single prompt to run.")
    parser.add_argument("--batch", help="JSONL file with prompts.")
    parser.add_argument("--out", default="out/predictions.jsonl")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("infer")

    if args.prompt:
        output = run_one(args.prompt)
        print(output)
        return

    if args.batch:
        run_batch(args.batch, args.out)
        logger.info("Wrote predictions to %s", args.out)
        return

    raise SystemExit("Provide --prompt or --batch.")


if __name__ == "__main__":
    main()
