import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_cmd(args: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str] | None:
    logging.getLogger("main").info("Running: %s", " ".join(args))
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if capture:
        return subprocess.run(args, check=True, cwd=cwd, env=env, capture_output=True, text=True)
    subprocess.run(args, check=True, cwd=cwd, env=env)
    return None


def ensure_env() -> None:
    load_dotenv()


def cmd_gen_dataset(cwd: Path) -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "docwain_ft.dataset_gen",
            "--out-sft",
            "data/docwain_sft.jsonl",
            "--out-eval",
            "data/docwain_eval.jsonl",
        ],
        cwd,
    )


def cmd_train_lora(cwd: Path) -> None:
    output_dir = os.getenv("OUTPUT_DIR", "out/")
    run_cmd(
        [
            sys.executable,
            "-m",
            "docwain_ft.train_lora",
            "--data",
            "data/docwain_sft.jsonl",
            "--out",
            output_dir,
        ],
        cwd,
    )


def cmd_export_adapter(cwd: Path) -> None:
    output_dir = os.getenv("OUTPUT_DIR", "out/")
    run_cmd(
        [
            sys.executable,
            "-m",
            "docwain_ft.export_adapter",
            "--lora-dir",
            f"{output_dir.rstrip('/')}/lora",
            "--out",
            "ollama/artifacts",
        ],
        cwd,
    )


def cmd_package_ollama(cwd: Path) -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "docwain_ft.ollama_packager",
            "--artifacts",
            "ollama/artifacts",
            "--modelfile",
            "ollama/Modelfile",
        ],
        cwd,
    )
    model_name = os.getenv("OLLAMA_MODEL_NAME", "DocWain-Agent")
    run_cmd(["ollama", "create", model_name, "--file", "ollama/Modelfile"], cwd)
    output_dir = os.getenv("OUTPUT_DIR", "out/")
    smoke_prompt = (
        "User Query: Provide invoice total.\n"
        "Retrieved Context:\n"
        "- doc_name=Doc_200 page=2 section=Summary chunk_kind=text profile_name=A. Patel\n"
        "  Total due is $250.00.\n"
        "Output Rules:\n"
        "Return a single sentence."
    )
    result = run_cmd(
        ["ollama", "run", model_name, smoke_prompt],
        cwd,
        capture=True,
    )
    smoke_path = Path(cwd) / output_dir / "smoke.txt"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    with open(smoke_path, "w", encoding="utf-8") as handle:
        handle.write(result.stdout if result else "")


def cmd_eval(cwd: Path, mock: bool) -> None:
    output_dir = os.getenv("OUTPUT_DIR", "out/")
    args = [
        sys.executable,
        "-m",
        "docwain_ft.eval_harness",
        "--data",
        "data/docwain_eval.jsonl",
        "--out",
        f"{output_dir.rstrip('/')}/eval_outputs.jsonl",
    ]
    if mock:
        args.append("--mock")
    run_cmd(args, cwd)


def cmd_tests(cwd: Path) -> None:
    run_cmd([sys.executable, "-m", "pytest", "-q"], cwd)


def cmd_all(cwd: Path) -> None:
    cmd_gen_dataset(cwd)
    cmd_train_lora(cwd)
    cmd_export_adapter(cwd)
    cmd_package_ollama(cwd)
    cmd_eval(cwd, mock=False)


def main() -> None:
    setup_logging()
    ensure_env()
    parser = argparse.ArgumentParser(description="DocWain GPT-OSS fine-tune pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gen-dataset")
    sub.add_parser("train-lora")
    sub.add_parser("export-adapter")
    sub.add_parser("package-ollama")
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--mock", action="store_true")
    sub.add_parser("tests")
    sub.add_parser("all")

    args = parser.parse_args()
    cwd = Path(__file__).resolve().parent

    if args.command == "gen-dataset":
        cmd_gen_dataset(cwd)
    elif args.command == "train-lora":
        cmd_train_lora(cwd)
    elif args.command == "export-adapter":
        cmd_export_adapter(cwd)
    elif args.command == "package-ollama":
        cmd_package_ollama(cwd)
    elif args.command == "eval":
        cmd_eval(cwd, mock=args.mock)
    elif args.command == "tests":
        cmd_tests(cwd)
    elif args.command == "all":
        cmd_all(cwd)


if __name__ == "__main__":
    main()
