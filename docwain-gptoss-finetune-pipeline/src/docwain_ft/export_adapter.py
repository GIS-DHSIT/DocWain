import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from docwain_ft.config import CONFIG
from docwain_ft.utils import ensure_dir, setup_logging


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def export_adapter_gguf(base_model: str, lora_dir: Path, out_path: Path) -> bool:
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False

    script = [
        "python",
        "-m",
        "llama_cpp.convert_lora_to_gguf",
        "--base",
        base_model,
        "--lora",
        str(lora_dir),
        "--out",
        str(out_path),
    ]
    try:
        _run(script)
        return out_path.exists()
    except Exception:
        return False


def export_merged_gguf(base_model: str, lora_dir: Path, merged_dir: Path, out_path: Path) -> bool:
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    model = PeftModel.from_pretrained(model, str(lora_dir))
    merged = model.merge_and_unload()
    ensure_dir(merged_dir)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.save_pretrained(merged_dir)

    script = [
        "python",
        "-m",
        "llama_cpp.convert",
        "--outfile",
        str(out_path),
        str(merged_dir),
    ]
    try:
        _run(script)
        return out_path.exists()
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DocWain LoRA adapter for Ollama.")
    parser.add_argument("--lora-dir", default="out/lora")
    parser.add_argument("--out", default="ollama/artifacts")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("export_adapter")

    lora_dir = Path(args.lora_dir)
    out_dir = ensure_dir(args.out)
    adapter_path = out_dir / "docwain_adapter.gguf"
    merged_path = out_dir / "docwain_merged.gguf"
    merged_dir = Path("out/merged")

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")

    logger.info("Attempting GGUF adapter export")
    if export_adapter_gguf(CONFIG.hf_base_model, lora_dir, adapter_path):
        logger.info("Adapter export succeeded: %s", adapter_path)
        return

    logger.warning("Adapter export failed; attempting merged GGUF export")
    if export_merged_gguf(CONFIG.hf_base_model, lora_dir, merged_dir, merged_path):
        logger.info("Merged GGUF export succeeded: %s", merged_path)
        return

    raise RuntimeError("Failed to export adapter or merged model. Ensure llama-cpp-python is installed.")


if __name__ == "__main__":
    main()
