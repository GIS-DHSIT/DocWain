import argparse
import logging
from pathlib import Path

from docwain_ft.config import CONFIG
from docwain_ft.prompts import DOCWAIN_SYSTEM_ANCHOR
from docwain_ft.utils import ensure_dir, setup_logging


STOP_TOKENS = ["<|system|>", "<|assistant|>", "<|user|>"]


def generate_modelfile(adapter_path: Path | None, merged_path: Path | None, out_path: Path) -> None:
    if adapter_path and adapter_path.exists():
        from_line = f"FROM {CONFIG.ollama_base}"
        adapter_line = f"ADAPTER ./artifacts/{adapter_path.name}"
    elif merged_path and merged_path.exists():
        from_line = f"FROM ./artifacts/{merged_path.name}"
        adapter_line = None
    else:
        raise FileNotFoundError("No adapter or merged GGUF file found for Modelfile.")

    lines = [from_line]
    if adapter_line:
        lines.append(adapter_line)

    lines.append("SYSTEM \"" + DOCWAIN_SYSTEM_ANCHOR.replace("\"", "'") + "\"")
    lines.append(f"PARAMETER temperature {CONFIG.temperature}")
    lines.append(f"PARAMETER top_p {CONFIG.top_p}")
    lines.append(f"PARAMETER repeat_penalty {CONFIG.repeat_penalty}")
    lines.append(f"PARAMETER num_ctx {CONFIG.num_ctx}")
    for token in STOP_TOKENS:
        lines.append(f"PARAMETER stop {token}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Ollama Modelfile for DocWain.")
    parser.add_argument("--artifacts", default="ollama/artifacts")
    parser.add_argument("--modelfile", default="ollama/Modelfile")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("ollama_packager")

    artifacts_dir = ensure_dir(args.artifacts)
    adapter_path = artifacts_dir / "docwain_adapter.gguf"
    merged_path = artifacts_dir / "docwain_merged.gguf"

    modelfile_path = Path(args.modelfile)
    ensure_dir(modelfile_path.parent)

    generate_modelfile(adapter_path, merged_path, modelfile_path)
    logger.info("Generated Modelfile at %s", modelfile_path)


if __name__ == "__main__":
    main()
