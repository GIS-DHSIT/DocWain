import json
import logging
import os
from pathlib import Path
from typing import Iterable


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path_obj = Path(path)
    if not path_obj.is_absolute() and not path_obj.exists():
        base_dir = Path(__file__).resolve().parents[2]
        candidate = base_dir / path_obj
        if candidate.exists():
            path_obj = candidate
    with open(path_obj, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}
