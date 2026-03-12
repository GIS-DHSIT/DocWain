import json
from src.utils.logging_utils import get_logger
from pathlib import Path
from typing import Dict, Iterable

logger = get_logger(__name__)

class DatasetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write_examples(self, examples: Iterable[Dict]):
        for ex in examples:
            self.file.write(json.dumps(ex, ensure_ascii=False) + "\n")
            self.count += 1

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
        logger.info("Wrote %d training pairs to %s", self.count, self.path)

def load_dataset_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)
