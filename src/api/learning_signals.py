import json
from src.utils.logging_utils import get_logger
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.config import Config

logger = get_logger(__name__)

class LearningSignalStore:
    """Persist lightweight online learning signals for fine-tuning and evaluation."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or (Config.Path.APP_HOME / "outputs" / "learning_signals")
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create learning signal directory: %s", exc)

    def record_high_quality(
        self,
        query: str,
        context: str,
        answer: str,
        sources: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        payload = {
            "query": query,
            "context": context,
            "answer": answer,
            "sources": sources,
            "metadata": metadata,
        }
        self._append("high_quality.jsonl", payload)
        training_example = {
            "instruction": query,
            "input": context,
            "output": answer,
            "source": {
                "strategy": "online_feedback",
                "document_ids": metadata.get("document_ids") or [],
                "profile_id": metadata.get("profile_id"),
            },
        }
        self._append("finetune_buffer.jsonl", training_example)

    def record_low_confidence(
        self,
        query: str,
        context: str,
        answer: str,
        reason: str,
        metadata: Dict[str, Any],
    ) -> None:
        payload = {
            "query": query,
            "context": context,
            "answer": answer,
            "reason": reason,
            "metadata": metadata,
        }
        self._append("low_confidence.jsonl", payload)

    def record_failure(
        self,
        query: str,
        reason: str,
        metadata: Dict[str, Any],
    ) -> None:
        payload = {
            "query": query,
            "reason": reason,
            "metadata": metadata,
        }
        self._append("failures.jsonl", payload)

    def _append(self, filename: str, payload: Dict[str, Any]) -> None:
        try:
            path = self.base_dir / filename
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to write learning signal %s: %s", filename, exc)
