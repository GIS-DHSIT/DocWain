"""Model registry — tracks trained models, handles promotion and rollback."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelEntry:
    tag: str
    base: str
    iteration: int
    composite_score: float
    scores: Dict[str, float]
    artifact_path: str
    status: str  # "production" | "available" | "rollback_ready"
    promoted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelRegistry:
    """YAML-backed model registry with promote/rollback operations."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._models: Dict[str, ModelEntry] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}
        for tag, data in raw.get("models", {}).items():
            data["tag"] = tag
            self._models[tag] = ModelEntry(**data)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Any] = {"models": {}}
        for tag, entry in self._models.items():
            d = asdict(entry)
            d.pop("tag")
            out["models"][tag] = d
        with open(self._path, "w") as f:
            yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    def register(self, entry: ModelEntry) -> None:
        self._models[entry.tag] = entry
        self._save()

    def get(self, tag: str) -> Optional[ModelEntry]:
        return self._models.get(tag)

    def list_models(self) -> List[ModelEntry]:
        return list(self._models.values())

    def promote(self, new_latest: ModelEntry) -> None:
        old = self._models.get("DocWain:latest")
        if old:
            old.tag = "DocWain:previous"
            old.status = "rollback_ready"
            self._models["DocWain:previous"] = old
        new_latest.tag = "DocWain:latest"
        new_latest.status = "production"
        new_latest.promoted_at = datetime.now(timezone.utc).isoformat()
        self._models["DocWain:latest"] = new_latest
        self._save()

    def rollback(self) -> None:
        prev = self._models.get("DocWain:previous")
        curr = self._models.get("DocWain:latest")
        if not prev:
            raise ValueError("No previous model to rollback to")
        prev.tag = "DocWain:latest"
        prev.status = "production"
        self._models["DocWain:latest"] = prev
        if curr:
            curr.tag = "DocWain:previous"
            curr.status = "rollback_ready"
            self._models["DocWain:previous"] = curr
        self._save()
