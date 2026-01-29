from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Registry:
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def update_model(self, name: str, payload: Dict[str, Any]) -> None:
        self.models[name] = payload

    def merge_models(self, models: List[Dict[str, Any]]) -> None:
        for model in models:
            name = model.get("name")
            if not name:
                continue
            self.models[name] = model

    def serialize(self) -> str:
        payload = {
            "models": list(self.models.values()),
            "meta": self.meta,
        }
        return json.dumps(payload, ensure_ascii=True)

    @classmethod
    def deserialize(cls, data: str) -> "Registry":
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialize registry JSON")
            return cls()
        models_raw = payload.get("models")
        meta_raw = payload.get("meta")
        registry = cls()
        if isinstance(meta_raw, dict):
            registry.meta = meta_raw
        if isinstance(models_raw, list):
            for model in models_raw:
                if isinstance(model, dict) and model.get("name"):
                    registry.models[model["name"]] = model
        return registry

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        return self.models.get(name)

    def list_models(self) -> List[str]:
        return sorted(self.models.keys())
