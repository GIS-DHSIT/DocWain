# tests/test_evolve_registry.py
import json
import pytest
from pathlib import Path
from datetime import datetime


class TestModelRegistry:
    """Tests for model registry CRUD operations."""

    def test_init_creates_empty_registry(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry
        reg = ModelRegistry(tmp_path / "registry.yaml")
        assert reg.list_models() == []

    def test_register_model(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry, ModelEntry
        reg = ModelRegistry(tmp_path / "registry.yaml")
        entry = ModelEntry(
            tag="DocWain:latest",
            base="qwen3-8b",
            iteration=1,
            composite_score=84.2,
            scores={"accuracy": 87, "groundedness": 83, "reasoning": 82, "formatting": 81, "tone": 79},
            artifact_path="finetune_artifacts/iter_1/",
            status="production",
        )
        reg.register(entry)
        models = reg.list_models()
        assert len(models) == 1
        assert models[0].tag == "DocWain:latest"

    def test_get_model_by_tag(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry, ModelEntry
        reg = ModelRegistry(tmp_path / "registry.yaml")
        entry = ModelEntry(tag="DocWain:latest", base="qwen3-8b", iteration=1,
                           composite_score=84.2, scores={}, artifact_path="", status="production")
        reg.register(entry)
        found = reg.get("DocWain:latest")
        assert found is not None
        assert found.base == "qwen3-8b"

    def test_get_nonexistent_returns_none(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry
        reg = ModelRegistry(tmp_path / "registry.yaml")
        assert reg.get("nonexistent") is None

    def test_promote_demotes_previous_latest(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry, ModelEntry
        reg = ModelRegistry(tmp_path / "registry.yaml")
        old = ModelEntry(tag="DocWain:latest", base="qwen3-8b", iteration=1,
                         composite_score=80.0, scores={}, artifact_path="iter_1/", status="production")
        reg.register(old)
        new = ModelEntry(tag="DocWain:latest", base="qwen3-8b", iteration=2,
                         composite_score=85.0, scores={}, artifact_path="iter_2/", status="production")
        reg.promote(new)
        prev = reg.get("DocWain:previous")
        assert prev is not None
        assert prev.iteration == 1
        assert prev.status == "rollback_ready"
        latest = reg.get("DocWain:latest")
        assert latest.iteration == 2

    def test_persist_and_reload(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry, ModelEntry
        path = tmp_path / "registry.yaml"
        reg = ModelRegistry(path)
        entry = ModelEntry(tag="DocWain:latest", base="qwen3-8b", iteration=1,
                           composite_score=84.2, scores={"accuracy": 87}, artifact_path="iter_1/", status="production")
        reg.register(entry)
        reg2 = ModelRegistry(path)
        found = reg2.get("DocWain:latest")
        assert found is not None
        assert found.composite_score == 84.2

    def test_list_available_models(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry, ModelEntry
        reg = ModelRegistry(tmp_path / "registry.yaml")
        reg.register(ModelEntry(tag="DocWain:latest", base="qwen3-8b", iteration=1,
                                composite_score=84.2, scores={}, artifact_path="", status="production"))
        reg.register(ModelEntry(tag="DocWain:llama", base="llama-3.1-8b", iteration=1,
                                composite_score=79.0, scores={}, artifact_path="", status="available"))
        available = reg.list_models()
        assert len(available) == 2

    def test_rollback_swaps_latest_and_previous(self, tmp_path):
        from src.finetune.evolve.registry import ModelRegistry, ModelEntry
        reg = ModelRegistry(tmp_path / "registry.yaml")
        reg.register(ModelEntry(tag="DocWain:previous", base="qwen3-8b", iteration=1,
                                composite_score=80.0, scores={}, artifact_path="iter_1/", status="rollback_ready"))
        reg.register(ModelEntry(tag="DocWain:latest", base="qwen3-8b", iteration=2,
                                composite_score=85.0, scores={}, artifact_path="iter_2/", status="production"))
        reg.rollback()
        latest = reg.get("DocWain:latest")
        assert latest.iteration == 1
