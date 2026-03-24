# tests/test_evolve_pipeline.py
import json
import pytest
from pathlib import Path


class TestPipelineState:
    def test_get_current_iteration_empty(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=tmp_path / "signals", artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        assert pipe.current_iteration() == 0

    def test_get_current_iteration_with_existing(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        signals = tmp_path / "signals"
        (signals / "iter_1").mkdir(parents=True)
        (signals / "iter_2").mkdir(parents=True)
        (signals / "iter_3").mkdir(parents=True)
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=signals, artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        assert pipe.current_iteration() == 3

    def test_next_iteration(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        signals = tmp_path / "signals"
        (signals / "iter_1").mkdir(parents=True)
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=signals, artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        assert pipe.next_iteration() == 2


class TestPipelineComponents:
    def test_creates_observer(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        from src.finetune.evolve.observer import Observer
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=tmp_path / "signals", artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        assert isinstance(pipe.observer, Observer)

    def test_creates_harvester(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        from src.finetune.evolve.harvester import Harvester
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=tmp_path / "signals", artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        assert isinstance(pipe.harvester, Harvester)

    def test_creates_teacher(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        from src.finetune.evolve.teacher import Teacher
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=tmp_path / "signals", artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        assert isinstance(pipe.teacher, Teacher)

    def test_creates_gate(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        from src.finetune.evolve.gate import QualityGate
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=tmp_path / "signals", artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        assert isinstance(pipe.gate, QualityGate)

    def test_pipeline_status_summary(self, tmp_path):
        from src.finetune.evolve.pipeline import EvolvePipeline
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig()
        pipe = EvolvePipeline(cfg, signals_dir=tmp_path / "signals", artifact_dir=tmp_path / "artifacts", registry_path=tmp_path / "registry.yaml")
        status = pipe.status()
        assert "current_iteration" in status
        assert "enabled_students" in status
        assert "registry_models" in status
