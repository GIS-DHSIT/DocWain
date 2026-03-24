# tests/test_evolve_config.py
import pytest
from pathlib import Path


class TestEvolveConfig:
    """Tests for evolve pipeline configuration."""

    def test_load_default_config(self):
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig.load_default()
        assert cfg.pipeline.scheduled_interval_hours == 24
        assert cfg.pipeline.signal_threshold == 50
        assert cfg.pipeline.distillation_every_n == 3
        assert cfg.pipeline.eval_prompt_count == 200

    def test_gate_weights_sum_to_one(self):
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig.load_default()
        total = sum(cfg.gate.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_gate_defaults(self):
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig.load_default()
        assert cfg.gate.composite_minimum == 80.0
        assert cfg.gate.criterion_floor == 60.0
        assert cfg.gate.must_beat_previous is True

    def test_models_has_primary_and_students(self):
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig.load_default()
        assert cfg.models.primary is not None
        assert len(cfg.models.students) >= 1

    def test_student_model_has_required_fields(self):
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig.load_default()
        for s in cfg.models.students:
            assert s.name
            assert s.repo
            assert isinstance(s.enabled, bool)

    def test_load_from_yaml(self, tmp_path):
        from src.finetune.evolve.config import EvolveConfig
        yaml_content = """
pipeline:
  scheduled_interval_hours: 12
  signal_threshold: 100
  distillation_every_n: 5
  eval_prompt_count: 50
models:
  primary: "test/model"
  students:
    - name: "test"
      repo: "test/repo"
      enabled: true
training:
  sft:
    epochs: 2
    learning_rate: 0.001
    lora_r: 8
    lora_alpha: 8
    batch_size: 2
  dpo:
    epochs: 1
    beta: 0.1
    learning_rate: 0.00005
gate:
  composite_minimum: 75.0
  criterion_floor: 55.0
  must_beat_previous: false
  weights:
    accuracy: 0.30
    groundedness: 0.25
    reasoning: 0.20
    formatting: 0.15
    tone: 0.10
deployment:
  target: ollama
  ollama_host: "http://localhost:11434"
  keep_previous: true
  max_stored_iterations: 5
docwain:
  endpoint: "http://localhost:11434"
  model_name: "DHS/DocWain"
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = EvolveConfig.load_from(yaml_file)
        assert cfg.pipeline.scheduled_interval_hours == 12
        assert cfg.pipeline.signal_threshold == 100
        assert cfg.gate.composite_minimum == 75.0

    def test_training_sft_defaults(self):
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig.load_default()
        assert cfg.training.sft.epochs == 3
        assert cfg.training.sft.lora_r == 16
        assert cfg.training.sft.batch_size == 4

    def test_training_dpo_defaults(self):
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig.load_default()
        assert cfg.training.dpo.epochs == 1
        assert cfg.training.dpo.beta == 0.1
