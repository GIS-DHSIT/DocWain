# tests/test_evolve_trainer.py
import pytest
from pathlib import Path


class TestTrainerConfig:
    def test_build_training_request(self):
        from src.finetune.evolve.trainer import MultiModelTrainer
        from src.finetune.evolve.config import EvolveConfig, StudentModel
        cfg = EvolveConfig()
        trainer = MultiModelTrainer(cfg)
        student = StudentModel(name="qwen3-8b", repo="unsloth/Qwen3-8B-bnb-4bit")
        req = trainer._build_request(student, dataset_path="/tmp/train.jsonl", iteration=1)
        assert req.base_model == "unsloth/Qwen3-8B-bnb-4bit"
        assert req.num_epochs == 3
        assert req.lora_r == 16

    def test_iteration_artifact_path(self):
        from src.finetune.evolve.trainer import MultiModelTrainer
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig()
        trainer = MultiModelTrainer(cfg, artifact_dir=Path("/tmp/artifacts"))
        path = trainer._artifact_path(iteration=5, model_name="qwen3-8b")
        assert path == Path("/tmp/artifacts/iter_5/models/qwen3-8b")

    def test_list_enabled_students(self):
        from src.finetune.evolve.trainer import MultiModelTrainer
        from src.finetune.evolve.config import EvolveConfig, StudentModel
        cfg = EvolveConfig()
        cfg.models.students = [
            StudentModel(name="qwen", repo="r1", enabled=True),
            StudentModel(name="llama", repo="r2", enabled=False),
            StudentModel(name="phi", repo="r3", enabled=True),
        ]
        trainer = MultiModelTrainer(cfg)
        enabled = trainer.list_enabled_students()
        assert len(enabled) == 2
        assert enabled[0].name == "qwen"
        assert enabled[1].name == "phi"


class TestTrainerDPO:
    def test_build_dpo_args(self):
        from src.finetune.evolve.trainer import MultiModelTrainer
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig()
        trainer = MultiModelTrainer(cfg)
        args = trainer._build_dpo_args()
        assert args["epochs"] == 1
        assert args["beta"] == 0.1
        assert args["learning_rate"] == 5e-5
