import pytest


class TestPhase1Config:
    def test_phase1_config_defaults(self):
        from src.finetune.v2.train_phase1 import Phase1Config
        cfg = Phase1Config()
        assert cfg.learning_rate == 1e-3
        assert cfg.epochs == 1
        assert cfg.batch_size == 32
        assert cfg.max_samples == 50000

    def test_phase1_output_dir(self):
        from src.finetune.v2.train_phase1 import Phase1Config
        cfg = Phase1Config()
        assert "phase1" in str(cfg.output_dir)
