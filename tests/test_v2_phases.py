import pytest


class TestPhase2:
    def test_phase2_config(self):
        from src.finetune.v2.train_phase2 import Phase2Config
        cfg = Phase2Config()
        assert cfg.lora_r == 16
        assert cfg.epochs == 2
        assert "phase2" in str(cfg.output_dir)

    def test_phase2_dataset_mix(self):
        from src.finetune.v2.train_phase2 import Phase2Config
        cfg = Phase2Config()
        total = sum(cfg.dataset_mix.values())
        assert abs(total - 1.0) < 0.01


class TestPhase3:
    def test_phase3_config(self):
        from src.finetune.v2.train_phase3 import Phase3Config
        cfg = Phase3Config()
        assert cfg.lora_r == 16
        assert cfg.epochs == 2
        assert "phase3" in str(cfg.output_dir)

    def test_phase3_tool_data_sources(self):
        from src.finetune.v2.train_phase3 import Phase3Config
        cfg = Phase3Config()
        assert "synthetic" in cfg.data_sources
        assert "toolbench" in cfg.data_sources
