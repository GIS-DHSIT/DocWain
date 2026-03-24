import pytest
from pathlib import Path


class TestV2Pipeline:
    def test_pipeline_phases(self):
        from src.finetune.v2.pipeline import V2Pipeline
        pipe = V2Pipeline()
        assert pipe.phases == ["phase1", "phase2", "phase3", "phase4"]

    def test_pipeline_status(self, tmp_path):
        from src.finetune.v2.pipeline import V2Pipeline
        pipe = V2Pipeline(base_dir=tmp_path)
        status = pipe.status()
        assert "current_phase" in status
        assert "completed_phases" in status
        assert status["current_phase"] is None
        assert status["completed_phases"] == []

    def test_pipeline_detects_completed_phase(self, tmp_path):
        from src.finetune.v2.pipeline import V2Pipeline
        (tmp_path / "phase1").mkdir()
        (tmp_path / "phase1" / "projection.pt").touch()
        pipe = V2Pipeline(base_dir=tmp_path)
        status = pipe.status()
        assert "phase1" in status["completed_phases"]

    def test_pipeline_next_phase(self, tmp_path):
        from src.finetune.v2.pipeline import V2Pipeline
        pipe = V2Pipeline(base_dir=tmp_path)
        assert pipe.next_phase() == "phase1"
        (tmp_path / "phase1").mkdir()
        (tmp_path / "phase1" / "projection.pt").touch()
        assert pipe.next_phase() == "phase2"
