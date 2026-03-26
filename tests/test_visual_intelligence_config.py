import os
import pytest


def test_visual_intelligence_config_defaults():
    from src.api.config import Config
    vi = Config.VisualIntelligence
    assert vi.ENABLED == "true"
    assert vi.GPU_DEVICE == "cuda:0"
    assert vi.CPU_FALLBACK == "true"
    assert vi.MAX_CONCURRENT_PAGES == 4
    assert "dit" in vi.TIER1_MODELS
    assert "layoutlmv3" in vi.TIER2_MODELS


def test_visual_intelligence_config_env_override(monkeypatch):
    monkeypatch.setenv("VISUAL_INTELLIGENCE_ENABLED", "false")
    monkeypatch.setenv("VISUAL_INTELLIGENCE_GPU_DEVICE", "cuda:1")
    assert os.getenv("VISUAL_INTELLIGENCE_ENABLED") == "false"
    assert os.getenv("VISUAL_INTELLIGENCE_GPU_DEVICE") == "cuda:1"
