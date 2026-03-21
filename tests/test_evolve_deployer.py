# tests/test_evolve_deployer.py
import pytest
from pathlib import Path


class TestDeployer:
    def test_build_model_tag(self):
        from src.finetune.evolve.deployer import Deployer
        d = Deployer(ollama_host="http://localhost:11434")
        assert d._build_tag("qwen3-8b", is_latest=True) == "DocWain:latest"
        assert d._build_tag("qwen3-8b", is_latest=False) == "DocWain:qwen3-8b"
        assert d._build_tag("llama-3.1-8b", is_latest=False) == "DocWain:llama-3.1-8b"

    def test_build_previous_tag(self):
        from src.finetune.evolve.deployer import Deployer
        d = Deployer(ollama_host="http://localhost:11434")
        assert d._build_previous_tag() == "DocWain:previous"

    def test_deploy_plan_single_model(self):
        from src.finetune.evolve.deployer import Deployer
        d = Deployer(ollama_host="http://localhost:11434")
        plan = d.plan_deployment(
            winner="qwen3-8b",
            all_models=["qwen3-8b", "llama-3.1-8b"],
            keep_previous=True,
        )
        assert len(plan) == 3
        tags = [p["tag"] for p in plan]
        assert "DocWain:latest" in tags
        assert "DocWain:llama-3.1-8b" in tags
        assert "DocWain:previous" in tags
