"""Configuration for the evolving fine-tune pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StudentModel:
    name: str
    repo: str
    enabled: bool = True


@dataclass
class ModelsConfig:
    primary: str = "unsloth/Qwen3-8B-bnb-4bit"
    students: List[StudentModel] = field(default_factory=lambda: [
        StudentModel(name="qwen3-8b", repo="unsloth/Qwen3-8B-bnb-4bit"),
        StudentModel(name="llama-3.1-8b", repo="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
    ])


@dataclass
class PipelineConfig:
    scheduled_interval_hours: int = 24
    signal_threshold: int = 50
    distillation_every_n: int = 3
    eval_prompt_count: int = 200


@dataclass
class SFTConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 16
    batch_size: int = 4


@dataclass
class DPOConfig:
    epochs: int = 1
    beta: float = 0.1
    learning_rate: float = 5e-5


@dataclass
class TrainingConfig:
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)


@dataclass
class GateConfig:
    composite_minimum: float = 80.0
    criterion_floor: float = 60.0
    must_beat_previous: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.30,
        "groundedness": 0.25,
        "reasoning": 0.20,
        "formatting": 0.15,
        "tone": 0.10,
    })


@dataclass
class DeploymentConfig:
    target: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    keep_previous: bool = True
    max_stored_iterations: int = 10


@dataclass
class DocWainConfig:
    endpoint: str = "http://localhost:11434"
    model_name: str = "DHS/DocWain"


@dataclass
class AzureFallbackConfig:
    enabled: bool = False
    endpoint: str = ""
    api_key: str = ""
    model: str = "gpt-4.1"
    deployment_date: str = "2025-04-14"


@dataclass
class EvolveConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    docwain: DocWainConfig = field(default_factory=DocWainConfig)
    azure_fallback: AzureFallbackConfig = field(default_factory=AzureFallbackConfig)

    @classmethod
    def load_default(cls) -> EvolveConfig:
        default_path = Path(__file__).parent.parent / "evolve_config.yaml"
        if default_path.exists():
            return cls.load_from(default_path)
        return cls()

    @classmethod
    def load_from(cls, path: Path) -> EvolveConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> EvolveConfig:
        cfg = cls()
        if "pipeline" in d:
            for k, v in d["pipeline"].items():
                setattr(cfg.pipeline, k, v)
        if "models" in d:
            cfg.models.primary = d["models"].get("primary", cfg.models.primary)
            if "students" in d["models"]:
                cfg.models.students = [
                    StudentModel(**s) for s in d["models"]["students"]
                ]
        if "training" in d:
            if "sft" in d["training"]:
                for k, v in d["training"]["sft"].items():
                    setattr(cfg.training.sft, k, v)
            if "dpo" in d["training"]:
                for k, v in d["training"]["dpo"].items():
                    setattr(cfg.training.dpo, k, v)
        if "gate" in d:
            for k, v in d["gate"].items():
                if k == "weights":
                    cfg.gate.weights = v
                else:
                    setattr(cfg.gate, k, v)
        if "deployment" in d:
            for k, v in d["deployment"].items():
                setattr(cfg.deployment, k, v)
        if "docwain" in d:
            for k, v in d["docwain"].items():
                setattr(cfg.docwain, k, v)
        if "azure_fallback" in d:
            for k, v in d["azure_fallback"].items():
                setattr(cfg.azure_fallback, k, v)
        return cfg
