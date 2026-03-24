# Evolving Fine-Tune Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an iterative fine-tuning pipeline where Claude Code acts as teacher/judge, training multiple student models in parallel and promoting the best to `DocWain:latest`.

**Architecture:** 5-stage pipeline (Observe → Harvest → Teach → Train → Gate) with multi-model tournament, hybrid distillation every 3rd iteration, and pluggable model registry. Builds on existing `src/finetune/` modules — no rewrites.

**Tech Stack:** Unsloth (LoRA training), TRL (DPO), Ollama (serving), pytest (testing), PyYAML (config), httpx (DocWain probing)

---

### Task 1: Pipeline Config Module

**Files:**
- Create: `src/finetune/evolve/__init__.py`
- Create: `src/finetune/evolve/config.py`
- Create: `src/finetune/evolve_config.yaml`
- Test: `tests/test_evolve_config.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_config.py
import json
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.finetune.evolve'`

**Step 3: Create package init**

```python
# src/finetune/evolve/__init__.py
"""Evolving fine-tune pipeline — iterative model improvement with Claude Code as teacher."""
```

**Step 4: Write the config module**

```python
# src/finetune/evolve/config.py
"""Configuration for the evolving fine-tune pipeline."""

from __future__ import annotations

import os
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
```

**Step 5: Write the YAML config file**

```yaml
# src/finetune/evolve_config.yaml
pipeline:
  scheduled_interval_hours: 24
  signal_threshold: 50
  distillation_every_n: 3
  eval_prompt_count: 200

models:
  primary: "unsloth/Qwen3-8B-bnb-4bit"
  students:
    - name: "qwen3-8b"
      repo: "unsloth/Qwen3-8B-bnb-4bit"
      enabled: true
    - name: "llama-3.1-8b"
      repo: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
      enabled: true

training:
  sft:
    epochs: 3
    learning_rate: 0.0002
    lora_r: 16
    lora_alpha: 16
    batch_size: 4
  dpo:
    epochs: 1
    beta: 0.1
    learning_rate: 0.00005

gate:
  composite_minimum: 80.0
  criterion_floor: 60.0
  must_beat_previous: true
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
  max_stored_iterations: 10

docwain:
  endpoint: "http://localhost:11434"
  model_name: "DHS/DocWain"

azure_fallback:
  enabled: false
  endpoint: "${AZURE_AI_ENDPOINT}"
  api_key: "${AZURE_AI_API_KEY}"
  model: "gpt-4.1"
  deployment_date: "2025-04-14"
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_evolve_config.py -v`
Expected: All 8 tests PASS

**Step 7: Commit**

```bash
git add src/finetune/evolve/__init__.py src/finetune/evolve/config.py src/finetune/evolve_config.yaml tests/test_evolve_config.py
git commit -m "feat(evolve): add pipeline config module with YAML loading"
```

---

### Task 2: Model Registry Module

**Files:**
- Create: `src/finetune/evolve/registry.py`
- Create: `registry.yaml`
- Test: `tests/test_evolve_registry.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the registry module**

```python
# src/finetune/evolve/registry.py
"""Model registry — tracks trained models, handles promotion and rollback."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelEntry:
    tag: str
    base: str
    iteration: int
    composite_score: float
    scores: Dict[str, float]
    artifact_path: str
    status: str  # "production" | "available" | "rollback_ready"
    promoted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ModelRegistry:
    """YAML-backed model registry with promote/rollback operations."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._models: Dict[str, ModelEntry] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}
        for tag, data in raw.get("models", {}).items():
            data["tag"] = tag
            self._models[tag] = ModelEntry(**data)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Any] = {"models": {}}
        for tag, entry in self._models.items():
            d = asdict(entry)
            d.pop("tag")
            out["models"][tag] = d
        with open(self._path, "w") as f:
            yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    def register(self, entry: ModelEntry) -> None:
        self._models[entry.tag] = entry
        self._save()

    def get(self, tag: str) -> Optional[ModelEntry]:
        return self._models.get(tag)

    def list_models(self) -> List[ModelEntry]:
        return list(self._models.values())

    def promote(self, new_latest: ModelEntry) -> None:
        old = self._models.get("DocWain:latest")
        if old:
            old.tag = "DocWain:previous"
            old.status = "rollback_ready"
            self._models["DocWain:previous"] = old
        new_latest.tag = "DocWain:latest"
        new_latest.status = "production"
        new_latest.promoted_at = datetime.utcnow().isoformat()
        self._models["DocWain:latest"] = new_latest
        self._save()

    def rollback(self) -> None:
        prev = self._models.get("DocWain:previous")
        curr = self._models.get("DocWain:latest")
        if not prev:
            raise ValueError("No previous model to rollback to")
        prev.tag = "DocWain:latest"
        prev.status = "production"
        self._models["DocWain:latest"] = prev
        if curr:
            curr.tag = "DocWain:previous"
            curr.status = "rollback_ready"
            self._models["DocWain:previous"] = curr
        self._save()
```

**Step 4: Create root-level registry.yaml**

```yaml
# registry.yaml
models: {}
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_evolve_registry.py -v`
Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add src/finetune/evolve/registry.py registry.yaml tests/test_evolve_registry.py
git commit -m "feat(evolve): add model registry with promote/rollback"
```

---

### Task 3: Observer Module

**Files:**
- Create: `src/finetune/evolve/observer.py`
- Create: `src/finetune/evolve/prompts/__init__.py`
- Create: `src/finetune/evolve/prompts/observer_prompts.py`
- Test: `tests/test_evolve_observer.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_observer.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


class TestObserverPrompts:
    """Tests for eval prompt generation."""

    def test_get_eval_prompts_returns_list(self):
        from src.finetune.evolve.prompts.observer_prompts import get_eval_prompts
        prompts = get_eval_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) >= 30

    def test_eval_prompts_have_required_fields(self):
        from src.finetune.evolve.prompts.observer_prompts import get_eval_prompts
        for p in get_eval_prompts():
            assert "query" in p
            assert "category" in p
            assert "subcategory" in p
            assert p["category"] in ("document_understanding", "interaction_quality")

    def test_prompts_cover_all_subcategories(self):
        from src.finetune.evolve.prompts.observer_prompts import get_eval_prompts
        subcats = {p["subcategory"] for p in get_eval_prompts()}
        assert "table_extraction" in subcats
        assert "layout_parsing" in subcats
        assert "cross_reference" in subcats
        assert "section_hierarchy" in subcats
        assert "uncertainty_handling" in subcats


class TestObserver:
    """Tests for the observer that probes DocWain."""

    def test_score_response_returns_criteria_dict(self):
        from src.finetune.evolve.observer import Observer
        obs = Observer.__new__(Observer)
        score = obs._score_response(
            query="What are the Q3 figures from the table?",
            response="The Q3 revenue was $1.2M as shown in the financial table on page 3.",
            category="document_understanding",
            subcategory="table_extraction",
        )
        assert "accuracy" in score
        assert "groundedness" in score
        assert "reasoning" in score
        assert "formatting" in score
        assert "tone" in score
        for v in score.values():
            assert 0.0 <= v <= 1.0

    def test_build_signal_from_observation(self):
        from src.finetune.evolve.observer import Observer, ObservationSignal
        obs = Observer.__new__(Observer)
        signal = obs._build_signal(
            query="test query",
            response="test response",
            category="document_understanding",
            subcategory="table_extraction",
            scores={"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.5, "formatting": 0.6, "tone": 0.7},
        )
        assert isinstance(signal, ObservationSignal)
        assert signal.signal_type == "table_extraction_weakness"
        assert signal.category == "document_understanding"
        assert signal.confidence_score < 1.0

    def test_classify_weakness_threshold(self):
        from src.finetune.evolve.observer import Observer
        obs = Observer.__new__(Observer)
        # Low scores = weakness
        assert obs._is_weak({"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.3, "formatting": 0.5, "tone": 0.5})
        # High scores = not weak
        assert not obs._is_weak({"accuracy": 0.9, "groundedness": 0.9, "reasoning": 0.9, "formatting": 0.9, "tone": 0.9})

    def test_save_signals_to_dir(self, tmp_path):
        from src.finetune.evolve.observer import Observer, ObservationSignal
        obs = Observer.__new__(Observer)
        obs._output_dir = tmp_path
        signals = [
            ObservationSignal(
                signal_type="table_extraction_weakness",
                query="test",
                model_response="test resp",
                category="document_understanding",
                subcategory="table_extraction",
                confidence_score=0.35,
                scores={"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.3, "formatting": 0.5, "tone": 0.5},
            )
        ]
        obs._save_signals(signals, iteration=1)
        signal_dir = tmp_path / "iter_1"
        assert signal_dir.exists()
        assert (signal_dir / "observation_signals.jsonl").exists()
        with open(signal_dir / "observation_signals.jsonl") as f:
            data = json.loads(f.readline())
        assert data["signal_type"] == "table_extraction_weakness"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_observer.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write observer prompts**

```python
# src/finetune/evolve/prompts/__init__.py
"""Prompt templates for the evolving fine-tune pipeline."""
```

```python
# src/finetune/evolve/prompts/observer_prompts.py
"""Evaluation prompts for probing DocWain's capabilities."""

from typing import Dict, List


def get_eval_prompts() -> List[Dict[str, str]]:
    """Return evaluation prompts covering document understanding and interaction quality."""
    prompts = []

    # Table extraction
    prompts.extend([
        {"query": "What are the quarterly revenue figures shown in the financial summary table?",
         "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "Compare the values in row 3 and row 7 of the data table.",
         "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "Summarize the key metrics from the performance dashboard table.",
         "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "Which column in the comparison table shows the highest growth rate?",
         "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "Extract all dates and corresponding values from the timeline table.",
         "category": "document_understanding", "subcategory": "table_extraction"},
    ])

    # Layout parsing
    prompts.extend([
        {"query": "What are the main sections covered in this document?",
         "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "List all bullet points under the 'Requirements' heading.",
         "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "What is the nested structure of the table of contents?",
         "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "Identify all numbered lists in section 3.",
         "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "How many sub-sections does the 'Implementation' chapter have?",
         "category": "document_understanding", "subcategory": "layout_parsing"},
    ])

    # Cross-reference
    prompts.extend([
        {"query": "The executive summary mentions a risk factor — where is that discussed in detail?",
         "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "Connect the findings in section 2 with the recommendations in section 5.",
         "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "Which appendix contains the data referenced in paragraph 3 of the overview?",
         "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "How do the conclusions relate to the methodology described earlier?",
         "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "The footnote on page 4 references another document — what does it say?",
         "category": "document_understanding", "subcategory": "cross_reference"},
    ])

    # Section hierarchy
    prompts.extend([
        {"query": "What is the organizational structure of this report?",
         "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "Which sections are at the same level as 'Budget Analysis'?",
         "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "Outline the document hierarchy from top-level to subsections.",
         "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "What parent section does 'Data Collection Methods' fall under?",
         "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "List all heading levels used in this document.",
         "category": "document_understanding", "subcategory": "section_hierarchy"},
    ])

    # Multi-page reasoning
    prompts.extend([
        {"query": "Trace the argument from the introduction through to the final conclusion.",
         "category": "document_understanding", "subcategory": "multi_page_reasoning"},
        {"query": "How does the data in chapter 2 support the claims in chapter 4?",
         "category": "document_understanding", "subcategory": "multi_page_reasoning"},
        {"query": "Summarize the evolution of the proposal across all sections.",
         "category": "document_understanding", "subcategory": "multi_page_reasoning"},
    ])

    # Uncertainty handling
    prompts.extend([
        {"query": "What is the exact market share percentage for Q4 2025?",
         "category": "interaction_quality", "subcategory": "uncertainty_handling"},
        {"query": "Is this policy still valid as of today?",
         "category": "interaction_quality", "subcategory": "uncertainty_handling"},
        {"query": "What does the document say about topics it doesn't cover?",
         "category": "interaction_quality", "subcategory": "uncertainty_handling"},
    ])

    # Adaptive tone
    prompts.extend([
        {"query": "tldr?",
         "category": "interaction_quality", "subcategory": "adaptive_tone"},
        {"query": "Please provide a comprehensive, detailed analysis of the methodology section including all statistical approaches, sample sizes, and confidence intervals.",
         "category": "interaction_quality", "subcategory": "adaptive_tone"},
        {"query": "hey whats in this doc",
         "category": "interaction_quality", "subcategory": "adaptive_tone"},
        {"query": "Summarize.",
         "category": "interaction_quality", "subcategory": "adaptive_tone"},
    ])

    return prompts
```

**Step 4: Write the observer module**

```python
# src/finetune/evolve/observer.py
"""Observer — probes DocWain model and identifies weaknesses."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .config import EvolveConfig
from .prompts.observer_prompts import get_eval_prompts


WEAKNESS_THRESHOLD = 0.6  # Weighted composite below this = weakness


@dataclass
class ObservationSignal:
    signal_type: str
    query: str
    model_response: str
    category: str
    subcategory: str
    confidence_score: float
    scores: Dict[str, float]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class Observer:
    """Sends evaluation prompts to DocWain and identifies weak patterns."""

    def __init__(self, config: EvolveConfig, output_dir: Path):
        self._config = config
        self._output_dir = output_dir
        self._weights = config.gate.weights

    def probe_model(self, prompts: Optional[List[Dict[str, str]]] = None) -> List[ObservationSignal]:
        """Send eval prompts to DocWain, score responses, return weakness signals."""
        prompts = prompts or get_eval_prompts()
        signals: List[ObservationSignal] = []

        for prompt in prompts:
            response = self._query_docwain(prompt["query"])
            scores = self._score_response(
                query=prompt["query"],
                response=response,
                category=prompt["category"],
                subcategory=prompt["subcategory"],
            )
            if self._is_weak(scores):
                signal = self._build_signal(
                    query=prompt["query"],
                    response=response,
                    category=prompt["category"],
                    subcategory=prompt["subcategory"],
                    scores=scores,
                )
                signals.append(signal)

        return signals

    def _query_docwain(self, query: str) -> str:
        """Send a query to DocWain's Ollama endpoint."""
        try:
            resp = httpx.post(
                f"{self._config.docwain.endpoint}/api/generate",
                json={
                    "model": self._config.docwain.model_name,
                    "prompt": query,
                    "stream": False,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            return f"[ERROR] {e}"

    def _score_response(
        self, query: str, response: str, category: str, subcategory: str,
    ) -> Dict[str, float]:
        """Score a response on the 5 criteria. Returns dict of criterion->score [0,1]."""
        scores: Dict[str, float] = {}

        # Accuracy: does it address the question directly?
        scores["accuracy"] = self._heuristic_accuracy(query, response)

        # Groundedness: does it cite evidence / avoid hallucination markers?
        scores["groundedness"] = self._heuristic_groundedness(response)

        # Reasoning: does it show logical structure?
        scores["reasoning"] = self._heuristic_reasoning(response)

        # Formatting: proper markdown structure?
        scores["formatting"] = self._heuristic_formatting(response)

        # Tone: appropriate length/style for the query?
        scores["tone"] = self._heuristic_tone(query, response)

        return scores

    def _heuristic_accuracy(self, query: str, response: str) -> float:
        """Basic heuristic: does the response contain relevant keywords from the query?"""
        if not response or response.startswith("[ERROR]"):
            return 0.0
        query_words = set(query.lower().split())
        stop_words = {"what", "is", "the", "a", "an", "in", "of", "and", "or", "how", "does", "are", "this", "that", "from", "to"}
        query_keywords = query_words - stop_words
        if not query_keywords:
            return 0.5
        resp_lower = response.lower()
        hits = sum(1 for w in query_keywords if w in resp_lower)
        return min(hits / max(len(query_keywords), 1), 1.0)

    def _heuristic_groundedness(self, response: str) -> float:
        """Penalize hallucination markers, reward evidence language."""
        if not response or response.startswith("[ERROR]"):
            return 0.0
        score = 0.5
        evidence_markers = ["according to", "based on", "as stated", "the document", "section", "page", "table", "paragraph"]
        hallucination_markers = ["i think", "i believe", "probably", "it seems like", "i'm not sure but"]
        for m in evidence_markers:
            if m in response.lower():
                score += 0.08
        for m in hallucination_markers:
            if m in response.lower():
                score -= 0.15
        return max(0.0, min(score, 1.0))

    def _heuristic_reasoning(self, response: str) -> float:
        """Reward structured thinking indicators."""
        if not response or response.startswith("[ERROR]"):
            return 0.0
        score = 0.4
        indicators = ["because", "therefore", "however", "in contrast", "specifically",
                       "first", "second", "finally", "this means", "as a result"]
        for ind in indicators:
            if ind in response.lower():
                score += 0.06
        return min(score, 1.0)

    def _heuristic_formatting(self, response: str) -> float:
        """Reward markdown structure."""
        if not response or response.startswith("[ERROR]"):
            return 0.0
        score = 0.5
        if "**" in response:
            score += 0.1
        if "\n-" in response or "\n*" in response:
            score += 0.1
        if "|" in response and "-|-" in response.replace(" ", ""):
            score += 0.15
        if response.count("\n#") >= 1:
            score += 0.1
        return min(score, 1.0)

    def _heuristic_tone(self, query: str, response: str) -> float:
        """Check if response length/style matches query complexity."""
        if not response or response.startswith("[ERROR]"):
            return 0.0
        query_len = len(query.split())
        resp_len = len(response.split())
        # Short query should get proportional response
        if query_len <= 5 and resp_len > 300:
            return 0.3  # Too verbose for short query
        if query_len > 30 and resp_len < 50:
            return 0.3  # Too terse for detailed query
        return 0.7

    def _is_weak(self, scores: Dict[str, float]) -> bool:
        """Check if weighted composite is below weakness threshold."""
        composite = sum(scores.get(k, 0) * w for k, w in self._weights.items())
        return composite < WEAKNESS_THRESHOLD

    def _build_signal(
        self, query: str, response: str, category: str, subcategory: str, scores: Dict[str, float],
    ) -> ObservationSignal:
        composite = sum(scores.get(k, 0) * w for k, w in self._weights.items())
        return ObservationSignal(
            signal_type=f"{subcategory}_weakness",
            query=query,
            model_response=response,
            category=category,
            subcategory=subcategory,
            confidence_score=round(composite, 3),
            scores=scores,
        )

    def _save_signals(self, signals: List[ObservationSignal], iteration: int) -> Path:
        """Save signals to iteration directory."""
        iter_dir = self._output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        out_path = iter_dir / "observation_signals.jsonl"
        with open(out_path, "w") as f:
            for s in signals:
                f.write(json.dumps(asdict(s)) + "\n")
        return out_path
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_evolve_observer.py -v`
Expected: All 7 tests PASS

**Step 6: Commit**

```bash
git add src/finetune/evolve/observer.py src/finetune/evolve/prompts/__init__.py src/finetune/evolve/prompts/observer_prompts.py tests/test_evolve_observer.py
git commit -m "feat(evolve): add observer module with eval prompts and heuristic scoring"
```

---

### Task 4: Harvester Module

**Files:**
- Create: `src/finetune/evolve/harvester.py`
- Test: `tests/test_evolve_harvester.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_harvester.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestHarvester:
    """Tests for signal harvesting from multiple sources."""

    def test_load_observation_signals(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        iter_dir = tmp_path / "iter_1"
        iter_dir.mkdir()
        signal = {"signal_type": "table_weakness", "query": "test", "model_response": "resp",
                  "category": "document_understanding", "subcategory": "table_extraction",
                  "confidence_score": 0.4, "scores": {}, "timestamp": "2026-03-21T00:00:00"}
        (iter_dir / "observation_signals.jsonl").write_text(json.dumps(signal) + "\n")
        h = Harvester(signals_dir=tmp_path)
        signals = h.load_observation_signals(iteration=1)
        assert len(signals) == 1
        assert signals[0]["signal_type"] == "table_weakness"

    def test_load_interaction_signals(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        # Write mock feedback signals
        feedback_file = tmp_path / "feedback.jsonl"
        entry = {"messages": [{"role": "user", "content": "test"}], "metadata": {"feedback": "negative", "correction": "x" * 60}}
        feedback_file.write_text(json.dumps(entry) + "\n")
        h = Harvester(signals_dir=tmp_path)
        signals = h.load_interaction_signals(feedback_path=feedback_file)
        assert len(signals) == 1

    def test_merge_and_dedup(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        obs = [
            {"query": "What are Q3 figures?", "category": "doc_understanding", "subcategory": "table"},
            {"query": "What are Q3 figures?", "category": "doc_understanding", "subcategory": "table"},
        ]
        interaction = [
            {"query": "Different question", "category": "interaction", "subcategory": "feedback"},
        ]
        merged = h.merge_and_dedup(obs, interaction)
        assert len(merged) == 2  # duplicate removed

    def test_category_balance(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        signals = [{"query": f"q{i}", "category": "doc_understanding", "subcategory": "table"} for i in range(100)]
        signals += [{"query": "layout_q", "category": "doc_understanding", "subcategory": "layout"}]
        balanced = h.balance_categories(signals, max_per_subcategory=20)
        table_count = sum(1 for s in balanced if s["subcategory"] == "table")
        assert table_count <= 20
        assert any(s["subcategory"] == "layout" for s in balanced)

    def test_save_harvest(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        signals = [{"query": "test", "category": "doc", "subcategory": "table"}]
        h.save_harvest(signals, iteration=1)
        path = tmp_path / "iter_1" / "harvested_signals.jsonl"
        assert path.exists()

    def test_harvest_summary(self, tmp_path):
        from src.finetune.evolve.harvester import Harvester
        h = Harvester(signals_dir=tmp_path)
        signals = [
            {"query": "q1", "category": "document_understanding", "subcategory": "table_extraction"},
            {"query": "q2", "category": "document_understanding", "subcategory": "layout_parsing"},
            {"query": "q3", "category": "interaction_quality", "subcategory": "feedback"},
        ]
        summary = h.summarize(signals)
        assert summary["total"] == 3
        assert summary["by_category"]["document_understanding"] == 2
        assert summary["by_category"]["interaction_quality"] == 1
        assert "table_extraction" in summary["by_subcategory"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_harvester.py -v`
Expected: FAIL

**Step 3: Write the harvester module**

```python
# src/finetune/evolve/harvester.py
"""Harvester — collects and merges signals from observer + stored feedback."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


class Harvester:
    """Collects signals from multiple sources, deduplicates, and balances categories."""

    def __init__(self, signals_dir: Path):
        self._signals_dir = Path(signals_dir)

    def load_observation_signals(self, iteration: int) -> List[Dict[str, Any]]:
        """Load observation signals from a completed observer run."""
        path = self._signals_dir / f"iter_{iteration}" / "observation_signals.jsonl"
        if not path.exists():
            return []
        signals = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    signals.append(json.loads(line))
        return signals

    def load_interaction_signals(
        self, feedback_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Load interaction signals from feedback JSONL files."""
        if feedback_path is None:
            feedback_path = Path("src/outputs/learning_signals/high_quality.jsonl")
        if not feedback_path.exists():
            return []
        signals = []
        with open(feedback_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                signals.append(self._feedback_to_signal(entry))
        return signals

    def _feedback_to_signal(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a feedback JSONL entry to a signal dict."""
        messages = entry.get("messages", [])
        query = ""
        for m in messages:
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        metadata = entry.get("metadata", {})
        return {
            "query": query,
            "category": "interaction_quality",
            "subcategory": "feedback",
            "signal_type": "user_feedback",
            "metadata": metadata,
        }

    def merge_and_dedup(
        self,
        observation_signals: List[Dict[str, Any]],
        interaction_signals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate signals by query text."""
        seen_queries: set = set()
        merged: List[Dict[str, Any]] = []
        for signal in observation_signals + interaction_signals:
            key = signal.get("query", "").strip().lower()
            if key and key not in seen_queries:
                seen_queries.add(key)
                merged.append(signal)
        return merged

    def balance_categories(
        self, signals: List[Dict[str, Any]], max_per_subcategory: int = 30,
    ) -> List[Dict[str, Any]]:
        """Ensure no single subcategory dominates the signal set."""
        by_subcat: Dict[str, List[Dict[str, Any]]] = {}
        for s in signals:
            sub = s.get("subcategory", "unknown")
            by_subcat.setdefault(sub, []).append(s)
        balanced: List[Dict[str, Any]] = []
        for sub, items in by_subcat.items():
            balanced.extend(items[:max_per_subcategory])
        return balanced

    def summarize(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of harvested signals."""
        by_cat: Counter = Counter()
        by_subcat: Counter = Counter()
        for s in signals:
            by_cat[s.get("category", "unknown")] += 1
            by_subcat[s.get("subcategory", "unknown")] += 1
        return {
            "total": len(signals),
            "by_category": dict(by_cat),
            "by_subcategory": dict(by_subcat),
        }

    def save_harvest(
        self, signals: List[Dict[str, Any]], iteration: int,
    ) -> Path:
        """Save merged signals to iteration directory."""
        iter_dir = self._signals_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        out_path = iter_dir / "harvested_signals.jsonl"
        with open(out_path, "w") as f:
            for s in signals:
                f.write(json.dumps(s) + "\n")
        return out_path
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evolve_harvester.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/finetune/evolve/harvester.py tests/test_evolve_harvester.py
git commit -m "feat(evolve): add harvester module for signal collection and dedup"
```

---

### Task 5: Teacher Module

**Files:**
- Create: `src/finetune/evolve/teacher.py`
- Create: `src/finetune/evolve/prompts/teacher_sft.py`
- Create: `src/finetune/evolve/prompts/teacher_dpo.py`
- Test: `tests/test_evolve_teacher.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_teacher.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestTeacherPrompts:
    """Tests for SFT and DPO prompt templates."""

    def test_sft_prompt_has_required_sections(self):
        from src.finetune.evolve.prompts.teacher_sft import build_sft_prompt
        prompt = build_sft_prompt(
            query="What are the Q3 figures?",
            category="document_understanding",
            subcategory="table_extraction",
        )
        assert "DocWain" in prompt
        assert "table" in prompt.lower() or "Q3" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_dpo_prompt_includes_model_response(self):
        from src.finetune.evolve.prompts.teacher_dpo import build_dpo_prompt
        prompt = build_dpo_prompt(
            query="What are the Q3 figures?",
            model_response="The figures are unavailable.",
            scores={"accuracy": 0.3, "groundedness": 0.4, "reasoning": 0.3, "formatting": 0.5, "tone": 0.5},
        )
        assert "figures are unavailable" in prompt
        assert "accuracy" in prompt.lower()
        assert isinstance(prompt, str)


class TestTeacher:
    """Tests for the teacher that generates training pairs."""

    def test_generate_sft_pair(self):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        t._docwain_system_prompt = "You are DocWain."
        pair = t._format_sft_pair(
            query="What are Q3 figures?",
            ideal_response="Based on the financial table, Q3 revenue was $1.2M.",
        )
        assert "messages" in pair
        msgs = pair["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "DocWain" in msgs[0]["content"]

    def test_generate_dpo_pair(self):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        t._docwain_system_prompt = "You are DocWain."
        pair = t._format_dpo_pair(
            query="What are Q3 figures?",
            chosen="Based on the financial table, Q3 revenue was $1.2M.",
            rejected="The figures are unavailable.",
        )
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen"][-1]["role"] == "assistant"
        assert pair["rejected"][-1]["role"] == "assistant"

    def test_filter_content_references(self):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        # Response that teaches patterns = OK
        assert t._is_pattern_not_content("When reading a table, first identify the headers and row structure.")
        # Response that recites document content = NOT OK
        assert not t._is_pattern_not_content("The company reported revenue of $45.2 million for fiscal year 2025 in their annual report filed on March 15.")

    def test_save_teaching_output(self, tmp_path):
        from src.finetune.evolve.teacher import Teacher
        t = Teacher.__new__(Teacher)
        t._output_dir = tmp_path
        sft_pairs = [{"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        dpo_pairs = [{"chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
                      "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}]}]
        t._save_output(sft_pairs, dpo_pairs, iteration=1)
        iter_dir = tmp_path / "iter_1"
        assert (iter_dir / "sft_pairs.jsonl").exists()
        assert (iter_dir / "dpo_pairs.jsonl").exists()
        with open(iter_dir / "sft_pairs.jsonl") as f:
            data = json.loads(f.readline())
        assert "messages" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_teacher.py -v`
Expected: FAIL

**Step 3: Write teacher prompt templates**

```python
# src/finetune/evolve/prompts/teacher_sft.py
"""SFT prompt templates for the teacher."""

DOCWAIN_PERSONA = (
    "You are DocWain — Document Wise AI Node. You are an expert document intelligence assistant "
    "that helps users understand complex documents. You adapt your tone to match the user's style. "
    "You cite evidence from documents, reason across sections, handle tables and layouts expertly, "
    "and clearly communicate uncertainty when information is incomplete."
)

CATEGORY_CONTEXT = {
    "table_extraction": "The user is asking about data in a table. Demonstrate expert table reasoning — identify headers, rows, relationships between cells, and present data clearly.",
    "layout_parsing": "The user is asking about document structure. Show understanding of headings, sections, lists, and hierarchical organization.",
    "cross_reference": "The user is asking about connections between different parts of a document. Demonstrate ability to link related information across sections.",
    "section_hierarchy": "The user is asking about document organization. Show understanding of parent-child section relationships and document outline structure.",
    "multi_page_reasoning": "The user needs information synthesized across multiple pages or sections. Demonstrate ability to follow arguments and connect evidence across the document.",
    "uncertainty_handling": "The query may touch on information the document doesn't fully cover. Demonstrate honest uncertainty while being helpful.",
    "adaptive_tone": "Match the tone and detail level of your response to the user's query style.",
    "feedback": "A user previously found this type of response unhelpful. Generate an improved version.",
}


def build_sft_prompt(query: str, category: str, subcategory: str) -> str:
    """Build a prompt asking the teacher to generate an ideal DocWain response."""
    context = CATEGORY_CONTEXT.get(subcategory, "Respond helpfully and accurately.")
    return f"""You are generating training data for DocWain, a document intelligence AI assistant.

DocWain's persona: {DOCWAIN_PERSONA}

Scenario category: {category} / {subcategory}
Context: {context}

User query: "{query}"

Generate the IDEAL DocWain response for this query. Requirements:
1. Respond as DocWain would — adaptive tone, grounded in evidence, well-structured
2. Focus on PATTERNS of document understanding, NOT specific document content
3. Show HOW to reason about this type of query, not memorized facts
4. Use markdown formatting where appropriate (tables, bold, lists)
5. If the query is short/casual, keep the response proportionally concise
6. If the query is detailed, provide thorough analysis

Write ONLY the ideal response, nothing else."""
```

```python
# src/finetune/evolve/prompts/teacher_dpo.py
"""DPO prompt templates for the teacher/judge."""


def build_dpo_prompt(query: str, model_response: str, scores: dict) -> str:
    """Build a prompt asking the teacher to judge and improve a response."""
    score_text = "\n".join(f"  - {k}: {v:.2f}/1.00" for k, v in scores.items())
    return f"""You are evaluating and improving a DocWain response.

User query: "{query}"

DocWain's response:
---
{model_response}
---

Current scores (0-1 scale):
{score_text}

Tasks:
1. Identify the specific weaknesses in this response
2. Generate a BETTER response that fixes these weaknesses
3. The improved response must:
   - Focus on document understanding PATTERNS, not specific content
   - Be grounded and cite evidence appropriately
   - Use adaptive tone matching the query style
   - Use proper markdown formatting

Output format:
WEAKNESSES: [brief list of issues]
---IMPROVED---
[your improved response here]"""
```

**Step 4: Write the teacher module**

```python
# src/finetune/evolve/teacher.py
"""Teacher — generates SFT and DPO training pairs using Claude Code's judgment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompts.teacher_sft import DOCWAIN_PERSONA


# Heuristic: responses with many specific numbers/dates/names are likely content, not patterns
_CONTENT_PATTERN = re.compile(
    r"(?:\$[\d,.]+\s*(?:million|billion|M|B|K))|"
    r"(?:fiscal year \d{4})|"
    r"(?:filed on [A-Z][a-z]+ \d{1,2})|"
    r"(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_CONTENT_THRESHOLD = 3  # More than this many matches = likely content


class Teacher:
    """Generates training pairs. In interactive mode, Claude Code IS the teacher."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._docwain_system_prompt = DOCWAIN_PERSONA

    def format_sft_pairs(
        self, signals: List[Dict[str, Any]], ideal_responses: List[str],
    ) -> List[Dict[str, Any]]:
        """Convert signals + ideal responses into SFT training pairs."""
        pairs = []
        for signal, response in zip(signals, ideal_responses):
            if not self._is_pattern_not_content(response):
                continue
            pair = self._format_sft_pair(signal["query"], response)
            pairs.append(pair)
        return pairs

    def format_dpo_pairs(
        self, signals: List[Dict[str, Any]], improved_responses: List[str],
    ) -> List[Dict[str, Any]]:
        """Convert signals + improved responses into DPO preference pairs."""
        pairs = []
        for signal, improved in zip(signals, improved_responses):
            if not self._is_pattern_not_content(improved):
                continue
            rejected = signal.get("model_response", "")
            if not rejected:
                continue
            pair = self._format_dpo_pair(signal["query"], chosen=improved, rejected=rejected)
            pairs.append(pair)
        return pairs

    def _format_sft_pair(self, query: str, ideal_response: str) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": self._docwain_system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": ideal_response},
            ]
        }

    def _format_dpo_pair(self, query: str, chosen: str, rejected: str) -> Dict[str, Any]:
        return {
            "chosen": [
                {"role": "system", "content": self._docwain_system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": chosen},
            ],
            "rejected": [
                {"role": "system", "content": self._docwain_system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": rejected},
            ],
        }

    def _is_pattern_not_content(self, response: str) -> bool:
        """Return True if the response teaches patterns rather than reciting document content."""
        matches = _CONTENT_PATTERN.findall(response)
        return len(matches) < _CONTENT_THRESHOLD

    def _save_output(
        self,
        sft_pairs: List[Dict[str, Any]],
        dpo_pairs: List[Dict[str, Any]],
        iteration: int,
    ) -> None:
        """Save training pairs to iteration directory."""
        iter_dir = self._output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        with open(iter_dir / "sft_pairs.jsonl", "w") as f:
            for p in sft_pairs:
                f.write(json.dumps(p) + "\n")
        with open(iter_dir / "dpo_pairs.jsonl", "w") as f:
            for p in dpo_pairs:
                f.write(json.dumps(p) + "\n")
        summary = {
            "sft_count": len(sft_pairs),
            "dpo_count": len(dpo_pairs),
            "iteration": iteration,
        }
        with open(iter_dir / "teach_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_evolve_teacher.py -v`
Expected: All 6 tests PASS

**Step 6: Commit**

```bash
git add src/finetune/evolve/teacher.py src/finetune/evolve/prompts/teacher_sft.py src/finetune/evolve/prompts/teacher_dpo.py tests/test_evolve_teacher.py
git commit -m "feat(evolve): add teacher module with SFT/DPO pair generation"
```

---

### Task 6: Tournament Module

**Files:**
- Create: `src/finetune/evolve/tournament.py`
- Test: `tests/test_evolve_tournament.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_tournament.py
import json
import pytest
from pathlib import Path


class TestTournament:
    """Tests for multi-model tournament scoring."""

    def test_compute_composite_score(self):
        from src.finetune.evolve.tournament import Tournament
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        scores = {"accuracy": 0.87, "groundedness": 0.83, "reasoning": 0.82, "formatting": 0.81, "tone": 0.79}
        composite = t.compute_composite(scores)
        expected = 0.87*0.30 + 0.83*0.25 + 0.82*0.20 + 0.81*0.15 + 0.79*0.10
        assert abs(composite - expected * 100) < 0.1

    def test_rank_models(self):
        from src.finetune.evolve.tournament import Tournament, ModelResult
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        results = [
            ModelResult(name="qwen3-8b", scores={"accuracy": 0.87, "groundedness": 0.83, "reasoning": 0.82, "formatting": 0.81, "tone": 0.79}),
            ModelResult(name="llama-8b", scores={"accuracy": 0.81, "groundedness": 0.78, "reasoning": 0.77, "formatting": 0.75, "tone": 0.73}),
        ]
        ranked = t.rank(results)
        assert ranked[0].name == "qwen3-8b"
        assert ranked[1].name == "llama-8b"
        assert ranked[0].composite > ranked[1].composite

    def test_empty_results(self):
        from src.finetune.evolve.tournament import Tournament
        t = Tournament(weights={"accuracy": 1.0})
        ranked = t.rank([])
        assert ranked == []

    def test_save_results(self, tmp_path):
        from src.finetune.evolve.tournament import Tournament, ModelResult
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        results = [
            ModelResult(name="qwen3-8b", scores={"accuracy": 0.87, "groundedness": 0.83, "reasoning": 0.82, "formatting": 0.81, "tone": 0.79}),
        ]
        ranked = t.rank(results)
        t.save_results(ranked, tmp_path / "tournament.json")
        assert (tmp_path / "tournament.json").exists()
        data = json.loads((tmp_path / "tournament.json").read_text())
        assert len(data["rankings"]) == 1
        assert data["rankings"][0]["name"] == "qwen3-8b"

    def test_per_category_best(self):
        from src.finetune.evolve.tournament import Tournament, ModelResult
        weights = {"accuracy": 0.30, "groundedness": 0.25, "reasoning": 0.20, "formatting": 0.15, "tone": 0.10}
        t = Tournament(weights=weights)
        results = [
            ModelResult(name="qwen3-8b", scores={"accuracy": 0.90, "groundedness": 0.70, "reasoning": 0.80, "formatting": 0.80, "tone": 0.80}),
            ModelResult(name="llama-8b", scores={"accuracy": 0.70, "groundedness": 0.95, "reasoning": 0.80, "formatting": 0.80, "tone": 0.80}),
        ]
        best = t.best_per_criterion(results)
        assert best["accuracy"] == "qwen3-8b"
        assert best["groundedness"] == "llama-8b"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_tournament.py -v`
Expected: FAIL

**Step 3: Write the tournament module**

```python
# src/finetune/evolve/tournament.py
"""Tournament — ranks trained models by weighted composite score."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelResult:
    name: str
    scores: Dict[str, float]
    composite: float = 0.0


class Tournament:
    """Runs all models against eval prompts and ranks by composite score."""

    def __init__(self, weights: Dict[str, float]):
        self._weights = weights

    def compute_composite(self, scores: Dict[str, float]) -> float:
        """Compute weighted composite score (0-100 scale)."""
        raw = sum(scores.get(k, 0) * w for k, w in self._weights.items())
        return round(raw * 100, 2)

    def rank(self, results: List[ModelResult]) -> List[ModelResult]:
        """Score and rank models by composite. Returns sorted list (best first)."""
        for r in results:
            r.composite = self.compute_composite(r.scores)
        return sorted(results, key=lambda r: r.composite, reverse=True)

    def best_per_criterion(self, results: List[ModelResult]) -> Dict[str, str]:
        """Return the best model name for each criterion."""
        best: Dict[str, str] = {}
        for criterion in self._weights:
            top = max(results, key=lambda r: r.scores.get(criterion, 0))
            best[criterion] = top.name
        return best

    def save_results(self, ranked: List[ModelResult], path: Path) -> None:
        """Save tournament results to JSON."""
        data = {
            "rankings": [
                {
                    "name": r.name,
                    "composite": r.composite,
                    "scores": r.scores,
                }
                for r in ranked
            ],
            "weights": self._weights,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evolve_tournament.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/finetune/evolve/tournament.py tests/test_evolve_tournament.py
git commit -m "feat(evolve): add tournament module for multi-model ranking"
```

---

### Task 7: Quality Gate Module

**Files:**
- Create: `src/finetune/evolve/gate.py`
- Test: `tests/test_evolve_gate.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_gate.py
import pytest


class TestQualityGate:
    """Tests for promotion quality gate."""

    def test_passes_when_all_criteria_met(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(
            composite=84.2,
            scores={"accuracy": 87, "groundedness": 83, "reasoning": 82, "formatting": 81, "tone": 79},
            previous_composite=81.0,
        )
        assert result.passed is True
        assert result.reason == ""

    def test_fails_composite_below_minimum(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(
            composite=75.0,
            scores={"accuracy": 80, "groundedness": 75, "reasoning": 70, "formatting": 70, "tone": 70},
            previous_composite=70.0,
        )
        assert result.passed is False
        assert "composite" in result.reason.lower()

    def test_fails_criterion_below_floor(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(
            composite=82.0,
            scores={"accuracy": 90, "groundedness": 85, "reasoning": 55, "formatting": 80, "tone": 80},
            previous_composite=79.0,
        )
        assert result.passed is False
        assert "reasoning" in result.reason.lower()

    def test_fails_not_beating_previous(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(
            composite=81.0,
            scores={"accuracy": 82, "groundedness": 80, "reasoning": 80, "formatting": 80, "tone": 80},
            previous_composite=82.0,
        )
        assert result.passed is False
        assert "previous" in result.reason.lower()

    def test_passes_without_previous_when_no_previous(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(
            composite=81.0,
            scores={"accuracy": 82, "groundedness": 80, "reasoning": 80, "formatting": 80, "tone": 80},
            previous_composite=None,
        )
        assert result.passed is True

    def test_multiple_failures_reported(self):
        from src.finetune.evolve.gate import QualityGate
        gate = QualityGate(composite_minimum=80.0, criterion_floor=60.0, must_beat_previous=True)
        result = gate.evaluate(
            composite=70.0,
            scores={"accuracy": 50, "groundedness": 55, "reasoning": 80, "formatting": 80, "tone": 80},
            previous_composite=75.0,
        )
        assert result.passed is False
        assert "accuracy" in result.reason.lower()
        assert "groundedness" in result.reason.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_gate.py -v`
Expected: FAIL

**Step 3: Write the gate module**

```python
# src/finetune/evolve/gate.py
"""Quality gate — decides whether a trained model can be promoted to production."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    composite: float = 0.0


class QualityGate:
    """Evaluates a model against promotion criteria."""

    def __init__(
        self,
        composite_minimum: float = 80.0,
        criterion_floor: float = 60.0,
        must_beat_previous: bool = True,
    ):
        self._composite_minimum = composite_minimum
        self._criterion_floor = criterion_floor
        self._must_beat_previous = must_beat_previous

    def evaluate(
        self,
        composite: float,
        scores: Dict[str, float],
        previous_composite: Optional[float] = None,
    ) -> GateResult:
        """Check all promotion criteria. Returns GateResult."""
        failures = []

        if composite < self._composite_minimum:
            failures.append(
                f"Composite {composite:.1f} below minimum {self._composite_minimum:.1f}"
            )

        for criterion, score in scores.items():
            if score < self._criterion_floor:
                failures.append(
                    f"{criterion} score {score:.1f} below floor {self._criterion_floor:.1f}"
                )

        if (
            self._must_beat_previous
            and previous_composite is not None
            and composite <= previous_composite
        ):
            failures.append(
                f"Composite {composite:.1f} does not beat previous {previous_composite:.1f}"
            )

        if failures:
            return GateResult(passed=False, reason="; ".join(failures), composite=composite)
        return GateResult(passed=True, reason="", composite=composite)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evolve_gate.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/finetune/evolve/gate.py tests/test_evolve_gate.py
git commit -m "feat(evolve): add quality gate with composite + floor + beat-previous checks"
```

---

### Task 8: Trainer Module (Multi-Model)

**Files:**
- Create: `src/finetune/evolve/trainer.py`
- Test: `tests/test_evolve_trainer.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_trainer.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestTrainerConfig:
    """Tests for multi-model trainer configuration."""

    def test_build_training_request(self):
        from src.finetune.evolve.trainer import MultiModelTrainer
        from src.finetune.evolve.config import EvolveConfig, StudentModel, SFTConfig
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
    """Tests for DPO training phase."""

    def test_build_dpo_args(self):
        from src.finetune.evolve.trainer import MultiModelTrainer
        from src.finetune.evolve.config import EvolveConfig
        cfg = EvolveConfig()
        trainer = MultiModelTrainer(cfg)
        args = trainer._build_dpo_args()
        assert args["epochs"] == 1
        assert args["beta"] == 0.1
        assert args["learning_rate"] == 5e-5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_trainer.py -v`
Expected: FAIL

**Step 3: Write the trainer module**

```python
# src/finetune/evolve/trainer.py
"""Multi-model trainer — runs SFT + DPO across configured student models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvolveConfig, StudentModel

logger = logging.getLogger(__name__)

# Default artifact directory
_DEFAULT_ARTIFACT_DIR = Path("finetune_artifacts")


class MultiModelTrainer:
    """Orchestrates training across multiple student models."""

    def __init__(self, config: EvolveConfig, artifact_dir: Optional[Path] = None):
        self._config = config
        self._artifact_dir = artifact_dir or _DEFAULT_ARTIFACT_DIR

    def list_enabled_students(self) -> List[StudentModel]:
        """Return only enabled student models."""
        return [s for s in self._config.models.students if s.enabled]

    def train_sft(
        self,
        student: StudentModel,
        dataset_path: str,
        iteration: int,
    ) -> Dict[str, Any]:
        """Run SFT training for a single student model. Returns training metrics."""
        from src.finetune.unsloth_trainer import get_finetune_manager

        request = self._build_request(student, dataset_path, iteration)
        manager = get_finetune_manager()
        status = manager.start_job(request)
        return {
            "job_id": status.job_id,
            "model": student.name,
            "status": status.status,
            "iteration": iteration,
        }

    def train_dpo(
        self,
        student: StudentModel,
        iteration: int,
        base_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run DPO alignment on SFT output. Returns training metrics."""
        from src.finetune.dpo_trainer import run_dpo_training

        args = self._build_dpo_args()
        if base_model_path:
            args["base_model"] = base_model_path
        else:
            args["base_model"] = student.repo

        result = run_dpo_training(**args)
        return {
            "model": student.name,
            "iteration": iteration,
            "dpo_result": result,
        }

    def train_all(
        self,
        dataset_path: str,
        iteration: int,
    ) -> List[Dict[str, Any]]:
        """Train all enabled students sequentially. Returns list of results."""
        results = []
        for student in self.list_enabled_students():
            logger.info("Training %s (iteration %d)", student.name, iteration)
            try:
                sft_result = self.train_sft(student, dataset_path, iteration)
                results.append(sft_result)
            except Exception as e:
                logger.error("SFT failed for %s: %s", student.name, e)
                results.append({
                    "model": student.name,
                    "iteration": iteration,
                    "status": "failed",
                    "error": str(e),
                })
        return results

    def _build_request(
        self, student: StudentModel, dataset_path: str, iteration: int,
    ):
        """Build a FinetuneRequest for a student model."""
        from src.finetune.models import FinetuneRequest

        sft = self._config.training.sft
        output_dir = str(self._artifact_path(iteration, student.name))
        return FinetuneRequest(
            profile_id=f"evolve_iter_{iteration}_{student.name}",
            base_model=student.repo,
            learning_rate=sft.learning_rate,
            num_epochs=sft.epochs,
            lora_r=sft.lora_r,
            lora_alpha=sft.lora_alpha,
            batch_size=sft.batch_size,
            dataset_path=dataset_path,
            output_dir=output_dir,
            run_name=f"evolve_iter{iteration}_{student.name}",
        )

    def _build_dpo_args(self) -> Dict[str, Any]:
        """Build DPO training arguments from config."""
        dpo = self._config.training.dpo
        return {
            "epochs": dpo.epochs,
            "beta": dpo.beta,
            "learning_rate": dpo.learning_rate,
            "lora_r": self._config.training.sft.lora_r,
        }

    def _artifact_path(self, iteration: int, model_name: str) -> Path:
        """Return the artifact directory for a specific iteration + model."""
        return self._artifact_dir / f"iter_{iteration}" / "models" / model_name
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evolve_trainer.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/finetune/evolve/trainer.py tests/test_evolve_trainer.py
git commit -m "feat(evolve): add multi-model trainer with SFT + DPO orchestration"
```

---

### Task 9: Distiller Module

**Files:**
- Create: `src/finetune/evolve/distiller.py`
- Test: `tests/test_evolve_distiller.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_distiller.py
import json
import pytest
from pathlib import Path


class TestDistiller:
    """Tests for hybrid distillation — cherry-picking best responses."""

    def test_should_distill_every_nth_iteration(self):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        assert not d.should_distill(iteration=1)
        assert not d.should_distill(iteration=2)
        assert d.should_distill(iteration=3)
        assert not d.should_distill(iteration=4)
        assert d.should_distill(iteration=6)

    def test_cherry_pick_best_responses(self):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        # Model A is best at accuracy, Model B at groundedness
        eval_results = {
            "model_a": [
                {"query": "q1", "response": "a_resp1", "scores": {"accuracy": 0.95, "groundedness": 0.60}},
                {"query": "q2", "response": "a_resp2", "scores": {"accuracy": 0.90, "groundedness": 0.65}},
            ],
            "model_b": [
                {"query": "q1", "response": "b_resp1", "scores": {"accuracy": 0.70, "groundedness": 0.95}},
                {"query": "q2", "response": "b_resp2", "scores": {"accuracy": 0.65, "groundedness": 0.90}},
            ],
        }
        best = d.cherry_pick(eval_results, criterion="accuracy")
        assert len(best) == 2
        assert best[0]["response"] == "a_resp1"  # Model A wins on accuracy

    def test_build_distillation_dataset(self):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        d._system_prompt = "You are DocWain."
        best_responses = [
            {"query": "q1", "response": "best resp 1"},
            {"query": "q2", "response": "best resp 2"},
        ]
        dataset = d.build_dataset(best_responses)
        assert len(dataset) == 2
        assert dataset[0]["messages"][0]["role"] == "system"
        assert dataset[0]["messages"][1]["role"] == "user"
        assert dataset[0]["messages"][2]["role"] == "assistant"

    def test_save_distillation_dataset(self, tmp_path):
        from src.finetune.evolve.distiller import Distiller
        d = Distiller(distill_every_n=3)
        dataset = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        path = d.save_dataset(dataset, tmp_path / "distill.jsonl")
        assert path.exists()
        with open(path) as f:
            data = json.loads(f.readline())
        assert "messages" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_distiller.py -v`
Expected: FAIL

**Step 3: Write the distiller module**

```python
# src/finetune/evolve/distiller.py
"""Distiller — cherry-picks best responses from all models for hybrid distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .prompts.teacher_sft import DOCWAIN_PERSONA


class Distiller:
    """Runs every Nth iteration to combine best outputs from all student models."""

    def __init__(self, distill_every_n: int = 3):
        self._every_n = distill_every_n
        self._system_prompt = DOCWAIN_PERSONA

    def should_distill(self, iteration: int) -> bool:
        """Check if this iteration should trigger distillation."""
        return iteration > 0 and iteration % self._every_n == 0

    def cherry_pick(
        self,
        eval_results: Dict[str, List[Dict[str, Any]]],
        criterion: str = "accuracy",
    ) -> List[Dict[str, Any]]:
        """Pick the best response for each query across all models by criterion.

        eval_results: {model_name: [{query, response, scores}, ...]}
        """
        # Group by query
        by_query: Dict[str, List[Dict[str, Any]]] = {}
        for model_name, results in eval_results.items():
            for r in results:
                q = r["query"]
                by_query.setdefault(q, []).append({**r, "model": model_name})

        best: List[Dict[str, Any]] = []
        for query, candidates in by_query.items():
            winner = max(candidates, key=lambda c: c["scores"].get(criterion, 0))
            best.append({"query": query, "response": winner["response"], "model": winner["model"]})
        return best

    def build_dataset(self, best_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert cherry-picked best responses into SFT training format."""
        dataset = []
        for item in best_responses:
            dataset.append({
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": item["query"]},
                    {"role": "assistant", "content": item["response"]},
                ]
            })
        return dataset

    def save_dataset(self, dataset: List[Dict[str, Any]], path: Path) -> Path:
        """Save distillation dataset to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")
        return path
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evolve_distiller.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/finetune/evolve/distiller.py tests/test_evolve_distiller.py
git commit -m "feat(evolve): add distiller module for hybrid model knowledge fusion"
```

---

### Task 10: Deployer Module

**Files:**
- Create: `src/finetune/evolve/deployer.py`
- Test: `tests/test_evolve_deployer.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_deployer.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDeployer:
    """Tests for GGUF export and Ollama registration."""

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
        assert len(plan) == 3  # previous backup + latest + alternative
        tags = [p["tag"] for p in plan]
        assert "DocWain:latest" in tags
        assert "DocWain:llama-3.1-8b" in tags
        assert "DocWain:previous" in tags
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_deployer.py -v`
Expected: FAIL

**Step 3: Write the deployer module**

```python
# src/finetune/evolve/deployer.py
"""Deployer — GGUF export and Ollama model registration."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class Deployer:
    """Handles GGUF export and Ollama registration for trained models."""

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self._ollama_host = ollama_host

    def _build_tag(self, model_name: str, is_latest: bool = False) -> str:
        if is_latest:
            return "DocWain:latest"
        return f"DocWain:{model_name}"

    def _build_previous_tag(self) -> str:
        return "DocWain:previous"

    def plan_deployment(
        self,
        winner: str,
        all_models: List[str],
        keep_previous: bool = True,
    ) -> List[Dict[str, str]]:
        """Create deployment plan without executing it."""
        plan = []
        if keep_previous:
            plan.append({"action": "backup", "tag": self._build_previous_tag()})
        plan.append({"action": "deploy", "tag": self._build_tag(winner, is_latest=True), "model": winner})
        for m in all_models:
            if m != winner:
                plan.append({"action": "deploy", "tag": self._build_tag(m, is_latest=False), "model": m})
        return plan

    def export_gguf(self, merged_dir: Path) -> Path:
        """Export merged model to GGUF format. Delegates to existing logic."""
        from src.finetune.docwain_finetune import _export_gguf
        return _export_gguf(merged_dir)

    def register_ollama(self, gguf_path: Path, tag: str) -> Dict[str, Any]:
        """Register a GGUF model with Ollama."""
        # Read and hash the file
        sha256 = hashlib.sha256()
        with open(gguf_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        digest = f"sha256:{sha256.hexdigest()}"

        # Upload blob
        resp = httpx.put(
            f"{self._ollama_host}/api/blobs/{digest}",
            content=open(gguf_path, "rb"),
            timeout=600.0,
        )
        resp.raise_for_status()

        # Create model
        modelfile = f"FROM @{digest}\n"
        resp = httpx.post(
            f"{self._ollama_host}/api/create",
            json={"model": tag, "modelfile": modelfile, "stream": False},
            timeout=300.0,
        )
        resp.raise_for_status()

        return {"tag": tag, "digest": digest, "status": "registered"}

    def deploy_all(
        self,
        plan: List[Dict[str, str]],
        artifact_dirs: Dict[str, Path],
    ) -> List[Dict[str, Any]]:
        """Execute a deployment plan."""
        results = []
        for step in plan:
            if step["action"] == "backup":
                logger.info("Backing up current DocWain:latest to %s", step["tag"])
                try:
                    resp = httpx.post(
                        f"{self._ollama_host}/api/copy",
                        json={"source": "DocWain:latest", "destination": step["tag"]},
                        timeout=120.0,
                    )
                    results.append({"tag": step["tag"], "status": "backed_up"})
                except Exception as e:
                    logger.warning("Backup failed (may not exist yet): %s", e)
                    results.append({"tag": step["tag"], "status": "skip_no_previous"})
            elif step["action"] == "deploy":
                model = step["model"]
                merged_dir = artifact_dirs.get(model)
                if not merged_dir:
                    results.append({"tag": step["tag"], "status": "skip_no_artifact"})
                    continue
                try:
                    gguf_path = self.export_gguf(merged_dir)
                    reg = self.register_ollama(gguf_path, step["tag"])
                    results.append(reg)
                except Exception as e:
                    logger.error("Deploy failed for %s: %s", step["tag"], e)
                    results.append({"tag": step["tag"], "status": "failed", "error": str(e)})
        return results
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evolve_deployer.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/finetune/evolve/deployer.py tests/test_evolve_deployer.py
git commit -m "feat(evolve): add deployer module for GGUF export and Ollama registration"
```

---

### Task 11: Pipeline Orchestrator (ties everything together)

**Files:**
- Create: `src/finetune/evolve/pipeline.py`
- Test: `tests/test_evolve_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_evolve_pipeline.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestPipelineState:
    """Tests for pipeline iteration state tracking."""

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
    """Tests for pipeline component access."""

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve_pipeline.py -v`
Expected: FAIL

**Step 3: Write the pipeline orchestrator**

```python
# src/finetune/evolve/pipeline.py
"""EvolvePipeline — orchestrates the full observe→harvest→teach→train→gate cycle."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvolveConfig
from .observer import Observer
from .harvester import Harvester
from .teacher import Teacher
from .trainer import MultiModelTrainer
from .tournament import Tournament
from .distiller import Distiller
from .gate import QualityGate, GateResult
from .registry import ModelRegistry
from .deployer import Deployer


class EvolvePipeline:
    """Main orchestrator — ties all pipeline stages together.

    Designed for interactive use in Claude Code sessions.
    Each stage returns data for review before proceeding.
    """

    def __init__(
        self,
        config: EvolveConfig,
        signals_dir: Path,
        artifact_dir: Path,
        registry_path: Path,
    ):
        self._config = config
        self._signals_dir = Path(signals_dir)
        self._artifact_dir = Path(artifact_dir)

        # Initialize components
        self.observer = Observer(config, output_dir=self._signals_dir)
        self.harvester = Harvester(signals_dir=self._signals_dir)
        self.teacher = Teacher(output_dir=self._signals_dir)
        self.trainer = MultiModelTrainer(config, artifact_dir=self._artifact_dir)
        self.tournament = Tournament(weights=config.gate.weights)
        self.distiller = Distiller(distill_every_n=config.pipeline.distillation_every_n)
        self.gate = QualityGate(
            composite_minimum=config.gate.composite_minimum,
            criterion_floor=config.gate.criterion_floor,
            must_beat_previous=config.gate.must_beat_previous,
        )
        self.registry = ModelRegistry(registry_path)
        self.deployer = Deployer(ollama_host=config.deployment.ollama_host)

    def current_iteration(self) -> int:
        """Return the highest completed iteration number, or 0 if none."""
        if not self._signals_dir.exists():
            return 0
        iters = []
        for d in self._signals_dir.iterdir():
            if d.is_dir() and re.match(r"iter_(\d+)", d.name):
                iters.append(int(d.name.split("_")[1]))
        return max(iters) if iters else 0

    def next_iteration(self) -> int:
        """Return the next iteration number."""
        return self.current_iteration() + 1

    def status(self) -> Dict[str, Any]:
        """Return pipeline status summary."""
        return {
            "current_iteration": self.current_iteration(),
            "next_iteration": self.next_iteration(),
            "enabled_students": [s.name for s in self.trainer.list_enabled_students()],
            "registry_models": [
                {"tag": m.tag, "score": m.composite_score, "status": m.status}
                for m in self.registry.list_models()
            ],
            "gate_config": {
                "composite_minimum": self._config.gate.composite_minimum,
                "criterion_floor": self._config.gate.criterion_floor,
            },
            "distillation_due": self.distiller.should_distill(self.next_iteration()),
        }

    # --- Stage methods (called interactively by Claude Code) ---

    def run_observe(self) -> Dict[str, Any]:
        """Stage 1: Probe DocWain and identify weaknesses."""
        iteration = self.next_iteration()
        signals = self.observer.probe_model()
        self.observer._save_signals(signals, iteration)
        return {
            "iteration": iteration,
            "signals_found": len(signals),
            "weak_areas": self._summarize_weak_areas(signals),
        }

    def run_harvest(self, iteration: int) -> Dict[str, Any]:
        """Stage 2: Collect and merge all signals."""
        obs_signals = self.harvester.load_observation_signals(iteration)
        int_signals = self.harvester.load_interaction_signals()
        merged = self.harvester.merge_and_dedup(obs_signals, int_signals)
        balanced = self.harvester.balance_categories(merged)
        self.harvester.save_harvest(balanced, iteration)
        return self.harvester.summarize(balanced)

    def run_gate(
        self, composite: float, scores: Dict[str, float],
    ) -> GateResult:
        """Stage 5: Evaluate against quality gate."""
        previous = self.registry.get("DocWain:latest")
        prev_composite = previous.composite_score if previous else None
        return self.gate.evaluate(composite, scores, prev_composite)

    def _summarize_weak_areas(self, signals) -> Dict[str, int]:
        """Count signals per subcategory."""
        areas: Dict[str, int] = {}
        for s in signals:
            sub = s.subcategory
            areas[sub] = areas.get(sub, 0) + 1
        return areas
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evolve_pipeline.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/finetune/evolve/pipeline.py tests/test_evolve_pipeline.py
git commit -m "feat(evolve): add pipeline orchestrator tying all stages together"
```

---

### Task 12: Update `__init__.py` exports and run full test suite

**Files:**
- Modify: `src/finetune/evolve/__init__.py`

**Step 1: Update exports**

```python
# src/finetune/evolve/__init__.py
"""Evolving fine-tune pipeline — iterative model improvement with Claude Code as teacher."""

from .config import EvolveConfig
from .pipeline import EvolvePipeline
from .registry import ModelRegistry, ModelEntry

__all__ = ["EvolveConfig", "EvolvePipeline", "ModelRegistry", "ModelEntry"]
```

**Step 2: Run all evolve tests**

Run: `pytest tests/test_evolve_*.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/finetune/evolve/__init__.py
git commit -m "feat(evolve): add public API exports"
```

---

### Task 13: Integration — Wire into existing config

**Files:**
- Modify: `src/api/config.py` — Add `Evolve` config class
- Test: Verify existing tests still pass

**Step 1: Read current config.py Finetune section to understand pattern**

Read: `src/api/config.py` lines 410-438

**Step 2: Add Evolve config class after the existing Finetune class**

Add a new `Evolve` inner class inside `Config`:
```python
class Evolve:
    ENABLED = os.getenv("EVOLVE_ENABLED", "false").lower() == "true"
    SIGNALS_DIR = os.getenv("EVOLVE_SIGNALS_DIR", "signals")
    ARTIFACT_DIR = os.getenv("EVOLVE_ARTIFACT_DIR", "finetune_artifacts")
    REGISTRY_PATH = os.getenv("EVOLVE_REGISTRY_PATH", "registry.yaml")
    CONFIG_PATH = os.getenv("EVOLVE_CONFIG_PATH", "src/finetune/evolve_config.yaml")
```

**Step 3: Run existing tests to verify no regression**

Run: `pytest tests/ -v --timeout=30 -x -q`
Expected: No regressions

**Step 4: Commit**

```bash
git add src/api/config.py
git commit -m "feat(config): add Evolve pipeline config to Config class"
```

---

## Summary

| Task | Module | Tests | Purpose |
|------|--------|-------|---------|
| 1 | config.py + YAML | 8 | Pipeline configuration with YAML loading |
| 2 | registry.py | 8 | Model registry with promote/rollback |
| 3 | observer.py + prompts | 7 | Probe DocWain, identify weaknesses |
| 4 | harvester.py | 6 | Collect + merge + dedup signals |
| 5 | teacher.py + prompts | 6 | Generate SFT + DPO training pairs |
| 6 | tournament.py | 5 | Multi-model ranking by composite score |
| 7 | gate.py | 6 | Quality gate for promotion decisions |
| 8 | trainer.py | 4 | Multi-model SFT + DPO orchestration |
| 9 | distiller.py | 4 | Hybrid distillation every Nth iteration |
| 10 | deployer.py | 3 | GGUF export + Ollama registration |
| 11 | pipeline.py | 7 | Orchestrator tying all stages together |
| 12 | __init__.py | - | Public API exports |
| 13 | config.py | - | Wire into existing app config |

**Total: 13 tasks, 64 tests, 12 new files, 1 modified file**
