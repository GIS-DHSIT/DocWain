"""Multi-model trainer — runs SFT + DPO across configured student models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvolveConfig, StudentModel

logger = logging.getLogger(__name__)

_DEFAULT_ARTIFACT_DIR = Path("finetune_artifacts")


class MultiModelTrainer:
    def __init__(self, config: EvolveConfig, artifact_dir: Optional[Path] = None):
        self._config = config
        self._artifact_dir = artifact_dir or _DEFAULT_ARTIFACT_DIR

    def list_enabled_students(self) -> List[StudentModel]:
        return [s for s in self._config.models.students if s.enabled]

    def train_sft(self, student: StudentModel, dataset_path: str, iteration: int) -> Dict[str, Any]:
        from src.finetune.unsloth_trainer import get_finetune_manager
        request = self._build_request(student, dataset_path, iteration)
        manager = get_finetune_manager()
        status = manager.start_job(request)
        return {"job_id": status.job_id, "model": student.name, "status": status.status, "iteration": iteration}

    def train_dpo(self, student: StudentModel, iteration: int, base_model_path: Optional[str] = None) -> Dict[str, Any]:
        from src.finetune.dpo_trainer import run_dpo_training
        args = self._build_dpo_args()
        args["base_model"] = base_model_path or student.repo
        result = run_dpo_training(**args)
        return {"model": student.name, "iteration": iteration, "dpo_result": result}

    def train_all(self, dataset_path: str, iteration: int) -> List[Dict[str, Any]]:
        results = []
        for student in self.list_enabled_students():
            logger.info("Training %s (iteration %d)", student.name, iteration)
            try:
                sft_result = self.train_sft(student, dataset_path, iteration)
                results.append(sft_result)
            except Exception as e:
                logger.error("SFT failed for %s: %s", student.name, e)
                results.append({"model": student.name, "iteration": iteration, "status": "failed", "error": str(e)})
        return results

    def _build_request(self, student: StudentModel, dataset_path: str, iteration: int):
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
        dpo = self._config.training.dpo
        return {
            "epochs": dpo.epochs,
            "beta": dpo.beta,
            "learning_rate": dpo.learning_rate,
            "lora_r": self._config.training.sft.lora_r,
        }

    def _artifact_path(self, iteration: int, model_name: str) -> Path:
        return self._artifact_dir / f"iter_{iteration}" / "models" / model_name
