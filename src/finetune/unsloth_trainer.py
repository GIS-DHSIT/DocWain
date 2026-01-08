import json
import math
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset

from src.api.config import Config
from src.finetune.models import FinetuneRequest, FinetuneStatus, ResolvedModel, TrainingExample

logger = logging.getLogger(__name__)

_MANAGER = None


def get_finetune_manager():
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = UnslothFinetuneManager()
    return _MANAGER


class UnslothFinetuneManager:
    """Coordinates Unsloth fine-tune jobs and activates results per profile."""

    BASE_MODEL_MAP = {
        "llama3.2": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "llama3": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    }

    def __init__(self):
        self.jobs: Dict[str, FinetuneStatus] = {}
        self.lock = threading.Lock()
        self.artifact_root = Path(Config.Path.APP_HOME) / "finetune_artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.artifact_root / "finetuned_models.json"
        self.state = self._load_state()
        self.base_models = [
            {"name": "llama3.2", "backend": "ollama", "path": None, "profile_id": None},
            {"name": "llama3.1", "backend": "ollama", "path": None, "profile_id": None},
        ]

    def start_job(self, request: FinetuneRequest) -> FinetuneStatus:
        """Start a fine-tune run asynchronously."""
        job_id = str(uuid.uuid4())
        status = FinetuneStatus(
            job_id=job_id,
            profile_id=request.profile_id,
            status="queued",
            message="scheduled",
            params=request.dict(),
        )
        with self.lock:
            self.jobs[job_id] = status

        thread = threading.Thread(target=self._run_job, args=(job_id, request), daemon=True)
        thread.start()
        return status

    def get_status(self, job_id: str) -> Optional[FinetuneStatus]:
        with self.lock:
            return self.jobs.get(job_id)

    def list_models(self) -> List[Dict[str, str]]:
        models = list(self.base_models)
        for profile_id, rec in self.state.items():
            models.append(
                {
                    "name": rec.get("served_model_name") or f"finetuned-{profile_id}",
                    "backend": rec.get("backend") or "unsloth",
                    "path": rec.get("model_path"),
                    "profile_id": profile_id,
                    "updated_at": rec.get("updated_at"),
                }
            )
        return models

    def resolve_model(self, profile_id: str | None, requested_model: str | None) -> ResolvedModel:
        if not profile_id:
            return ResolvedModel(model_name=requested_model, profile_id=None)
        record = self.state.get(profile_id)
        if not record:
            return ResolvedModel(model_name=requested_model, profile_id=profile_id)
        return ResolvedModel(
            model_name=record.get("served_model_name") or record.get("model_path") or requested_model,
            backend=record.get("backend") or "unsloth",
            model_path=record.get("model_path"),
            profile_id=profile_id,
        )

    def _run_job(self, job_id: str, request: FinetuneRequest):
        """Execute fine-tune and activate the resulting model for the profile."""
        self._update_status(job_id, status="running", message="initializing")
        try:
            dataset = self._prepare_dataset(request)
            if len(dataset) < 5:
                raise ValueError("Dataset too small for finetuning (min 5 rows)")
            eval_ds = None
            train_ds = dataset
            if len(dataset) > 20:
                split = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
                train_ds = split["train"]
                eval_ds = split["test"]

            output_dir = self._build_output_dir(request.profile_id, job_id, request.output_dir)

            model_id = self.BASE_MODEL_MAP.get(request.base_model, request.base_model)
            self._update_status(job_id, status="running", message=f"loading base model {model_id}")

            model, tokenizer = self._load_model(model_id, request)
            self._update_status(job_id, status="running", message="training with Unsloth SFTTrainer")

            eval_metrics = self._train(model, tokenizer, train_ds, request, output_dir, eval_dataset=eval_ds)
            if eval_ds:
                eval_loss = eval_metrics.get("eval_loss")
                if eval_loss is None or math.isnan(eval_loss):
                    raise ValueError("Evaluation failed; not promoting model")
                self._update_status(job_id, status="running", message=f"eval_loss={eval_loss:.4f}")

            merged_dir = output_dir / "merged"
            self._activate_profile_model(request.profile_id, merged_dir)

            self._update_status(
                job_id,
                status="completed",
                message="fine-tune finished",
                output_model=str(merged_dir),
            )
        except Exception as exc:
            logger.error("Finetune job %s failed: %s", job_id, exc, exc_info=True)
            self._update_status(job_id, status="failed", message=str(exc))

    def _prepare_dataset(self, request: FinetuneRequest) -> Dataset:
        """Build a Dataset with instruction/output pairs; pulls profile data only when allowed."""
        examples: List[TrainingExample] = []
        if request.training_examples:
            examples.extend(request.training_examples)

        if request.include_actual_data:
            examples.extend(self._load_profile_examples(request.profile_id))

        if request.dataset_path:
            ds = load_dataset("json", data_files=request.dataset_path, split="train")
            ds = ds.map(self._format_row, remove_columns=ds.column_names)
            return ds

        if not examples:
            examples = [
                TrainingExample(
                    instruction=f"Provide a concise summary for profile {request.profile_id}",
                    output="This is a placeholder summary used to warm up the fine-tune pipeline.",
                ),
            ]

        dedup = {}
        for ex in examples:
            key = (ex.instruction.strip(), ex.output.strip(), (ex.input or "").strip())
            dedup[key] = ex
        rows = [self._format_row(ex.model_dump()) for ex in dedup.values()]
        return Dataset.from_list(rows)

    @staticmethod
    def _format_row(row: Dict) -> Dict:
        """Normalize raw row to text for SFTTrainer."""
        instr = row.get("instruction") or ""
        inp = row.get("input") or ""
        out = row.get("output") or ""
        parts = [
            "### Instruction:",
            instr,
        ]
        if inp:
            parts.extend(["### Input:", inp])
        parts.extend(["### Response:", out])
        return {"text": "\n".join(parts).strip()}

    def _load_profile_examples(self, profile_id: str, limit: int = 50) -> List[TrainingExample]:
        """Pull profile-specific rows from Mongo when permitted."""
        try:
            from src.api.dataHandler import db  # lazy import to avoid startup cost

            collection = db[getattr(Config.MongoDB, "DOCUMENTS", "documents")]
            cursor = collection.find({"profileId": profile_id}).limit(limit)
            rows: List[TrainingExample] = []
            for doc in cursor:
                content = (
                    doc.get("content")
                    or doc.get("text")
                    or doc.get("body")
                    or doc.get("raw_text")
                    or ""
                )
                if not content:
                    continue
                instruction = f"Summarize document {doc.get('_id')} for profile {profile_id}"
                rows.append(TrainingExample(instruction=instruction, output=str(content)[:4000]))
            if rows:
                logger.info("Loaded %d examples from profile %s", len(rows), profile_id)
            return rows
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load profile data for finetune: %s", exc)
            return []

    @staticmethod
    def _build_output_dir(profile_id: str, job_id: str, root_dir: str) -> Path:
        out_dir = Path(root_dir)
        if not out_dir.is_absolute():
            out_dir = Path(Config.Path.APP_HOME) / out_dir
        out_dir = out_dir / profile_id / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _load_model(self, model_id: str, request: FinetuneRequest):
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer

    def _train(self, model, tokenizer, dataset: Dataset, request: FinetuneRequest, output_dir: Path, eval_dataset: Optional[Dataset] = None):
        import torch
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel

        FastLanguageModel.for_training(model)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=4096,
            packing=True,
            args=TrainingArguments(
                output_dir=str(output_dir / "checkpoints"),
                per_device_train_batch_size=request.batch_size,
                gradient_accumulation_steps=request.gradient_accumulation,
                num_train_epochs=request.num_epochs,
                max_steps=request.max_steps,
                learning_rate=request.learning_rate,
                logging_steps=5,
                save_strategy="no",
                bf16=torch.cuda.is_available(),
                fp16=not torch.cuda.is_available(),
                evaluation_strategy="no",
            ),
        )
        trainer.train()
        eval_metrics = {}
        if eval_dataset:
            eval_metrics = trainer.evaluate()
        merged_dir = output_dir / "merged"
        FastLanguageModel.save_pretrained(
            model,
            tokenizer,
            save_directory=str(merged_dir),
            save_method="merged_16bit",
        )
        return eval_metrics

    def _activate_profile_model(self, profile_id: str, model_dir: Path):
        entry = {
            "profile_id": profile_id,
            "model_path": str(model_dir),
            "backend": "unsloth",
            "served_model_name": f"finetuned-{profile_id}",
            "updated_at": time.time(),
        }
        self.state[profile_id] = entry
        self._persist_state()

    def _update_status(self, job_id: str, status: str, message: str = "", output_model: Optional[str] = None):
        with self.lock:
            if job_id not in self.jobs:
                return
            status_obj = self.jobs[job_id]
            status_obj.status = status
            status_obj.message = message
            if output_model:
                status_obj.output_model = output_model
            if status in {"completed", "failed"}:
                status_obj.finished_at = time.time()
            self.jobs[job_id] = status_obj

    def _load_state(self) -> Dict[str, Dict]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {}

    def _persist_state(self):
        try:
            self.state_path.write_text(json.dumps(self.state, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist finetune state: %s", exc)
