import hashlib
import json
import math
import logging
import inspect
import threading
import time
import uuid
import os
import traceback
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from src.api.config import Config
from src.finetune.models import FinetuneRequest, FinetuneStatus, ResolvedModel, TrainingExample
from transformers import __version__ as transformers_version

logger = logging.getLogger(__name__)

_MANAGER = None


def build_training_arguments(request: FinetuneRequest, output_dir: Path, has_eval_dataset: bool):
    """
    Build TrainingArguments in a version-tolerant way by filtering unsupported kwargs.
    """
    from transformers import TrainingArguments
    import torch

    # Device capability checks
    has_cuda = torch.cuda.is_available()
    bf16_supported = False
    if has_cuda:
        try:
            if hasattr(torch.cuda, "is_bf16_supported"):
                bf16_supported = torch.cuda.is_bf16_supported()
            else:
                major, minor = torch.cuda.get_device_capability()
                bf16_supported = major >= 8  # Ampere+
        except Exception:
            bf16_supported = False

    fp16_supported = has_cuda

    args_dict = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": request.batch_size,
        "gradient_accumulation_steps": request.gradient_accumulation,
        "num_train_epochs": request.num_epochs,
        "max_steps": request.max_steps,
        "learning_rate": request.learning_rate,
        "logging_steps": 5,
        "save_strategy": "no",
        "bf16": bf16_supported,
        "fp16": fp16_supported,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    eval_kwargs = {}
    eval_config_used = "none"
    if has_eval_dataset:
        if "evaluation_strategy" in supported:
            eval_kwargs["evaluation_strategy"] = "steps"
            if "eval_steps" in supported:
                eval_kwargs["eval_steps"] = max(1, request.max_steps // 5)
            eval_config_used = "evaluation_strategy"
        elif "eval_strategy" in supported:
            eval_kwargs["eval_strategy"] = "steps"
            if "eval_steps" in supported:
                eval_kwargs["eval_steps"] = max(1, request.max_steps // 5)
            eval_config_used = "eval_strategy"
        elif "eval_steps" in supported:
            eval_kwargs["eval_steps"] = max(1, request.max_steps // 5)
            eval_config_used = "eval_steps_only"

    filtered_args = {k: v for k, v in args_dict.items() if k in supported}
    filtered_args.update({k: v for k, v in eval_kwargs.items() if k in supported})

    unsupported = {k: v for k, v in {**args_dict, **eval_kwargs}.items() if k not in supported}
    if unsupported:
        logger.debug("TrainingArguments unsupported kwargs filtered: %s", list(unsupported.keys()))

    if has_eval_dataset and not eval_kwargs:
        logger.warning(
            "Eval dataset provided but evaluation scheduling args not supported by this transformers version; "
            "continuing without scheduled evaluation."
        )

    logger.info(
        "Building TrainingArguments with transformers %s | eval_config=%s | supported_keys=%s",
        transformers_version,
        eval_config_used,
        sorted(filtered_args.keys()),
    )
    return TrainingArguments(**filtered_args)


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
        self._concurrency_sem = threading.Semaphore(int(os.getenv("FINETUNE_MAX_CONCURRENT", "2")))
        self._redis_lock_client = None
        self.artifact_root = Path(Config.Path.APP_HOME) / "finetune_artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.artifact_root / "finetuned_models.json"
        self.state = self._load_state()
        self.run_index_path = self.artifact_root / "finetune_runs.json"
        self.run_index = self._load_runs()
        self.base_models = [
            {"name": "llama3.2", "backend": "ollama", "path": None, "profile_id": None},
            {"name": "llama3.1", "backend": "ollama", "path": None, "profile_id": None},
        ]

    def start_job(self, request: FinetuneRequest) -> FinetuneStatus:
        """Start a fine-tune run asynchronously."""
        params_hash = self._params_hash(request)
        if not request.retrain:
            existing = self._existing_active_run(request.profile_id, params_hash)
            if existing:
                existing.message = existing.message or "Existing training run reused"
                with self.lock:
                    self.jobs.setdefault(existing.job_id, existing)
                return existing

        job_id = str(uuid.uuid4())
        run_id = request.run_name or job_id
        status = FinetuneStatus(
            job_id=job_id,
            profile_id=request.profile_id,
            status="queued",
            message="scheduled",
            params=request.dict(),
            training_run_id=run_id,
            params_hash=params_hash,
        )
        with self.lock:
            self.jobs[job_id] = status
            self.run_index[request.profile_id] = {
                "run_id": run_id,
                "job_id": job_id,
                "params_hash": params_hash,
                "status": "queued",
                "started_at": status.started_at,
            }
            self._persist_runs()

        thread = threading.Thread(target=self._run_job, args=(job_id, request), daemon=True)
        thread.start()
        return status

    def get_status(self, job_id: str) -> Optional[FinetuneStatus]:
        with self.lock:
            if job_id in self.jobs:
                return self.jobs.get(job_id)
            for profile_id, entry in self.run_index.items():
                if entry.get("job_id") == job_id or entry.get("run_id") == job_id:
                    return FinetuneStatus(
                        job_id=job_id,
                        profile_id=profile_id,
                        status=entry.get("status", "completed"),
                        message=entry.get("message", ""),
                        output_model=entry.get("output_model"),
                        started_at=entry.get("started_at", time.time()),
                        finished_at=entry.get("finished_at"),
                        params={},
                        training_run_id=entry.get("run_id"),
                        params_hash=entry.get("params_hash"),
                    )
            return None

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
        import torch

        redis_lock = self._acquire_redis_lock(job_id)
        if redis_lock is False:
            self._update_status(job_id, status="failed", message="Unable to acquire training lock; try again later.")
            return

        acquired = self._concurrency_sem.acquire(blocking=False)
        if not acquired:
            self._update_status(job_id, status="failed", message="Too many concurrent finetune jobs; please retry later.")
            return
        self._update_status(job_id, status="running", message="initializing")
        manifest = {
            "job_id": job_id,
            "profile_id": request.profile_id,
            "run_name": request.run_name or job_id,
            "base_model": request.base_model,
            "generation_model": None,
            "started_at": time.time(),
            "finished_at": None,
            "status": "running",
            "error": None,
            "traceback": None,
            "config": {k: v for k, v in request.dict().items() if k != "training_examples"},
            "dataset": {},
            "versions": {
                "transformers": transformers_version,
            },
        }
        try:
            from unsloth import __version__ as unsloth_version
            manifest["versions"]["unsloth"] = unsloth_version
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                self._ensure_gpu_headroom()
            dataset = self._prepare_dataset(request)
            dataset_hash = self._hash_dataset(request.dataset_path)
            if len(dataset) < 5:
                raise ValueError("Dataset too small for finetuning (min 5 rows)")
            eval_ds = None
            train_ds = dataset
            if len(dataset) > 20:
                split = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
                train_ds = split["train"]
                eval_ds = split["test"]

            if len(train_ds) == 0:
                raise ValueError("Training dataset is empty after preprocessing")
            if eval_ds is not None and len(eval_ds) == 0:
                eval_ds = None

            manifest["dataset"]["train_size"] = len(train_ds)
            manifest["dataset"]["eval_size"] = len(eval_ds) if eval_ds else 0
            manifest["dataset"]["columns"] = train_ds.column_names

            output_dir = self._build_output_dir(request, job_id)

            model_id = self.BASE_MODEL_MAP.get(request.base_model, request.base_model)
            self._update_status(job_id, status="running", message=f"loading base model {model_id}")
            if torch.cuda.is_available():
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            load_notes: Dict[str, Any] = {}
            try:
                model, tokenizer = self._load_model(model_id, request, load_notes)
            except torch.cuda.OutOfMemoryError as oom:
                logger.error("OOM during model load; retrying with memory-safe settings: %s", oom)
                torch.cuda.empty_cache()
                request = self._memory_safe_request(request)
                manifest["oom_retry_load"] = {
                    "batch_size": request.batch_size,
                    "gradient_accumulation": request.gradient_accumulation,
                }
                load_notes = {}
                model, tokenizer = self._load_model(model_id, request, load_notes)
            manifest["base_model_resolved"] = model_id
            manifest["load"] = load_notes
            logger.info(
                "[FINETUNE] job_id=%s | base_model=%s | dataset(train=%d, eval=%d) | batch=%d | grad_accum=%d",
                job_id,
                model_id,
                len(train_ds),
                len(eval_ds) if eval_ds else 0,
                request.batch_size,
                request.gradient_accumulation,
            )
            self._update_status(job_id, status="running", message="training with Unsloth SFTTrainer")

            try:
                eval_metrics = self._train(model, tokenizer, train_ds, request, output_dir, eval_dataset=eval_ds)
            except torch.cuda.OutOfMemoryError as oom:
                logger.error("OOM during training; retrying with memory-safe settings: %s", oom)
                self._update_status(job_id, status="running", message="OOM hit; retrying with smaller batch")
                torch.cuda.empty_cache()
                try:
                    del model
                    del tokenizer
                except Exception:
                    pass
                request = self._memory_safe_request(request)
                manifest["oom_retry_train"] = {
                    "batch_size": request.batch_size,
                    "gradient_accumulation": request.gradient_accumulation,
                }
                load_notes = {}
                model, tokenizer = self._load_model(model_id, request, load_notes)
                manifest["load_retry_after_oom"] = load_notes
                eval_metrics = self._train(model, tokenizer, train_ds, request, output_dir, eval_dataset=eval_ds)
            if eval_ds:
                eval_loss = eval_metrics.get("eval_loss")
                if eval_loss is None or math.isnan(eval_loss):
                    raise ValueError("Evaluation failed; not promoting model")
                self._update_status(job_id, status="running", message=f"eval_loss={eval_loss:.4f}")

            merged_dir = output_dir / "merged"
            self._activate_profile_model(
                request.profile_id,
                merged_dir,
                base_model=request.base_model,
                dataset_hash=dataset_hash,
                qdrant_snapshot=request.qdrant_snapshot,
                eval_loss=eval_loss,
            )

            self._update_status(
                job_id,
                status="completed",
                message="fine-tune finished",
                output_model=str(merged_dir),
            )
            manifest["status"] = "completed"
            manifest["finished_at"] = time.time()
            manifest["eval_metrics"] = eval_metrics or {}
            self._persist_manifest(output_dir, manifest)
        except Exception as exc:
            logger.error("Finetune job %s failed: %s", job_id, exc, exc_info=True)
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            manifest["traceback"] = traceback.format_exc()
            manifest["finished_at"] = time.time()
            try:
                self._persist_manifest(output_dir if 'output_dir' in locals() else Path(Config.Path.APP_HOME) / "finetune_artifacts", manifest)
            except Exception:
                pass
            self._update_status(job_id, status="failed", message=str(exc))
        finally:
            try:
                self._concurrency_sem.release()
            except Exception:
                pass
            if redis_lock not in {None, False}:
                with contextlib.suppress(Exception):
                    redis_lock.release()

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

    def _build_output_dir(self, request: FinetuneRequest, job_id: str) -> Path:
        run_name = request.run_name or f"run-{request.profile_id}"
        out_dir = Path(request.output_dir)
        if not out_dir.is_absolute():
            out_dir = Path(Config.Path.APP_HOME) / out_dir
        out_dir = out_dir / run_name / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _load_model(self, model_id: str, request: FinetuneRequest, load_notes: Optional[Dict[str, Any]] = None):
        from unsloth import FastLanguageModel
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for 4-bit finetuning. No GPU detected. "
                "Use a CUDA-capable host or disable finetuning."
            )

        notes = load_notes if load_notes is not None else {}
        notes.setdefault("meta_retry", False)

        def _attempt_load(safe_mode: bool, overrides: Optional[Dict[str, Any]] = None):
            return self._load_model_once(
                model_id,
                request,
                FastLanguageModel,
                safe_mode=safe_mode,
                notes=notes,
                overrides=overrides,
            )

        try:
            model, tokenizer = _attempt_load(safe_mode=False)
            fell_back = False
        except ValueError as exc:
            if self._is_bnb_dispatch_error(exc) and request.allow_offload:
                fallback_overrides: Dict[str, Any] = {
                    "device_map": request.device_map or "auto",
                    "llm_int8_enable_fp32_cpu_offload": True,
                }
                max_memory = request.max_memory or self._default_max_memory()
                if max_memory:
                    fallback_overrides["max_memory"] = max_memory
                logger.warning(
                    "BNB dispatch error when loading %s; retrying with offload and device_map=%s",
                    model_id,
                    fallback_overrides.get("device_map"),
                )
                notes["bnb_dispatch_retry"] = fallback_overrides
                try:
                    model, tokenizer = _attempt_load(safe_mode=False, overrides=fallback_overrides)
                    fell_back = True
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"Failed to load {model_id} with offload after BNB dispatch error. "
                        f"Consider a smaller base model or lower max_seq_length. "
                        f"GPU memory status: {self._gpu_memory_status()}"
                    ) from retry_exc
            else:
                raise
        except Exception as exc:
            if self._is_meta_tensor_error(exc):
                logger.warning(
                    "Encountered meta tensor error while loading %s; retrying with safer settings.", model_id
                )
                notes["meta_retry"] = True
                self._cleanup_cuda()
                model, tokenizer = _attempt_load(safe_mode=True)
                fell_back = True
            else:
                raise

        if self._has_meta_tensors(model):
            if fell_back:
                params, buffers = self._find_meta_tensors(model)
                raise RuntimeError(
                    f"Model still has meta tensors after safe load: params={params[:3]}, buffers={buffers[:3]}"
                )
            logger.warning("Meta tensors detected after loading %s; retrying with safer settings.", model_id)
            notes["meta_retry"] = True
            self._cleanup_cuda()
            model, tokenizer = _attempt_load(safe_mode=True)
            if self._has_meta_tensors(model):
                params, buffers = self._find_meta_tensors(model)
                raise RuntimeError(
                    f"Model still has meta tensors after safe reload: params={params[:3]}, buffers={buffers[:3]}"
                )

        return model, tokenizer

    def _load_model_once(
        self,
        model_id: str,
        request: FinetuneRequest,
        fast_language_model_cls,
        safe_mode: bool,
        notes: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ):
        import torch

        load_kwargs: Dict[str, Any] = {
            "model_name": model_id,
            "max_seq_length": 4096,
            "load_in_4bit": True,
        }

        # Respect explicit overrides first.
        overrides = overrides or {}

        if request.max_memory and "max_memory" not in overrides:
            load_kwargs["max_memory"] = request.max_memory

        if request.device_map is not None and "device_map" not in overrides:
            load_kwargs["device_map"] = request.device_map
        elif torch.cuda.is_available() and "device_map" not in overrides:
            load_kwargs["device_map"] = {"": self._preferred_device()}

        target_device = None
        if safe_mode:
            target_device = self._preferred_device()
            load_kwargs["device_map"] = {"": target_device}
            load_kwargs["low_cpu_mem_usage"] = False
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        load_kwargs.update(overrides)

        logger.info(
            "Loading model %s with kwargs %s",
            model_id,
            {k: v for k, v in load_kwargs.items() if k not in {"token"}},
        )

        model, tokenizer = fast_language_model_cls.from_pretrained(**load_kwargs)
        model = fast_language_model_cls.get_peft_model(
            model,
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )

        if notes is not None:
            attempt = {
                "mode": "safe" if safe_mode else "default",
                "device_map": load_kwargs.get("device_map"),
                "low_cpu_mem_usage": load_kwargs.get("low_cpu_mem_usage", True),
            }
            if target_device is not None:
                attempt["target_device"] = str(target_device)
            notes.setdefault("attempts", []).append(attempt)

        return model, tokenizer

    def _train(self, model, tokenizer, dataset: Dataset, request: FinetuneRequest, output_dir: Path, eval_dataset: Optional[Dataset] = None):
        import torch
        from trl import SFTTrainer
        from unsloth import FastLanguageModel

        if "text" not in dataset.column_names:
            raise ValueError(f"Dataset missing required 'text' column: {dataset.column_names}")
        if eval_dataset is not None and len(eval_dataset) > 0 and "text" not in eval_dataset.column_names:
            raise ValueError("Eval dataset missing required 'text' column")

        FastLanguageModel.for_training(model)

        try:
            training_args = build_training_arguments(request, output_dir, has_eval_dataset=bool(eval_dataset))
        except ValueError as arg_exc:
            logger.warning("TrainingArguments build failed (%s); retrying with fp32 CPU defaults", arg_exc)
            # Fallback: disable mixed precision flags and rebuild
            request_fp32 = request.model_copy()
            training_args = build_training_arguments(request_fp32, output_dir, has_eval_dataset=bool(eval_dataset))
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=4096,
            packing=True,
            args=training_args,
        )
        logger.info(
            "Starting training job | device=%s | bf16=%s | fp16=%s",
            "cuda" if torch.cuda.is_available() else "cpu",
            getattr(training_args, "bf16", None),
            getattr(training_args, "fp16", None),
        )
        trainer.train()
        eval_metrics: Dict[str, float] = {}
        if eval_dataset:
            try:
                eval_metrics = trainer.evaluate() or {}
            except Exception as eval_exc:
                logger.warning("Evaluation failed to run; continuing without eval metrics: %s", eval_exc)
                eval_metrics = {}
        merged_dir = output_dir / "merged"
        self._save_model(model, tokenizer, merged_dir)
        return eval_metrics

    def _save_model(self, model, tokenizer, output_dir: Path):
        """
        Save a trained model+tokenizer using the best available Unsloth/HF hook.

        Unsloth 2026.1.x removed FastLanguageModel.save_pretrained, so we try the
        model-level merged save first, then fall back to the class helper (if present),
        and finally to vanilla HF save_pretrained.
        """
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except Exception:
            FastLanguageModel = None  # noqa: N806

        output_dir.mkdir(parents=True, exist_ok=True)

        # Preferred: Unsloth merged save on the model instance
        if hasattr(model, "save_pretrained_merged"):
            logger.info("Saving model via save_pretrained_merged to %s", output_dir)
            model.save_pretrained_merged(str(output_dir), tokenizer=tokenizer, save_method="merged_16bit")
            return

        # Secondary: class-level helper if the installed Unsloth exposes it
        if FastLanguageModel and hasattr(FastLanguageModel, "save_pretrained"):
            logger.info("Saving model via FastLanguageModel.save_pretrained to %s", output_dir)
            FastLanguageModel.save_pretrained(
                model,
                tokenizer,
                save_directory=str(output_dir),
                save_method="merged_16bit",
            )
            return

        # Fallback: plain HF save + tokenizer save
        logger.warning(
            "Unsloth merged save helper not available; falling back to model.save_pretrained/tokenizer.save_pretrained."
        )
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(output_dir))
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(output_dir))

    def _activate_profile_model(self, profile_id: str, model_dir: Path):
        entry = {
            "profile_id": profile_id,
            "model_path": str(model_dir),
            "base_model": base_model,
            "dataset_hash": dataset_hash,
            "qdrant_snapshot": qdrant_snapshot,
            "eval_loss": eval_loss,
            "created_at": time.time(),
        }
        runs.append(run_record)
        entry.update(
            {
                "profile_id": profile_id,
                "model_path": str(model_dir),
                "backend": "unsloth",
                "served_model_name": f"finetuned-{profile_id}",
                "updated_at": time.time(),
                "runs": runs,
            }
        )
        self.state[profile_id] = entry
        self._persist_state()
        self._cleanup_old_runs(profile_id)

    def _cleanup_old_runs(self, profile_id: str) -> None:
        if not getattr(Config.Finetune, "CLEANUP_ENABLED", False):
            return
        keep = int(getattr(Config.Finetune, "CLEANUP_KEEP_LAST", 3))
        entry = self.state.get(profile_id, {})
        runs = entry.get("runs", [])
        if len(runs) <= keep:
            return
        runs_sorted = sorted(runs, key=lambda r: r.get("created_at", 0), reverse=True)
        to_keep = runs_sorted[:keep]
        to_remove = runs_sorted[keep:]
        for run in to_remove:
            model_path = run.get("model_path")
            if model_path:
                try:
                    shutil.rmtree(Path(model_path).parent, ignore_errors=True)
                except Exception:
                    pass
        entry["runs"] = to_keep
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
            run_entry = self.run_index.get(status_obj.profile_id)
            if run_entry and run_entry.get("job_id") == job_id:
                run_entry["status"] = status
                run_entry["message"] = message
                run_entry["output_model"] = output_model or run_entry.get("output_model")
                run_entry["finished_at"] = status_obj.finished_at
                self.run_index[status_obj.profile_id] = run_entry
                self._persist_runs()

    def _load_state(self) -> Dict[str, Dict]:
        if not self.state_path.exists():
            return {}
        try:
            raw = json.loads(self.state_path.read_text())
            for profile_id, rec in raw.items():
                rec.setdefault("runs", [])
            return raw
        except Exception:
            return {}

    def _persist_state(self):
        try:
            self.state_path.write_text(json.dumps(self.state, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist finetune state: %s", exc)

    def _load_runs(self) -> Dict[str, Dict]:
        if not self.run_index_path.exists():
            return {}
        try:
            return json.loads(self.run_index_path.read_text())
        except Exception:
            return {}

    def _persist_runs(self):
        try:
            self.run_index_path.write_text(json.dumps(self.run_index, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist finetune run index: %s", exc)

    @staticmethod
    def _persist_manifest(output_dir: Path, manifest: Dict[str, Any]):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "manifest.json"
            tmp_path = manifest_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(manifest, indent=2, default=str))
            tmp_path.replace(manifest_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist finetune manifest: %s", exc)

    @staticmethod
    def _memory_safe_request(request: FinetuneRequest) -> FinetuneRequest:
        safe = request.model_copy()
        safe.batch_size = 1
        safe.gradient_accumulation = max(
            request.gradient_accumulation * request.batch_size,
            request.gradient_accumulation,
            1,
        )
        safe.learning_rate = min(request.learning_rate, 2e-4)
        return safe

    @staticmethod
    def _preferred_device():
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    @staticmethod
    def _cleanup_cuda():
        import torch

        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.ipc_collect()

    @staticmethod
    def _find_meta_tensors(model) -> tuple[list[str], list[str]]:
        meta_params = [name for name, param in model.named_parameters() if getattr(param, "device", None) and param.device.type == "meta"]
        meta_buffers = [name for name, buf in model.named_buffers() if getattr(buf, "device", None) and buf.device.type == "meta"]
        return meta_params, meta_buffers

    def _has_meta_tensors(self, model) -> bool:
        params, buffers = self._find_meta_tensors(model)
        return bool(params or buffers)

    @staticmethod
    def _is_meta_tensor_error(exc: Exception) -> bool:
        return "meta tensor" in str(exc).lower()

    @staticmethod
    def _is_bnb_dispatch_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "some modules are dispatched on the cpu or the disk" in msg

    @staticmethod
    def _default_max_memory():
        import torch

        if not torch.cuda.is_available():
            return None
        try:
            free, total = torch.cuda.mem_get_info()
            free_gb = max(int((free / (1024**3)) * 0.8), 1)
            return {0: f"{free_gb}GiB", "cpu": "48GiB"}
        except Exception:
            return None

    @staticmethod
    def _gpu_memory_status():
        import torch

        if not torch.cuda.is_available():
            return "cuda_unavailable"
        try:
            free, total = torch.cuda.mem_get_info()
            return f"free={free/(1024**3):.2f}GiB total={total/(1024**3):.2f}GiB"
        except Exception:
            return "unavailable"

    def _acquire_redis_lock(self, job_id: str):
        lock_key = os.getenv("FINETUNE_REDIS_LOCK_KEY")
        if not lock_key:
            return None
        try:
            import redis

            if not self._redis_lock_client:
                self._redis_lock_client = redis.Redis(
                    host=getattr(Config.Redis, "HOST", "localhost"),
                    port=int(getattr(Config.Redis, "PORT", 6379)),
                    password=getattr(Config.Redis, "PASSWORD", None),
                    ssl=getattr(Config.Redis, "SSL", False),
                )
            lock = self._redis_lock_client.lock(
                name=lock_key,
                timeout=int(os.getenv("FINETUNE_REDIS_LOCK_TIMEOUT", "1800")),
                blocking_timeout=int(os.getenv("FINETUNE_REDIS_LOCK_WAIT", "30")),
            )
            acquired = lock.acquire(blocking=True)
            if not acquired:
                return False
            logger.info("Acquired Redis training lock for job %s under key %s", job_id, lock_key)
            return lock
        except Exception as exc:
            logger.warning("Redis lock unavailable; continuing without distributed lock: %s", exc)
            return None

    @staticmethod
    def _ensure_gpu_headroom(min_free_mb: int = None):
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024 * 1024)
        min_required = float(min_free_mb or os.getenv("FINETUNE_MIN_FREE_MEM_MB", "2048"))
        if free_mb < min_required:
            logger.info(
                "Waiting for GPU memory to free up (free=%.1fMB, required=%.1fMB)", free_mb, min_required
            )
            waited = 0
            wait_cap = int(os.getenv("FINETUNE_GPU_WAIT_SEC", "300"))
            interval = 5
            while free_mb < min_required and waited < wait_cap:
                time.sleep(interval)
                waited += interval
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                free_mb = free / (1024 * 1024)
            if free_mb < min_required:
                raise RuntimeError(
                    f"Insufficient GPU memory after waiting {waited}s "
                    f"(free={free_mb:.1f}MB, required={min_required:.1f}MB)"
                )

    @staticmethod
    def _params_hash(request: FinetuneRequest) -> str:
        payload = request.dict()
        payload.pop("run_name", None)
        payload.pop("retrain", None)
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _existing_active_run(self, profile_id: str, params_hash: str) -> Optional[FinetuneStatus]:
        with self.lock:
            for status in self.jobs.values():
                if (
                    status.profile_id == profile_id
                    and status.params_hash == params_hash
                    and status.status in {"queued", "running"}
                ):
                    return status
            run_entry = self.run_index.get(profile_id)
            if run_entry and run_entry.get("params_hash") == params_hash and run_entry.get("status") in {"queued", "running", "completed"}:
                return FinetuneStatus(
                    job_id=run_entry.get("job_id") or run_entry.get("run_id") or str(uuid.uuid4()),
                    profile_id=profile_id,
                    status=run_entry.get("status") or "completed",
                    message=run_entry.get("message") or "Existing training run already recorded",
                    output_model=run_entry.get("output_model"),
                    params={},
                    training_run_id=run_entry.get("run_id"),
                    params_hash=params_hash,
                )
        return None
