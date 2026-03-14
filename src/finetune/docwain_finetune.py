"""Fine-tuning script for DocWain-Agent-v2 (TaskSpec understanding).

Integrates autonomous experiment loop patterns from karpathy/autoresearch:
- Results tracking via TSV (each iteration logged with metrics)
- Advance-or-revert strategy (keep best model, revert on regression)
- Hyperparameter experimentation (vary LoRA/LR/epochs between iterations)
- Fast-fail (abort if training loss explodes)
- Log capture (training output to files, not flooding stdout)
- Reliable GGUF export (convert_hf_to_gguf.py primary, Unsloth fallback)

Train → evaluate → retrain loop:
1. Load base model (Meta-Llama-3.1-8B-Instruct via Unsloth)
2. Apply LoRA with experiment-specific hyperparameters
3. Train on TaskSpec dataset
4. Export to GGUF via convert_hf_to_gguf.py
5. Register as DocWain-Agent-v2 in Ollama via blob API
6. Evaluate against ground truth
7. If improved → keep (advance). If worse → revert to best.
8. Augment data for weak categories, try new hyperparams, repeat.

Usage::

    python -m src.finetune.docwain_finetune
    python -m src.finetune.docwain_finetune --max-iterations 10 --target-score 85
"""

from __future__ import annotations

# Prevent DeepSpeed from failing when CUDA toolkit isn't installed system-wide.
import os as _os
if not _os.environ.get("CUDA_HOME") and not _os.path.exists("/usr/local/cuda/bin/nvcc"):
    try:
        import torch as _torch
        _cuda_ver = _torch.version.cuda or "12.8"
        _shim = "/tmp/cuda_home"
        _os.makedirs(f"{_shim}/bin", exist_ok=True)
        _nvcc = f"{_shim}/bin/nvcc"
        if not _os.path.exists(_nvcc):
            with open(_nvcc, "w") as _f:
                _f.write(
                    "#!/bin/bash\n"
                    'echo "nvcc: NVIDIA (R) Cuda compiler driver"\n'
                    f'echo "Cuda compilation tools, release {_cuda_ver}, V{_cuda_ver}.61"\n'
                    f'echo "Build cuda_{_cuda_ver}.r{_cuda_ver}/compiler.0"\n'
                )
            _os.chmod(_nvcc, 0o755)
        _os.environ["CUDA_HOME"] = _shim
    except Exception:
        pass

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "docwain-agent-v2"
BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
ARTIFACT_DIR = Path(os.getenv("DOCWAIN_HOME", ".")) / "finetune_artifacts" / "taskspec"
DATASET_DIR = Path("finetune_data")
RESULTS_TSV = ARTIFACT_DIR / "results.tsv"

# Default LoRA hyperparameters (fits T4 16GB)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LENGTH = 4096

# Default training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01

# Evaluation targets
DEFAULT_TARGET_SCORE = 80
DEFAULT_MAX_ITERATIONS = 5

# Fast-fail: abort training if loss exceeds this early in training
LOSS_EXPLOSION_THRESHOLD = 10.0

# ── Hyperparameter schedules (autoresearch-style experimentation) ──────────
# Each iteration can try different hyperparams. Index by (iteration - 1) % len.
_EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    # Iter 1: Baseline
    {"lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 3, "description": "baseline r=16 a=32 lr=2e-4"},
    # Iter 2: Higher rank, lower LR
    {"lora_r": 32, "lora_alpha": 64, "lr": 1e-4, "epochs": 3, "description": "higher rank r=32 a=64 lr=1e-4"},
    # Iter 3: More epochs, warmup
    {"lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 5, "warmup_steps": 20, "description": "more epochs=5 warmup=20"},
    # Iter 4: Extended targets + gate projection
    {"lora_r": 16, "lora_alpha": 32, "lr": 1.5e-4, "epochs": 4, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "description": "extended targets + gate/up/down proj"},
    # Iter 5: Lower LR, more accumulation
    {"lora_r": 32, "lora_alpha": 64, "lr": 5e-5, "epochs": 5, "grad_accum": 8, "description": "r=32 lr=5e-5 accum=8 epochs=5"},
]


@dataclass
class FinetuneResult:
    """Outcome of the full fine-tuning pipeline."""

    success: bool = False
    model_name: str = ""
    iterations_run: int = 0
    final_score: float = 0.0
    best_score: float = 0.0
    eval_details: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: str = ""
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Results tracking (autoresearch pattern)
# ═══════════════════════════════════════════════════════════════════════════════


def _init_results_tsv() -> None:
    """Initialize results.tsv if it doesn't exist."""
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "iteration\tcomposite\tparse_rate\tintent_acc\tdomain_acc\tentity_recall\t"
            "constraint_acc\tstatus\tdescription\n"
        )


def _log_result(
    iteration: int,
    eval_result: Dict[str, Any],
    status: str,
    description: str,
) -> None:
    """Append an experiment result to results.tsv."""
    _init_results_tsv()
    row = (
        f"{iteration}\t"
        f"{eval_result.get('composite_score', 0.0):.1f}\t"
        f"{eval_result.get('json_parse_rate', 0.0):.1f}\t"
        f"{eval_result.get('intent_accuracy', 0.0):.1f}\t"
        f"{eval_result.get('domain_accuracy', 0.0):.1f}\t"
        f"{eval_result.get('entity_recall', 0.0):.1f}\t"
        f"{eval_result.get('constraint_accuracy', 0.0):.1f}\t"
        f"{status}\t{description}\n"
    )
    with open(RESULTS_TSV, "a") as f:
        f.write(row)
    log.info("Logged result: iter=%d composite=%.1f status=%s",
             iteration, eval_result.get("composite_score", 0.0), status)


# ═══════════════════════════════════════════════════════════════════════════════
# Training pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_finetune_pipeline(
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    target_score: int = DEFAULT_TARGET_SCORE,
    dataset_dir: Optional[Path] = None,
) -> FinetuneResult:
    """Execute the autonomous train → evaluate → advance/revert loop.

    Inspired by karpathy/autoresearch: each iteration tries a different
    hyperparameter config. If the score improves, keep (advance). If worse,
    revert to the best model and try a different config.
    """
    dataset_dir = dataset_dir or DATASET_DIR
    result = FinetuneResult()
    _init_results_tsv()

    # Step 1: Generate training data if not present
    train_path = dataset_dir / "taskspec_train.jsonl"
    eval_path = dataset_dir / "taskspec_eval.jsonl"

    if not train_path.exists():
        log.info("Generating training dataset...")
        from src.finetune.training_data_generator import build_training_dataset
        build_result = build_training_dataset(output_dir=dataset_dir)
        log.info(
            "Dataset: %d train, %d eval",
            build_result.train_count, build_result.eval_count,
        )

    best_score = 0.0
    best_iter = 0

    for iteration in range(1, max_iterations + 1):
        # Select experiment config (cycle through configs)
        config_idx = (iteration - 1) % len(_EXPERIMENT_CONFIGS)
        exp_config = _EXPERIMENT_CONFIGS[config_idx]
        description = exp_config.get("description", f"iter_{iteration}")

        log.info("═══ Iteration %d/%d — %s ═══", iteration, max_iterations, description)
        result.iterations_run = iteration

        try:
            # Step 2: Train with experiment-specific hyperparameters
            output_dir = ARTIFACT_DIR / f"iter_{iteration}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Capture training logs to file (autoresearch pattern)
            train_log_path = output_dir / "train.log"

            _train_model(
                train_path, eval_path, output_dir,
                lora_r=exp_config.get("lora_r", LORA_R),
                lora_alpha=exp_config.get("lora_alpha", LORA_ALPHA),
                learning_rate=exp_config.get("lr", LEARNING_RATE),
                num_epochs=exp_config.get("epochs", NUM_EPOCHS),
                warmup_steps=exp_config.get("warmup_steps", WARMUP_STEPS),
                grad_accum=exp_config.get("grad_accum", GRADIENT_ACCUMULATION),
                target_modules=exp_config.get("target_modules", TARGET_MODULES),
                train_log_path=train_log_path,
            )

            # Step 3: Export to GGUF
            gguf_path = _export_gguf(output_dir)

            # Step 4: Register in Ollama
            _register_ollama(gguf_path, output_dir)

            # Step 5: Evaluate
            eval_result = _evaluate_with_retry(eval_path)
            score = eval_result.get("composite_score", 0.0)
            result.final_score = score
            result.eval_details = eval_result

            # Save eval results to file
            eval_out = output_dir / "eval_results.json"
            with open(eval_out, "w") as f:
                json.dump(eval_result, f, indent=2)

            log.info(
                "Iteration %d: composite=%.1f%% parse=%.1f%% intent=%.1f%% domain=%.1f%% (target: %d%%)",
                iteration, score,
                eval_result.get("json_parse_rate", 0.0),
                eval_result.get("intent_accuracy", 0.0),
                eval_result.get("domain_accuracy", 0.0),
                target_score,
            )

            # Step 6: Advance-or-revert (autoresearch pattern)
            if score >= target_score and eval_result.get("json_parse_rate", 0) >= 95:
                # Target met — we're done
                _log_result(iteration, eval_result, "pass", description)
                log.info("TARGET MET! Score %.1f%% >= %d%%", score, target_score)
                result.success = True
                result.best_score = score
                result.model_name = f"{MODEL_NAME}:latest"
                result.artifact_dir = str(output_dir)
                return result

            if score > best_score:
                # Improved — advance (keep this model as best)
                _log_result(iteration, eval_result, "keep", description)
                best_score = score
                best_iter = iteration
                result.best_score = best_score
                log.info("IMPROVED: %.1f%% > previous best. Advancing.", score)
            else:
                # Equal or worse — revert (restore best model)
                _log_result(iteration, eval_result, "discard", description)
                log.info("NO IMPROVEMENT: %.1f%% <= best %.1f%%. Discarding.", score, best_score)
                if best_iter > 0:
                    _revert_to_best(best_iter, output_dir)

            # Step 7: Augment dataset for next iteration
            if iteration < max_iterations and score < target_score:
                log.info("Score %.1f%% < target %d%% — augmenting dataset...", score, target_score)
                _augment_dataset(eval_result, train_path, dataset_dir)

        except _TrainingDiverged as exc:
            # Fast-fail: training loss exploded
            _log_result(iteration, {}, "crash", f"DIVERGED: {description}")
            log.warning("Training diverged at iteration %d: %s", iteration, exc)
            result.error = str(exc)
            if iteration == max_iterations:
                return result
            log.info("Trying next hyperparameter config...")

        except Exception as exc:
            _log_result(iteration, {}, "crash", f"ERROR: {description}")
            log.error("Iteration %d failed: %s", iteration, exc, exc_info=True)
            result.error = str(exc)
            if iteration == max_iterations:
                return result
            log.info("Retrying with next iteration...")

    if not result.success:
        result.error = f"Max iterations ({max_iterations}) reached. Best score: {best_score:.1f}%"
    return result


class _TrainingDiverged(RuntimeError):
    """Raised when training loss explodes (fast-fail)."""


def _revert_to_best(best_iter: int, current_dir: Path) -> None:
    """Restore the best model from a previous iteration (autoresearch revert)."""
    best_dir = ARTIFACT_DIR / f"iter_{best_iter}"
    best_gguf = None
    for f in best_dir.rglob("*.gguf"):
        best_gguf = f
        break
    if best_gguf and best_gguf.exists():
        log.info("Reverting to best model from iteration %d", best_iter)
        _register_ollama(best_gguf, best_dir)
    else:
        log.warning("Could not find best model artifacts at %s", best_dir)


def _train_model(
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    *,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    warmup_steps: int = WARMUP_STEPS,
    grad_accum: int = GRADIENT_ACCUMULATION,
    target_modules: List[str] = None,
    train_log_path: Optional[Path] = None,
) -> None:
    """Load base model, apply LoRA, train on dataset.

    Supports per-experiment hyperparameter overrides (autoresearch pattern).
    """
    if target_modules is None:
        target_modules = TARGET_MODULES

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise RuntimeError(
            "Unsloth not installed. Install with: pip install unsloth"
        )

    log.info("Loading base model: %s", BASE_MODEL_ID)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    log.info("Applying LoRA (r=%d, alpha=%d, modules=%s)", lora_r, lora_alpha, target_modules)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(train_path), split="train")
    eval_dataset = None
    if eval_path.exists():
        eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")

    # Format examples for training
    def _format_example(example):
        messages = example.get("messages", [])
        text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        return {"text": text}

    dataset = dataset.map(_format_example)
    if eval_dataset:
        eval_dataset = eval_dataset.map(_format_example)

    # Training arguments
    from transformers import TrainingArguments

    training_args_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "seed": 42,
        "report_to": "none",
    }

    # Auto-detect precision
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            training_args_kwargs["bf16"] = True
        else:
            training_args_kwargs["fp16"] = True

    if eval_dataset:
        training_args_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    # Create trainer with fast-fail callback
    from trl import SFTTrainer
    from transformers import TrainerCallback

    class FastFailCallback(TrainerCallback):
        """Abort training if loss explodes + log to file (autoresearch patterns)."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            # Log to file (autoresearch log capture pattern)
            if train_log_path:
                with open(train_log_path, "a") as f:
                    f.write(json.dumps({
                        "step": state.global_step,
                        "epoch": state.epoch,
                        **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
                    }) + "\n")
            # Fast-fail check (autoresearch pattern: abort if loss > threshold)
            if "loss" in logs:
                loss = logs["loss"]
                if loss > LOSS_EXPLOSION_THRESHOLD and state.global_step > 20:
                    log.error("FAST-FAIL: loss=%.2f at step %d", loss, state.global_step)
                    control.should_training_stop = True

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        dataset_text_field="text",
        callbacks=[FastFailCallback()],
    )

    log.info("Starting training (%d examples, %d epochs, lr=%.2e)...",
             len(dataset), num_epochs, learning_rate)
    train_result = trainer.train()

    # Check if training was aborted by fast-fail
    final_loss = train_result.metrics.get("train_loss", 0.0)
    if final_loss > LOSS_EXPLOSION_THRESHOLD:
        raise _TrainingDiverged(f"Training loss {final_loss:.2f} exceeds threshold {LOSS_EXPLOSION_THRESHOLD}")

    # Log training summary
    training_summary = {
        "train_loss": final_loss,
        "eval_loss": train_result.metrics.get("eval_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_steps": train_result.metrics.get("total_flos"),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)

    # Save model
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving merged model to %s", merged_dir)

    try:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except AttributeError:
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    log.info("Training complete (loss=%.4f)", final_loss)


def _export_gguf(output_dir: Path) -> Path:
    """Export the merged model to GGUF f16 format via convert_hf_to_gguf.py.

    Primary: convert_hf_to_gguf.py (reliable, worked on this system).
    Fallback 1: Unsloth's save_pretrained_gguf.
    Fallback 2: Return merged_dir for Ollama native handling.
    """
    merged_dir = output_dir / "merged"
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / f"{MODEL_NAME}.f16.gguf"

    if gguf_path.exists():
        log.info("GGUF already exists: %s", gguf_path)
        return gguf_path

    # Primary: convert_hf_to_gguf.py from gguf package
    try:
        import gguf as _gguf
        gguf_pkg_dir = Path(_gguf.__file__).parent
        convert_script = gguf_pkg_dir / "scripts" / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            # Try alternative location
            for parent in gguf_pkg_dir.parents:
                candidate = parent / "scripts" / "convert_hf_to_gguf.py"
                if candidate.exists():
                    convert_script = candidate
                    break

        if convert_script.exists():
            log.info("Exporting GGUF via convert_hf_to_gguf.py...")
            result = subprocess.run(
                [sys.executable, str(convert_script),
                 str(merged_dir), "--outfile", str(gguf_path), "--outtype", "f16"],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode == 0 and gguf_path.exists():
                log.info("GGUF exported: %s (%.1f GB)",
                         gguf_path, gguf_path.stat().st_size / 1e9)
                return gguf_path
            else:
                log.warning("convert_hf_to_gguf.py failed: %s", result.stderr[-500:] if result.stderr else "unknown")
        else:
            log.debug("convert_hf_to_gguf.py not found at %s", convert_script)
    except Exception as exc:
        log.warning("GGUF conversion via convert_hf_to_gguf failed: %s", exc)

    # Fallback 1: Unsloth GGUF export
    try:
        from unsloth import FastLanguageModel
        log.info("Exporting GGUF via Unsloth (fallback)...")
        env = os.environ.copy()
        env["DEBIAN_FRONTEND"] = "noninteractive"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(merged_dir),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,
        )
        model.save_pretrained_gguf(
            str(gguf_dir), tokenizer, quantization_method="f16",
        )
        for f in gguf_dir.iterdir():
            if f.suffix == ".gguf":
                if f != gguf_path:
                    f.rename(gguf_path)
                return gguf_path
    except Exception as exc:
        log.warning("Unsloth GGUF export failed: %s", exc)

    # Fallback 2: Return merged dir for Ollama native conversion
    log.info("Using merged safetensors directory for Ollama (native conversion)")
    return merged_dir


def _register_ollama(model_path: Path, output_dir: Path) -> None:
    """Register the fine-tuned model in Ollama via blob upload API.

    Two-step process:
    1. Upload the GGUF blob via POST /api/blobs/{digest}
    2. Create the model via POST /api/create with files parameter
    """
    import hashlib
    import requests

    model_tag = f"{MODEL_NAME}:latest"

    # Find the GGUF file
    gguf_file = None
    if model_path.is_file() and model_path.suffix == ".gguf":
        gguf_file = model_path
    elif model_path.is_dir():
        for f in model_path.rglob("*.gguf"):
            gguf_file = f
            break
        if gguf_file is None:
            gguf_file = _export_gguf(output_dir)
            if gguf_file.is_dir():
                log.warning("No GGUF available, skipping Ollama registration")
                return

    if gguf_file is None or not gguf_file.exists():
        log.warning("No valid model artifact at %s", model_path)
        return

    # Step 1: Calculate SHA256 and upload blob
    log.info("Calculating SHA256 of %s (%.1f GB)...",
             gguf_file, gguf_file.stat().st_size / 1e9)
    sha = hashlib.sha256()
    with open(gguf_file, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            sha.update(chunk)
    digest = f"sha256:{sha.hexdigest()}"

    # Check if blob already exists
    try:
        head_resp = requests.head(
            f"http://localhost:11434/api/blobs/{digest}", timeout=10
        )
        if head_resp.status_code == 200:
            log.info("Blob already exists in Ollama, skipping upload")
        else:
            raise requests.RequestException("not found")
    except requests.RequestException:
        # Upload blob
        log.info("Uploading GGUF to Ollama blob store...")
        try:
            with open(gguf_file, "rb") as f:
                resp = requests.post(
                    f"http://localhost:11434/api/blobs/{digest}",
                    data=f, timeout=1800,
                )
            if resp.status_code not in (200, 201):
                log.warning("Blob upload failed: HTTP %d", resp.status_code)
                return
        except Exception as exc:
            log.warning("Blob upload failed: %s", exc)
            return

    # Step 2: Create model from blob with training system prompt
    from src.finetune.training_data_generator import SYSTEM_PROMPT

    try:
        resp = requests.post("http://localhost:11434/api/create", json={
            "model": model_tag,
            "files": {"model.gguf": digest},
            "system": SYSTEM_PROMPT,
            "parameters": {"temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.0},
        }, stream=True, timeout=300)

        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if "error" in data:
                    log.warning("ollama create error: %s", data["error"])
                    return
                log.info("  %s", status)

        log.info("Model registered in Ollama as %s", model_tag)
    except Exception as exc:
        log.warning("Ollama registration failed: %s", exc)
        return

    # Step 3: Create Q4_K_M quantized variant for production
    q4_tag = f"{MODEL_NAME}:q4_k_m"
    try:
        log.info("Creating quantized Q4_K_M variant...")
        resp = requests.post("http://localhost:11434/api/create", json={
            "model": q4_tag,
            "from": model_tag,
            "quantize": "q4_K_M",
        }, stream=True, timeout=600)

        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if "error" in data:
                    log.warning("Q4_K_M quantization error: %s", data["error"])
                    break
                log.info("  %s", status)

        log.info("Quantized model registered as %s", q4_tag)
    except Exception as exc:
        log.warning("Q4_K_M quantization failed: %s — F16 model still usable", exc)


def _evaluate_with_retry(eval_path: Path, retries: int = 2) -> Dict[str, Any]:
    """Run evaluation with retries (model may need warmup)."""
    from src.finetune.evaluate_model import evaluate_model

    # Prefer quantized model for evaluation (production config)
    q4_tag = f"{MODEL_NAME}:q4_k_m"
    f16_tag = f"{MODEL_NAME}:latest"
    try:
        import ollama as _ollama
        models = _ollama.list()
        model_names = [m.model for m in getattr(models, "models", [])]
        eval_tag = q4_tag if any(q4_tag.split(":")[0] in n and "q4" in n for n in model_names) else f16_tag
    except Exception:
        eval_tag = f16_tag

    for attempt in range(retries + 1):
        try:
            return evaluate_model(
                model_name=eval_tag,
                eval_set_path=eval_path,
            )
        except Exception as exc:
            if attempt < retries:
                log.warning("Eval attempt %d failed: %s — retrying in 10s", attempt + 1, exc)
                time.sleep(10)
            else:
                raise


def _augment_dataset(
    eval_result: Dict[str, Any],
    train_path: Path,
    dataset_dir: Path,
) -> None:
    """Generate additional targeted training examples for weak categories."""
    from src.finetune.evaluate_model import identify_weak_categories

    weak = identify_weak_categories(eval_result)
    if not weak:
        log.info("No specific weak categories identified")
        return

    log.info("Weak categories: %s — generating augmentation data", weak)

    from src.finetune.training_data_generator import (
        _DOMAIN_TASK_SPECS,
        _SEED_PARAPHRASES,
        _chat_example,
        _expand_paraphrases_offline,
    )

    augmented: List[Dict[str, Any]] = []
    per_category = max(1, 200 // max(len(weak), 1))

    for category in weak:
        generated = 0
        for task_key, ts in _DOMAIN_TASK_SPECS.items():
            if generated >= per_category:
                break
            domain = ts.get("domain", "")
            intent = ts.get("intent", "")
            if category in (domain, intent, task_key):
                seeds = _SEED_PARAPHRASES.get(task_key, [])
                ts_with_conf = {**ts, "confidence": 0.90}
                for seed in seeds:
                    if generated >= per_category:
                        break
                    expanded = _expand_paraphrases_offline(seed, count=10)
                    for q in expanded:
                        augmented.append(_chat_example(q, ts_with_conf))
                        generated += 1
                        if generated >= per_category:
                            break

    if augmented:
        with open(train_path, "a", encoding="utf-8") as f:
            for ex in augmented:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("Added %d augmentation examples to training set", len(augmented))


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DocWain-Agent-v2 (autonomous experiment loop)"
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--target-score", type=int, default=DEFAULT_TARGET_SCORE)
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    result = run_finetune_pipeline(
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        dataset_dir=Path(args.dataset_dir),
    )

    print("\n" + "═" * 60)
    if result.success:
        print(f"SUCCESS — {result.model_name} ready")
        print(f"  Final score: {result.final_score:.1f}%")
        print(f"  Best score:  {result.best_score:.1f}%")
        print(f"  Iterations:  {result.iterations_run}")
        print(f"  Artifacts:   {result.artifact_dir}")
    else:
        print(f"INCOMPLETE — {result.error}")
        print(f"  Best score:  {result.best_score:.1f}%")
        print(f"  Last score:  {result.final_score:.1f}%")
        print(f"  Iterations:  {result.iterations_run}")
    print(f"  Results:     {RESULTS_TSV}")
    print("═" * 60)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
