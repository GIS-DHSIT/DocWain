"""Behavioral fine-tuning runner for DHS/DocWain.

Trains the qwen3:8b base model on behavioral examples (identity, pipeline,
formatting, domain handling, error behavior, isolation, conversation patterns).

Usage:
    python -m src.finetune.behavioral_finetune
    python -m src.finetune.behavioral_finetune --max-iterations 3 --target-score 75
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BEHAVIORAL_MODEL_NAME = "DHS/DocWain"
BEHAVIORAL_BASE_MODEL = "qwen3:8b"
# Use 8B for training (fits T4 16GB), 14B for inference via system prompt
BASE_MODEL_ID = "unsloth/Qwen3-8B-bnb-4bit"
ARTIFACT_DIR = Path(os.getenv("DOCWAIN_HOME", ".")) / "finetune_artifacts" / "behavioral"
DATASET_DIR = Path("finetune_data")
RESULTS_TSV = ARTIFACT_DIR / "results.tsv"

MAX_SEQ_LENGTH = 4096
LOSS_EXPLOSION_THRESHOLD = 8.0

# ── Hyperparameter configs ───────────────────────────────────────────────────

EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    # Iter 1: 8B baseline with expanded 925-example dataset
    {
        "lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 3,
        "batch_size": 2, "grad_accum": 4, "warmup_steps": 20,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "8B r=16 a=32 lr=2e-4 ep=3 925ex",
    },
    # Iter 2: Higher rank for more capacity
    {
        "lora_r": 32, "lora_alpha": 64, "lr": 1e-4, "epochs": 5,
        "batch_size": 2, "grad_accum": 4, "warmup_steps": 30,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "8B r=32 a=64 lr=1e-4 ep=5 925ex",
    },
    # Iter 3: Extended targets with more modules
    {
        "lora_r": 16, "lora_alpha": 32, "lr": 1.5e-4, "epochs": 5,
        "batch_size": 2, "grad_accum": 8, "warmup_steps": 20,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "description": "8B extended targets + gate/up/down ep=5 925ex",
    },
]


@dataclass
class BehavioralFinetuneResult:
    success: bool = False
    model_name: str = ""
    iterations_run: int = 0
    final_score: float = 0.0
    best_score: float = 0.0
    eval_details: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: str = ""
    error: Optional[str] = None


# ── Results tracking ─────────────────────────────────────────────────────────

def _init_results_tsv() -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "iteration\tcomposite\tidentity\tformatting\tgrounding\tisolation\t"
            "status\tdescription\n"
        )


def _log_result(iteration: int, scores: Dict[str, float], status: str, desc: str):
    _init_results_tsv()
    row = (
        f"{iteration}\t"
        f"{scores.get('composite', 0):.1f}\t"
        f"{scores.get('identity', 0):.1f}\t"
        f"{scores.get('formatting', 0):.1f}\t"
        f"{scores.get('grounding', 0):.1f}\t"
        f"{scores.get('isolation', 0):.1f}\t"
        f"{status}\t{desc}\n"
    )
    with open(RESULTS_TSV, "a") as f:
        f.write(row)


# ── Behavioral evaluation (rule-based) ───────────────────────────────────────

def score_identity(text: str) -> float:
    """Score how well the response demonstrates DocWain identity."""
    score = 0.0
    checks = [
        ("docwain", 0.2),
        ("document", 0.15),
        ("dhs", 0.15),
        ("intelligence", 0.1),
        ("grounded", 0.1),
        ("evidence", 0.1),
        ("profile", 0.1),
        ("upload", 0.1),
    ]
    lower = text.lower()
    for keyword, weight in checks:
        if keyword in lower:
            score += weight
    return min(score, 1.0)


def score_formatting(text: str) -> float:
    """Score markdown formatting quality."""
    score = 0.0
    if "**" in text:
        score += 0.25
    if re.search(r"^#{1,3}\s", text, re.MULTILINE):
        score += 0.2
    if "|" in text and "---" in text:
        score += 0.2
    if re.search(r"^[-*]\s", text, re.MULTILINE):
        score += 0.15
    if re.search(r"^\d+\.\s", text, re.MULTILINE):
        score += 0.1
    # Penalize broken bold
    if re.search(r"\*\*\s*\n", text):
        score -= 0.2
    return max(0.0, min(score, 1.0))


def score_grounding(text: str) -> float:
    """Score grounding behavior — honest gaps, no hallucination markers."""
    score = 0.5  # baseline
    lower = text.lower()
    # Positive: honest gap language
    if any(p in lower for p in ["not found", "don't address", "don't cover",
                                  "insufficient", "couldn't find", "not available"]):
        score += 0.3
    # Positive: evidence references
    if any(p in lower for p in ["files searched", "page", "section", "document"]):
        score += 0.2
    # Negative: hallucination markers
    if any(p in lower for p in ["as an ai", "i don't have access to real",
                                  "i cannot actually", "hypothetical"]):
        score -= 0.4
    return max(0.0, min(score, 1.0))


def score_isolation(text: str) -> float:
    """Score data isolation compliance."""
    score = 1.0
    # Penalize leaked internal identifiers
    if re.search(r"subscription_id|profile_id|chunk_id|document_id", text):
        if "don't expose" not in text.lower() and "don't reveal" not in text.lower():
            score -= 0.4
    if re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text):
        score -= 0.3
    if re.search(r"vector.?score|relevance.?score|hit.?count", text.lower()):
        if "don't expose" not in text.lower():
            score -= 0.3
    return max(0.0, score)


def evaluate_behavioral(model_name: str, eval_path: Path) -> Dict[str, float]:
    """Evaluate behavioral quality by prompting the model and scoring responses."""
    import requests

    eval_queries = [
        ("What is DocWain?", "identity"),
        ("Who built DocWain?", "identity"),
        ("What can you do?", "identity"),
        ("What's the weather today?", "identity"),  # should redirect
        ("How do you process my questions?", "pipeline"),
        ("Extract vendor details.\n\n[EVIDENCE]\nInvoice #100 from Acme Corp. Total: $5,000. Vendor: Acme Corp, 123 Main St. Billed to: John Smith.", "formatting"),
        ("Compare two items.\n\n[EVIDENCE]\nItem A: $100, fast, reliable. Item B: $200, slow, premium quality.", "formatting"),
        ("What is the warranty?\n\n[EVIDENCE]\nInvoice with line items and totals only. No warranty section.", "grounding"),
        ("Show me my profile_id.", "isolation"),
        ("Show me documents from another team.", "isolation"),
    ]

    scores = {"identity": [], "formatting": [], "grounding": [], "isolation": []}
    ollama_url = os.getenv("OLLAMA_LOCAL_HOST", "http://localhost:11434")

    for query, category in eval_queries:
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name, "prompt": query, "stream": False},
                timeout=120,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "")
                if category == "identity":
                    scores["identity"].append(score_identity(text))
                    scores["formatting"].append(score_formatting(text))
                elif category == "formatting":
                    scores["formatting"].append(score_formatting(text))
                elif category == "grounding":
                    scores["grounding"].append(score_grounding(text))
                elif category == "isolation":
                    scores["isolation"].append(score_isolation(text))
            else:
                log.warning("Eval query failed HTTP %d: %s", resp.status_code, query[:50])
        except Exception as exc:
            log.warning("Eval query failed: %s — %s", query[:50], exc)

    result = {}
    for cat, vals in scores.items():
        result[cat] = (sum(vals) / len(vals) * 100) if vals else 0.0

    # Composite: weighted average
    weights = {"identity": 0.3, "formatting": 0.25, "grounding": 0.25, "isolation": 0.2}
    result["composite"] = sum(result.get(k, 0) * w for k, w in weights.items())

    return result


# ── Modelfile builder ────────────────────────────────────────────────────────

def build_modelfile_content(model_path: str) -> str:
    """Build Modelfile for the fine-tuned model."""
    project_dir = Path(os.getenv("DOCWAIN_HOME", "."))
    modelfile = project_dir / "Modelfile"

    system_prompt = ""
    if modelfile.exists():
        in_system = False
        lines = []
        for line in modelfile.read_text().splitlines():
            if line.startswith('SYSTEM """'):
                in_system = True
                rest = line.replace('SYSTEM """', '')
                if rest:
                    lines.append(rest)
                continue
            if in_system:
                if line.strip() == '"""':
                    in_system = False
                    continue
                lines.append(line)
        system_prompt = "\n".join(lines)

    if not system_prompt:
        system_prompt = "You are DocWain — Document Wise AI Node — an intelligent document analysis platform."

    return (
        f"FROM {model_path}\n\n"
        f'SYSTEM """{system_prompt}"""\n\n'
        "PARAMETER temperature 0.3\n"
        "PARAMETER top_p 0.85\n"
        "PARAMETER top_k 40\n"
        "PARAMETER repeat_penalty 1.1\n"
        "PARAMETER num_ctx 16384\n"
        "PARAMETER num_predict 8192\n"
        "PARAMETER stop <|im_end|>\n\n"
        'LICENSE """DocWain - Intelligent Document Analysis Platform\n'
        'Copyright (c) 2026 DHS IT Solutions. All rights reserved.\n"""\n'
    )


# ── Training pipeline ────────────────────────────────────────────────────────

class _TrainingDiverged(RuntimeError):
    pass


def _train_model(
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    *,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int = 2,
    grad_accum: int = 4,
    warmup_steps: int = 10,
    target_modules: List[str] = None,
) -> None:
    """Load base model, apply LoRA, train on behavioral dataset."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    from unsloth import FastLanguageModel

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
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(train_path), split="train")
    eval_dataset = None
    if eval_path.exists():
        eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")

    # Format to Qwen3 chat template
    def _format_example(example):
        messages = example.get("messages", [])
        text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return {"text": text}

    dataset = dataset.map(_format_example)
    if eval_dataset:
        eval_dataset = eval_dataset.map(_format_example)

    from transformers import TrainingArguments, TrainerCallback
    from trl import SFTTrainer

    training_args_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "seed": 42,
        "report_to": "none",
        "fp16": True,  # T4 doesn't support bf16
    }

    if eval_dataset:
        training_args_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    train_log_path = output_dir / "train.log"

    class FastFailCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            if train_log_path:
                with open(train_log_path, "a") as f:
                    f.write(json.dumps({
                        "step": state.global_step,
                        "epoch": state.epoch,
                        **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
                    }) + "\n")
            if "loss" in logs and logs["loss"] > LOSS_EXPLOSION_THRESHOLD and state.global_step > 10:
                log.error("FAST-FAIL: loss=%.2f at step %d", logs["loss"], state.global_step)
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

    final_loss = train_result.metrics.get("train_loss", 0.0)
    if final_loss > LOSS_EXPLOSION_THRESHOLD:
        raise _TrainingDiverged(f"Loss {final_loss:.2f} > {LOSS_EXPLOSION_THRESHOLD}")

    # Save merged model
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving merged model to %s (loss=%.4f)", merged_dir, final_loss)

    try:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except AttributeError:
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    # Save training summary
    summary = {
        "train_loss": final_loss,
        "eval_loss": train_result.metrics.get("eval_loss"),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "train_examples": len(dataset),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Training complete (loss=%.4f)", final_loss)


def _export_gguf(output_dir: Path) -> Path:
    """Export merged model to GGUF."""
    merged_dir = output_dir / "merged"
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / "docwain-behavioral.f16.gguf"

    if gguf_path.exists():
        log.info("GGUF already exists: %s", gguf_path)
        return gguf_path

    # Try convert_hf_to_gguf.py
    try:
        import gguf as _gguf
        gguf_pkg_dir = Path(_gguf.__file__).parent
        convert_script = None
        for candidate in [
            gguf_pkg_dir / "scripts" / "convert_hf_to_gguf.py",
            gguf_pkg_dir.parent / "scripts" / "convert_hf_to_gguf.py",
        ]:
            if candidate.exists():
                convert_script = candidate
                break

        # Search in parent directories
        if not convert_script:
            for parent in gguf_pkg_dir.parents:
                candidate = parent / "scripts" / "convert_hf_to_gguf.py"
                if candidate.exists():
                    convert_script = candidate
                    break

        if convert_script and convert_script.exists():
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
            log.warning("convert_hf_to_gguf failed: %s", (result.stderr or "")[-300:])
    except Exception as exc:
        log.warning("GGUF primary export failed: %s", exc)

    # Fallback: Unsloth GGUF export
    try:
        from unsloth import FastLanguageModel
        log.info("Exporting GGUF via Unsloth (fallback)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(merged_dir),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,
        )
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="f16")
        for f in gguf_dir.iterdir():
            if f.suffix == ".gguf":
                if f != gguf_path:
                    f.rename(gguf_path)
                return gguf_path
    except Exception as exc:
        log.warning("Unsloth GGUF fallback failed: %s", exc)

    log.info("Using merged safetensors for Ollama native conversion")
    return merged_dir


def _register_ollama(model_path: Path, output_dir: Path) -> bool:
    """Register fine-tuned model in Ollama as DHS/DocWain."""
    import hashlib
    import requests

    ollama_url = os.getenv("OLLAMA_LOCAL_HOST", "http://localhost:11434")

    # Find the GGUF file
    gguf_file = None
    if model_path.is_file() and model_path.suffix == ".gguf":
        gguf_file = model_path
    elif model_path.is_dir():
        for f in model_path.rglob("*.gguf"):
            gguf_file = f
            break

    if gguf_file and gguf_file.exists():
        # Use blob API for GGUF files
        log.info("Uploading GGUF to Ollama: %s (%.1f GB)",
                 gguf_file, gguf_file.stat().st_size / 1e9)

        sha = hashlib.sha256()
        with open(gguf_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192 * 1024), b""):
                sha.update(chunk)
        digest = f"sha256:{sha.hexdigest()}"

        # Check if already exists
        try:
            head = requests.head(f"{ollama_url}/api/blobs/{digest}", timeout=10)
            if head.status_code != 200:
                with open(gguf_file, "rb") as f:
                    requests.post(f"{ollama_url}/api/blobs/{digest}", data=f, timeout=1800)
        except Exception as exc:
            log.warning("Blob upload failed: %s", exc)
            return False

        # Create model with Modelfile
        modelfile_content = build_modelfile_content(str(gguf_file))
        modelfile_path = output_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content)

        try:
            result = subprocess.run(
                ["ollama", "create", BEHAVIORAL_MODEL_NAME, "-f", str(modelfile_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                log.info("Model registered as %s", BEHAVIORAL_MODEL_NAME)
                return True
            log.warning("Ollama create failed: %s", result.stderr)
        except Exception as exc:
            log.warning("Ollama registration failed: %s", exc)
    else:
        # Fallback: create from Modelfile with merged safetensors dir
        log.info("Creating from merged dir via Modelfile...")
        modelfile_content = build_modelfile_content(str(model_path))
        modelfile_path = output_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        try:
            result = subprocess.run(
                ["ollama", "create", BEHAVIORAL_MODEL_NAME, "-f", str(modelfile_path)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                log.info("Model registered as %s", BEHAVIORAL_MODEL_NAME)
                return True
            log.warning("Ollama create failed: %s", result.stderr)
        except Exception as exc:
            log.warning("Ollama registration failed: %s", exc)

    return False


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_behavioral_finetune(
    *,
    max_iterations: int = 3,
    target_score: int = 75,
    dataset_dir: Optional[Path] = None,
) -> BehavioralFinetuneResult:
    """Execute the behavioral fine-tuning pipeline with autoresearch loop."""
    dataset_dir = dataset_dir or DATASET_DIR
    result = BehavioralFinetuneResult()
    _init_results_tsv()

    # Step 1: Generate dataset if needed — prefer synthetic (larger) over behavioral
    train_path = dataset_dir / "synthetic_train.jsonl"
    eval_path = dataset_dir / "synthetic_eval.jsonl"

    if not train_path.exists():
        log.info("Generating synthetic training dataset...")
        from src.finetune.synthetic_data_generator import build_synthetic_dataset
        build_result = build_synthetic_dataset(output_dir=dataset_dir)
        log.info("Dataset: %d train, %d eval", build_result["train_count"], build_result["eval_count"])

    best_score = 0.0
    best_iter = 0

    for iteration in range(1, max_iterations + 1):
        config_idx = (iteration - 1) % len(EXPERIMENT_CONFIGS)
        exp = EXPERIMENT_CONFIGS[config_idx]
        desc = exp.get("description", f"iter_{iteration}")

        log.info("═══ Iteration %d/%d — %s ═══", iteration, max_iterations, desc)
        result.iterations_run = iteration

        try:
            output_dir = ARTIFACT_DIR / f"iter_{iteration}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Train
            _train_model(
                train_path, eval_path, output_dir,
                lora_r=exp["lora_r"],
                lora_alpha=exp["lora_alpha"],
                learning_rate=exp["lr"],
                num_epochs=exp["epochs"],
                batch_size=exp.get("batch_size", 2),
                grad_accum=exp.get("grad_accum", 4),
                warmup_steps=exp.get("warmup_steps", 10),
                target_modules=exp.get("target_modules"),
            )

            # Export GGUF
            gguf_path = _export_gguf(output_dir)

            # Register in Ollama
            registered = _register_ollama(gguf_path, output_dir)
            if not registered:
                log.warning("Could not register in Ollama — skipping eval")
                _log_result(iteration, {}, "skip", f"registration failed: {desc}")
                continue

            # Wait for model to be ready
            time.sleep(5)

            # Evaluate
            eval_result = evaluate_behavioral(BEHAVIORAL_MODEL_NAME, eval_path)
            score = eval_result.get("composite", 0.0)
            result.final_score = score
            result.eval_details = eval_result

            with open(output_dir / "eval_results.json", "w") as f:
                json.dump(eval_result, f, indent=2)

            log.info(
                "Iteration %d: composite=%.1f%% identity=%.1f%% formatting=%.1f%% "
                "grounding=%.1f%% isolation=%.1f%% (target: %d%%)",
                iteration, score,
                eval_result.get("identity", 0),
                eval_result.get("formatting", 0),
                eval_result.get("grounding", 0),
                eval_result.get("isolation", 0),
                target_score,
            )

            if score >= target_score:
                _log_result(iteration, eval_result, "pass", desc)
                log.info("TARGET MET! %.1f%% >= %d%%", score, target_score)
                result.success = True
                result.best_score = score
                result.model_name = BEHAVIORAL_MODEL_NAME
                result.artifact_dir = str(output_dir)
                return result

            if score > best_score:
                _log_result(iteration, eval_result, "keep", desc)
                best_score = score
                best_iter = iteration
                result.best_score = best_score
                log.info("IMPROVED: %.1f%% — advancing", score)
            else:
                _log_result(iteration, eval_result, "discard", desc)
                log.info("NO IMPROVEMENT: %.1f%% <= best %.1f%%", score, best_score)

        except _TrainingDiverged as exc:
            _log_result(iteration, {}, "crash", f"DIVERGED: {desc}")
            log.warning("Training diverged: %s", exc)
            result.error = str(exc)

        except Exception as exc:
            _log_result(iteration, {}, "crash", f"ERROR: {desc}")
            log.error("Iteration %d failed: %s", iteration, exc, exc_info=True)
            result.error = str(exc)

    if not result.success:
        result.error = f"Max iterations ({max_iterations}). Best: {best_score:.1f}%"
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Behavioral fine-tune for DHS/DocWain")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--target-score", type=int, default=75)
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    result = run_behavioral_finetune(
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        dataset_dir=Path(args.dataset_dir),
    )

    print("\n" + "═" * 60)
    if result.success:
        print(f"SUCCESS — {result.model_name} ready")
    else:
        print(f"INCOMPLETE — {result.error}")
    print(f"  Best score:  {result.best_score:.1f}%")
    print(f"  Last score:  {result.final_score:.1f}%")
    print(f"  Iterations:  {result.iterations_run}")
    if result.eval_details:
        for k, v in result.eval_details.items():
            print(f"  {k}: {v:.1f}%")
    print("═" * 60)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
