"""Recursive visualization fine-tuning loop for DocWain.

Iteratively: generates fresh synthetic data -> trains SFT -> trains DPO ->
evaluates -> retrains if below threshold.  Orchestrates the entire
visualization fine-tuning pipeline.

Usage::

    python -m src.finetune.viz_finetune_loop
    python -m src.finetune.viz_finetune_loop --max-iterations 3 --threshold 85
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

ARTIFACT_DIR = Path("finetune_artifacts") / "viz_loop"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Result of a single fine-tuning iteration."""

    iteration: int
    composite_score: float
    verdict: str
    chart_avg: float
    text_avg: float
    sft_count: int
    dpo_count: int
    duration_seconds: float


@dataclass
class VizFinetuneState:
    """Tracks state across fine-tuning iterations."""

    iterations: List[IterationResult] = field(default_factory=list)
    best_score: float = 0.0
    best_iteration: int = 0

    def record_iteration(
        self,
        iteration: int,
        score: float,
        verdict: str,
        **kwargs: Any,
    ) -> IterationResult:
        """Record the result of an iteration and update best tracking."""
        result = IterationResult(
            iteration=iteration,
            composite_score=score,
            verdict=verdict,
            chart_avg=kwargs.get("chart_avg", 0.0),
            text_avg=kwargs.get("text_avg", 0.0),
            sft_count=kwargs.get("sft_count", 0),
            dpo_count=kwargs.get("dpo_count", 0),
            duration_seconds=kwargs.get("duration_seconds", 0.0),
        )
        self.iterations.append(result)
        if score > self.best_score:
            self.best_score = score
            self.best_iteration = iteration
        return result

    def has_passed(self, threshold: float) -> bool:
        """Check if the best score meets or exceeds the threshold."""
        return self.best_score >= threshold


@dataclass
class VizFinetuneConfig:
    """Configuration for the recursive fine-tuning loop."""

    max_iterations: int = 5
    pass_threshold: float = 80.0
    marginal_threshold: float = 60.0
    data_refresh_each_iteration: bool = True
    base_model: str = "unsloth/Qwen3-8B-bnb-4bit"
    model_name: str = "DHS/DocWain"
    sft_epochs: int = 3
    dpo_epochs: int = 2
    dpo_beta: float = 0.1
    learning_rate: float = 2e-4
    dpo_learning_rate: float = 5e-5
    lora_r: int = 16


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_iteration_dataset(
    iteration: int,
    seed: int = 42,
    output_dir: str = "finetune_data",
    weak_areas: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate fresh SFT + DPO datasets for a single iteration.

    Args:
        iteration: Current iteration number (1-based).
        seed: Base random seed.
        output_dir: Directory to write dataset files.
        weak_areas: Optional list of chart types that need more examples.

    Returns:
        Dict with sft_path, dpo_path, sft_count, dpo_count, iteration, seed.
    """
    from src.finetune.viz_training_data import write_viz_sft_dataset
    from src.finetune.dpo_data_generator import build_dpo_dataset

    iter_seed = seed + iteration * 1000
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info(
        "Generating iteration %d dataset (seed=%d, weak_areas=%s)",
        iteration, iter_seed, weak_areas,
    )

    # SFT data
    sft_result = write_viz_sft_dataset(output_dir=out, seed=iter_seed)
    sft_path = sft_result["train_path"]
    sft_count = sft_result["train_count"]

    # DPO data
    dpo_result = build_dpo_dataset(output_dir=out)
    dpo_path = str(out / "dpo_train.jsonl")
    dpo_count = dpo_result["train_count"]

    log.info(
        "Iteration %d dataset: %d SFT examples, %d DPO pairs",
        iteration, sft_count, dpo_count,
    )

    return {
        "sft_path": sft_path,
        "dpo_path": dpo_path,
        "sft_count": sft_count,
        "dpo_count": dpo_count,
        "iteration": iteration,
        "seed": iter_seed,
    }


# ---------------------------------------------------------------------------
# Training helpers (lazy imports for GPU-dependent libraries)
# ---------------------------------------------------------------------------

def _run_sft_training(config: VizFinetuneConfig, dataset_path: str) -> Dict[str, Any]:
    """Run SFT training using Unsloth FastLanguageModel.

    Args:
        config: Fine-tuning configuration.
        dataset_path: Path to the SFT training JSONL file.

    Returns:
        Dict with training summary (loss, epochs, etc.).
    """
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    log.info("Loading base model for SFT: %s", config.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_r * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Format as Qwen3 chat template
    def _format_chat(example: Dict[str, Any]) -> Dict[str, str]:
        messages = example["messages"]
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return {"text": text}

    dataset = dataset.map(_format_chat)

    output_dir = ARTIFACT_DIR / "sft_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=config.learning_rate,
            num_train_epochs=config.sft_epochs,
            warmup_steps=5,
            logging_steps=1,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
            seed=42,
            max_seq_length=4096,
        )
    except (TypeError, ImportError):
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=config.learning_rate,
            num_train_epochs=config.sft_epochs,
            warmup_steps=5,
            logging_steps=1,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
            seed=42,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    log.info("Starting SFT training (%d examples, %d epochs)...", len(dataset), config.sft_epochs)
    train_result = trainer.train()
    final_loss = train_result.metrics.get("train_loss", 0.0)

    # Save merged model + GGUF for Ollama
    merged_dir = ARTIFACT_DIR / "sft_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except AttributeError:
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    # Export to GGUF for Ollama consumption
    gguf_dir = ARTIFACT_DIR / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        log.info("GGUF exported to %s", gguf_dir)
    except Exception as exc:
        log.warning("GGUF export failed (will use Modelfile fallback): %s", exc)

    log.info("SFT training complete (loss=%.4f)", final_loss)
    return {"train_loss": final_loss, "epochs": config.sft_epochs, "examples": len(dataset)}


def _run_dpo_training(config: VizFinetuneConfig, dpo_path: str) -> Dict[str, Any]:
    """Run DPO training on top of the SFT-trained model.

    Args:
        config: Fine-tuning configuration.
        dpo_path: Path to the DPO training JSONL file.

    Returns:
        Dict with DPO training summary.
    """
    from src.finetune.dpo_trainer import run_dpo_training

    # Use SFT merged model if available, otherwise fall back to base
    sft_merged = ARTIFACT_DIR / "sft_merged"
    base = str(sft_merged) if sft_merged.exists() else config.base_model

    log.info("Starting DPO training on %s (epochs=%d, beta=%.2f)...", base, config.dpo_epochs, config.dpo_beta)
    result = run_dpo_training(
        base_model=base,
        epochs=config.dpo_epochs,
        learning_rate=config.dpo_learning_rate,
        lora_r=config.lora_r,
        beta=config.dpo_beta,
    )
    log.info("DPO training complete (loss=%.4f)", result.get("train_loss", 0.0))
    return result


def _update_ollama_model(config: VizFinetuneConfig) -> bool:
    """Update the Ollama model from trained GGUF weights or Modelfile fallback.

    Looks for GGUF files from training. If found, creates a temporary Modelfile
    pointing to the GGUF. Otherwise falls back to the project Modelfile.

    Args:
        config: Fine-tuning configuration.

    Returns:
        True if the model was updated successfully.
    """
    import glob

    # Look for GGUF from training
    gguf_dir = ARTIFACT_DIR / "gguf"
    gguf_files = sorted(glob.glob(str(gguf_dir / "*.gguf"))) if gguf_dir.exists() else []

    if gguf_files:
        gguf_path = gguf_files[-1]  # latest GGUF
        log.info("Found trained GGUF: %s", gguf_path)

        # Read existing Modelfile for system prompt and params
        base_modelfile = Path("Modelfile")
        system_block = ""
        param_lines = []
        if base_modelfile.exists():
            content = base_modelfile.read_text()
            # Extract everything after FROM line
            lines = content.split("\n")
            for line in lines:
                if line.startswith("SYSTEM") or line.startswith("PARAMETER") or line.startswith("LICENSE"):
                    break
            # Get SYSTEM, PARAMETER, LICENSE blocks
            in_system = False
            for line in lines:
                if line.startswith("SYSTEM"):
                    in_system = True
                    system_block += line + "\n"
                elif in_system:
                    system_block += line + "\n"
                    if '"""' in line and not line.startswith("SYSTEM"):
                        in_system = False
                elif line.startswith("PARAMETER") or line.startswith("LICENSE"):
                    param_lines.append(line)

        # Create temporary Modelfile pointing to trained GGUF
        tmp_modelfile = ARTIFACT_DIR / "Modelfile.trained"
        with open(tmp_modelfile, "w") as f:
            f.write(f"FROM {gguf_path}\n\n")
            if system_block:
                f.write(system_block + "\n")
            for pline in param_lines:
                f.write(pline + "\n")

        modelfile_path = str(tmp_modelfile)
        log.info("Using trained GGUF Modelfile: %s", modelfile_path)
    else:
        modelfile_path = "Modelfile"
        if not Path(modelfile_path).exists():
            log.warning("Modelfile not found, skipping Ollama update")
            return False
        log.info("No GGUF found, falling back to base Modelfile")

    try:
        log.info("Updating Ollama model: %s", config.model_name)
        result = subprocess.run(
            ["ollama", "create", config.model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            log.info("Ollama model updated successfully")
            return True
        else:
            log.error("Ollama model update failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        log.warning("ollama CLI not found, skipping model update")
        return False
    except subprocess.TimeoutExpired:
        log.error("Ollama model update timed out")
        return False


def _identify_weak_areas(eval_result: Dict[str, Any]) -> List[str]:
    """Identify chart types that scored below 60%.

    Args:
        eval_result: Evaluation result from run_viz_evaluation.

    Returns:
        List of chart type names that need improvement.
    """
    weak = []
    per_example = eval_result.get("per_example", [])

    # Group scores by chart type
    type_scores: Dict[str, List[float]] = {}
    for ex in per_example:
        chart_type = ex.get("expected_chart_type")
        if chart_type and chart_type != "flow" and ex.get("expects_chart"):
            type_scores.setdefault(chart_type, []).append(ex["score"]["composite"])

    for chart_type, scores in type_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg < 60.0:
            weak.append(chart_type)
            log.info("Weak area: %s (avg=%.1f)", chart_type, avg)

    return weak


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_viz_finetune_loop(
    config: Optional[VizFinetuneConfig] = None,
) -> Dict[str, Any]:
    """Run the recursive visualization fine-tuning loop.

    For each iteration:
    1. Generate fresh synthetic data
    2. Run SFT training
    3. Run DPO training
    4. Update Ollama model
    5. Evaluate
    6. Record results, persist iteration JSON
    7. If passed: break. If failed: identify weak areas and continue.

    Args:
        config: Fine-tuning configuration. Uses defaults if None.

    Returns:
        Summary dict with total_iterations, best_score, best_iteration,
        passed, and iterations list.
    """
    if config is None:
        config = VizFinetuneConfig()

    from src.finetune.viz_evaluator import run_viz_evaluation

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    state = VizFinetuneState()
    weak_areas: Optional[List[str]] = None

    log.info(
        "Starting viz finetune loop (max_iterations=%d, threshold=%.1f)",
        config.max_iterations, config.pass_threshold,
    )

    for iteration in range(1, config.max_iterations + 1):
        iter_start = time.time()
        log.info("=== Iteration %d/%d ===", iteration, config.max_iterations)

        # Step 1: Generate data
        if config.data_refresh_each_iteration or iteration == 1:
            dataset = generate_iteration_dataset(
                iteration=iteration,
                seed=42,
                output_dir="finetune_data",
                weak_areas=weak_areas,
            )
        sft_path = dataset["sft_path"]
        dpo_path = dataset["dpo_path"]

        # Step 2: SFT training
        try:
            _run_sft_training(config, sft_path)
        except Exception as exc:
            log.error("SFT training failed in iteration %d: %s", iteration, exc)
            state.record_iteration(
                iteration=iteration,
                score=0.0,
                verdict="fail",
                sft_count=dataset["sft_count"],
                dpo_count=dataset["dpo_count"],
                duration_seconds=time.time() - iter_start,
            )
            continue

        # Step 3: DPO training
        try:
            _run_dpo_training(config, dpo_path)
        except Exception as exc:
            log.error("DPO training failed in iteration %d: %s", iteration, exc)
            state.record_iteration(
                iteration=iteration,
                score=0.0,
                verdict="fail",
                sft_count=dataset["sft_count"],
                dpo_count=dataset["dpo_count"],
                duration_seconds=time.time() - iter_start,
            )
            continue

        # Step 4: Update Ollama model
        _update_ollama_model(config)

        # Step 5: Evaluate
        try:
            eval_result = run_viz_evaluation(model_name=config.model_name)
        except Exception as exc:
            log.error("Evaluation failed in iteration %d: %s", iteration, exc)
            state.record_iteration(
                iteration=iteration,
                score=0.0,
                verdict="fail",
                sft_count=dataset["sft_count"],
                dpo_count=dataset["dpo_count"],
                duration_seconds=time.time() - iter_start,
            )
            continue

        composite = eval_result["composite_score"]
        verdict = eval_result["verdict"]
        duration = time.time() - iter_start

        # Step 6: Record results
        state.record_iteration(
            iteration=iteration,
            score=composite,
            verdict=verdict,
            chart_avg=eval_result.get("chart_avg", 0.0),
            text_avg=eval_result.get("text_avg", 0.0),
            sft_count=dataset["sft_count"],
            dpo_count=dataset["dpo_count"],
            duration_seconds=duration,
        )

        # Persist iteration JSON
        iter_file = ARTIFACT_DIR / f"iteration_{iteration}.json"
        iter_data = {
            "iteration": iteration,
            "composite_score": composite,
            "verdict": verdict,
            "chart_avg": eval_result.get("chart_avg", 0.0),
            "text_avg": eval_result.get("text_avg", 0.0),
            "sft_count": dataset["sft_count"],
            "dpo_count": dataset["dpo_count"],
            "duration_seconds": round(duration, 1),
        }
        with open(iter_file, "w", encoding="utf-8") as f:
            json.dump(iter_data, f, indent=2)

        log.info(
            "Iteration %d: composite=%.1f, verdict=%s, duration=%.1fs",
            iteration, composite, verdict, duration,
        )

        # Step 7: Check pass/fail
        if verdict == "pass":
            log.info("Passed threshold (%.1f >= %.1f) -- stopping.", composite, config.pass_threshold)
            break

        # Identify weak areas for next iteration
        weak_areas = _identify_weak_areas(eval_result)

    # Persist final summary
    summary = {
        "total_iterations": len(state.iterations),
        "best_score": state.best_score,
        "best_iteration": state.best_iteration,
        "passed": state.has_passed(config.pass_threshold),
        "iterations": [
            {
                "iteration": r.iteration,
                "composite_score": r.composite_score,
                "verdict": r.verdict,
                "chart_avg": r.chart_avg,
                "text_avg": r.text_avg,
                "sft_count": r.sft_count,
                "dpo_count": r.dpo_count,
                "duration_seconds": round(r.duration_seconds, 1),
            }
            for r in state.iterations
        ],
    }

    summary_file = ARTIFACT_DIR / "loop_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Finetune loop complete: %d iterations, best=%.1f (iter %d), passed=%s",
        summary["total_iterations"],
        summary["best_score"],
        summary["best_iteration"],
        summary["passed"],
    )

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the visualization fine-tuning loop."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Recursive visualization fine-tuning loop for DocWain",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Maximum number of training iterations (default: 5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=80.0,
        help="Composite score threshold to pass (default: 80.0)",
    )
    parser.add_argument(
        "--base-model", type=str, default="unsloth/Qwen3-8B-bnb-4bit",
        help="Base model for fine-tuning",
    )
    parser.add_argument(
        "--model-name", type=str, default="DHS/DocWain",
        help="Ollama model name to create/update",
    )
    parser.add_argument(
        "--sft-epochs", type=int, default=3,
        help="Number of SFT training epochs (default: 3)",
    )
    parser.add_argument(
        "--dpo-epochs", type=int, default=2,
        help="Number of DPO training epochs (default: 2)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = VizFinetuneConfig(
        max_iterations=args.max_iterations,
        pass_threshold=args.threshold,
        base_model=args.base_model,
        model_name=args.model_name,
        sft_epochs=args.sft_epochs,
        dpo_epochs=args.dpo_epochs,
    )

    result = run_viz_finetune_loop(config)

    print(f"\n{'=' * 50}")
    print(f"Total iterations:  {result['total_iterations']}")
    print(f"Best score:        {result['best_score']:.1f}")
    print(f"Best iteration:    {result['best_iteration']}")
    print(f"Passed:            {result['passed']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
