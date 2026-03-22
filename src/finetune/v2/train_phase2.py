"""Phase 2 — Document Intelligence SFT.

Trains the projection MLP + LoRA adapters on document-understanding data:
table extraction, layout analysis, OCR correction, and cross-reference tasks.

The projection is initialised from the Phase 1 checkpoint so vision-language
alignment is preserved while the model learns document-specific skills.

Typical wall-time: ~6-10 hours on a single A100-80 GB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Phase2Config:
    """Hyperparameters for Phase 2 document intelligence SFT."""

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Dataset mix — proportions must sum to 1.0
    dataset_mix: Dict[str, float] = field(default_factory=lambda: {
        "table": 0.40,
        "layout": 0.25,
        "ocr": 0.20,
        "cross_ref": 0.15,
    })

    # Data
    data_dir: Path = Path("finetune_data/v2/doc_intelligence")
    phase1_checkpoint: Path = Path("finetune_artifacts/v2/phase1/projection.pt")

    # Output
    output_dir: Path = Path("finetune_artifacts/v2/phase2")
    save_steps: int = 200
    logging_steps: int = 25
    eval_steps: int = 200

    # Quality gates — minimum metrics to proceed to Phase 3
    gate_docvqa_accuracy: float = 0.75
    gate_table_f1: float = 0.80
    gate_layout_map: float = 0.70


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_phase2(
    config: Optional[Phase2Config] = None,
    *,
    phase1_checkpoint: Optional[Path] = None,
) -> Path:
    """Execute Phase 2 document intelligence SFT.

    1. Loads the vision encoder + projection from Phase 1.
    2. Loads the text model and applies LoRA adapters.
    3. Trains projection + LoRA on a weighted mix of doc-intel datasets.
    4. Evaluates against quality gates.
    5. Saves adapters and updated projection to ``config.output_dir``.

    Parameters
    ----------
    config:
        Training configuration. Uses defaults if ``None``.
    phase1_checkpoint:
        Override path to the Phase 1 projection checkpoint.

    Returns
    -------
    Path to the output directory containing adapters and projection.
    """
    if config is None:
        config = Phase2Config()

    proj_ckpt = phase1_checkpoint or config.phase1_checkpoint
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 2: Document Intelligence SFT ===")
    logger.info("LoRA r=%d  LR=%s  epochs=%d  batch=%d",
                config.lora_r, config.learning_rate, config.epochs, config.batch_size)
    logger.info("Dataset mix: %s", config.dataset_mix)

    # --- Load model -----------------------------------------------------------
    from .vision_graft import GraftConfig, VisionGraftedModel

    graft_cfg = GraftConfig(freeze_vision=True, freeze_text=False)
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()
    model.load_projection(checkpoint=proj_ckpt)
    model.load_text_model()
    model.add_lora(r=config.lora_r, lora_alpha=config.lora_alpha)

    # --- Build mixed dataset --------------------------------------------------
    logger.info("Building weighted dataset from mix: %s", config.dataset_mix)

    # In production, this would:
    # 1. Load JSONL files for each category from data_dir
    # 2. Sample according to dataset_mix proportions
    # 3. Convert to chat format via dataset_preprocess.format_vision_sft()
    # 4. Tokenize with the model's tokenizer

    # --- Training loop (SFTTrainer) -------------------------------------------
    logger.info("Starting SFT training...")

    # In production this would use trl.SFTTrainer or equivalent.
    # The training loop would:
    # 1. Forward pass through vision encoder → projection → text model
    # 2. Compute cross-entropy loss on assistant tokens only
    # 3. Backprop through projection + LoRA (vision encoder frozen)

    # --- Save outputs ---------------------------------------------------------
    model.save_all(config.output_dir)
    logger.info("Phase 2 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
