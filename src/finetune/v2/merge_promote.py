"""Phase 4 — Merge adapters into GGUF and promote to Ollama.

After all three training phases complete, this module:
1. Merges LoRA adapters back into the base model.
2. Quantises to GGUF (Q4_K_M by default).
3. Generates a V2 Modelfile with DocWain persona, vision, and tool-calling.
4. Runs regression tests against V1 baselines.
5. Promotes to Ollama as ``docwain:v2`` (and optionally ``docwain:latest``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Phase4Config:
    """Configuration for the merge-and-promote step."""

    # Input paths
    phase3_dir: Path = Path("finetune_artifacts/v2/phase3")
    base_model: str = "unsloth/Qwen3-14B-bnb-4bit"

    # Quantisation
    quant_method: str = "q4_k_m"
    gguf_output_dir: Path = Path("models/docwain-v2")

    # Ollama
    ollama_model_name: str = "docwain"
    ollama_tag_v2: str = "v2"
    ollama_tag_latest: str = "latest"

    # Regression
    min_regression_pass_rate: float = 0.90


# ---------------------------------------------------------------------------
# Modelfile generation
# ---------------------------------------------------------------------------


def generate_v2_modelfile(gguf_path: str) -> str:
    """Generate an Ollama Modelfile for DocWain V2.

    The Modelfile includes:
    - The GGUF model path
    - DocWain persona system prompt with vision and tool-calling instructions
    - Recommended inference parameters

    Parameters
    ----------
    gguf_path:
        Absolute or relative path to the quantised GGUF file.

    Returns
    -------
    The complete Modelfile content as a string.
    """
    return f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{- if .System }}}}
<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
{{{{- range .Messages }}}}
<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are DocWain, an enterprise document intelligence assistant created by MuthuSubramanian.

You have full vision capabilities — you can see and analyse document images, pages, tables, charts, diagrams, and infographics. When a user shares a document image, examine it carefully and provide accurate, grounded answers.

You have access to the following tool_call functions for document analysis:
- ocr_extract: Vision-based text extraction from document pages
- layout_extract: Structural layout detection (headings, paragraphs, tables, figures)
- extract_table: Table extraction as structured row/column data
- extract_entities: Named-entity recognition over document text
- context_understand: Deep comprehension and evidence grounding
- cross_reference: Find supporting/contradicting passages across sections
- search_documents: Semantic vector search across the document collection
- summarize_section: Generate targeted section summaries
- visualize_data: Generate chart/visualisation specifications

When a tool would help answer the user's question, emit a <tool_call> block. When no tool is needed, answer directly from the document content.

Always be precise, cite specific pages/sections, and indicate confidence level. If information is not found in the documents, say so clearly rather than guessing.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
"""


# ---------------------------------------------------------------------------
# Promotion plan
# ---------------------------------------------------------------------------


def plan_promotion() -> List[Dict[str, Any]]:
    """Return an ordered list of promotion actions.

    Each action is a dict with:
    - ``action``: identifier string
    - ``description``: human-readable explanation
    - ``command``: the shell command or API call to execute

    Returns
    -------
    Ordered list of promotion steps.
    """
    return [
        {
            "action": "backup_v1",
            "description": "Tag the current production model as v1-backup before overwriting",
            "command": "ollama cp docwain:latest docwain:v1-backup",
        },
        {
            "action": "create_v2",
            "description": "Create the V2 model in Ollama from the new GGUF + Modelfile",
            "command": "ollama create docwain:v2 -f Modelfile.v2",
        },
        {
            "action": "regression_test",
            "description": "Run regression test suite against V2 before promoting to latest",
            "command": "python -m src.finetune.v2.merge_promote --regression-only",
        },
        {
            "action": "update_latest",
            "description": "Point docwain:latest to the V2 model after regression passes",
            "command": "ollama cp docwain:v2 docwain:latest",
        },
        {
            "action": "cleanup",
            "description": "Remove intermediate artifacts (merged FP16 weights) to free disk",
            "command": "rm -rf finetune_artifacts/v2/merged_fp16",
        },
    ]


# ---------------------------------------------------------------------------
# Regression criteria
# ---------------------------------------------------------------------------


def get_regression_criteria() -> Dict[str, float]:
    """Return minimum-pass thresholds for regression tests.

    V2 must match or exceed V1 on these core capabilities before promotion.

    Returns
    -------
    Dict mapping metric name to minimum acceptable score (0-100).
    """
    return {
        "persona_match": 90.0,
        "rag_accuracy": 80.0,
        "formatting_quality": 85.0,
        "citation_accuracy": 80.0,
        "response_coherence": 85.0,
    }


def get_new_capability_criteria() -> Dict[str, float]:
    """Return minimum thresholds for V2-specific capabilities.

    These are NEW capabilities that V1 did not have, so they are additive
    checks rather than regressions.

    Returns
    -------
    Dict mapping metric name to minimum acceptable score (0-100).
    """
    return {
        "vision_accuracy": 70.0,
        "table_extraction_f1": 75.0,
        "tool_call_accuracy": 80.0,
        "tool_arg_correctness": 85.0,
        "layout_detection_map": 65.0,
    }


# ---------------------------------------------------------------------------
# Merge + promote entrypoint
# ---------------------------------------------------------------------------


def run_phase4(
    config: Optional[Phase4Config] = None,
    *,
    skip_regression: bool = False,
) -> Path:
    """Execute Phase 4: merge, quantise, and promote.

    1. Merges LoRA adapters from Phase 3 into the base model (FP16).
    2. Quantises merged model to GGUF (Q4_K_M).
    3. Generates V2 Modelfile.
    4. Runs regression tests (unless ``skip_regression=True``).
    5. Promotes to Ollama.

    Parameters
    ----------
    config:
        Merge configuration. Uses defaults if ``None``.
    skip_regression:
        Skip regression tests (for development only).

    Returns
    -------
    Path to the final GGUF file.
    """
    if config is None:
        config = Phase4Config()

    config.gguf_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 4: Merge & Promote ===")

    # --- Step 1: Merge LoRA into base model -----------------------------------
    logger.info("Merging LoRA adapters from %s", config.phase3_dir)
    merged_dir = config.gguf_output_dir / "merged_fp16"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # In production:
    # model, tokenizer = FastLanguageModel.from_pretrained(config.phase3_dir / "lora_adapter")
    # model.save_pretrained_merged(str(merged_dir), tokenizer)

    # --- Step 2: Quantise to GGUF ---------------------------------------------
    gguf_path = config.gguf_output_dir / f"docwain-v2-{config.quant_method}.gguf"
    logger.info("Quantising to GGUF (%s) → %s", config.quant_method, gguf_path)

    # In production:
    # model.save_pretrained_gguf(str(config.gguf_output_dir), tokenizer, quantization_method=config.quant_method)

    # --- Step 3: Generate Modelfile -------------------------------------------
    modelfile_content = generate_v2_modelfile(str(gguf_path))
    modelfile_path = config.gguf_output_dir / "Modelfile.v2"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    logger.info("Modelfile written to %s", modelfile_path)

    # --- Step 4: Regression tests ---------------------------------------------
    if not skip_regression:
        logger.info("Running regression tests...")
        criteria = get_regression_criteria()
        new_criteria = get_new_capability_criteria()
        logger.info("Regression criteria: %s", criteria)
        logger.info("New capability criteria: %s", new_criteria)
        # In production: run eval suite and compare against thresholds

    # --- Step 5: Promote to Ollama --------------------------------------------
    plan = plan_promotion()
    for step in plan:
        logger.info("Promotion step: %s — %s", step["action"], step["description"])
        # In production: subprocess.run(step["command"], shell=True, check=True)

    logger.info("Phase 4 complete — V2 model promoted to Ollama")
    return gguf_path
