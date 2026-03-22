"""Dataset registry and downloader for DocWain V2 training.

Provides a curated registry of HuggingFace datasets organised into three
training phases (vision_pretrain, doc_intelligence, tool_calling) along
with helpers to list, inspect, and download them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Phase 1 — Vision pre-training
    "llava_pretrain": {
        "hf_id": "liuhaotian/LLaVA-Pretrain",
        "split": "train",
        "phase": "vision_pretrain",
        "sample_size": 558_000,
        "description": "LLaVA pre-training image-caption pairs for vision-language alignment.",
    },
    # Phase 2 — Document intelligence
    "docvqa": {
        "hf_id": "lmms-lab/DocVQA",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 39_463,
        "description": "Document Visual Question Answering on industry documents.",
    },
    "chartvqa": {
        "hf_id": "HuggingFaceM4/ChartQA",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 28_299,
        "description": "Chart-based visual question answering.",
    },
    "infographicsvqa": {
        "hf_id": "lmms-lab/InfographicsVQA",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 24_000,
        "description": "Visual QA over infographic images.",
    },
    "pubtabnet": {
        "hf_id": "apoidea/pubtabnet-html",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 516_738,
        "description": "Table recognition and structure extraction from PubMed articles.",
    },
    "fintabnet": {
        "hf_id": "yifeihu/FinTabNet.c",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 89_646,
        "description": "Financial table recognition from SEC filings.",
    },
    "wikitablequestions": {
        "hf_id": "Stanford/web_questions",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 22_033,
        "description": "Question answering on Wikipedia tables.",
    },
    "doclaynet": {
        "hf_id": "ds4sd/DocLayNet",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 80_863,
        "description": "Document layout analysis across multiple domains.",
    },
    "publaynet": {
        "hf_id": "COCO-DOTA/PubLayNet",
        "split": "train",
        "phase": "doc_intelligence",
        "sample_size": 335_703,
        "description": "Large-scale document layout analysis from PubMed Central.",
    },
    # Phase 3 — Tool calling
    "toolbench": {
        "hf_id": "sahil2801/ToolBench",
        "split": "train",
        "phase": "tool_calling",
        "sample_size": 126_486,
        "description": "Multi-tool planning and API call generation.",
    },
    "gorilla": {
        "hf_id": "gorilla-llm/APIBench",
        "split": "train",
        "phase": "tool_calling",
        "sample_size": 16_450,
        "description": "API call generation from natural language.",
    },
    "nexusraven": {
        "hf_id": "Nexusflow/NexusRaven_API_evaluation",
        "split": "train",
        "phase": "tool_calling",
        "sample_size": 6_000,
        "description": "Function-call evaluation dataset for LLM tool use.",
    },
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_datasets() -> List[str]:
    """Return the names of all registered datasets."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Return metadata dict for a single dataset.

    Raises ``KeyError`` if *name* is not in the registry.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return dict(DATASET_REGISTRY[name])


def list_datasets_by_phase(phase: str) -> List[str]:
    """Return dataset names belonging to *phase*.

    Valid phases: ``vision_pretrain``, ``doc_intelligence``, ``tool_calling``.
    """
    return [
        name
        for name, info in DATASET_REGISTRY.items()
        if info["phase"] == phase
    ]


def download_dataset(
    name: str,
    output_dir: str | Path,
    *,
    max_samples: Optional[int] = None,
    streaming: bool = True,
) -> Path:
    """Download (or stream) a dataset from HuggingFace and save to *output_dir*.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"docvqa"``).
    output_dir:
        Local directory where the dataset will be saved as JSONL.
    max_samples:
        Cap the number of rows written. ``None`` means use the full split.
    streaming:
        If ``True`` (default), use HF streaming to avoid downloading the
        entire dataset up-front.

    Returns
    -------
    Path to the written JSONL file.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for downloading. "
            "Install with: pip install datasets"
        ) from exc

    info = get_dataset_info(name)
    hf_id = info["hf_id"]
    split = info["split"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.jsonl"

    logger.info("Downloading %s (%s, split=%s) → %s", name, hf_id, split, output_path)

    ds = load_dataset(hf_id, split=split, streaming=streaming)

    cap = max_samples or info["sample_size"]
    count = 0
    import json

    with output_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(json.dumps(row, default=str) + "\n")
            count += 1
            if count >= cap:
                break

    logger.info("Wrote %d rows to %s", count, output_path)
    return output_path
