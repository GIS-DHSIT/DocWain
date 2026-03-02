import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    instruction: str
    output: str
    input: Optional[str] = ""


class FinetuneRequest(BaseModel):
    profile_id: str = Field(..., description="Domain/profile id to specialize the model")
    base_model: str = Field("llama3.2", description="Base model label to fine-tune")
    learning_rate: float = Field(2e-4, gt=0)
    num_epochs: int = Field(1, ge=1)
    max_steps: int = Field(200, ge=1)
    batch_size: int = Field(4, ge=1)
    gradient_accumulation: int = Field(2, ge=1)
    lora_r: int = Field(16, ge=1)
    lora_alpha: int = Field(16, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    include_actual_data: bool = Field(False, description="Whether to pull profile data into the training set")
    dataset_path: Optional[str] = Field(None, description="Optional path to a JSON/JSONL dataset file")
    training_examples: List[TrainingExample] = Field(default_factory=list)
    output_dir: str = Field("finetune_artifacts", description="Root directory for finetune outputs")
    run_name: Optional[str] = Field(None, description="Optional identifier for the run")
    retrain: bool = Field(False, description="Force a new run even if the same parameters already completed")
    allow_offload: bool = Field(True, description="Permit CPU offload when GPU memory is insufficient")
    device_map: Optional[object] = Field(None, description="Optional device_map override for model loading")
    max_memory: Optional[object] = Field(None, description="Optional max_memory override for device_map dispatch")
    agentic: bool = Field(False, description="Enable agentic finetune orchestration when supported")
    orchestrator_model: Optional[str] = Field(None, description="Optional override for orchestrator LLM")
    qdrant_snapshot: Optional[Dict[str, object]] = Field(
        None, description="Optional snapshot metadata for the Qdrant collection"
    )
    collection_name: Optional[str] = Field(
        None, description="Optional collection name for agentic finetuning context enforcement"
    )
    training_run_id: Optional[str] = Field(
        None, description="Optional training run id for diagnostics/artifact paths"
    )


class FinetuneStatus(BaseModel):
    job_id: str
    profile_id: str
    status: str
    message: str = ""
    output_model: Optional[str] = None
    started_at: float = Field(default_factory=lambda: time.time())
    finished_at: Optional[float] = None
    params: dict = Field(default_factory=dict)
    training_run_id: Optional[str] = None
    params_hash: Optional[str] = None
    ollama: Optional[Dict[str, Any]] = None


class ResolvedModel(BaseModel):
    model_name: Optional[str]
    backend: Optional[str] = None
    model_path: Optional[str] = None
    profile_id: Optional[str] = None


class AutoFinetuneRequest(FinetuneRequest):
    # Make profile/subscription optional for discovery-based auto runs
    profile_id: Optional[str] = Field(
        None,
        description="Optional profile to target. When omitted, profiles are discovered from Qdrant.",
    )
    subscription_id: Optional[str] = Field(
        None,
        description="Optional single subscription/collection to target. When omitted, all collections are scanned.",
    )
    subscription_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of subscriptions/collections to target. When omitted, all collections are scanned.",
    )
    max_points: int = Field(120, ge=1, description="Max chunks to sample from Qdrant")
    questions_per_chunk: int = Field(2, ge=1, le=5, description="QA pairs per chunk")
    generation_model: Optional[str] = Field(None, description="Model used to generate synthetic QA")

    def finetune_payload(self, profile_id: str, dataset_path: str) -> dict:
        """
        Build a FinetuneRequest payload by merging auto-discovery data with user overrides.
        Auto-only fields are stripped before constructing the finetune job.
        """
        payload = self.dict()
        payload.update(
            {
                "profile_id": profile_id,
                "dataset_path": str(dataset_path),
                # Auto runs should not silently pull production data into the dataset.
                "include_actual_data": False,
            }
        )
        for field in ("subscription_id", "subscription_ids", "max_points", "questions_per_chunk", "generation_model"):
            payload.pop(field, None)
        return payload


class AutoFinetuneRunRequest(BaseModel):
    collection_name: Optional[str] = Field(None, description="Optional collection to target for the run")
    profile_id: Optional[str] = Field(None, description="Optional profile to target")
    dry_run: bool = Field(False, description="When true, only plan the run without executing jobs")
    max_profiles_per_run: Optional[int] = Field(
        None,
        description="Optional cap on number of profiles processed per run",
    )
    max_points: int = Field(120, ge=1, description="Max chunks to sample from Qdrant")
    questions_per_chunk: int = Field(2, ge=1, le=5, description="QA pairs per chunk")
    generation_model: Optional[str] = Field(None, description="Model used to generate synthetic QA")


class CollectionFinetuneRequest(AutoFinetuneRequest):
    """
    Request model for end-to-end finetuning driven solely by a Qdrant collection name.
    """

    collection_name: str = Field(..., description="Qdrant collection to read profile chunks from")
    profile_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of profile ids to target inside the collection",
    )
    fail_fast: bool = Field(False, description="Abort processing on first failure instead of continuing")

    def finetune_payload(self, profile_id: str, dataset_path: str) -> dict:
        payload = super().finetune_payload(profile_id, dataset_path)
        payload.pop("collection_name", None)
        payload.pop("profile_ids", None)
        payload.pop("fail_fast", None)
        payload["retrain"] = self.retrain
        return payload


class CollectionOnlyFinetuneRequest(BaseModel):
    """
    Collection-driven fine-tune request that discovers all profiles in the Qdrant collection.

    Example request body:
    {
        "collection_name": "my-qdrant-collection",
        "base_model": "llama3.2",
        "learning_rate": 0.0002,
        "num_epochs": 1,
        "max_steps": 200,
        "batch_size": 4,
        "gradient_accumulation": 2,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "include_actual_data": false,
        "dataset_path": "string",
        "training_examples": [
            {"instruction": "string", "output": "string", "input": ""}
        ],
        "output_dir": "finetune_artifacts",
        "run_name": "string",
        "retrain": false,
        "subscription_id": "string",
        "subscription_ids": ["string"],
        "max_points": 120,
        "questions_per_chunk": 2,
        "generation_model": "string",
        "fail_fast": false
    }
    """

    collection_name: str = Field(..., description="Qdrant collection to read profile chunks from")
    base_model: str = Field("llama3.2", description="Base model label to fine-tune")
    learning_rate: float = Field(2e-4, gt=0)
    num_epochs: int = Field(1, ge=1)
    max_steps: int = Field(200, ge=1)
    batch_size: int = Field(4, ge=1)
    gradient_accumulation: int = Field(2, ge=1)
    lora_r: int = Field(16, ge=1)
    lora_alpha: int = Field(16, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    include_actual_data: bool = Field(False, description="Whether to pull profile data into the training set")
    dataset_path: Optional[str] = Field(None, description="Optional path to a JSON/JSONL dataset file")
    training_examples: List[TrainingExample] = Field(default_factory=list)
    output_dir: str = Field("finetune_artifacts", description="Root directory for finetune outputs")
    run_name: Optional[str] = Field(None, description="Optional identifier for the run")
    retrain: bool = Field(False, description="Force a new run even if the same parameters already completed")
    profile_ids: Optional[List[str]] = Field(None, description="Optional list of profile ids to target")
    subscription_id: Optional[str] = Field(None, description="Optional subscription identifier for auditing")
    subscription_ids: Optional[List[str]] = Field(None, description="Optional subscription identifiers for auditing")
    max_points: int = Field(120, ge=1, description="Max chunks to sample from Qdrant")
    questions_per_chunk: int = Field(2, ge=1, le=5, description="QA pairs per chunk")
    generation_model: Optional[str] = Field(None, description="Model used to generate synthetic QA")
    fail_fast: bool = Field(False, description="Abort processing on first failure instead of continuing")
    agentic: bool = Field(False, description="Enable agentic finetune orchestration when supported")
    orchestrator_model: Optional[str] = Field(None, description="Optional override for orchestrator LLM")

    def finetune_payload(self, profile_id: str, dataset_path: str) -> dict:
        """
        Build a FinetuneRequest payload by merging discovery data with user overrides.
        """
        payload = self.dict()
        payload.update(
            {
                "profile_id": profile_id,
                "dataset_path": str(dataset_path),
            }
        )
        for field in (
            "collection_name",
            "subscription_id",
            "subscription_ids",
            "max_points",
            "questions_per_chunk",
            "generation_model",
            "fail_fast",
        ):
            payload.pop(field, None)
        return payload
