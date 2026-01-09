import time
from typing import List, Optional
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


class FinetuneStatus(BaseModel):
    job_id: str
    profile_id: str
    status: str
    message: str = ""
    output_model: Optional[str] = None
    started_at: float = Field(default_factory=lambda: time.time())
    finished_at: Optional[float] = None
    params: dict = Field(default_factory=dict)


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
