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
    subscription_id: str = Field("default", description="Tenant/subscription")
    max_points: int = Field(120, ge=1, description="Max chunks to sample from Qdrant")
    questions_per_chunk: int = Field(2, ge=1, le=5, description="QA pairs per chunk")
    generation_model: Optional[str] = Field(None, description="Model used to generate synthetic QA")
