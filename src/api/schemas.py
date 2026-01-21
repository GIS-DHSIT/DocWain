from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelInfo(BaseModel):
    model: str
    source: str = Field(..., description="Provider of the model (ollama, finetune, gemini, etc.)")
    backend: Optional[str] = None
    profile_id: Optional[str] = None
    path: Optional[str] = None
    updated_at: Optional[Any] = None
    size: Optional[int] = None
    digest: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class DocumentSummary(BaseModel):
    doc_id: str
    document_name: str
    doc_type: Optional[str] = None
    profile_id: Optional[str] = None
    subscription_id: Optional[str] = None
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None


class DocumentListResponse(BaseModel):
    items: List[DocumentSummary]
    limit: int
    offset: int
    total: int
