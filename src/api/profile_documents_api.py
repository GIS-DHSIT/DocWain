from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.document_understanding_service import extract_and_understand, run_document_understanding
from src.profiles.profile_store import resolve_profile_name
from src.retrieval.profile_query import query_profile

profile_docs_router = APIRouter(prefix="/profiles", tags=["Profiles"])


class UnderstandRequest(BaseModel):
    subscription_id: str = Field(..., description="Subscription identifier")
    profile_name: Optional[str] = Field(None, description="Profile display name")
    model_name: Optional[str] = Field(None, description="Ollama model name override")
    embed_after: bool = Field(False, description="Whether to embed after understanding (HITL: must be False by default)")


class QueryRequest(BaseModel):
    subscription_id: str = Field(..., description="Subscription identifier")
    query: str = Field(..., description="User query")
    model_name: Optional[str] = Field(None, description="Ollama model name override")
    top_k: int = Field(6, ge=1, le=20)


@profile_docs_router.post("/{profile_id}/documents/upload", summary="Upload a document for a profile")
async def upload_document_for_profile(
    profile_id: str,
    subscription_id: str = Form(...),
    profile_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    doc_id = str(uuid.uuid4())
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail={"error": {"code": "empty_file", "message": "Empty file"}})

    profile_name = profile_name or resolve_profile_name(subscription_id=subscription_id, profile_id=profile_id)

    result = extract_and_understand(
        document_id=doc_id,
        file_bytes=contents,
        filename=file.filename or "document",
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        content_type=file.content_type,
        content_size=len(contents),
        embed_after=False,
    )
    return {"document_id": doc_id, "status": "COMPLETED", "result": result}


@profile_docs_router.post("/{profile_id}/documents/{document_id}/understand", summary="Run understanding on a document")
def understand_document(
    profile_id: str,
    document_id: str,
    payload: UnderstandRequest = Body(...),
):
    profile_name = payload.profile_name or resolve_profile_name(
        subscription_id=payload.subscription_id, profile_id=profile_id
    )
    result = run_document_understanding(
        document_id=document_id,
        subscription_id=payload.subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        model_name=payload.model_name,
        embed_after=payload.embed_after,
    )
    return result


@profile_docs_router.post("/{profile_id}/query", summary="Query a profile")
def query_profile_endpoint(profile_id: str, payload: QueryRequest = Body(...)):
    response = query_profile(
        subscription_id=payload.subscription_id,
        profile_id=profile_id,
        query=payload.query,
        model_name=payload.model_name,
        top_k=payload.top_k,
    )
    return response
