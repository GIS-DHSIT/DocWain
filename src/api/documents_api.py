from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.schemas import DocumentListResponse
from src.documents.document_store_adapter import list_documents
from src.api.blob_store import BlobConfigurationError
from src.api.embedding_service import embed_documents, embedding_integrity_report
from src.security.response_sanitizer import sanitize_user_payload

documents_router = APIRouter()


class DocumentEmbedRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="Document identifier")
    document_ids: Optional[List[str]] = Field(None, description="Multiple document identifiers")
    subscription_id: Optional[str] = Field(None, description="Optional subscription id override")
    profile_id: Optional[str] = Field(None, description="Optional profile id override")
    doc_type: Optional[str] = Field(None, description="Optional document type override")
    max_blobs: Optional[int] = Field(None, description="Optional max blobs to process per request")
    force_reembed: bool = Field(False, description="Whether to force re-embedding if points already exist")
    collection_name: Optional[str] = Field(None, description="Optional collection name override")


class DocumentIntegrityRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="Document identifier")
    document_ids: Optional[List[str]] = Field(None, description="Multiple document identifiers")
    subscription_id: Optional[str] = Field(None, description="Optional subscription id override")
    profile_id: Optional[str] = Field(None, description="Optional profile id override")
    limit: Optional[int] = Field(None, description="Max documents to inspect when ids are not provided")


@documents_router.get("/documents", response_model=DocumentListResponse)
def get_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, description="Optional name search query"),
    doc_type: str | None = Query(None, description="Optional document type filter"),
    profile_id: str | None = Query(None, description="Optional profile id filter"),
    created_after: str | None = Query(None, description="ISO timestamp filter"),
    created_before: str | None = Query(None, description="ISO timestamp filter"),
):
    """
    List documents with stable pagination and optional filters.
    Returns doc_id, document_name, doc_type, profile_id, subscription_id, created_at, updated_at when available.
    """
    try:
        items, total = list_documents(
            limit=limit,
            offset=offset,
            q=q,
            doc_type=doc_type,
            profile_id=profile_id,
            created_after=created_after,
            created_before=created_before,
        )
        return {
            "items": items,
            "limit": limit,
            "offset": offset,
            "total": total,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "documents_list_failed", "message": str(exc)}},
        )


@documents_router.post("/documents/embed")
def embed_document(request: DocumentEmbedRequest = Body(...)):
    try:
        if not request.subscription_id:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "subscription_id_required", "message": "subscription_id is required"}},
            )
        if not (request.profile_id or request.document_id or request.document_ids):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "code": "profile_or_document_required",
                        "message": "profile_id or document_id(s) must be provided",
                    }
                },
            )
        result = embed_documents(
            document_id=request.document_id,
            document_ids=request.document_ids,
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
            doc_type=request.doc_type,
            max_blobs=request.max_blobs,
            force_reembed=request.force_reembed,
        )
        return sanitize_user_payload(result)
    except HTTPException:
        raise
    except BlobConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@documents_router.post("/documents/embed/report")
def embed_report(request: DocumentIntegrityRequest = Body(...)):
    try:
        result = embedding_integrity_report(
            document_id=request.document_id,
            document_ids=request.document_ids,
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
            limit=request.limit,
        )
        return sanitize_user_payload(result)
    except BlobConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
