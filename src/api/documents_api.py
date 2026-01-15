from fastapi import APIRouter, HTTPException, Query

from src.documents.document_store_adapter import list_documents

documents_router = APIRouter()


@documents_router.get("/documents")
def get_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, description="Optional name search query"),
    doc_type: str | None = Query(None, description="Optional document type filter"),
    created_after: str | None = Query(None, description="ISO timestamp filter"),
    created_before: str | None = Query(None, description="ISO timestamp filter"),
):
    """
    List documents with stable pagination and optional filters.
    Returns doc_id, document_name, doc_type, created_at, updated_at when available.
    """
    try:
        items, total = list_documents(
            limit=limit,
            offset=offset,
            q=q,
            doc_type=doc_type,
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
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}")
