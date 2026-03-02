from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from src.api.config import Config
from src.kg.entity_extractor import EntityExtractor
from src.kg.neo4j_store import Neo4jStore
from src.kg.qdrant_reader import QdrantKGReader
from src.kg.sync_service import KGSyncService

knowledge_graph_router = APIRouter(prefix="/knowledge-graph", tags=["KnowledgeGraph"])


class KGUpdateRequest(BaseModel):
    mode: str = Field("incremental", description="Sync mode; only incremental supported")
    batch_size: int = Field(200, ge=1, le=1000)
    max_points: int = Field(5000, ge=1, le=100000)
    state_name: str = Field("default", min_length=1)
    collection_name: Optional[str] = Field(None, description="Optional Qdrant collection override")
    subscription_id: str = Field(..., description="Subscription identifier")
    profile_id: str = Field(..., description="Profile identifier")


class KGUpdateResponse(BaseModel):
    status: str
    processed_points: int
    skipped_points: int
    last_qdrant_offset: Optional[Any]
    nodes_edges_estimate: dict


class KGStatusResponse(BaseModel):
    status: str
    state_name: str
    last_qdrant_offset: Optional[Any]
    last_sync_at: Optional[str]
    counts: dict


@knowledge_graph_router.post("/update", response_model=KGUpdateResponse)
def update_knowledge_graph(request: KGUpdateRequest = Body(...)):
    if request.mode != "incremental":
        raise HTTPException(status_code=400, detail="Only incremental mode is supported")

    collection_name = request.collection_name or Config.KnowledgeGraph.QDRANT_COLLECTION

    store = None
    try:
        reader = QdrantKGReader(collection_name=collection_name)
        store = Neo4jStore()
        extractor = EntityExtractor()
        service = KGSyncService(qdrant_reader=reader, neo4j_store=store, entity_extractor=extractor)
        stats = service.run(
            batch_size=request.batch_size,
            max_points=request.max_points,
            state_name=request.state_name,
            subscription_id=request.subscription_id,
            profile_id=request.profile_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    return {
        "status": "ok",
        "processed_points": stats.processed_points,
        "skipped_points": stats.skipped_points,
        "last_qdrant_offset": stats.last_qdrant_offset,
        "nodes_edges_estimate": {
            "chunks_upserted": stats.chunks_upserted,
            "entities_seen": stats.entities_seen,
            "mentions_created": stats.mentions_created,
        },
    }


@knowledge_graph_router.get("/status", response_model=KGStatusResponse)
def knowledge_graph_status(state_name: str = "default"):
    store = None
    try:
        store = Neo4jStore()
        status = store.get_status(state_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
    return {
        "status": "ok",
        "state_name": status["state_name"],
        "last_qdrant_offset": status["last_qdrant_offset"],
        "last_sync_at": status["last_sync_at"],
        "counts": status["counts"],
    }
