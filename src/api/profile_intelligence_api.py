"""Profile Intelligence API — serves auto-generated document insights."""

from fastapi import APIRouter, HTTPException

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

profile_intelligence_router = APIRouter(prefix="/profiles", tags=["Profile Intelligence"])


@profile_intelligence_router.get("/{profile_id}/intelligence", summary="Get profile intelligence report")
async def get_profile_intelligence(profile_id: str):
    """Return the auto-generated intelligence report for a profile.

    Contains:
    - Profile overview (summary, key metrics, overall insights)
    - Per-document briefs (key facts, entities, insights per document)
    - Cross-document analysis (comparisons, trends, anomalies, rankings)
    """
    from pymongo import MongoClient
    from src.api.config import Config

    client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
    db = client[Config.MongoDB.DB]
    report = db["profile_intelligence"].find_one(
        {"profile_id": profile_id},
        {"_id": 0},
    )

    if not report:
        return {
            "profile_id": profile_id,
            "status": "pending",
            "message": "No intelligence report yet. Reports generate automatically after documents complete processing.",
            "profile_overview": None,
            "document_briefs": [],
            "cross_document_analysis": None,
        }

    return report
