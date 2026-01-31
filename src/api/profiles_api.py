from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from src.profiles.profile_store import create_profile, get_profile, update_profile

profiles_router = APIRouter(prefix="/profiles", tags=["Profiles"])


class ProfileCreateRequest(BaseModel):
    subscription_id: str = Field(..., description="Subscription identifier")
    profile_name: str = Field(..., description="Display name for the profile")
    profile_id: Optional[str] = Field(None, description="Optional explicit profile id")


class ProfileUpdateRequest(BaseModel):
    subscription_id: str = Field(..., description="Subscription identifier")
    profile_name: Optional[str] = Field(None, description="Display name for the profile")
    status: Optional[str] = Field(None, description="Profile status")


@profiles_router.post("", summary="Create a profile")
def create_profile_endpoint(payload: ProfileCreateRequest = Body(...)):
    profile_id = payload.profile_id or str(uuid.uuid4())
    record = create_profile(
        subscription_id=payload.subscription_id,
        profile_name=payload.profile_name,
        profile_id=profile_id,
        status="READY",
    )
    return record


@profiles_router.get("/{profile_id}", summary="Get a profile")
def get_profile_endpoint(profile_id: str, subscription_id: str):
    record = get_profile(subscription_id=subscription_id, profile_id=profile_id)
    if not record:
        raise HTTPException(status_code=404, detail={"error": {"code": "profile_not_found", "message": "Profile not found"}})
    return record


@profiles_router.put("/{profile_id}", summary="Update a profile")
def update_profile_endpoint(profile_id: str, payload: ProfileUpdateRequest = Body(...)):
    patch = {k: v for k, v in payload.model_dump().items() if v is not None and k != "subscription_id"}
    if not patch:
        raise HTTPException(status_code=400, detail={"error": {"code": "profile_update_empty", "message": "No fields to update"}})
    record = update_profile(subscription_id=payload.subscription_id, profile_id=profile_id, patch=patch)
    return record
