from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from src.api.config import Config
from src.api.dataHandler import db


def _profiles_collection():
    return db[Config.MongoDB.PROFILES]


def create_profile(
    *,
    subscription_id: str,
    profile_name: str,
    profile_id: Optional[str] = None,
    status: str = "READY",
) -> Dict[str, Any]:
    now = time.time()
    profile_id = profile_id or str(uuid.uuid4())
    profile_id = str(profile_id)
    record = {
        "profile_id": profile_id,
        "subscription_id": str(subscription_id),
        "profile_name": profile_name,
        "status": status,
        "updated_at": now,
    }
    _profiles_collection().update_one(
        {"profile_id": profile_id, "subscription_id": str(subscription_id)},
        {"$set": record, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    record["created_at"] = now
    return record


def update_profile(
    *,
    subscription_id: str,
    profile_id: str,
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    update = dict(patch)
    update["updated_at"] = time.time()
    _profiles_collection().update_one(
        {"profile_id": str(profile_id), "subscription_id": str(subscription_id)},
        {"$set": update},
        upsert=True,
    )
    return get_profile(subscription_id=subscription_id, profile_id=profile_id) or update


def get_profile(*, subscription_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
    return _profiles_collection().find_one({"profile_id": str(profile_id), "subscription_id": str(subscription_id)})


def resolve_profile_name(*, subscription_id: str, profile_id: str) -> Optional[str]:
    profile = get_profile(subscription_id=subscription_id, profile_id=profile_id)
    if profile:
        return profile.get("profile_name")
    return None
