from __future__ import annotations

from src.utils.logging_utils import get_logger
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.api.dw_newron import get_redis_client

logger = get_logger(__name__)

@dataclass
class LimitResult:
    allowed: bool
    message: str
    reason: str
    meta: Optional[dict] = None

def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _settings() -> dict:
    return {
        "enabled": _truthy(os.getenv("AGENT_MODE_ENABLED", "true")),
        "max_users": _env_int("AGENT_MODE_MAX_USERS_PER_SUBSCRIPTION", 3),
        "max_user_daily": _env_int("AGENT_MODE_MAX_PROMPTS_PER_USER_PER_DAY", 20),
        "max_sub_daily": _env_int("AGENT_MODE_MAX_PROMPTS_PER_SUBSCRIPTION_PER_DAY", 100),
    }

def _audit_collection():
    from src.api.dataHandler import db

    return db["agent_audit"]

def check_and_count(
    *,
    subscription_id: str,
    profile_id: str,
    user_id: str,
    session_id: Optional[str],
    query: str,
    redis_client: Any | None = None,
) -> LimitResult:
    settings = _settings()
    if not settings["enabled"]:
        return LimitResult(
            allowed=False,
            reason="disabled",
            message=(
                "Agent mode is currently disabled for this account. "
                "Please use the standard mode or contact support to enable agent access."
            ),
        )

    if redis_client is None:
        try:
            redis_client = get_redis_client()
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent limits: Redis unavailable: %s", exc, exc_info=True)
            redis_client = None

    if redis_client is None:
        logger.warning("Agent limits: Redis client missing; allowing request without enforcement.")
        return LimitResult(allowed=True, reason="redis_unavailable", message="ok", meta={"skipped": True})

    day = _today_key()
    users_key = f"agent:users:{subscription_id}"
    user_daily_key = f"agent:count:user:{subscription_id}:{user_id}:daily:{day}"
    sub_daily_key = f"agent:count:sub:{subscription_id}:daily:{day}"
    user_lifetime_key = f"agent:count:user:{subscription_id}:{user_id}:lifetime"
    sub_lifetime_key = f"agent:count:sub:{subscription_id}:lifetime"

    try:
        if not redis_client.sismember(users_key, user_id):
            current_users = int(redis_client.scard(users_key) or 0)
            if current_users >= settings["max_users"]:
                return LimitResult(
                    allowed=False,
                    reason="user_cap",
                    message=(
                        "This subscription has reached the agent user cap. "
                        "Please remove an existing agent user or upgrade your plan."
                    ),
                )
        redis_client.sadd(users_key, user_id)

        user_daily = int(redis_client.get(user_daily_key) or 0)
        if user_daily >= settings["max_user_daily"]:
            return LimitResult(
                allowed=False,
                reason="user_daily_limit",
                message=(
                    "You have reached your daily agent prompt limit. "
                    "Please try again tomorrow or request a higher limit."
                ),
            )

        sub_daily = int(redis_client.get(sub_daily_key) or 0)
        if sub_daily >= settings["max_sub_daily"]:
            return LimitResult(
                allowed=False,
                reason="subscription_daily_limit",
                message=(
                    "This subscription has reached its daily agent prompt limit. "
                    "Please retry tomorrow or contact support for more capacity."
                ),
            )

        pipe = redis_client.pipeline()
        pipe.incr(user_daily_key)
        pipe.expire(user_daily_key, 2 * 86400)
        pipe.incr(sub_daily_key)
        pipe.expire(sub_daily_key, 2 * 86400)
        pipe.incr(user_lifetime_key)
        pipe.incr(sub_lifetime_key)
        pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent limits: Redis enforcement failed: %s", exc, exc_info=True)
        return LimitResult(
            allowed=False,
            reason="redis_error",
            message=(
                "We could not validate agent access at the moment. "
                "Please try again shortly."
            ),
        )

    try:
        _audit_collection().insert_one(
            {
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "user_id": user_id,
                "session_id": session_id,
                "query": query,
                "ts": datetime.now(timezone.utc),
                "mode": "agent",
                "allowed": True,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent audit insert failed: %s", exc, exc_info=True)

    return LimitResult(allowed=True, reason="ok", message="ok")

__all__ = ["check_and_count", "LimitResult"]
