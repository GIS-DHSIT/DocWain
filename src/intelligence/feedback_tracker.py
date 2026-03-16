"""Implicit query-signal tracker for intelligence monitoring and fine-tuning.

Captures signals from every query automatically — no explicit user feedback
required. Metrics are aggregated per-profile in Redis and used to identify
fine-tuning candidates and surface quality trends.

All keys are profile-scoped and expire after 30 days.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_PREFIX = "intel:metrics"
_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
_LOW_CONFIDENCE_MAX = 100
_LOW_CONFIDENCE_THRESHOLD = 0.5


def _key(profile_id: str, *parts: str) -> str:
    """Build a Redis key with profile scoping."""
    return ":".join([_PREFIX, profile_id] + list(parts))


class FeedbackTracker:
    """Records implicit query signals and aggregates profile-level metrics."""

    def __init__(self, redis_client):
        self._r = redis_client

    # ------------------------------------------------------------------
    # Signal recording
    # ------------------------------------------------------------------

    def record_query_signal(
        self,
        profile_id: str,
        query: str,
        response: str,
        evidence: List[str],
        grounded: bool,
        confidence: float,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a query signal to Redis for aggregation."""
        try:
            # Guard all inputs — Redis rejects None values
            confidence = confidence if confidence is not None else 0.0
            task_type = task_type or "unknown"
            query = query or ""
            grounded = bool(grounded)

            pipe = self._r.pipeline(transaction=False)

            # Total query count
            k_total = _key(profile_id, "total_queries")
            pipe.incr(k_total)
            pipe.expire(k_total, _TTL_SECONDS)

            # Confidence distribution
            k_conf_sum = _key(profile_id, "confidence_sum")
            pipe.incrbyfloat(k_conf_sum, float(confidence))
            pipe.expire(k_conf_sum, _TTL_SECONDS)

            k_conf_count = _key(profile_id, "confidence_count")
            pipe.incr(k_conf_count)
            pipe.expire(k_conf_count, _TTL_SECONDS)

            # Grounding rate
            if grounded:
                k_grounded = _key(profile_id, "grounded_count")
                pipe.incr(k_grounded)
                pipe.expire(k_grounded, _TTL_SECONDS)

            # Task type distribution
            k_task_types = _key(profile_id, "task_types")
            pipe.hincrby(k_task_types, str(task_type), 1)
            pipe.expire(k_task_types, _TTL_SECONDS)

            # Low-confidence queries
            if confidence < _LOW_CONFIDENCE_THRESHOLD:
                k_low = _key(profile_id, "low_confidence_queries")
                entry = json.dumps({
                    "query": str(query),
                    "confidence": float(confidence),
                    "task_type": str(task_type),
                    "grounded": bool(grounded),
                    "ts": time.time(),
                })
                pipe.lpush(k_low, entry)
                pipe.ltrim(k_low, 0, _LOW_CONFIDENCE_MAX - 1)
                pipe.expire(k_low, _TTL_SECONDS)

            pipe.execute()
            logger.debug(
                "Recorded query signal for profile %s (confidence=%.2f, grounded=%s, task=%s)",
                profile_id, confidence, grounded, task_type,
            )
        except Exception:
            logger.exception("Failed to record query signal for profile %s", profile_id)

    # ------------------------------------------------------------------
    # Metrics retrieval
    # ------------------------------------------------------------------

    def get_profile_metrics(self, profile_id: str) -> Dict[str, Any]:
        """Return aggregated metrics for a profile."""
        try:
            pipe = self._r.pipeline(transaction=False)
            pipe.get(_key(profile_id, "total_queries"))
            pipe.get(_key(profile_id, "confidence_sum"))
            pipe.get(_key(profile_id, "confidence_count"))
            pipe.get(_key(profile_id, "grounded_count"))
            pipe.hgetall(_key(profile_id, "task_types"))
            pipe.llen(_key(profile_id, "low_confidence_queries"))
            results = pipe.execute()

            total_queries = int(results[0] or 0)
            confidence_sum = float(results[1] or 0.0)
            confidence_count = int(results[2] or 0)
            grounded_count = int(results[3] or 0)
            task_types_raw = results[4] or {}
            low_confidence_count = int(results[5] or 0)

            avg_confidence = (
                confidence_sum / confidence_count if confidence_count > 0 else 0.0
            )
            grounded_ratio = (
                grounded_count / total_queries if total_queries > 0 else 0.0
            )

            # Decode task type keys/values (Redis returns bytes in some clients)
            task_type_distribution = {}
            for k, v in task_types_raw.items():
                key = k.decode() if isinstance(k, bytes) else k
                task_type_distribution[key] = int(v)

            return {
                "total_queries": total_queries,
                "avg_confidence": round(avg_confidence, 4),
                "grounded_ratio": round(grounded_ratio, 4),
                "task_type_distribution": task_type_distribution,
                "low_confidence_count": low_confidence_count,
            }
        except Exception:
            logger.exception("Failed to retrieve metrics for profile %s", profile_id)
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "grounded_ratio": 0.0,
                "task_type_distribution": {},
                "low_confidence_count": 0,
            }

    # ------------------------------------------------------------------
    # Fine-tuning candidate detection
    # ------------------------------------------------------------------

    def get_tuning_candidates(
        self,
        profile_id: str,
        min_queries: int = 50,
        max_low_confidence_ratio: float = 0.3,
    ) -> Dict[str, Any]:
        """Determine whether a profile is a candidate for fine-tuning."""
        metrics = self.get_profile_metrics(profile_id)
        total = metrics["total_queries"]

        if total < min_queries:
            return {
                "is_candidate": False,
                "reason": f"Insufficient queries ({total}/{min_queries})",
                "metrics": metrics,
            }

        low_ratio = (
            metrics["low_confidence_count"] / total if total > 0 else 0.0
        )

        if low_ratio > max_low_confidence_ratio:
            return {
                "is_candidate": True,
                "reason": (
                    f"High low-confidence ratio ({low_ratio:.2%}) "
                    f"exceeds threshold ({max_low_confidence_ratio:.2%})"
                ),
                "metrics": metrics,
            }

        if metrics["grounded_ratio"] < 0.5:
            return {
                "is_candidate": True,
                "reason": (
                    f"Low grounding ratio ({metrics['grounded_ratio']:.2%}) "
                    f"indicates retrieval quality issues"
                ),
                "metrics": metrics,
            }

        return {
            "is_candidate": False,
            "reason": "Profile metrics are within acceptable thresholds",
            "metrics": metrics,
        }
