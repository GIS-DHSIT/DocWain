"""Pipeline quality metrics — tracks latency, grounding, judge verdicts per query.

Usage:
    from src.metrics.quality_metrics import record_query_metrics, get_quality_summary

    record_query_metrics(QueryMetrics(
        query="summarize Dhayal's profile",
        latency_ms=1200.0,
        retrieval_count=8,
        ...
    ))
    summary = get_quality_summary(hours=24)
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

_MAX_METRICS_BUFFER = 2000

@dataclass
class QueryMetrics:
    query: str
    latency_ms: float = 0.0
    retrieval_count: int = 0
    entity_detected: bool = False
    llm_backend: str = "unknown"
    llm_latency_ms: float = 0.0
    answer_length: int = 0
    grounding_score: float = 0.0
    judge_verdict: str = "unknown"
    scope_mode: str = "unknown"
    domain: str = "unknown"
    timestamp: float = field(default_factory=time.time)

_BUFFER: deque = deque(maxlen=_MAX_METRICS_BUFFER)
_LOCK = threading.Lock()

def record_query_metrics(metrics: QueryMetrics) -> None:
    with _LOCK:
        _BUFFER.append(metrics)

def get_recent_metrics(hours: int = 24) -> List[QueryMetrics]:
    cutoff = time.time() - (hours * 3600)
    with _LOCK:
        return [m for m in _BUFFER if m.timestamp >= cutoff]

def get_quality_summary(hours: int = 24) -> Dict[str, Any]:
    """Aggregate quality stats over the last N hours."""
    recent = get_recent_metrics(hours)
    if not recent:
        return {
            "queries_count": 0,
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "avg_retrieval_count": 0,
            "entity_detection_rate": 0,
            "grounding_avg": 0,
            "judge_pass_rate": 0,
            "llm_backends": {},
            "scope_modes": {},
        }

    latencies = sorted(m.latency_ms for m in recent)
    n = len(latencies)
    p50 = latencies[n // 2] if n else 0
    p95 = latencies[int(n * 0.95)] if n else 0

    llm_latencies = [m.llm_latency_ms for m in recent if m.llm_latency_ms > 0]

    grounding_scores = [m.grounding_score for m in recent if m.grounding_score > 0]
    judge_passed = sum(1 for m in recent if m.judge_verdict == "pass")

    entity_detected = sum(1 for m in recent if m.entity_detected)

    backends: Dict[str, int] = {}
    scopes: Dict[str, int] = {}
    for m in recent:
        backends[m.llm_backend] = backends.get(m.llm_backend, 0) + 1
        scopes[m.scope_mode] = scopes.get(m.scope_mode, 0) + 1

    return {
        "queries_count": n,
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "avg_retrieval_count": round(sum(m.retrieval_count for m in recent) / n, 1),
        "entity_detection_rate": round(entity_detected / n, 3) if n else 0,
        "grounding_avg": round(sum(grounding_scores) / len(grounding_scores), 3) if grounding_scores else 0,
        "judge_pass_rate": round(judge_passed / n, 3) if n else 0,
        "avg_llm_latency_ms": round(sum(llm_latencies) / len(llm_latencies), 1) if llm_latencies else 0,
        "llm_backends": backends,
        "scope_modes": scopes,
    }

__all__ = ["QueryMetrics", "record_query_metrics", "get_quality_summary", "get_recent_metrics"]
