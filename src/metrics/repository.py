from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any, Dict, Optional

from src.metrics.ai_metrics import AIMetricsStore, get_metrics_store
from src.metrics.aggregation import iter_hour_keys


def _parse_bucket_data(raw: Dict[str, Any]) -> Dict[str, Any]:
    totals: Dict[str, float] = defaultdict(float)
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    dists: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hists: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for field, value in (raw or {}).items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if field.startswith("min:"):
            name = field.split("min:", 1)[1]
            if name not in mins or numeric < mins[name]:
                mins[name] = numeric
            continue
        if field.startswith("max:"):
            name = field.split("max:", 1)[1]
            if name not in maxs or numeric > maxs[name]:
                maxs[name] = numeric
            continue
        if field.startswith("dist:"):
            _, rest = field.split("dist:", 1)
            name, label = rest.split(":", 1)
            dists[name][label] += numeric
            continue
        if field.startswith("hist:"):
            _, rest = field.split("hist:", 1)
            name, label = rest.split(":", 1)
            hists[name][label] += numeric
            continue
        totals[field] += numeric

    return {
        "totals": totals,
        "mins": mins,
        "maxs": maxs,
        "dists": dists,
        "hists": hists,
    }


class MetricsRepository:
    """Repository interface for reading stored metrics for a time range."""

    def __init__(self, store: Optional[AIMetricsStore] = None) -> None:
        self.store = store or get_metrics_store()
        self.redis = self.store.redis

    def fetch_hourly(
        self,
        start_utc: dt.datetime,
        end_utc: dt.datetime,
        *,
        document_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not self.redis:
            return {}
        hour_keys = list(iter_hour_keys(start_utc, end_utc))
        redis_keys = [
            self.store.key_for(
                hour_key,
                document_id=document_id,
                model_id=model_id,
                agent=agent,
                tool=tool,
            )
            for hour_key in hour_keys
        ]
        pipe = self.redis.pipeline()
        for key in redis_keys:
            pipe.hgetall(key)
        raw_results = pipe.execute()
        hourly: Dict[str, Dict[str, Any]] = {}
        for hour_key, raw in zip(hour_keys, raw_results):
            if raw:
                hourly[hour_key] = _parse_bucket_data(raw)
        return hourly
