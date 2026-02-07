from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional, Tuple


DEFAULT_RETENTION_HOURS = int(os.getenv("METRICS_RETENTION_HOURS", "168"))
DEFAULT_LATENCY_BUCKETS_MS = os.getenv("METRICS_LATENCY_BUCKETS_MS", "50,100,250,500,1000,2000,5000,10000")


def _slug(value: Optional[str]) -> str:
    if not value:
        return "default"
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
    cleaned = cleaned.strip("-")
    return cleaned or "default"


def _parse_latency_buckets(raw: str) -> Tuple[int, ...]:
    buckets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            buckets.append(int(part))
        except ValueError:
            continue
    if not buckets:
        buckets = [50, 100, 250, 500, 1000, 2000, 5000, 10000]
    return tuple(sorted(set(buckets)))


def _hour_bucket(ts: Optional[float] = None) -> str:
    return datetime.utcfromtimestamp(ts or time.time()).strftime("%Y%m%d%H")


def _bucket_label(value_ms: float, buckets: Tuple[int, ...]) -> str:
    for edge in buckets:
        if value_ms <= edge:
            return f"le_{edge}"
    return f"gt_{buckets[-1]}"


class AIMetricsStore:
    """Redis-backed metrics store with hourly buckets and lightweight aggregation."""

    def __init__(
        self,
        redis_client: Any,
        retention_hours: int = DEFAULT_RETENTION_HOURS,
        latency_buckets_ms: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.redis = redis_client
        self.retention_seconds = max(1, int(retention_hours)) * 3600
        self.latency_buckets_ms = latency_buckets_ms or _parse_latency_buckets(DEFAULT_LATENCY_BUCKETS_MS)

    @property
    def available(self) -> bool:
        return self.redis is not None

    def _key(self, bucket: str, dims: Iterable[Tuple[str, str]]) -> str:
        key = f"ai:metrics:hour:{bucket}"
        for name, value in dims:
            key += f":{name}:{_slug(value)}"
        return key

    def key_for(
        self,
        bucket: str,
        *,
        document_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> str:
        dims = []
        if document_id:
            dims.append(("document", document_id))
        if model_id:
            dims.append(("model", model_id))
        if agent:
            dims.append(("agent", agent))
        if tool:
            dims.append(("tool", tool))
        return self._key(bucket, dims)

    def _key_variants(
        self,
        bucket: str,
        *,
        document_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> Tuple[str, ...]:
        dims = []
        if document_id:
            dims.append(("document", document_id))
        if model_id:
            dims.append(("model", model_id))
        if agent:
            dims.append(("agent", agent))
        if tool:
            dims.append(("tool", tool))

        keys = [self._key(bucket, [])]
        if not dims:
            return tuple(keys)

        total = 1 << len(dims)
        for mask in range(1, total):
            subset = [dims[i] for i in range(len(dims)) if mask & (1 << i)]
            keys.append(self._key(bucket, subset))
        return tuple(keys)

    def _expire_keys(self, pipe: Any, keys: Iterable[str]) -> None:
        for key in keys:
            pipe.expire(key, self.retention_seconds)

    def record(
        self,
        *,
        counters: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, float]] = None,
        distributions: Optional[Dict[str, Dict[str, float]]] = None,
        histograms: Optional[Dict[str, float]] = None,
        minmax: Optional[Dict[str, float]] = None,
        ts: Optional[float] = None,
        document_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> None:
        if not self.redis or not hasattr(self.redis, "pipeline"):
            return
        bucket = _hour_bucket(ts)
        keys = self._key_variants(
            bucket,
            document_id=document_id,
            model_id=model_id,
            agent=agent,
            tool=tool,
        )
        pipe = self.redis.pipeline()
        counters = counters or {}
        values = values or {}
        distributions = distributions or {}
        histograms = histograms or {}
        for key in keys:
            for name, amount in counters.items():
                pipe.hincrbyfloat(key, f"count:{name}", float(amount))
            for name, value in values.items():
                pipe.hincrbyfloat(key, f"sum:{name}", float(value))
                pipe.hincrby(key, f"count:{name}", 1)
            for name, bins in distributions.items():
                for label, amount in bins.items():
                    pipe.hincrbyfloat(key, f"dist:{name}:{_slug(label)}", float(amount))
            for name, value in histograms.items():
                if value is None:
                    continue
                label = _bucket_label(float(value), self.latency_buckets_ms)
                pipe.hincrby(key, f"hist:{name}:{label}", 1)
        self._expire_keys(pipe, keys)
        pipe.execute()

        if minmax:
            for key in keys:
                for name, value in minmax.items():
                    self._update_minmax(key, name, float(value))

    def _update_minmax(self, key: str, name: str, value: float) -> None:
        if not self.redis:
            return
        min_field = f"min:{name}"
        max_field = f"max:{name}"
        current_min = self.redis.hget(key, min_field)
        current_max = self.redis.hget(key, max_field)
        if current_min is None or value < float(current_min):
            self.redis.hset(key, min_field, value)
        if current_max is None or value > float(current_max):
            self.redis.hset(key, max_field, value)

    def aggregate(
        self,
        *,
        window_hours: int,
        document_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.redis:
            return {"available": False, "reason": "redis_unavailable"}
        window_hours = max(1, int(window_hours))
        totals: Dict[str, float] = defaultdict(float)
        mins: Dict[str, float] = {}
        maxs: Dict[str, float] = {}
        dists: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        hists: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        now = datetime.utcnow()
        for offset in range(window_hours):
            bucket = (now - timedelta(hours=offset)).strftime("%Y%m%d%H")
            key = self.key_for(
                bucket,
                document_id=document_id,
                model_id=model_id,
                agent=agent,
                tool=tool,
            )
            data = self.redis.hgetall(key) or {}
            for field, raw in data.items():
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if field.startswith("min:"):
                    name = field.split("min:", 1)[1]
                    if name not in mins or value < mins[name]:
                        mins[name] = value
                    continue
                if field.startswith("max:"):
                    name = field.split("max:", 1)[1]
                    if name not in maxs or value > maxs[name]:
                        maxs[name] = value
                    continue
                if field.startswith("dist:"):
                    _, rest = field.split("dist:", 1)
                    name, label = rest.split(":", 1)
                    dists[name][_slug(label)] += value
                    continue
                if field.startswith("hist:"):
                    _, rest = field.split("hist:", 1)
                    name, label = rest.split(":", 1)
                    hists[name][label] += value
                    continue
                totals[field] += value

        return {
            "available": True,
            "totals": totals,
            "mins": mins,
            "maxs": maxs,
            "dists": {k: dict(v) for k, v in dists.items()},
            "hists": {k: dict(v) for k, v in hists.items()},
        }

    def _avg(self, totals: Dict[str, float], name: str) -> float:
        total = totals.get(f"sum:{name}", 0.0)
        count = totals.get(f"count:{name}", 0.0)
        if not count:
            return 0.0
        return total / count

    def _rate(self, totals: Dict[str, float], numer: str, denom: str) -> float:
        num = totals.get(f"count:{numer}", 0.0)
        den = totals.get(f"count:{denom}", 0.0) or 0.0
        if not den:
            return 0.0
        return num / den

    def _hist_quantile(self, hist: Dict[str, float], quantile: float) -> float:
        if not hist:
            return 0.0
        buckets = list(self.latency_buckets_ms)
        labels = [f"le_{edge}" for edge in buckets] + [f"gt_{buckets[-1]}"]
        counts = [hist.get(label, 0.0) for label in labels]
        total = sum(counts)
        if total <= 0:
            return 0.0
        target = total * quantile
        cumulative = 0.0
        for label, edge, count in zip(labels, buckets + [buckets[-1]], counts):
            cumulative += count
            if cumulative >= target:
                return float(edge)
        return float(buckets[-1])

    def build_snapshot(
        self,
        *,
        window_hours: int,
        document_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> Dict[str, Any]:
        aggregated = self.aggregate(
            window_hours=window_hours,
            document_id=document_id,
            model_id=model_id,
            agent=agent,
            tool=tool,
        )
        if not aggregated.get("available"):
            return aggregated

        totals = aggregated["totals"]
        mins = aggregated["mins"]
        maxs = aggregated["maxs"]
        dists = aggregated["dists"]
        hists = aggregated["hists"]

        tool_calls = dists.get("tool_usage", {})
        tool_total = sum(tool_calls.values()) or 0.0
        tool_usage_pct = {
            k: round((v / tool_total) * 100, 2) for k, v in tool_calls.items()
        } if tool_total else {}

        model_usage = dists.get("model_usage", {})
        failure_types = dists.get("failure_type", {})
        agent_dist = dists.get("agent_execution", {})

        request_latency_hist = hists.get("request_latency_ms", {})
        llm_latency_hist = hists.get("llm_latency_ms", {})
        tool_latency_hist = hists.get("tool_latency_ms", {})

        prompt_total = totals.get("sum:prompt_tokens", 0.0)
        completion_total = totals.get("sum:completion_tokens", 0.0)
        token_efficiency = 0.0
        if (prompt_total + completion_total) > 0:
            token_efficiency = completion_total / (prompt_total + completion_total)

        return {
            "available": True,
            "window_hours": window_hours,
            "filters": {
                "document_id": document_id,
                "model_id": model_id,
                "agent": agent,
                "tool": tool,
            },
            "document_extraction": {
                "text_extraction_accuracy_pct": round(self._avg(totals, "text_extraction_accuracy_pct") * 100, 2),
                "ocr_confidence_avg": round(self._avg(totals, "ocr_confidence"), 2),
                "ocr_confidence_min": round(mins.get("ocr_confidence", 0.0), 2),
                "ocr_confidence_max": round(maxs.get("ocr_confidence", 0.0), 2),
                "structure_extraction_accuracy_pct": round(
                    self._avg(totals, "structure_extraction_accuracy_pct") * 100, 2
                ),
                "missing_content_ratio": round(self._avg(totals, "missing_content_ratio"), 4),
                "corrupted_page_ratio": round(self._avg(totals, "corrupted_page_ratio"), 4),
            },
            "semantic_quality": {
                "sentence_transformation_similarity_score": round(
                    self._avg(totals, "sentence_transformation_similarity_score"), 4
                ),
                "semantic_preservation_score": round(self._avg(totals, "semantic_preservation_score"), 4),
                "embedding_coherence_score": round(self._avg(totals, "embedding_coherence_score"), 4),
                "chunk_semantic_drift_score": round(self._avg(totals, "chunk_semantic_drift_score"), 4),
                "hallucination_risk_score": round(self._avg(totals, "hallucination_risk_score"), 4),
            },
            "retrieval": {
                "retrieval_accuracy_pct": round(self._rate(totals, "retrieval_grounded", "retrieval_total") * 100, 2),
                "top_k_hit_rate": round(self._rate(totals, "retrieval_hits", "retrieval_total"), 4),
                "mean_reciprocal_rank": round(self._avg(totals, "mean_reciprocal_rank"), 4),
                "context_grounding_score": round(self._avg(totals, "context_grounding_score"), 4),
                "answer_faithfulness_score": round(self._avg(totals, "answer_faithfulness_score"), 4),
                "answer_relevance_score": round(self._avg(totals, "answer_relevance_score"), 4),
            },
            "tool_usage": {
                "tool_usage_distribution_pct": tool_usage_pct,
                "tool_success_rate": round(self._rate(totals, "tool_success", "tool_calls"), 4),
                "tool_failure_rate": round(self._rate(totals, "tool_failure", "tool_calls"), 4),
                "agent_execution_distribution": agent_dist,
                "avg_tool_latency_ms": round(self._avg(totals, "tool_latency_ms"), 2),
                "llm_without_tool_fallback_rate": round(
                    self._rate(totals, "llm_without_tool_fallback", "requests_with_tools"), 4
                ),
            },
            "llm_performance": {
                "prompt_tokens_avg": round(self._avg(totals, "prompt_tokens"), 2),
                "completion_tokens_avg": round(self._avg(totals, "completion_tokens"), 2),
                "token_efficiency_score": round(token_efficiency, 4),
                "latency_p50_ms": round(self._hist_quantile(llm_latency_hist, 0.50), 2),
                "latency_p95_ms": round(self._hist_quantile(llm_latency_hist, 0.95), 2),
                "latency_p99_ms": round(self._hist_quantile(llm_latency_hist, 0.99), 2),
                "model_usage_breakdown": model_usage,
                "response_consistency_score": round(self._avg(totals, "response_consistency_score"), 4),
                "temperature_sensitivity_score": round(self._avg(totals, "temperature_sensitivity_score"), 4),
            },
            "system_health": {
                "end_to_end_success_rate": round(self._rate(totals, "requests_success", "requests_total"), 4),
                "ingest_to_answer_latency_ms": round(self._avg(totals, "request_latency_ms"), 2),
                "failure_type_distribution": failure_types,
                "cold_start_vs_warm_start_latency": {
                    "cold_start_avg_ms": round(self._avg(totals, "cold_start_latency_ms"), 2),
                    "warm_start_avg_ms": round(self._avg(totals, "warm_start_latency_ms"), 2),
                },
                "retry_rate": round(self._rate(totals, "llm_retry_count", "llm_request_count"), 4),
            },
            "latency_distributions": {
                "request_latency_ms": request_latency_hist,
                "tool_latency_ms": tool_latency_hist,
            },
        }


_STORE: Optional[AIMetricsStore] = None


def get_metrics_store(redis_client: Optional[Any] = None) -> AIMetricsStore:
    global _STORE
    if _STORE is None or (redis_client is not None and _STORE.redis is not redis_client):
        if redis_client is None:
            try:
                from src.api.dw_newron import get_redis_client
            except Exception:
                redis_client = None
            else:
                redis_client = get_redis_client()
        _STORE = AIMetricsStore(redis_client)
    return _STORE
