import os
import time
import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple

DEFAULT_ROLLING_WINDOW = int(os.getenv("METRICS_ROLLING_WINDOW", "200"))
DEFAULT_PER_DOC_LIMIT = int(os.getenv("METRICS_PER_DOC_LIMIT", "50"))
METRICS_V2_ENABLED = os.getenv("METRICS_V2_ENABLED", "true").lower() in {"1", "true", "yes"}


def _label_key(labels: Optional[Dict[str, Any]]) -> Tuple:
    if not labels:
        return tuple()
    return tuple(sorted(labels.items()))


class TelemetryStore:
    """
    Lightweight in-memory telemetry store with rolling timers and per-doc metrics.
    Thread-safe for simple counters/timers.
    """

    def __init__(self, rolling_window: int = DEFAULT_ROLLING_WINDOW, per_doc_limit: int = DEFAULT_PER_DOC_LIMIT):
        self.lock = threading.Lock()
        self.rolling_window = max(1, rolling_window)
        self.per_doc_limit = max(1, per_doc_limit)
        self.counters: Dict[Tuple[str, Tuple], float] = defaultdict(float)
        self.timers: Dict[Tuple[str, Tuple], Deque[float]] = defaultdict(deque)
        self.gauges: Dict[Tuple[str, Tuple], float] = {}
        self.per_doc: Dict[str, Dict[str, Any]] = {}
        self.doc_order: Deque[str] = deque()

    def increment(self, name: str, labels: Optional[Dict[str, Any]] = None, amount: float = 1.0) -> None:
        key = (name, _label_key(labels))
        with self.lock:
            self.counters[key] += amount

    def observe(self, name: str, value_ms: float, labels: Optional[Dict[str, Any]] = None) -> None:
        key = (name, _label_key(labels))
        with self.lock:
            bucket = self.timers[key]
            bucket.append(float(value_ms))
            if len(bucket) > self.rolling_window:
                bucket.popleft()

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
        key = (name, _label_key(labels))
        with self.lock:
            self.gauges[key] = float(value)

    def record_doc_metric(self, doc_id: str, metric_name: str, value: Any) -> None:
        with self.lock:
            if doc_id not in self.per_doc and len(self.per_doc) >= self.per_doc_limit:
                oldest = self.doc_order.popleft()
                self.per_doc.pop(oldest, None)
            if doc_id not in self.per_doc:
                self.per_doc[doc_id] = {}
                self.doc_order.append(doc_id)
            self.per_doc[doc_id][metric_name] = value
            self.per_doc[doc_id]["last_updated"] = time.time()

    def record_metadata_quality(self, doc_id: str, metadata: Dict[str, Any], expected_fields: Optional[list[str]] = None):
        expected = expected_fields or list(metadata.keys())
        present = 0
        missing_fields = []
        for field in expected:
            if metadata.get(field) is not None:
                present += 1
            else:
                missing_fields.append(field)
        expected_count = len(expected)
        completeness = (present / expected_count) if expected_count else 1.0

        with self.lock:
            self.increment("metadata_fields_expected", amount=expected_count)
            self.increment("metadata_fields_present", amount=present)
            if missing_fields:
                for field in missing_fields:
                    self.increment(f"metadata_missing::{field}")
            self.record_doc_metric(doc_id, "metadata_completeness", completeness)
            self.record_doc_metric(doc_id, "metadata_fields_present", present)
            self.record_doc_metric(doc_id, "metadata_fields_expected", expected_count)
            self.record_doc_metric(doc_id, "last_metadata_extraction_time", time.time())

    @staticmethod
    def _timer_stats(values: Deque[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0}
        arr = list(values)
        arr.sort()
        avg = sum(arr) / len(arr)
        idx = int(0.95 * (len(arr) - 1))
        return {"count": len(arr), "avg_ms": avg, "p95_ms": arr[idx]}

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            embedding = {
                "embedding_requests_count": self._get_counter("embedding_requests_count"),
                "total_chunks_embedded": self._get_counter("total_chunks_embedded"),
                "embedding_failures_count": self._get_counter("embedding_failures_count"),
                "last_embedding_time": self.gauges.get(("last_embedding_time", tuple()), None),
            }
            embed_stats = self._timer_stats(self.timers.get(("embedding_latency_ms", tuple()), deque()))
            embedding.update(
                {
                    "avg_embedding_latency_ms": round(embed_stats["avg_ms"], 3),
                    "p95_embedding_latency_ms": round(embed_stats["p95_ms"], 3),
                }
            )

            retrieval = {
                "retrieval_requests_count": self._get_counter("retrieval_requests_count"),
                "retrieval_failures_count": self._get_counter("retrieval_failures_count"),
                "avg_topk_returned": self._avg_timer("retrieval_topk"),
                "avg_context_tokens_returned": self._avg_timer("retrieval_context_tokens"),
                "hit_rate": self._hit_rate(),
                "last_retrieval_time": self.gauges.get(("last_retrieval_time", tuple()), None),
            }
            ret_stats = self._timer_stats(self.timers.get(("retrieval_latency_ms", tuple()), deque()))
            retrieval.update(
                {
                    "avg_retrieval_latency_ms": round(ret_stats["avg_ms"], 3),
                    "p95_retrieval_latency_ms": round(ret_stats["p95_ms"], 3),
                }
            )

            metadata = {
                "metadata_fields_expected": self._get_counter("metadata_fields_expected"),
                "metadata_fields_present_avg": self._avg_counter("metadata_fields_present", "metadata_fields_expected"),
                "metadata_completeness_ratio_avg": self._avg_counter("metadata_fields_present", "metadata_fields_expected"),
                "metadata_parse_failures_count": self._get_counter("metadata_parse_failures_count"),
                "last_metadata_extraction_time": self._latest_doc_field("last_metadata_extraction_time"),
                "per_field_missing_rate": self._per_field_missing_rate(),
            }

            per_doc_snapshot = {}
            for doc_id, metrics in self.per_doc.items():
                per_doc_snapshot[doc_id] = dict(metrics)

            return {
                "embedding": embedding,
                "retrieval": retrieval,
                "metadata_quality": metadata,
                "per_doc": per_doc_snapshot,
            }

    def _get_counter(self, name: str) -> float:
        return self.counters.get((name, tuple()), 0.0)

    def _avg_timer(self, name: str) -> float:
        stats = self._timer_stats(self.timers.get((name, tuple()), deque()))
        return round(stats["avg_ms"], 3)

    def _avg_counter(self, num_name: str, denom_name: str) -> float:
        num = self._get_counter(num_name)
        denom = self._get_counter(denom_name) or 1.0
        return round(num / denom, 4)

    def _hit_rate(self) -> float:
        hits = self._get_counter("retrieval_hits")
        total = self._get_counter("retrieval_requests_count") or 1.0
        return round(hits / total, 4)

    def _latest_doc_field(self, field: str) -> Optional[Any]:
        latest = None
        for doc_metrics in self.per_doc.values():
            if field in doc_metrics:
                if latest is None or doc_metrics.get("last_updated", 0) > latest.get("last_updated", 0):
                    latest = doc_metrics
        if latest:
            return latest.get(field)
        return None

    def _per_field_missing_rate(self) -> Dict[str, float]:
        rates = {}
        for (name, _), value in self.counters.items():
            if not name.startswith("metadata_missing::"):
                continue
            field = name.split("::", 1)[1]
            total = self._get_counter("metadata_fields_expected") or 1.0
            rates[field] = round(value / total, 4)
        return rates


_TELEMETRY = TelemetryStore()


def telemetry_store() -> TelemetryStore:
    return _TELEMETRY
