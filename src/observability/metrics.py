from __future__ import annotations

import threading
from collections import defaultdict
from typing import DefaultDict


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._gauges: DefaultDict[str, float] = defaultdict(float)

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe_ms(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._gauges[name] = value_ms

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }


_METRICS = MetricsStore()


def metrics_store() -> MetricsStore:
    return _METRICS


__all__ = ["metrics_store", "MetricsStore"]
