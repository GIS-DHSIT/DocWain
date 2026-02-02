from __future__ import annotations

from typing import Any, Dict, List

from src.observability.metrics import metrics_store


def rerank_chunks(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    try:
        _ = query
        return chunks
    except Exception:  # noqa: BLE001
        metrics_store().increment("rerank_fail_count")
        return chunks


__all__ = ["rerank_chunks"]
