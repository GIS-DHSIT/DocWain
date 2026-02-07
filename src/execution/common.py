from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from src.mode.execution_mode import ExecutionMode


@dataclass
class ExecutionResult:
    answer: Dict[str, Any]
    mode: ExecutionMode
    debug: Dict[str, Any]
    stream: Optional[Iterable[str]] = None


def normalize_answer(answer: Any) -> Dict[str, Any]:
    """Ensure downstream consumers get a consistent answer structure."""
    if isinstance(answer, dict):
        meta = {}
        raw_meta = answer.get("metadata")
        if isinstance(raw_meta, dict):
            meta.update(raw_meta)
        structured = {
            "response": answer.get("response") or answer.get("answer"),
            "sources": answer.get("sources", []),
            "grounded": answer.get("grounded", False),
            "context_found": answer.get("context_found", False),
            "metadata": meta,
        }
        for k, v in answer.items():
            if k in {"response", "answer", "sources", "metadata"}:
                continue
            structured["metadata"][k] = v
        structured["metadata"] = structured.get("metadata") or {}
        return structured
    return {
        "response": str(answer),
        "sources": [],
        "grounded": False,
        "context_found": False,
        "metadata": {},
    }


def chunk_text_stream(text: str, chunk_size: int = 256) -> Iterable[str]:
    if not text:
        yield ""
        return
    for idx in range(0, len(text), chunk_size):
        yield text[idx: idx + chunk_size]
