from __future__ import annotations

import re
from typing import Any, Dict, Optional

from src.orchestrator.orchestrator import run_query
from src.profiles.profile_store import resolve_profile_name


def query_profile(
    *,
    subscription_id: str,
    profile_id: str,
    query: str,
    model_name: Optional[str] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    profile_name = resolve_profile_name(subscription_id=subscription_id, profile_id=profile_id)
    return run_query(
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_name=profile_name,
        query=query,
        model_name=model_name,
        top_k=top_k,
    )

def build_grounded_answer(
    *,
    query: str,
    intent: Any,
    retrieved: list[Dict[str, Any]],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    _ = (query, intent, model_name)
    raw_texts = [str(item.get("text") or "") for item in retrieved if item.get("text")]
    answer = " ".join(raw_texts).strip() if raw_texts else "No grounded evidence found in the retrieved context."
    answer = re.sub(r"document_id\s*=\s*\w+", "[redacted]", answer, flags=re.IGNORECASE)

    citations = []
    for item in retrieved:
        file_name = item.get("file_name") or item.get("source_name") or "document"
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        page_span = ""
        if page_start is not None:
            page_span = f"p{page_start}"
            if page_end and page_end != page_start:
                page_span = f"p{page_start}-{page_end}"
        citation = f"{file_name} {page_span}".strip()
        citation = re.sub(r"document_id\s*=\s*\w+", "[redacted]", citation, flags=re.IGNORECASE)
        citations.append(citation)
    return {"answer": answer, "citations": citations}


__all__ = ["query_profile", "build_grounded_answer"]
