from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.api.config import Config
from src.api.dataHandler import encode_with_fallback
from src.api.vector_store import build_collection_name
from src.retrieval.intent_router import analyze_intent, IntentResult

logger = logging.getLogger(__name__)

_INTERNAL_ID_RE = re.compile(
    r"\b(?:subscription_id|profile_id|document_id|chunk_id|point_id|qdrant_id)\s*[:=]\s*[A-Za-z0-9_-]{6,}\b",
    re.IGNORECASE,
)


def build_profile_filter(subscription_id: str, profile_id: str, doc_types: Optional[List[str]]) -> Filter:
    must = [
        FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
        FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
    ]
    if doc_types:
        must.append(FieldCondition(key="document_type", match=MatchAny(any=doc_types)))
    return Filter(must=must)


def retrieve_profile_chunks(
    *,
    subscription_id: str,
    profile_id: str,
    query: str,
    intent: IntentResult,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
    collection = build_collection_name(subscription_id)

    query_vec = encode_with_fallback([query], normalize_embeddings=True, convert_to_numpy=False)[0]
    q_filter = build_profile_filter(subscription_id, profile_id, intent.target_doc_types)

    results = client.search(
        collection_name=collection,
        query_vector=("content_vector", list(query_vec)),
        query_filter=q_filter,
        limit=top_k,
    )

    hits: List[Dict[str, Any]] = []
    for hit in results:
        payload = hit.payload or {}
        hits.append(
            {
                "text": payload.get("text") or "",
                "file_name": payload.get("file_name") or payload.get("filename") or payload.get("source_file"),
                "section_title": payload.get("section_title") or "",
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "chunk_kind": payload.get("chunk_kind") or payload.get("chunk_type"),
            }
        )
    return hits


def build_grounded_answer(
    *,
    query: str,
    intent: IntentResult,
    retrieved: List[Dict[str, Any]],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not retrieved:
        return {
            "answer": "Not found in profile documents. Consider uploading relevant document types or sections.",
            "citations": [],
        }

    citations = []
    context_parts = []
    for hit in retrieved:
        file_name = hit.get("file_name") or "Unknown"
        section_title = hit.get("section_title") or "Section"
        page_start = hit.get("page_start")
        page_end = hit.get("page_end") or page_start
        page_range = f"{page_start}-{page_end}" if page_start is not None else "N/A"
        citation = f"[Source: {file_name}, Section: {section_title}, Page: {page_range}]"
        citations.append(citation)
        context_parts.append(f"{hit.get('text')}\n{citation}")

    context = "\n\n".join(context_parts)

    if model_name and _ollama_available():
        answer = _generate_with_ollama(query, intent, context, model_name)
    else:
        answer = _fallback_answer(intent, context)

    answer = _sanitize_output(answer)
    citations = [_sanitize_output(citation) for citation in citations]

    return {"answer": answer, "citations": citations}


def _fallback_answer(intent: IntentResult, context: str) -> str:
    if intent.intent == "summarize":
        return f"Summary based on profile documents:\n{context}"
    return f"Answer based on profile documents:\n{context}"


def _ollama_available() -> bool:
    return True


def _generate_with_ollama(query: str, intent: IntentResult, context: str, model_name: str) -> str:
    import ollama  # noqa: WPS433

    prompt = (
        "You are DocWain. Answer the user using ONLY the provided context. "
        "Do not mention internal IDs. If the answer is not supported, say "
        "'Not found in profile documents.'\n\n"
        f"Intent: {intent.intent}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    resp = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0})
    return (resp.get("response") or "").strip()


def _sanitize_output(text: str) -> str:
    if not text:
        return ""
    return _INTERNAL_ID_RE.sub("[redacted]", text)


def query_profile(
    *,
    subscription_id: str,
    profile_id: str,
    query: str,
    model_name: Optional[str] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    intent = analyze_intent(query, model_name=model_name)
    retrieved = retrieve_profile_chunks(
        subscription_id=subscription_id,
        profile_id=profile_id,
        query=query,
        intent=intent,
        top_k=top_k,
    )
    response = build_grounded_answer(query=query, intent=intent, retrieved=retrieved, model_name=model_name)
    return {
        "intent": {
            "intent": intent.intent,
            "target_doc_types": intent.target_doc_types,
            "constraints": intent.constraints,
            "need_tables": intent.need_tables,
            "source": intent.source,
        },
        **response,
    }


__all__ = ["query_profile", "retrieve_profile_chunks", "build_grounded_answer", "build_profile_filter"]
