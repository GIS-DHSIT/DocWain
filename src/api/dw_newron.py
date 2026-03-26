import re
import json
import time
import threading
import logging

from src.utils.logging_utils import get_logger
import hashlib
import concurrent.futures
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque, Counter
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from fastapi import HTTPException, status
from qdrant_client.models import Distance, PointStruct, SparseVector, Range
import warnings
warnings.filterwarnings("ignore", message=r".*_target_device.*has been deprecated", category=FutureWarning)
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import os
import redis
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from src.api.config import Config
from src.api.vector_store import QdrantVectorStore
from src.api.qdrant_indexes import REQUIRED_PAYLOAD_INDEX_FIELDS, autoheal_missing_index, ensure_payload_indexes
from src.api.enhanced_context_builder import IntelligentContextBuilder
from src.api.query_intelligence import QueryIntelligence
from src.api.enhanced_retrieval import GraphGuidedRetriever
from src.api.reasoning_layer import EvidencePlanner, AnswerVerifier, ConfidenceScorer
from src.api.learning_signals import LearningSignalStore
from src.agentic.memory import AgentMemory
# Legacy agentic imports removed — handled by ReasoningEngine intelligence pipeline
from src.chat.companion_classifier import CompanionClassifier
from src.chat.opener_generator import contains_banned_opener, generate_opener
from src.kg.neo4j_store import Neo4jStore
from src.kg.retrieval import GraphAugmenter, GraphHints
from src.kg.score import GraphSupportScorer
from src.intelligence.retrieval import run_intelligent_pipeline
from src.utils.redis_cache import RedisJsonCache, hash_query, stamp_cache_payload
from src.utils.payload_utils import get_canonical_text, get_content_text, get_embedding_text, get_source_name
from src.agentic.retriever_manager import RetrieverManager
from src.agentic.verification_agent import VerificationAgent
from src.services.retrieval import (
    QueryUnderstanding,
    HybridRetriever,
    HybridRetrieverConfig,
    Reranker,
    RerankerConfig,
    RetrievalConfidenceScorer,
)
from src.services.retrieval.score_utils import RerankShapeError, normalize_scores, to_py_scalar
from src.retrieval import (
    QueryAnalyzer,
    EvidenceConstraints,
    EvidenceRequirements,
    HybridRanker,
    HybridRankerConfig,
    RetrievalQualityScorer,
    ContextAssembler,
    FallbackRepair,
    EvidenceSynthesizer,
    extract_required_attributes,
    filter_chunks_by_intent,
    extract_answer_requirements,
    validate_answer_requirements,
    build_intent_miss_response,
)
from src.retrieval.errors import RetrievalFilterError
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from sklearn.feature_extraction.text import HashingVectorizer
from src.api.vector_store import build_collection_name
from src.retrieval.filter_builder import build_qdrant_filter
from src.api.genai_client import generate_text, get_genai_client
from src.finetune import resolve_model_for_profile
from src.metrics.ai_metrics import get_metrics_store
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store
from src.security.response_sanitizer import sanitize_user_payload
from src.embedding.pipeline.chunk_integrity import is_valid_chunk_text
from src.quality.bad_answer_evaluator import BadAnswerEvaluator, EvalConfig
from src.quality.auto_repair import AutoRepairEngine, RepairConfig
from src.quality.telemetry import emit_quality_telemetry
from src.retrieval.profile_document_index import build_profile_document_index
from src.retrieval.retrieval_planner import RetrievalPlanner
from src.intent.llm_intent import parse_intent
from src.intelligence.conversation_state import ConversationState, ProgressiveSummarizer
from src.generation.prompts import build_system_prompt

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

logger = get_logger(__name__)

STOP_TOKENS = {
    "<|endoftext|>",
    "<|eot_id|>",
    "</s>",
    "<|end|>",
}

def _resolve_model_alias(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return model_name
    normalized = str(model_name).strip().lower()
    # Strip Ollama tag suffix (e.g. "gpt-oss:latest" → "gpt-oss")
    base_name = normalized.split(":")[0]
    if base_name in ("docwain-agent", "gpt-oss", "dhs/docwain"):
        return "DHS/DocWain"
    return model_name

def _is_generation_empty(text: Optional[str], stop_tokens: Optional[Set[str]] = None) -> bool:
    if text is None:
        return True
    cleaned = str(text).strip()
    if not cleaned:
        return True
    tokens = stop_tokens or STOP_TOKENS
    if cleaned in tokens:
        return True
    for token in tokens:
        if token and cleaned.replace(token, "").strip() == "":
            return True
    return False

def _sanitize_snippet(text: Optional[str], limit: int = 80) -> str:
    sample = re.sub(r"\s+", " ", (text or "").strip())
    if len(sample) > limit:
        sample = sample[:limit].rstrip() + "..."
    return sample

def _collect_documents_seen(chunks: List[Any]) -> List[str]:
    names: List[str] = []
    seen = set()
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or {}
        name = get_source_name(meta)
        if not name:
            continue
        base = os.path.basename(str(name))
        if base and base not in seen:
            names.append(base)
            seen.add(base)
    return names

def _coerce_page_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:
        logger.debug("Failed to coerce page number from %r", value, exc_info=True)
        return None

def _build_evidence_packets_from_chunks(
    chunks: List[Any],
    *,
    max_excerpts_per_file: Optional[int] = None,
    excerpt_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    if max_excerpts_per_file is None:
        max_excerpts_per_file = int(getattr(Config.Retrieval, "EVIDENCE_SYNTHESIZER_MAX_EXCERPTS_PER_FILE", 6))
    if excerpt_chars is None:
        excerpt_chars = int(getattr(Config.Retrieval, "EVIDENCE_SYNTHESIZER_EXCERPT_CHARS", 800))

    packets: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    seen: Set[Tuple[str, Optional[int], str]] = set()

    for chunk in chunks:
        meta = getattr(chunk, "metadata", None) or {}
        source = get_source_name(meta) or getattr(chunk, "source", None) or meta.get("source") or meta.get("file_name")
        if not source:
            continue
        file_name = os.path.basename(str(source))
        if not file_name:
            continue
        if file_name not in packets:
            packets[file_name] = {"file_name": file_name, "excerpts": []}
            order.append(file_name)
            doc_domain = meta.get("doc_domain") or meta.get("document_type") or meta.get("doc_type")
            if doc_domain:
                packets[file_name]["doc_domain"] = str(doc_domain)
        if max_excerpts_per_file and len(packets[file_name]["excerpts"]) >= max_excerpts_per_file:
            continue
        text = get_content_text(meta) or getattr(chunk, "text", "") or ""
        text = str(text or "").strip()
        if not text:
            continue
        if excerpt_chars and len(text) > excerpt_chars:
            text = text[:excerpt_chars].rstrip() + "..."
        page = meta.get("page")
        if page is None:
            page = meta.get("page_start")
        if page is None:
            page = meta.get("page_end")
        page = _coerce_page_number(page)
        dedupe_key = (file_name, page, text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        excerpt_payload = {"text": text}
        if page is not None:
            excerpt_payload["page"] = page
        packets[file_name]["excerpts"].append(excerpt_payload)

    return [packets[name] for name in order if packets[name]["excerpts"]]

def _collect_profile_documents(
    subscription_id: str,
    profile_id: str,
    redis_client: Optional[Any],
    target_doc_ids: Optional[List[str]] = None,
) -> List[str]:
    try:
        from src.intelligence.redis_intel_cache import RedisIntelCache

        cache = RedisIntelCache(redis_client)
        catalog = cache.get_json(cache.catalog_key(subscription_id, profile_id)) or {}
        docs = catalog.get("documents") or []
        if target_doc_ids:
            target_set = {str(d) for d in target_doc_ids if d}
            docs = [doc for doc in docs if str(doc.get("document_id")) in target_set]
        docs = [doc.get("source_name") for doc in docs if doc.get("source_name")]
        return [str(doc) for doc in docs if doc]
    except Exception:
        return []

def _filter_invalid_retrieved_chunks(
    chunks: List[Any],
    *,
    min_chars: int,
    min_tokens: int,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    valid: List[Any] = []
    invalid_samples: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        if is_valid_chunk_text(text, min_chars=min_chars, min_tokens=min_tokens):
            valid.append(chunk)
        else:
            meta = getattr(chunk, "metadata", None) or {}
            invalid_samples.append(
                {
                    "chunk_id": meta.get("chunk_id") or getattr(chunk, "id", None),
                    "length": len(text or ""),
                    "sample": _sanitize_snippet(text),
                }
            )
    return valid, invalid_samples

def _build_retrieval_empty_text_response(
    *,
    query: str,
    user_id: str,
    collection_name: str,
    request_id: Optional[str],
    index_version: Optional[str],
    preprocessing_metadata: Dict[str, Any],
    retrieval_attempts: List[Dict[str, Any]],
    selected_strategy: str,
    profile_context_data: Dict[str, Any],
    documents_seen: List[str],
    processing_time: float,
) -> Dict[str, Any]:
    likely_causes = [
        "OCR/extraction produced empty text",
        "chunking produced empty payload.text",
        "preprocessing removed all tokens",
        "wrong payload key used for context",
    ]
    next_actions = [
        "Verify extraction output contains resume body text (not only filename).",
        "Verify Qdrant payload key is `text` and is populated.",
        "Check chunking thresholds and ensure MIN_CHARS >= 50 for saved chunks.",
        "Re-run extraction+embed for affected docs.",
    ]
    response_text = (
        "I couldn’t access extracted content for these documents (retrieval returned empty text). "
        "Diagnostic code: RETRIEVAL_EMPTY_TEXT. "
        "Next steps: "
        + " ".join(next_actions)
    )
    return {
        "status": "RETRIEVAL_EMPTY_TEXT",
        "response": response_text,
        "sources": [],
        "user_id": user_id,
        "collection": collection_name,
        "request_id": request_id,
        "index_version": index_version,
        "context_found": False,
        "query_type": "retrieval_empty_text",
        "preprocessing": preprocessing_metadata,
        "retrieval_attempts": retrieval_attempts,
        "selected_strategy": selected_strategy,
        "profile_context": profile_context_data,
        "grounded": False,
        "processing_time": processing_time,
        "likely_causes": likely_causes,
        "next_actions": next_actions,
        "documents_seen": documents_seen,
        "retrieval_stats": {"initial_retrieved": 0, "final_context": 0},
    }

def _build_retrieval_filter_error_response(
    *,
    query: str,
    user_id: str,
    collection_name: str,
    request_id: Optional[str],
    index_version: Optional[str],
    details: Optional[str],
    error_code: str = "RETRIEVAL_FILTER_FAILED",
    documents_searched: Optional[List[str]] = None,
) -> Dict[str, Any]:
    documents_searched = documents_searched or []
    missing_field = None
    if details and "missing_index=" in details:
        try:
            missing_field = details.split("missing_index=", 1)[1].split()[0].strip()
        except Exception:
            missing_field = None
    if error_code == "RETRIEVAL_INDEX_MISSING":
        field_label = missing_field or "required field"
        message = (
            f"I couldn’t retrieve evidence due to an indexing issue ({field_label} not indexed)."
        )
    elif error_code == "RETRIEVAL_QDRANT_UNAVAILABLE":
        message = "I couldn’t retrieve evidence because the document index is unavailable."
    else:
        message = "Profile isolation enforced; cannot search outside profile."
    return {
        "ok": False,
        "error_code": error_code,
        "message": message,
        "details": details or "",
        "response": message,
        "sources": [],
        "user_id": user_id,
        "collection": collection_name,
        "request_id": request_id,
        "correlation_id": request_id,
        "index_version": index_version,
        "context_found": False,
        "query_type": "retrieval_filter_failed",
        "grounded": False,
        "documents_searched": documents_searched,
    }

def _requires_detailed_summary(query: str) -> bool:
    lowered = (query or "").lower()
    summary_cues = ("summarize", "summary", "summarise", "overview", "key points", "recap", "compare")
    return any(cue in lowered for cue in summary_cues)

def _extract_requested_fields(query: str) -> List[str]:
    if not query:
        return []
    lowered = query.lower()
    patterns = [
        r"fields?\s*[:\-]\s*(.+)",
        r"extract\s+(.+)",
        r"provide\s+(.+)",
        r"include\s+(.+)",
        r"list\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            raw = match.group(1)
            raw = re.split(r"\bfrom\b|\bin\b|\bfor\b|\busing\b|\bof\b", raw)[0]
            parts = re.split(r",|/|;|\band\b", raw)
            fields = []
            for part in parts:
                item = part.strip(" .:-\t\n")
                if item and len(item) > 1:
                    fields.append(item)
            return fields
    return []

def build_summary_from_chunks(query: str, chunks: List[Any]) -> str:
    if not chunks:
        return "I couldn't find enough document context to summarize."

    lowered = (query or "").lower()
    name_match = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})'s\b", query or "")
    target_name = name_match.group(1).strip() if name_match else None

    filtered = list(chunks)
    if target_name:
        name_filtered = [
            chunk
            for chunk in filtered
            if str((chunk.metadata or {}).get("candidate_name") or "").lower() == target_name.lower()
            or target_name.lower() in (chunk.text or "").lower()
        ]
        if name_filtered:
            filtered = name_filtered

    if "profile" in lowered or "resume" in lowered or "candidate" in lowered:
        resume_filtered = [
            chunk
            for chunk in filtered
            if str((chunk.metadata or {}).get("doc_type") or "").lower() in {"resume", "cv"}
            or str((chunk.metadata or {}).get("doc_domain") or "").lower() == "resume"
            or "resume" in ((chunk.metadata or {}).get("source_name") or "").lower()
        ]
        if resume_filtered:
            filtered = resume_filtered

    if not filtered:
        return "I couldn't find enough document context to summarize."

    sentences: List[str] = []
    seen = set()
    for chunk in filtered:
        text = get_content_text(chunk.metadata or {}) or chunk.text or ""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        for part in parts:
            clean = part.strip()
            if not clean or clean.lower() in seen:
                continue
            seen.add(clean.lower())
            sentences.append(clean)
            if len(sentences) >= 3:
                break
        if len(sentences) >= 3:
            break

    summary = " ".join(sentences).strip()
    if target_name and target_name.lower() not in summary.lower():
        summary = f"{target_name}: {summary}"
    return summary or "I couldn't find enough document context to summarize."

def sanitize_response_obj(response_obj: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = sanitize_user_payload(response_obj or {})
    if not isinstance(cleaned, dict):
        return response_obj

    response_text = cleaned.get("response")
    if isinstance(response_text, str):
        response_text = re.sub(r"\[SOURCE[^\]]*\]", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"(?i)citations?:", "", response_text)
        response_text = re.sub(r"/(?:[\w\-.]+/)+[\w\-.]+", "", response_text)
        # Strip AI disclaimer sentences that leak from LLM responses
        # Only match "As an AI" when followed by disclaimer patterns, not job titles like "AI Engineer"
        response_text = re.sub(r"\bAs an AI(?:\s+(?:language\s+model|assistant|model|chatbot|system))\b[^.]*\.", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"\bAs a language model\b[^.]*\.", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"\bI(?:'m| am) (?:just )?an? AI(?:\s+(?:language\s+model|assistant|model|chatbot))\b[^.]*\.", "", response_text, flags=re.IGNORECASE)
        # Strip extraction placeholder phrases that leak through rendering
        response_text = re.sub(r"Not explicitly mentioned(?:\s+in\s+documents)?\.?", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"MISSING_REASON", "", response_text)
        lines = response_text.splitlines()
        cleaned_lines: List[str] = []
        for line in lines:
            cleaned_lines.append(re.sub(r"[ \t]+", " ", line).strip())
        normalized: List[str] = []
        blank = False
        for line in cleaned_lines:
            if line == "":
                if normalized and not blank:
                    normalized.append("")
                blank = True
                continue
            blank = False
            normalized.append(line)
        response_text = "\n".join(normalized).strip()
        cleaned["response"] = response_text

    sources = cleaned.get("sources")
    if isinstance(sources, list):
        normalized_sources = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            entry = {}
            name = src.get("source_name") or src.get("source")
            if name:
                entry["source_name"] = os.path.basename(str(name))
            page = src.get("page")
            if page is not None:
                entry["page"] = page
            if entry:
                normalized_sources.append(entry)
        cleaned["sources"] = normalized_sources

    for key in ("request_id", "collection", "user_id"):
        cleaned.pop(key, None)
    return cleaned

def render_source_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    if not answer or not sources:
        return answer
    citation_map: Dict[int, str] = {}
    for idx, src in enumerate(sources, start=1):
        citation = src.get("citation")
        if not citation:
            name = src.get("source_name") or "Document"
            name = os.path.basename(str(name))
            page = src.get("page")
            citation = f"{name}, p. {page}" if page else f"{name}"
        citation_map[idx] = citation

    def _replace(match: re.Match) -> str:
        content = match.group(1)
        ids = [int(val) for val in re.findall(r"SOURCE-(\d+)", content)]
        citations = []
        for ident in ids:
            cite = citation_map.get(ident)
            if cite:
                citations.append(cite)
        if not citations:
            return ""
        unique = list(dict.fromkeys(citations))
        if len(unique) == 1:
            return f"({unique[0]})"
        return "(" + "; ".join(unique) + ")"

    rendered = re.sub(r"\[(SOURCE-[^\]]+)\]", _replace, answer)
    return rendered.strip()

def _build_evidence_ledger(chunks: List[Any], max_snippet_chars: int = 240) -> List[Dict[str, str]]:
    ledger = []
    for chunk in chunks:
        meta = getattr(chunk, "metadata", {}) or {}
        doc_name = get_source_name(meta) or meta.get("source_name") or meta.get("file_name") or "Unknown"
        text = (getattr(chunk, "text", "") or "").strip()
        snippet = text[:max_snippet_chars].strip()
        ledger.append({"doc_name": doc_name, "snippet": snippet})
    return ledger

def _find_field_value(field: str, ledger: List[Dict[str, str]]) -> Optional[str]:
    if not field:
        return None
    pattern = re.compile(rf"{re.escape(field)}\s*[:\-]\s*([^\n\r;,.]+)", re.IGNORECASE)
    for entry in ledger:
        match = pattern.search(entry.get("snippet", ""))
        if match:
            return match.group(1).strip()
    return None

def _format_evidence_fallback(query: str, ledger: List[Dict[str, str]]) -> str:
    requested_fields = _extract_requested_fields(query)
    lines = ["Here is what I can verify from the retrieved context."]
    if requested_fields:
        lines.append("Requested fields:")
        for field in requested_fields:
            value = _find_field_value(field, ledger)
            if value:
                lines.append(f"- {field}: {value}")
            # else: omit field entirely
    if ledger:
        lines.append("Evidence snippets:")
        for entry in ledger:
            lines.append(f"- {entry['doc_name']}: {entry['snippet']}")
    else:
        lines.append("No matching information was found in the documents.")
    return "\n".join(lines).strip()

class QdrantCollectionNotFoundError(ValueError):
    def __init__(self, collection_name: str, available_collections: Optional[List[str]] = None):
        available = available_collections or []
        detail = f"qdrant_collection_not_found: collection '{collection_name}' not found"
        if available:
            detail = f"{detail}; available_collections={available}"
        super().__init__(detail)
        self.collection_name = collection_name
        self.available_collections = available

_QDRANT_INDEX_CACHE: Set[str] = set()
_QDRANT_FILTER_INDEX_CACHE: Set[Tuple[str, str]] = set()
_COLLECTION_COUNT_CACHE: Dict[str, Tuple[float, int]] = {}
_COLLECTION_COUNT_TTL_SEC = int(os.getenv("QDRANT_COUNT_TTL_SEC", "120"))

def _ensure_qdrant_indexes(client: QdrantClient, collection_name: str) -> None:
    if not collection_name:
        return
    if collection_name in _QDRANT_INDEX_CACHE:
        return
    store = QdrantVectorStore(client=client)
    try:
        store.ensure_payload_indexes(collection_name, REQUIRED_PAYLOAD_INDEX_FIELDS, create_missing=True)
        _QDRANT_INDEX_CACHE.add(collection_name)
    except Exception as exc:  # noqa: BLE001
        # Best-effort: indexes are created at startup. Log and continue —
        # the actual query will fail more specifically if indexes are missing.
        logger.warning("Index bootstrap check failed for %s: %s", collection_name, exc)
        _QDRANT_INDEX_CACHE.add(collection_name)  # Cache to avoid retry spam

def _resolve_collection_name(
    *,
    collection_name: Optional[str] = None,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> str:
    if collection_name and str(collection_name).strip():
        return str(collection_name).strip()
    if subscription_id and str(subscription_id).strip():
        return build_collection_name(str(subscription_id).strip())
    if profile_id and str(profile_id).strip():
        return str(profile_id).strip()
    raise ValueError("collection_name could not be resolved; provide collection_name, subscription_id, or profile_id")

# Initialize models and clients (lazy loading to avoid startup errors)
_MODEL = None
_CROSS_ENCODER = None
_QDRANT_CLIENT = None
_REDIS_CLIENT = None
_MODEL_CACHE: Dict[int, SentenceTransformer] = {}
_MODEL_BY_NAME: Dict[str, SentenceTransformer] = {}
_METRICS_TRACKER = None
REDIS_MEMORY_TTL = int(os.getenv("REDIS_MEMORY_TTL", "86400"))
ANSWER_CACHE_TTL = int(os.getenv("RAG_ANSWER_CACHE_TTL", "1800"))
ANSWER_CACHE_VERSION = "v3"
NO_ANSWER_CACHE = os.getenv("NO_ANSWER_CACHE", "true").strip().lower() in {"true", "1", "yes", "on"}
ENABLE_ANSWER_CACHE = False if NO_ANSWER_CACHE else os.getenv("ENABLE_ANSWER_CACHE", "false").strip().lower() == "true"

def _torch_cuda_available() -> bool:
    try:
        return bool(torch) and bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False

def _embedding_device() -> str:
    # Always use CPU for embeddings — the embedding model (bge-large, ~3GB GPU)
    # competes with DocWain-Agent for T4 16GB VRAM.  CPU embedding adds <1s
    # latency but frees the GPU entirely for LLM generation.
    env_device = (os.getenv("EMBEDDING_DEVICE") or "cpu").strip().lower()
    return env_device

def _is_meta_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "meta tensor" in msg or "cannot copy out of meta tensor" in msg

def _resolve_torch_dtype():
    if not torch:
        return None
    raw = (os.getenv("EMBEDDING_TORCH_DTYPE") or "").strip().lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32", "full"}:
        return torch.float32
    return torch.float32

def _model_kwargs_for_device(device: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"device_map": None, "low_cpu_mem_usage": False}
    dtype = _resolve_torch_dtype()
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return kwargs

def _parse_redis_connection_string(conn_str: str):
    """
    Lightweight parser for Azure Redis Cache connection strings.

    Returns a dict with host, port, password, and ssl settings. Falls
    back to conservative defaults when parsing fails so the app can still start.
    """
    settings = {
        "host": (getattr(Config.Redis, "HOST", "localhost") or "").strip(),
        "port": getattr(Config.Redis, "PORT", 6380),
        "password": getattr(Config.Redis, "PASSWORD", None) or None,
        "ssl": getattr(Config.Redis, "SSL", True),
    }

    if not conn_str:
        return settings

    conn_str = conn_str.strip()
    try:
        if conn_str.startswith(("redis://", "rediss://")):
            parsed_url = urlparse(conn_str)
            if parsed_url.hostname:
                settings["host"] = parsed_url.hostname.strip()
            if parsed_url.port:
                settings["port"] = int(parsed_url.port)
            if parsed_url.password:
                settings["password"] = parsed_url.password.strip()
            settings["ssl"] = conn_str.startswith("rediss://") or getattr(Config.Redis, "SSL", True)
            return settings

        parts = conn_str.split(",")
        host_port = parts[0]
        if ":" in host_port:
            host, port = host_port.split(":", 1)
            settings["host"] = host.strip() or settings["host"]
            settings["port"] = int(port or settings["port"])
        elif host_port:
            settings["host"] = host_port.strip() or settings["host"]

        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "password":
                settings["password"] = value
            elif key == "ssl":
                settings["ssl"] = value.lower() in {"true", "1", "yes", "on"}
    except Exception as exc:
        logger.debug("Failed to parse Redis connection string", exc_info=True)
        return settings

    if isinstance(settings.get("password"), str):
        settings["password"] = settings["password"].strip() or None
    settings["port"] = int(settings.get("port") or getattr(Config.Redis, "PORT", 6380))
    settings["ssl"] = bool(settings.get("ssl"))
    return settings

def _slug(value: str) -> str:
    """Safe, lowercase slug for cache keys."""
    if not value:
        return "default"
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "default"

def _clean_password(password: Optional[str]) -> Optional[str]:
    """Normalize password strings, trimming accidental whitespace and empty values."""
    if password is None:
        return None
    cleaned = password.strip()
    return cleaned or None

def _configure_hf_env() -> None:
    """Set HuggingFace hub timeouts/retries to avoid default short timeouts."""
    if getattr(Config.Model, "OFFLINE_ONLY", False) or getattr(Config.Model, "DISABLE_HF", False):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        return
    if os.getenv("HF_HUB_READ_TIMEOUT") is None:
        os.environ["HF_HUB_READ_TIMEOUT"] = str(getattr(Config.Model, "HF_HUB_READ_TIMEOUT", 30))
    if os.getenv("HF_HUB_CONNECT_TIMEOUT") is None:
        os.environ["HF_HUB_CONNECT_TIMEOUT"] = str(getattr(Config.Model, "HF_HUB_CONNECT_TIMEOUT", 10))
    if os.getenv("HF_HUB_MAX_RETRIES") is None:
        os.environ["HF_HUB_MAX_RETRIES"] = str(getattr(Config.Model, "HF_HUB_MAX_RETRIES", 3))
    if getattr(Config.Model, "HF_DISABLE_TELEMETRY", True):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if getattr(Config.Model, "TRANSFORMERS_OFFLINE", False):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _coerce_bool(value: Optional[Any], default: Optional[bool] = None) -> bool:
    """Coerce truthy string/env values into a boolean with a sensible default."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}

_METRICS_EMBED_SAMPLE_RATE = float(os.getenv("METRICS_EMBED_SAMPLE_RATE", "0.0"))
_METRICS_EMBED_MAX_CHARS = int(os.getenv("METRICS_EMBED_MAX_CHARS", "1200"))
_FIRST_METRICS_REQUEST = True

def _is_gemini_backend(llm_client: Any) -> bool:
    backend = getattr(llm_client, "backend", None)
    if backend:
        return str(backend).lower() == "gemini"
    model_name = getattr(llm_client, "model_name", "") or ""
    return str(model_name).lower().startswith("gemini")

def _approx_token_count(text: Optional[str]) -> int:
    return len((text or "").split())

def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", (left or "").lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", (right or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(overlap) / max(len(union), 1)

def _embedding_similarity_score(left: str, right: str) -> Optional[float]:
    if not left or not right:
        return None
    sample_rate = max(0.0, min(1.0, _METRICS_EMBED_SAMPLE_RATE))
    if sample_rate < 1.0 and random.random() > sample_rate:
        return None
    left_trimmed = (left or "")[:_METRICS_EMBED_MAX_CHARS]
    right_trimmed = (right or "")[:_METRICS_EMBED_MAX_CHARS]
    try:
        model = get_model()
        vectors = model.encode([left_trimmed, right_trimmed], normalize_embeddings=True)
        sim = float(np.dot(vectors[0], vectors[1]))
        return max(0.0, min(1.0, sim))
    except Exception as exc:
        logger.debug("Failed to compute embedding similarity score", exc_info=True)
        return None

def _semantic_similarity(left: str, right: str) -> float:
    sim = _embedding_similarity_score(left, right)
    if sim is not None:
        return sim
    return _token_overlap_score(left, right)

def _build_namespace(
        subscription_id: str,
        profile_id: str,
        model_name: str = "",
        session_id: Optional[str] = None
) -> str:
    """
    Compose a stable namespace for Redis keys so history/feedback/cache
    never bleed across tenants, profiles, models, or chat sessions.
    """
    parts = [
        _slug(subscription_id),
        _slug(profile_id),
        _slug(model_name or "default"),
    ]
    if session_id:
        parts.append(_slug(session_id))
    return ":".join(parts)

def _load_model_candidates(required_dim: Optional[int] = None) -> SentenceTransformer:
    candidates = []
    for name in getattr(Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", []):
        if name and name not in candidates:
            candidates.append(name)
    if not candidates:
        candidates.append(getattr(Config.Model, "SENTENCE_TRANSFORMERS", "sentence-transformers/all-mpnet-base-v2"))

    last_error = None
    _configure_hf_env()

    # Always use CPU for embeddings — GPU is reserved for DocWain-Agent on T4 16GB.
    # bge-large-en-v1.5 (335M params) runs fine on CPU with minimal latency impact.
    device = "cpu"

    for name in candidates:
        try:
            logger.info("Loading sentence transformer model: %s (device=%s)", name, device)
            model = SentenceTransformer(name, device=device, local_files_only=True)
            model.show_progress_bar = False
            dim = model.get_sentence_embedding_dimension()
            logger.info("Loaded model '%s' with dim=%s on %s", name, dim, device)
            if required_dim is None or dim == required_dim:
                return model
            _MODEL_CACHE[dim] = model  # cache for possible future use
        except Exception as e:
            last_error = e
            logger.warning("Failed to load model '%s' on %s: %s", name, device, e)
            # If GPU failed, retry on CPU
            if device != "cpu":
                try:
                    logger.info("Retrying '%s' on CPU...", name)
                    model = SentenceTransformer(name, device="cpu", local_files_only=True)
                    model.show_progress_bar = False
                    dim = model.get_sentence_embedding_dimension()
                    logger.info("Loaded model '%s' with dim=%s on CPU (fallback)", name, dim)
                    if required_dim is None or dim == required_dim:
                        return model
                    _MODEL_CACHE[dim] = model
                except Exception as cpu_e:
                    last_error = cpu_e
                    logger.warning("Failed to load model '%s' on CPU: %s", name, cpu_e)
    raise RuntimeError(f"Could not load any sentence transformer model from {candidates}: {last_error}")

def resolve_embedding_model_for_generation(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    mapping = getattr(Config.ModelArbitration, "EMBEDDING_MAP", {}) or {}
    if not mapping:
        return None
    key = str(model_name).lower()
    return mapping.get(key) or mapping.get(model_name) or mapping.get(key.split(":")[0])

def get_model_by_name(model_name: str, required_dim: Optional[int] = None) -> SentenceTransformer:
    """Load a specific embedding model by name with optional dimension enforcement."""
    _configure_hf_env()
    try:
        from src.api.rag_state import singleton_guard_active

        if singleton_guard_active() and _MODEL is not None and model_name not in _MODEL_BY_NAME:
            logger.error("Embedding model reinit requested (%s); reusing existing singleton", model_name)
            return get_model(required_dim=required_dim)
    except Exception as exc:
        logger.debug("Failed to check singleton guard for embedding model", exc_info=True)
    if model_name in _MODEL_BY_NAME:
        model = _MODEL_BY_NAME[model_name]
    else:
        device = _embedding_device()
        model_kwargs = _model_kwargs_for_device(device)
        try:
            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs,
                local_files_only=True,
            )
            _MODEL_BY_NAME[model_name] = model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding model %s failed to load: %s", model_name, exc)
            return get_model(required_dim=required_dim)
    if required_dim is None:
        return model
    dim = model.get_sentence_embedding_dimension()
    if dim == required_dim:
        return model
    logger.debug("Embedding model %s dim=%s does not match required=%s; using default embedder", model_name, dim, required_dim)
    return get_model(required_dim=required_dim)

def get_model(required_dim: Optional[int] = None):
    """Lazy load sentence transformer model."""
    global _MODEL
    try:
        from src.api.rag_state import singleton_guard_active

        if singleton_guard_active() and _MODEL is None:
            logger.error("Embedding model requested after startup; initializing lazily (bug)")
    except Exception as exc:
        logger.debug("Failed to check singleton guard for model init", exc_info=True)
    if _MODEL is None:
        _MODEL = _load_model_candidates()

    if required_dim is None:
        return _MODEL

    dim = _MODEL.get_sentence_embedding_dimension()
    if dim == required_dim:
        return _MODEL

    # Try cached models by dimension
    if required_dim in _MODEL_CACHE:
        logger.info(f"Using cached model with dim {required_dim}")
        return _MODEL_CACHE[required_dim]

    # Try loading a candidate that matches required_dim
    try:
        candidate = _load_model_candidates(required_dim=required_dim)
        _MODEL_CACHE[required_dim] = candidate
        logger.info(f"Loaded fallback model with dim {required_dim}")
        return candidate
    except Exception as exc:
        logger.warning(f"Failed to load fallback model for dim {required_dim}: {exc}; continuing with primary model")
        return _MODEL

class _FallbackEmbedder:
    """Local stub to avoid HuggingFace network calls when disabled."""

    def __init__(self):
        self._dim = int(getattr(Config.Model, "EMBEDDING_DIM", 0) or 0)

    def encode(self, *args, **kwargs):
        raise RuntimeError("Embedding models are disabled (HF connections disabled).")

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

def get_cross_encoder():
    """Lazy load cross encoder model."""
    global _CROSS_ENCODER
    try:
        from src.api.rag_state import singleton_guard_active

        if singleton_guard_active() and _CROSS_ENCODER is None:
            logger.error("Cross-encoder requested after startup; initializing lazily (bug)")
    except Exception as exc:
        logger.debug("Failed to check singleton guard for cross-encoder", exc_info=True)
    if _CROSS_ENCODER is None:
        if not getattr(Config.Retrieval, "RERANKER_ENABLED", True):
            logger.info("Cross-encoder reranker disabled via configuration")
            return None
        model_name = getattr(Config.Model, "RERANKER_MODEL", "")
        if not model_name:
            logger.info("Cross-encoder reranker disabled via configuration")
            return None
        try:
            _configure_hf_env()
            # Force CPU for cross-encoder to avoid GPU contention with Ollama.
            # The reranker is small (~420MB) and runs faster on CPU (1-2s)
            # than on a contended GPU (14-38s when Ollama is generating).
            ce_device = getattr(getattr(Config, "Reranker", None), "DEVICE", "cpu") or "cpu"
            _CROSS_ENCODER = CrossEncoder(model_name, device=ce_device, local_files_only=True)
            logger.info("Loaded cross-encoder model: %s on %s", model_name, ce_device)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load cross-encoder '%s', continuing without reranker: %s", model_name, e)
            _CROSS_ENCODER = None
    return _CROSS_ENCODER

def get_qdrant_client():
    """Lazy load Qdrant client."""
    global _QDRANT_CLIENT
    try:
        from src.api.rag_state import singleton_guard_active

        if singleton_guard_active() and _QDRANT_CLIENT is None:
            logger.error("Qdrant client requested after startup; initializing lazily (bug)")
    except Exception as exc:
        logger.debug("Failed to check singleton guard for Qdrant client", exc_info=True)
    if _QDRANT_CLIENT is None:
        try:
            _QDRANT_CLIENT = QdrantClient(
                url=Config.Qdrant.URL,
                api_key=Config.Qdrant.API,
                timeout=120
            )
            logger.info("Initialized Qdrant client")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    return _QDRANT_CLIENT

def get_redis_client():
    """Lazy init for Redis client with comprehensive error handling."""
    global _REDIS_CLIENT
    try:
        from src.api.rag_state import singleton_guard_active

        if singleton_guard_active() and _REDIS_CLIENT is None:
            logger.error("Redis client requested after startup; initializing lazily (bug)")
    except Exception as exc:
        logger.debug("Failed to check singleton guard for Redis client", exc_info=True)
    if _REDIS_CLIENT is None:
        try:
            raw_conn = (
                os.getenv("REDIS_URL")
                or os.getenv("REDIS_CONNECTION_STRING")
                or getattr(Config.Redis, "CONNECTION_STRING", "")
            )

            parsed = _parse_redis_connection_string(raw_conn)
            host = (
                os.getenv("REDIS_HOST")
                or parsed.get("host")
                or getattr(Config.Redis, "HOST", "rediscache.redis.cache.windows.net")
                or ""
            ).strip()
            if not host:
                host = getattr(Config.Redis, "HOST", "localhost")

            password = _clean_password(
                os.getenv("REDIS_PASSWORD")
                or parsed.get("password")
                or getattr(Config.Redis, "PASSWORD", "")
            )

            db_idx = int(os.getenv("REDIS_DB", getattr(Config.Redis, "DB", 0)))
            port = int(os.getenv("REDIS_PORT", parsed.get("port", getattr(Config.Redis, "PORT", 6380))))
            ssl_enabled = _coerce_bool(
                os.getenv("REDIS_SSL"),
                parsed.get("ssl", getattr(Config.Redis, "SSL", True)),
            )
            socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "10"))
            socket_connect_timeout = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", str(socket_timeout)))

            client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db_idx,
                ssl=ssl_enabled,
                decode_responses=True,
                socket_connect_timeout=socket_connect_timeout,
                socket_timeout=socket_timeout,
            )
            client.ping()
            _REDIS_CLIENT = client
            logger.info("Redis connected via SSL=%s to %s:%s/%s", ssl_enabled, host, port, db_idx)

        except Exception as e:
            logger.error(f" Redis connection failed: {e}")
            logger.warning("RAG system will work WITHOUT caching/history persistence. "
                           "Check REDIS_CONNECTION_STRING/REDIS_PASSWORD and network reachability.")
            _REDIS_CLIENT = None  # Graceful degradation

    return _REDIS_CLIENT

def configure_gemini():
    """Configure Gemini API with proper error handling."""
    try:
        api_key = getattr(Config.Model, "GEMINI_API_KEY", None) or getattr(Config.Gemini, "GEMINI_API_KEY", None)
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in configuration")
        get_genai_client(api_key)
        logger.info("Gemini API configured successfully")
        return api_key
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        raise

@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict
    source: str | None = None
    method: str = "dense"

@dataclass
class ConversationTurn:
    """Data class for conversation history."""
    user_message: str
    assistant_response: str
    timestamp: float

@dataclass
class ProfileContextSnapshot:
    """Lightweight snapshot of profile-specific vocabulary and hints."""
    top_keywords: List[str]
    document_hints: List[str]
    total_chunks: int
    last_updated: float

class ChatFeedbackMemory:
    """Stores compact Q/A feedback to steer future responses."""

    def __init__(
            self,
            max_items: int = 12,
            redis_client: Optional[redis.Redis] = None,
            ttl_seconds: int = REDIS_MEMORY_TTL
    ):
        self.max_items = max_items
        self.memories: Dict[Tuple[str, str], deque] = {}
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _cache_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:feedback:{namespace}:{user_id}"

    def _load_from_cache(self, namespace: str, user_id: str):
        key = (namespace, user_id)
        if key in self.memories or not self.redis:
            return
        try:
            cached = self.redis.get(self._cache_key(namespace, user_id))
        except Exception as exc:
            logger.warning(f"Failed to read feedback memory from Redis: {exc}")
            return
        if not cached:
            return
        try:
            items = json.loads(cached)
            if isinstance(items, list):
                self.memories[key] = deque(items, maxlen=self.max_items)
        except Exception as exc:
            logger.warning(f"Failed to parse feedback memory from Redis: {exc}")

    def _persist(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            payload = list(self.memories.get((namespace, user_id), []))
            self.redis.setex(self._cache_key(namespace, user_id), self.ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Failed to persist feedback memory to Redis: {exc}")

    def clear(self, namespace: str, user_id: str):
        """Remove cached feedback so new sessions start fresh."""
        key = (namespace, user_id)
        if key in self.memories:
            self.memories[key].clear()
        if self.redis:
            try:
                self.redis.delete(self._cache_key(namespace, user_id))
            except Exception as exc:
                logger.warning(f"Failed to clear feedback memory from Redis: {exc}")

    def add_feedback(self, namespace: str, user_id: str, query: str, answer: str, sources: List[Dict[str, Any]]):
        self._load_from_cache(namespace, user_id)
        key = (namespace, user_id)
        if key not in self.memories:
            self.memories[key] = deque(maxlen=self.max_items)
        src_names = [s.get("source_name") for s in (sources or []) if s.get("source_name")]
        self.memories[key].append({
            "q": query.strip(),
            "a": answer.strip(),
            "sources": src_names,
            "ts": time.time()
        })
        self._persist(namespace, user_id)

    def build_feedback_context(self, namespace: str, user_id: str, limit: int = 5) -> str:
        self._load_from_cache(namespace, user_id)
        key = (namespace, user_id)
        if key not in self.memories or not self.memories[key]:
            return ""
        recent = list(self.memories[key])[-limit:]
        lines = []
        for idx, item in enumerate(recent, 1):
            source_hint = f" | sources: {', '.join(item['sources'][:3])}" if item.get("sources") else ""
            lines.append(f"{idx}) Q: {item['q']} | A: {item['a'][:180]}{source_hint}")
        return "RECENT CHAT FEEDBACK (reuse tone/precision):\n" + "\n".join(lines) + "\n"

class ConversationHistory:
    """Manages conversation history with a sliding window."""

    def __init__(
            self,
            max_turns: int = 3,
            redis_client: Optional[redis.Redis] = None,
            ttl_seconds: int = REDIS_MEMORY_TTL
    ):
        self.max_turns = max_turns
        self.histories: Dict[Tuple[str, str], deque] = {}
        self.recent_docs: Dict[Tuple[str, str], deque] = {}
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _history_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:history:{namespace}:{user_id}"

    def _docs_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:recent_docs:{namespace}:{user_id}"

    def _load_history(self, namespace: str, user_id: str):
        key = (namespace, user_id)
        if key in self.histories or not self.redis:
            return
        try:
            cached = self.redis.get(self._history_key(namespace, user_id))
        except Exception as exc:
            logger.warning(f"Failed to read conversation history from Redis: {exc}")
            return
        if not cached:
            return
        try:
            items = json.loads(cached)
            if isinstance(items, list):
                turns = deque(maxlen=self.max_turns)
                for item in items[-self.max_turns:]:
                    if not isinstance(item, dict):
                        continue
                    turns.append(ConversationTurn(
                        user_message=item.get("user_message", ""),
                        assistant_response=item.get("assistant_response", ""),
                        timestamp=float(item.get("timestamp", 0.0))
                    ))
                self.histories[key] = turns
        except Exception as exc:
            logger.warning(f"Failed to parse conversation history from Redis: {exc}")

    def _load_recent_docs(self, namespace: str, user_id: str):
        key = (namespace, user_id)
        if key in self.recent_docs or not self.redis:
            return
        try:
            cached = self.redis.get(self._docs_key(namespace, user_id))
        except Exception as exc:
            logger.warning(f"Failed to read recent docs from Redis: {exc}")
            return
        if not cached:
            return
        try:
            items = json.loads(cached)
            if isinstance(items, list):
                self.recent_docs[key] = deque(items, maxlen=10)
        except Exception as exc:
            logger.warning(f"Failed to parse recent docs from Redis: {exc}")

    def _persist_history(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            payload = []
            for turn in self.histories.get((namespace, user_id), []):
                payload.append({
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "timestamp": turn.timestamp
                })
            self.redis.setex(self._history_key(namespace, user_id), self.ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Failed to persist conversation history to Redis: {exc}")

    def _persist_recent_docs(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            payload = list(self.recent_docs.get((namespace, user_id), []))
            self.redis.setex(self._docs_key(namespace, user_id), self.ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Failed to persist recent docs to Redis: {exc}")

    def add_turn(self, namespace: str, user_id: str, user_message: str, assistant_response: str):
        """Add a conversation turn to history."""
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories:
            self.histories[key] = deque(maxlen=self.max_turns)

        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=time.time()
        )
        self.histories[key].append(turn)
        self._persist_history(namespace, user_id)

    def add_sources(self, namespace: str, user_id: str, doc_ids: List[str]):
        """Track recently used document IDs for recency-based boosting."""
        key = (namespace, user_id)
        self._load_recent_docs(namespace, user_id)
        if key not in self.recent_docs:
            self.recent_docs[key] = deque(maxlen=10)
        for doc_id in doc_ids:
            if doc_id:
                self.recent_docs[key].append(doc_id)
        self._persist_recent_docs(namespace, user_id)

    def get_context(
            self,
            namespace: str,
            user_id: str,
            max_turns: int = 2,
            include_assistant: bool = True,
            max_chars: int = 1200,
    ) -> str:
        """Get recent conversation context as formatted string."""
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories or not self.histories[key]:
            return ""

        context_parts = []
        recent_turns = list(self.histories[key])[-max_turns:]

        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            if include_assistant and turn.assistant_response:
                trimmed = " ".join(turn.assistant_response.split())
                if len(trimmed) > 280:
                    trimmed = trimmed[:277].rsplit(" ", 1)[0] + "..."
                context_parts.append(f"Assistant: {trimmed}")

        context = "\n".join(context_parts)
        if len(context) > max_chars:
            return context[-max_chars:]
        return context

    def get_recent_doc_ids(self, namespace: str, user_id: str) -> List[str]:
        """Return a list of recently cited document IDs for this user."""
        key = (namespace, user_id)
        self._load_recent_docs(namespace, user_id)
        if key not in self.recent_docs:
            return []
        return list(self.recent_docs[key])

    def get_last_user_message(self, namespace: str, user_id: str) -> str:
        """Return the most recent user message in history."""
        key = (namespace, user_id)
        self._load_history(namespace, user_id)
        if key not in self.histories or not self.histories[key]:
            return ""
        return (self.histories[key][-1].user_message or "").strip()

    def clear_history(self, namespace: str, user_id: str):
        """Clear conversation history for a user."""
        key = (namespace, user_id)
        if key in self.histories:
            self.histories[key].clear()
        if key in self.recent_docs:
            self.recent_docs[key].clear()
        if self.redis:
            try:
                self.redis.delete(self._history_key(namespace, user_id))
                self.redis.delete(self._docs_key(namespace, user_id))
            except Exception as exc:
                logger.warning(f"Failed to clear Redis memory for user {user_id}: {exc}")

class TextPreprocessor:
    """Handles text preprocessing for consistent tokenization."""

    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'about', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'that', 'this', 'these', 'those', 'it', 'its', 'their', 'them', 'they', 'you',
            'your', 'we', 'our', 'us', 'i', 'me', 'my', 'he', 'she', 'his', 'her', 'not',
            'do', 'does', 'did', 'done', 'have', 'has', 'had', 'will', 'would', 'can', 'could',
            'should', 'may', 'might', 'must', 'if', 'then', 'so', 'than', 'such', 'also'
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by lowercasing and removing extra whitespace."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with lemmatization and stopword removal for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        lemmatized = []
        for token in tokens:
            if token.endswith('ing'):
                lemmatized.append(token[:-3])
            elif token.endswith('ed'):
                lemmatized.append(token[:-2])
            elif token.endswith('s') and len(token) > 3:
                lemmatized.append(token[:-1])
            else:
                lemmatized.append(token)

        return lemmatized

class GreetingHandler:
    """Handles greeting and farewell detection."""

    GREETINGS = {
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'greetings', 'howdy', 'what\'s up', 'how are you', 'how do you do',
        'nice to meet you', 'good day', 'hiya', 'sup', 'yo', 'salutations',
        'welcome', 'hola', 'bonjour', 'namaste', 'aloha', 'hii'
    }

    FAREWELLS = {
        'bye', 'goodbye', 'good bye', 'see you', 'see ya', 'farewell',
        'adieu', 'cheerio', 'tata', 'ta ta', 'catch you later', 'later',
        'take care', 'until next time', 'signing off', 'gotta go', 'gtg',
        'peace out', 'ciao', 'au revoir', 'sayonara', 'hasta la vista',
        'talk to you later', 'ttyl', 'see you soon', 'see you around',
        'good night', 'goodnight', 'have a good day', 'have a great day',
        'thanks for your help', 'that\'s all', 'i\'m done', 'end chat',
        'quit', 'exit', 'close', 'finish', 'terminate', 'stop'
    }

    POSITIVE_FEEDBACK = {
        'thanks', 'thank you', 'thanks a lot', 'thank you so much',
        'appreciate it', 'much appreciated', 'thx', 'ty', 'tysm',
        'great answer', 'good answer', 'very good', 'awesome', 'perfect',
        'well done', 'nice', 'excellent'
    }

    NEGATIVE_FEEDBACK = {
        'bad answer', 'wrong answer', 'not right', 'not correct', 'incorrect',
        'not accurate', 'this is bad', 'this is wrong', 'not good',
        'doesn\'t make sense', 'does not make sense', 'poor answer',
        'not helpful', 'this is not right response', 'this is not right',
        'this is not correct', 'this is not accurate', 'bad response'
    }

    QUESTION_CUES = {
        'can you', 'could you', 'would you', 'should you', 'help me', 'help with', 'help',
        'how', 'what', 'why', 'when', 'where', 'which', 'who', 'can u', 'could u',
        'tell me', 'show me', 'explain', 'walk me through', 'guide me', 'need help'
    }

    FILLER_WORDS = {'there', 'team', 'docwain', 'assistant', 'bot', 'buddy', 'friend', 'please'}

    @staticmethod
    def _clean_message(message: str) -> str:
        cleaned = re.sub(r"[^\w\s?.!]", " ", message.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _strip_phrases(cls, message: str, phrases: Set[str]) -> str:
        cleaned = cls._clean_message(message)
        for phrase in sorted(phrases, key=len, reverse=True):
            cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _has_question_cue(cls, message: str) -> bool:
        if "?" in message:
            return True
        normalized = cls._clean_message(message)
        return any(cue in normalized for cue in cls.QUESTION_CUES)

    @classmethod
    def _has_follow_on_content(cls, message: str, phrases: Set[str]) -> bool:
        """
        Check if there is meaningful content beyond the matched phrase.
        Prevents treating greetings/thanks with real questions as small talk.
        """
        remainder = cls._strip_phrases(message, phrases)
        if not remainder:
            return False

        if cls._has_question_cue(remainder):
            return True

        tokens = [t for t in remainder.split() if t not in cls.FILLER_WORDS]
        return len(tokens) >= 2

    @classmethod
    def is_greeting(cls, message: str) -> bool:
        """Check if message is a greeting."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(g) for g in cls.GREETINGS) + r')\b'
        if not re.search(pattern, normalized):
            return False

        if cls._has_follow_on_content(message, cls.GREETINGS):
            return False

        return True

    @classmethod
    def is_farewell(cls, message: str) -> bool:
        """Check if message is a farewell."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.FAREWELLS) + r')\b'
        if not re.search(pattern, normalized):
            return False

        if cls._has_follow_on_content(message, cls.FAREWELLS):
            return False

        return True

    @classmethod
    def is_positive_feedback(cls, message: str) -> bool:
        """Check if message is positive feedback."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.POSITIVE_FEEDBACK) + r')\b'
        if not re.search(pattern, normalized):
            return False

        if cls._has_follow_on_content(message, cls.POSITIVE_FEEDBACK):
            return False

        return True

    @classmethod
    def is_negative_feedback(cls, message: str) -> bool:
        """Check if message is negative feedback."""
        normalized = cls._clean_message(message)
        if not normalized:
            return False

        pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.NEGATIVE_FEEDBACK) + r')\b'
        if not re.search(pattern, normalized):
            return False

        return True

class OllamaClient:
    """Handles local Ollama model calls with controlled generation to reduce hallucinations."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = _resolve_model_alias(model_name) or os.getenv("OLLAMA_MODEL", "DHS/DocWain")
        if not self.model_name:
            raise ValueError("OLLAMA_MODEL environment variable is not set")
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 1,
        backoff: float = 0.5,
    ) -> Tuple[str, Dict[str, Any]]:
        metrics_store = get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        generation_options = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.2),
            "top_p": getattr(Config.LLM, "TOP_P", 0.85),
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
            "num_predict": getattr(Config.LLM, "MAX_TOKENS", 2048),
        }
        if options:
            generation_options.update({k: v for k, v in options.items() if v is not None})

        last_response: Dict[str, Any] = {}
        for attempt in range(1, max_retries + 1):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=generation_options,
                )
                last_response = response or {}
                text = (last_response.get("response") or "").strip()
                if metrics_store.available:
                    latency_ms = (time.time() - request_started) * 1000
                    metrics_store.record(
                        values={"llm_latency_ms": latency_ms},
                        histograms={"llm_latency_ms": latency_ms},
                        model_id=self.model_name,
                    )
                    if attempt > 1:
                        metrics_store.record(
                            counters={"llm_retry_count": attempt - 1},
                            model_id=self.model_name,
                        )
                return text, last_response
            except Exception as exc:
                logger.warning(f"Ollama attempt {attempt}/{max_retries} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error("All Ollama retries failed")
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "", last_response

    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
        backoff: float = 1.0
    ) -> str:
        text, response = self.generate_with_metadata(
            prompt,
            max_retries=max_retries,
            backoff=backoff,
        )
        if not text:
            logger.debug(f"Ollama returned empty response: {response}")
            return "I don’t have enough information in the documents to answer that."
        return text

class RateLimitCooldownError(RuntimeError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.code = 429
        self.retry_after = retry_after

class GeminiClient:
    def __init__(self, model_name: Optional[str] = None):
        self.api_key = configure_gemini()
        self.model_name = model_name or Config.Model.GEMINI_MODEL_NAME
        if not self.model_name:
            raise ValueError("Gemini model name is not configured")
        self.generation_config = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.3),
            "top_p": getattr(Config.LLM, "TOP_P", 0.95),
            "top_k": 40,
            "max_output_tokens": getattr(Config.LLM, "MAX_TOKENS", 2048),
        }
        logger.info(f"Initialized GeminiClient with model: {self.model_name}")
        self._cache: Dict[str, Tuple[float, str]] = {}
        self._cache_ttl = int(os.getenv("GEMINI_CACHE_TTL_SECONDS", "900"))
        self._circuit_open_until = 0.0
        self._circuit_failures = 0
        self._circuit_threshold = int(os.getenv("GEMINI_CIRCUIT_BREAKER_THRESHOLD", "2"))
        self._circuit_timeout = int(os.getenv("GEMINI_CIRCUIT_BREAKER_TIMEOUT", "60"))
        self._rate_limit_window = int(os.getenv("GEMINI_RATE_LIMIT_WINDOW", "60"))
        self._rate_limit_threshold = int(os.getenv("GEMINI_RATE_LIMIT_THRESHOLD", str(self._circuit_threshold)))
        self._rate_limit_cooldown = int(os.getenv("GEMINI_RATE_LIMIT_COOLDOWN", str(self._circuit_timeout)))
        self._rate_limit_hits = 0
        self._rate_limit_last_ts = 0.0
        self._redis_client = None
        self._cooldown_key = f"llm:cooldown:gemini:{self.model_name}"

    def generate_with_metadata(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: float = 1.0,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        text = self.generate(prompt, max_retries=max_retries, backoff=backoff)
        return text, {"response": text}

    def _get_redis_client(self):
        if self._redis_client is None:
            try:
                self._redis_client = get_redis_client()
            except Exception as exc:
                logger.debug("Failed to initialize Redis client for LLM wrapper", exc_info=True)
                self._redis_client = None
        return self._redis_client

    def _get_cooldown_until(self) -> Optional[float]:
        redis_client = self._get_redis_client()
        if not redis_client:
            return None
        try:
            raw = redis_client.get(self._cooldown_key)
        except Exception as exc:
            logger.debug("Failed to read cooldown from Redis", exc_info=True)
            return None
        if not raw:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return float(raw)
        except Exception as exc:
            logger.debug("Failed to parse cooldown timestamp from Redis", exc_info=True)
            return None

    def in_cooldown(self) -> bool:
        now = time.time()
        if self._circuit_open_until and now < self._circuit_open_until:
            return True
        redis_until = self._get_cooldown_until()
        if redis_until and redis_until > now:
            self._circuit_open_until = max(self._circuit_open_until, redis_until)
            return True
        return False

    def _set_cooldown(self, until_ts: float) -> None:
        self._circuit_open_until = max(self._circuit_open_until, until_ts)
        redis_client = self._get_redis_client()
        if redis_client:
            try:
                ttl = max(1, int(until_ts - time.time()))
                redis_client.setex(self._cooldown_key, ttl, str(int(until_ts)))
            except Exception as exc:
                logger.debug("Failed to persist cooldown to Redis", exc_info=True)

    def _record_rate_limit_hit(self) -> bool:
        now = time.time()
        if now - self._rate_limit_last_ts > self._rate_limit_window:
            self._rate_limit_hits = 0
        self._rate_limit_hits += 1
        self._rate_limit_last_ts = now
        if self._rate_limit_hits >= self._rate_limit_threshold:
            self._set_cooldown(now + self._rate_limit_cooldown)
            return True
        return False

    @staticmethod
    def _extract_status(exc: Exception) -> Optional[int]:
        for attr in ("code", "status", "status_code"):
            value = getattr(exc, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception as exc:
                    logger.debug("Failed to call status extraction method", exc_info=True)
                    value = None
            if isinstance(value, int):
                return value
            try:
                if hasattr(value, "value"):
                    return int(value.value)
            except Exception as exc:
                logger.debug("Failed to extract status code from exception attribute", exc_info=True)
                continue
        msg = str(exc)
        if "429" in msg or "too many requests" in msg.lower():
            return 429
        return None

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None) or getattr(exc, "resp", None) or getattr(exc, "http_response", None)
        headers = None
        if response is not None:
            headers = getattr(response, "headers", None) or getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    if isinstance(retry_after, (int, float)):
                        return float(retry_after)
                    retry_after = str(retry_after).strip()
                    if retry_after.isdigit():
                        return float(retry_after)
                    parsed = parsedate_to_datetime(retry_after)
                    if parsed:
                        delta = (parsed - datetime.now(parsed.tzinfo)).total_seconds()
                        return max(0.0, delta)
                except Exception as exc:
                    logger.debug("Failed to parse Retry-After header", exc_info=True)
                    return None
        retry_delay = getattr(exc, "retry_delay", None)
        try:
            if retry_delay is not None:
                return float(retry_delay)
        except Exception as exc:
            logger.debug("Failed to parse retry_delay from exception", exc_info=True)
        return None

    def generate(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: float = 1.0
    ) -> str:
        """Generate response with retry logic and robust parsing."""
        metrics_store = get_metrics_store()
        request_started = time.time()
        cache_key = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            ts, text = cached
            if (time.time() - ts) <= self._cache_ttl:
                logger.info("Gemini cache hit")
                return text
            self._cache.pop(cache_key, None)
        if self.in_cooldown():
            raise RateLimitCooldownError("Gemini cooldown active; skipping request")

        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        max_retries = max(1, min(int(max_retries or 1), 3))
        total_sleep = 0.0
        max_sleep = float(os.getenv("GEMINI_MAX_BACKOFF_SECONDS", "2.5"))
        total_cap = float(os.getenv("GEMINI_TOTAL_BACKOFF_CAP_SECONDS", "3.5"))
        rate_limit_seen = False
        for attempt in range(1, max_retries + 1):
            try:
                text, response = generate_text(
                    api_key=self.api_key,
                    model=self.model_name,
                    prompt=prompt,
                    generation_config=self.generation_config,
                )
                if text:
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                        if attempt > 1:
                            metrics_store.record(
                                counters={"llm_retry_count": attempt - 1},
                                model_id=self.model_name,
                        )
                    self._circuit_failures = 0
                    self._circuit_open_until = 0.0
                    self._rate_limit_hits = 0
                    self._cache[cache_key] = (time.time(), text)
                    return text
                logger.warning(f"No text in response: {response}")
                return "I apologize, but I couldn't generate a proper response."
            except Exception as e:
                status = self._extract_status(e)
                retry_after = self._extract_retry_after(e)
                if status == 429 and not rate_limit_seen:
                    rate_limit_seen = True
                    triggered = self._record_rate_limit_hit()
                    logger.warning(
                        "Gemini rate limit encountered (attempt %s/%s)",
                        attempt,
                        max_retries,
                        extra={
                            "stage": "generate",
                            "provider": "gemini",
                            "model": self.model_name,
                            "retry_after": retry_after,
                        },
                    )
                    if triggered:
                        raise RateLimitCooldownError("Gemini rate limit cooldown activated", retry_after=self._rate_limit_cooldown)
                elif status is not None:
                    logger.warning(
                        "Gemini API attempt %s/%s failed (status=%s): %s",
                        attempt,
                        max_retries,
                        status,
                        e,
                        extra={"stage": "generate", "provider": "gemini", "model": self.model_name},
                    )
                else:
                    logger.warning(
                        "Gemini API attempt %s/%s failed: %s",
                        attempt,
                        max_retries,
                        e,
                        extra={"stage": "generate", "provider": "gemini", "model": self.model_name},
                    )
                if status in {500, 502, 503, 504}:
                    self._circuit_failures += 1
                    if self._circuit_failures >= self._circuit_threshold:
                        self._set_cooldown(time.time() + self._circuit_timeout)
                if attempt < max_retries:
                    sleep_for = None
                    if retry_after is not None:
                        sleep_for = min(max_sleep, max(0.0, float(retry_after)))
                    else:
                        base = max(0.1, float(backoff))
                        sleep_for = min(max_sleep, base * (2 ** (attempt - 1)))
                        sleep_for += random.uniform(0, sleep_for * 0.25)
                    remaining = max(0.0, total_cap - total_sleep)
                    if sleep_for and remaining > 0:
                        sleep_for = min(sleep_for, remaining)
                        time.sleep(sleep_for)
                        total_sleep += sleep_for
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "I apologize, but I encountered an error generating a response."

class OpenAICompatibleClient:
    """Handles OpenAI-compatible local LLM endpoints (chat completions)."""

    def __init__(
            self,
            model_name: Optional[str] = None,
            endpoint: Optional[str] = None,
            api_key: Optional[str] = None
    ):
        self.endpoint = endpoint or os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        if self.endpoint.rstrip("/").endswith("/v1"):
            self.endpoint = self.endpoint.rstrip("/") + "/chat/completions"
        self.model_name = model_name or os.getenv("LOCAL_LLM_MODEL", "local-model")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "")
        self.temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", str(getattr(Config.LLM, "TEMPERATURE", 0.0))))
        self.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", str(getattr(Config.LLM, "MAX_TOKENS", 2048))))
        self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "30"))
        logger.info("Initialized OpenAI-compatible client at %s with model %s", self.endpoint, self.model_name)

    def warm_up(self):
        try:
            self.generate("ping", max_retries=1, backoff=0.0)
        except Exception as exc:
            logger.warning("Local LLM warm-up failed (continuing): %s", exc)

    def generate(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: float = 1.0
    ) -> str:
        metrics_store = get_metrics_store()
        request_started = time.time()
        if metrics_store.available:
            metrics_store.record(
                counters={"llm_request_count": 1},
                distributions={"model_usage": {self.model_name: 1}},
                model_id=self.model_name,
            )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, max_retries + 1):
            try:
                req = request.Request(self.endpoint, data=data, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                response = json.loads(body)
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or choice.get("text") or ""
                text = text.strip()
                if text:
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                        if attempt > 1:
                            metrics_store.record(
                                counters={"llm_retry_count": attempt - 1},
                                model_id=self.model_name,
                            )
                    return text
                logger.warning("No text in local LLM response: %s", response)
                return "I apologize, but I couldn't generate a proper response."
            except (HTTPError, URLError, ValueError) as e:
                logger.warning("Local LLM attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error("All local LLM retry attempts failed: %s", e)
                    if metrics_store.available:
                        latency_ms = (time.time() - request_started) * 1000
                        metrics_store.record(
                            counters={"llm_failure": 1},
                            values={"llm_latency_ms": latency_ms},
                            histograms={"llm_latency_ms": latency_ms},
                            model_id=self.model_name,
                        )
                    raise

        return "I apologize, but I encountered an error generating a response."

class QueryReformulator:
    """Reformulates conversational queries into clear, concise search queries."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def reformulate(self, query: str, conversation_context: str = "") -> str:
        """Reformulate query using LLM to make it more search-friendly."""
        if len(query.split()) <= 5 and not conversation_context:
            return query
        if _is_gemini_backend(self.llm_client):
            return query

        prompt = f"""You are a query reformulation assistant. Convert the user's conversational question into a clear, concise search query optimized for semantic search.

RULES:
1. Extract the core information need
2. Remove filler words and conversational elements
3. Keep domain-specific terms and technical vocabulary
4. Make it 3-10 words maximum
5. If conversation context is provided, resolve pronouns and references
6. Output ONLY the reformulated query, nothing else

{("CONVERSATION CONTEXT:" + chr(10) + conversation_context + chr(10)) if conversation_context else ""}
USER QUERY: {query}

REFORMULATED QUERY:"""

        try:
            reformulated = self.llm_client.generate(
                prompt,
                system=build_system_prompt(),
                max_retries=2,
                backoff=0.5
            )

            reformulated = reformulated.strip().strip('"\'')

            if 2 <= len(reformulated.split()) <= 15 and reformulated.lower() != query.lower():
                logger.info(f"Reformulated: '{query}' � '{reformulated}'")
                return reformulated
            else:
                return query

        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return query

class QdrantRetriever:
    """Handles retrieval from Qdrant using native search functionality."""

    def __init__(self, client: QdrantClient, model: SentenceTransformer):
        self.client = client
        self.model = model
        self.preprocessor = TextPreprocessor()
        self.profile_context_cache: Dict[Tuple[str, str], ProfileContextSnapshot] = {}
        self.collection_dims: Dict[str, int] = {}
        self._hash_vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    @staticmethod
    def _ensure_profile(profile_id: Optional[str]):
        if not profile_id:
            logger.error("Security: retrieval attempted without profile_id filter")
            raise ValueError("profile_id is required for retrieval to enforce isolation")

    def _build_filter(
        self,
        subscription_id: str,
        profile_id: str,
        document_ids: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None,
        doc_domains: Optional[List[str]] = None,
        section_kinds: Optional[List[str]] = None,
    ) -> Filter:
        self._ensure_profile(profile_id)
        if not subscription_id or not str(subscription_id).strip():
            logger.error("Security: retrieval attempted without subscription_id filter")
            raise ValueError("subscription_id is required for retrieval to enforce isolation")
        base = build_qdrant_filter(
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
            doc_domain=doc_domains,
            section_kind=section_kinds,
        )
        conditions = list(getattr(base, "must", []) or [])
        should = list(getattr(base, "should", []) or [])
        if document_ids:
            conditions.append(FieldCondition(key="document_id", match=MatchAny(any=[str(d) for d in document_ids])))
        if source_files:
            values = [str(s) for s in source_files]
            should.append(FieldCondition(key="source.name", match=MatchAny(any=values)))
            should.append(FieldCondition(key="source_file", match=MatchAny(any=values)))
        return Filter(must=conditions, should=should or None)

    def _build_sparse_query(self, query: str) -> SparseVector:
        matrix = self._hash_vectorizer.transform([query])
        coo = matrix.tocoo()
        return SparseVector(indices=coo.col.tolist(), values=coo.data.astype(np.float32).tolist())

    def ensure_collection_exists(self, collection_name: str) -> None:
        if not collection_name:
            raise QdrantCollectionNotFoundError(collection_name or "")
        try:
            self.client.get_collection(collection_name)
            return
        except Exception:  # noqa: BLE001
            try:
                collections = self.client.get_collections()
                available = [c.name for c in getattr(collections, "collections", []) if getattr(c, "name", None)]
            except Exception:  # noqa: BLE001
                available = []
            raise QdrantCollectionNotFoundError(collection_name, available_collections=available)

    @staticmethod
    def _points_to_chunks(results, method: str) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        points = []
        if hasattr(results, "points"):
            points = results.points or []
        elif isinstance(results, tuple) and results:
            points = results[0] or []

        for pt in points:
            payload = pt.payload or {}
            source = get_source_name(payload) or payload.get("source") or ""
            chunks.append(
                RetrievedChunk(
                    id=str(pt.id),
                    text=get_embedding_text(payload),
                    score=float(pt.score),
                    metadata=payload,
                    source=source or None,
                    method=method,
                )
            )
        return chunks

    @staticmethod
    def _rrf_merge(dense: List[RetrievedChunk], sparse: List[RetrievedChunk], k: int = 60) -> List[RetrievedChunk]:
        scored: Dict[str, RetrievedChunk] = {}
        weights = getattr(Config.Retrieval, "HYBRID_WEIGHTS", {"dense": 0.6, "sparse": 0.4})
        dense_w = float(weights.get("dense", 0.6))
        sparse_w = float(weights.get("sparse", 0.4))

        for rank, chunk in enumerate(dense, start=1):
            score = dense_w * (1.0 / (k + rank))
            scored.setdefault(chunk.id, chunk)
            scored[chunk.id].score = scored[chunk.id].score + score if scored[chunk.id].score else score

        for rank, chunk in enumerate(sparse, start=1):
            score = sparse_w * (1.0 / (k + rank))
            if chunk.id in scored:
                scored[chunk.id].score += score
                scored[chunk.id].metadata.setdefault("methods", []).append("sparse")
            else:
                chunk.score = score
                scored[chunk.id] = chunk

        return sorted(scored.values(), key=lambda c: to_py_scalar(c.score), reverse=True)

    def run_search(
            self,
            collection_name: str,
            query_vector: List[float],
            query_filter: Optional[dict] = None,
            limit: int = 50,
            vector_name: str = "content_vector",
            score_threshold: Optional[float] = None,
            *,
            _allow_retry: bool = True,
    ):
        """Execute a vector search against Qdrant using query_points."""
        try:
            if query_filter is not None:
                try:
                    filter_fields: List[str] = []
                    stack = [query_filter]
                    while stack:
                        node = stack.pop()
                        if hasattr(node, "must") or hasattr(node, "should") or hasattr(node, "must_not"):
                            for key in ("must", "should", "must_not"):
                                children = getattr(node, key, None) or []
                                stack.extend(children)
                        if hasattr(node, "key"):
                            filter_fields.append(getattr(node, "key"))
                    if "doc_type" in filter_fields:
                        cache_key = (collection_name, "doc_type")
                        if cache_key not in _QDRANT_FILTER_INDEX_CACHE:
                            try:
                                ensure_payload_indexes(
                                    client=self.client,
                                    collection_name=collection_name,
                                    required_fields=["doc_type"],
                                    create_missing=True,
                                )
                            finally:
                                _QDRANT_FILTER_INDEX_CACHE.add(cache_key)
                except Exception as exc:
                    logger.debug("Failed to ensure Qdrant payload indexes", exc_info=True)
            kwargs = dict(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Add query_filter only if provided (not as 'filter')
            if query_filter is not None:
                kwargs["query_filter"] = query_filter

            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold

            results = self.client.query_points(**kwargs)
            return results
        except UnexpectedResponse as exc:
            detail = getattr(exc, "content", None) or str(exc)
            logger.error("Qdrant query_points error (%s): %s", exc.status_code, detail, exc_info=True)
            missing_field = autoheal_missing_index(self.client, collection_name, detail)
            if missing_field and _allow_retry:
                logger.warning(
                    "Retrying Qdrant query after auto-heal for missing index: %s",
                    missing_field,
                )
                return self.run_search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    vector_name=vector_name,
                    score_threshold=score_threshold,
                    _allow_retry=False,
                )
            if missing_field:
                raise RetrievalFilterError(
                    "Index required but not found; retrieval blocked.",
                    code="RETRIEVAL_INDEX_MISSING",
                    details=f"missing_index={missing_field}",
                ) from exc
            raise RetrievalFilterError(
                "Profile isolation enforced; cannot search outside profile.",
                code="RETRIEVAL_FILTER_FAILED",
                details=str(detail),
            ) from exc
        except Exception as e:
            logger.error("Qdrant query_points error: %s", e, exc_info=True)
            raise RetrievalFilterError(
                "Qdrant unavailable; retrieval blocked.",
                code="RETRIEVAL_QDRANT_UNAVAILABLE",
                details=str(e),
            ) from e

    def get_collection_vector_dim(self, collection_name: str) -> Optional[int]:
        """Fetch and cache the expected vector dimension for a collection."""
        if collection_name in self.collection_dims:
            return self.collection_dims[collection_name]
        try:
            info = self.client.get_collection(collection_name)
            cfg = getattr(info, "config", None) or {}
            params = getattr(cfg, "params", None) or {}
            vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
            dim = None
            if hasattr(vectors, "size"):
                dim = vectors.size
            elif isinstance(vectors, dict):
                if "size" in vectors:
                    dim = vectors.get("size")
                elif "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
                    dim = vectors["content_vector"].get("size")
            if dim is None:
                dim = getattr(Config.Model, "EMBEDDING_DIM", None) or 1024
            dim = int(dim)
            self.collection_dims[collection_name] = dim
            logger.info(f"Collection '{collection_name}' expects dim={dim}")
            return dim
        except Exception as e:
            fallback_dim = getattr(Config.Model, "EMBEDDING_DIM", None) or 768
            logger.warning(
                "Could not fetch vector dim for collection '%s': %s; using fallback dim=%s",
                collection_name,
                e,
                fallback_dim,
            )
            self.collection_dims[collection_name] = int(fallback_dim)
            return int(fallback_dim)

    def retrieve(
            self,
            collection_name: str,
            query: str,
            subscription_id: str,
            filter_profile: str = None,
            top_k: int = 50,
            score_threshold: float = 0.10  # ✅ CHANGED FROM 0.15
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using Qdrant's native search."""
        try:
            self._ensure_profile(filter_profile)
            target_dim = self.get_collection_vector_dim(collection_name)
            model = get_model(required_dim=target_dim)
            q_dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
            if target_dim and q_dim and target_dim != q_dim:
                raise ValueError(f"Embedding dim mismatch: model produces {q_dim} but collection expects {target_dim}")
            query_vector = model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()
            if target_dim and len(query_vector) != target_dim:
                raise ValueError(f"Embedding dim {len(query_vector)} does not match collection dim {target_dim}")
        except Exception as err:
            logger.error(f"Failed to embed query for retrieval: {err}", exc_info=True)
            return []

        logger.info(f"Searching collection '{collection_name}' for query: {query[:100]}")

        query_filter = self._build_filter(subscription_id, str(filter_profile))

        results = self.run_search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="content_vector",
            score_threshold=score_threshold
        )

        if not results or not getattr(results, "points", []):
            logger.debug(f"No results found in collection '{collection_name}'")
            return []

        points = results.points or []
        logger.info("Qdrant returned %d hits", len(points))
        logger.info("Top scores: %s", [p.score for p in points[:3]])

        chunks: List[RetrievedChunk] = []
        for pt in points:
            payload = pt.payload or {}
            text = get_embedding_text(payload)
            snippet = (get_content_text(payload) or text)[:120].replace("\n", " ")
            logger.debug("Hit score=%.4f snippet=%s", pt.score, snippet)
            source = get_source_name(payload) or payload.get("source") or "unknown"
            chunk = RetrievedChunk(
                id=str(pt.id),
                text=text,
                score=float(pt.score),
                source=source,
                metadata=payload
            )
            chunks.append(chunk)

        return chunks

    def hybrid_retrieve(
            self,
            collection_name: str,
            query: str,
            subscription_id: str,
            profile_id: str,
            top_k: int = 50,
            document_ids: Optional[List[str]] = None,
            source_files: Optional[List[str]] = None,
            doc_domains: Optional[List[str]] = None,
            section_kinds: Optional[List[str]] = None,
            score_threshold: float = 0.05,
    ) -> List[RetrievedChunk]:
        """Hybrid dense + sparse retrieval with reciprocal rank fusion."""
        self._ensure_profile(profile_id)
        query_filter = self._build_filter(
            subscription_id,
            profile_id,
            document_ids=document_ids,
            source_files=source_files,
            doc_domains=doc_domains,
            section_kinds=section_kinds,
        )

        target_dim = self.get_collection_vector_dim(collection_name)
        model = get_model(required_dim=target_dim)
        dense_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32).tolist()
        if target_dim and len(dense_vector) != target_dim:
            logger.error(
                "Embedding dim %s does not match collection dim %s for collection '%s'",
                len(dense_vector),
                target_dim,
                collection_name,
            )
            return []
        sparse_vector = self._build_sparse_query(query)

        dense_results = self.run_search(
            collection_name=collection_name,
            query_vector=dense_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="content_vector",
            score_threshold=score_threshold,
        )
        sparse_results = self.run_search(
            collection_name=collection_name,
            query_vector=sparse_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="keywords_vector",
            score_threshold=None,
        )

        dense_chunks = self._points_to_chunks(dense_results, method="dense")
        sparse_chunks = self._points_to_chunks(sparse_results, method="sparse")

        if not dense_chunks and not sparse_chunks:
            return []

        merged = self._rrf_merge(dense_chunks, sparse_chunks)
        return merged[:top_k]

    def expand_with_neighbors(
            self,
            collection_name: str,
            seed_chunks: List[RetrievedChunk],
            subscription_id: str,
            profile_id: Optional[str],
            window: int = 1,
            max_new: int = 6
    ) -> List[RetrievedChunk]:
        """
        Pull immediate neighbor chunks (same document_id, adjacent chunk_index)
        to improve local context continuity.
        """
        if not seed_chunks or window <= 0:
            return seed_chunks

        seen = set()
        expanded = []

        for chunk in seed_chunks:
            meta = chunk.metadata or {}
            doc_id = meta.get("document_id")
            idx = meta.get("chunk_index")
            if doc_id is None or idx is None:
                expanded.append(chunk)
                continue

            key = (str(doc_id), str(idx))
            seen.add(key)
            expanded.append(chunk)

            if len(expanded) >= len(seed_chunks) + max_new:
                continue

            neighbor_filters = []

            # Prefer exact prev/next chunk ids to avoid schema/type issues on chunk_index.
            neighbor_ids = []
            for key_name in ("prev_chunk_id", "next_chunk_id"):
                neighbor_val = meta.get(key_name)
                if neighbor_val:
                    neighbor_ids.append(str(neighbor_val))
            if neighbor_ids:
                base = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id))
                must = list(getattr(base, "must", []) or [])
                must.append(FieldCondition(key="document_id", match=MatchValue(value=str(doc_id))))
                must.append(FieldCondition(key="chunk_id", match=MatchAny(any=neighbor_ids)))
                neighbor_filters.append(Filter(must=must))

            # Fallback to chunk_index equality (not range) to avoid 400s when stored as keyword.
            try:
                idx_int = int(idx)
            except Exception as exc:
                logger.debug("Failed to parse chunk index as integer", exc_info=True)
                idx_int = None
            if idx_int is not None and window > 0:
                neighbor_values = []
                for delta in range(-window, window + 1):
                    if delta == 0:
                        continue
                    candidate = idx_int + delta
                    if candidate < 0:
                        continue
                    neighbor_values.extend([candidate, str(candidate)])
                # Deduplicate while preserving order
                seen_values = set()
                neighbor_values = [v for v in neighbor_values if not (v in seen_values or seen_values.add(v))]
                if neighbor_values:
                    base = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id))
                    must = list(getattr(base, "must", []) or [])
                    must.append(FieldCondition(key="document_id", match=MatchValue(value=str(doc_id))))
                    must.append(FieldCondition(key="chunk_index", match=MatchAny(any=neighbor_values)))
                    neighbor_filters.append(Filter(must=must))

            if not neighbor_filters:
                continue

            for scroll_filter in neighbor_filters:
                try:
                    scroll = self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=scroll_filter,
                        with_payload=True,
                        with_vectors=False,
                        limit=max(window * 4, 2),  # small fetch per seed
                    )
                except Exception as exc:
                    logger.debug("Neighbor fetch failed for doc %s idx %s: %s", doc_id, idx, exc)
                    continue

                points = []
                if hasattr(scroll, "points"):
                    points = scroll.points or []
                elif isinstance(scroll, tuple) and scroll:
                    points = scroll[0] or []

                for pt in points:
                    payload = pt.payload or {}
                    neighbor_idx = payload.get("chunk_index")
                    neighbor_key = None
                    if neighbor_idx is not None:
                        neighbor_key = (str(payload.get("document_id", doc_id)), str(neighbor_idx))
                    if neighbor_key and neighbor_key in seen:
                        continue
                    text = get_embedding_text(payload)
                    if not text:
                        continue
                    if neighbor_key:
                        seen.add(neighbor_key)
                    neighbor_score = max(float(chunk.score) - 0.05, 0.0)
                    neighbor_source = (
                        get_source_name(payload)
                        or payload.get("source")
                        or chunk.source
                        or "unknown"
                    )
                    expanded.append(
                        RetrievedChunk(
                            id=str(pt.id),
                            text=text,
                            score=neighbor_score,
                            source=neighbor_source,
                            metadata=payload
                        )
                    )
                    if len(expanded) >= len(seed_chunks) + max_new:
                        break

        return expanded

    def get_profile_context(
            self,
            collection_name: str,
            subscription_id: str,
            profile_id: str,
            max_points: int = 400,
            refresh_seconds: int = 300
    ) -> ProfileContextSnapshot:
        """Build lightweight context from existing embeddings to guide vague queries."""
        cache_key = (collection_name, str(profile_id))
        now = time.time()
        cached = self.profile_context_cache.get(cache_key)
        if cached and (now - cached.last_updated) < refresh_seconds:
            return cached

        filter_ = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id))
        collected_points = []
        next_offset = None
        batch_size = min(120, max_points)

        try:
            while len(collected_points) < max_points:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_,
                    offset=next_offset,
                    limit=min(batch_size, max_points - len(collected_points)),
                    with_payload=True,
                    with_vectors=False
                )

                if hasattr(scroll_result, "points"):
                    batch = scroll_result.points or []
                    next_offset = getattr(scroll_result, "next_page_offset", None)
                elif isinstance(scroll_result, tuple):
                    batch = scroll_result[0] if len(scroll_result) > 0 else []
                    next_offset = scroll_result[1] if len(scroll_result) > 1 else None
                else:
                    batch = []
                    next_offset = None

                if not batch:
                    break

                collected_points.extend(batch)

                if not next_offset:
                    break
        except Exception as e:
            logger.warning(f"Failed to build profile context for {profile_id}: {e}")
            snapshot = ProfileContextSnapshot([], [], 0, now)
            self.profile_context_cache[cache_key] = snapshot
            return snapshot

        token_counts: Counter = Counter()
        doc_hints: List[str] = []
        seen_hints = set()

        for pt in collected_points:
            payload = pt.payload or {}
            text = get_embedding_text(payload)
            if text:
                token_counts.update(self.preprocessor.tokenize(text))

            hint_values = [
                get_source_name(payload),
                payload.get("document_id"),
                payload.get("section"),
            ]
            for hint_val in hint_values:
                if hint_val:
                    hint_val = str(hint_val)
                    if hint_val not in seen_hints:
                        doc_hints.append(hint_val)
                        seen_hints.add(hint_val)

        top_keywords = [w for w, _ in token_counts.most_common(40)]
        snapshot = ProfileContextSnapshot(
            top_keywords=top_keywords,
            document_hints=doc_hints[:12],
            total_chunks=len(collected_points),
            last_updated=now
        )
        self.profile_context_cache[cache_key] = snapshot
        return snapshot

class HybridReranker:
    """Reranks retrieved chunks using BM25 + vector scores with dynamic weighting."""

    def __init__(self, alpha: float = 0.7, cross_encoder: Optional[CrossEncoder] = None):
        """Initialize reranker."""
        self.alpha = alpha
        self.preprocessor = TextPreprocessor()
        self.cross_encoder = cross_encoder

    def adjust_alpha(self, query: str) -> float:
        """Dynamically adjust alpha based on query characteristics."""
        words = query.lower().split()

        question_words = {'what', 'why', 'how', 'explain', 'describe', 'understand'}
        if any(qw in words for qw in question_words):
            return 0.75

        if any(re.search(r'\d+', word) for word in words) or len(words) <= 3:
            return 0.6

        return self.alpha

    def rerank(
            self,
            chunks: List[RetrievedChunk],
            query: str,
            top_k: int = 10,
            use_cross_encoder: bool = True,
            diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Rerank chunks using hybrid BM25 + vector scoring."""
        if not chunks:
            return []

        try:
            alpha = self.adjust_alpha(query)
            logger.info(f"Using alpha={alpha:.2f} for reranking")

            texts = [get_embedding_text(chunk.metadata or {}) or chunk.text for chunk in chunks]
            normalized_texts = [self.preprocessor.normalize_text(text or "") for text in texts]
            tokenized_corpus = [self.preprocessor.tokenize(text) for text in normalized_texts]
            tokenized_query = self.preprocessor.tokenize(self.preprocessor.normalize_text(query))

            if any(len(doc) > 0 for doc in tokenized_corpus):
                bm25 = BM25Okapi(tokenized_corpus)
                bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float64)
            else:
                invalid_samples = [
                    {
                        "chunk_id": (chunk.metadata or {}).get("chunk_id") or chunk.id,
                        "length": len(chunk.text or ""),
                        "sample": _sanitize_snippet(chunk.text),
                    }
                    for chunk in chunks
                ]
                diag = diagnostics or {}
                retrieved_count = int(diag.get("retrieved_count", len(chunks)))
                dropped_invalid = int(diag.get("dropped_invalid_count", len(invalid_samples)))
                remaining = int(diag.get("remaining_chunks", max(0, retrieved_count - dropped_invalid)))
                samples = diag.get("invalid_samples") or invalid_samples
                logger.warning(
                    "BM25 skipped: all documents empty after tokenization | retrieved=%s "
                    "dropped_invalid=%s remaining=%s samples=%s",
                    retrieved_count,
                    dropped_invalid,
                    remaining,
                    samples[:3],
                )
                bm25_scores = np.zeros(len(chunks), dtype=np.float64)

            vector_scores = np.array([float(chunk.score) for chunk in chunks], dtype=np.float64)

            if len(vector_scores) > 1:
                v_min, v_max = vector_scores.min(), vector_scores.max()
                b_min, b_max = bm25_scores.min(), bm25_scores.max()

                v_range = v_max - v_min if v_max > v_min else 1.0
                b_range = b_max - b_min if b_max > b_min else 1.0

                vector_scores_norm = (vector_scores - v_min) / v_range
                bm25_scores_norm = (bm25_scores - b_min) / b_range
            else:
                vector_scores_norm = np.ones_like(vector_scores)
                bm25_scores_norm = np.ones_like(bm25_scores)

            hybrid_scores = (
                    alpha * vector_scores_norm +
                    (1 - alpha) * bm25_scores_norm
            )

            top_n_for_ce = min(top_k * 2, len(chunks))
            sorted_indices = np.argsort(hybrid_scores)[::-1][:top_n_for_ce]

            candidate_chunks = [chunks[idx] for idx in sorted_indices]

            if use_cross_encoder and self.cross_encoder and len(candidate_chunks) > 1:
                try:
                    pairs = [[query, chunk.text] for chunk in candidate_chunks]
                    ce_scores = self.cross_encoder.predict(pairs)

                    normalized = normalize_scores(ce_scores, expected_k=len(candidate_chunks))
                    for i, score in enumerate(normalized):
                        candidate_chunks[i].score = float(score)

                    candidate_chunks.sort(key=lambda c: to_py_scalar(c.score), reverse=True)
                    logger.info(f"Applied CrossEncoder reranking to {len(candidate_chunks)} chunks")

                except RerankShapeError as exc:
                    logger.warning(
                        "CrossEncoder rerank shape mismatch; falling back to hybrid scores: %s",
                        exc,
                        extra={
                            "stage": "rerank",
                            "provider": "cross_encoder",
                            "expected_k": exc.expected_k,
                            "actual_len": exc.actual_len,
                            "score_shape": exc.score_shape,
                        },
                    )
                    for i, chunk in enumerate(candidate_chunks):
                        chunk.score = float(hybrid_scores[sorted_indices[i]])
                except Exception as e:
                    logger.warning(f"CrossEncoder reranking failed: {e}")
                    for i, chunk in enumerate(candidate_chunks):
                        chunk.score = float(hybrid_scores[sorted_indices[i]])
            else:
                for i, chunk in enumerate(candidate_chunks):
                    chunk.score = float(hybrid_scores[sorted_indices[i]])

            reranked_chunks = candidate_chunks[:top_k]

            logger.info(f"Reranked to {len(reranked_chunks)} chunks (alpha={alpha:.2f})")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Error in reranking: {e}", exc_info=True)
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]

class ContextBuilder:
    """Builds formatted context for LLM with source citations."""

    @staticmethod
    def build_source_hints(chunks: List[RetrievedChunk]) -> str:
        """Create a compact source map to bias model attention to top evidence."""
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks[:5], 1):
            meta = chunk.metadata or {}
            source_name = chunk.source or get_source_name(meta) or f"Document {i}"
            page = meta.get('page') or meta.get('page_start') or meta.get('page_number')
            parts = [f"{i}) {source_name}"]
            if page is not None:
                parts.append(f"page={page}")
            lines.append(" | ".join(parts))
        return "SOURCE MAP:\n" + "\n".join(lines) + "\n"

    @staticmethod
    def build_context(chunks: List[RetrievedChunk], max_chunks: int = 3) -> str:
        """Build context string with source citations."""
        if not chunks:
            return ""

        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            text_value = get_content_text(chunk.metadata or {}) or chunk.text
            normalized = ' '.join((text_value or "").split())
            if normalized not in seen_texts and normalized.strip():
                seen_texts.add(normalized)
                unique_chunks.append(chunk)

        selected_chunks = unique_chunks[:max_chunks]

        context_parts = []
        source_map = ContextBuilder.build_source_hints(selected_chunks)
        if source_map:
            context_parts.append(source_map)
        for i, chunk in enumerate(selected_chunks, 1):
            # Use source_name directly from metadata as fallback
            source_name = chunk.source or get_source_name(chunk.metadata or {}) or f"Document {i}"
            content_text = get_content_text(chunk.metadata or {}) or chunk.text
            context_parts.append(
                f"[SOURCE: {source_name}]\n{content_text}\n[/SOURCE]"
            )

        return "\n".join(context_parts)

    @staticmethod
    def extract_sources(chunks: List[RetrievedChunk], max_sources: int = 3) -> List[Dict[str, Any]]:
        """Extract source information for response metadata."""
        sources = []
        for i, chunk in enumerate(chunks[:max_sources], 1):
            meta = chunk.metadata or {}
            page = meta.get("page") or meta.get("page_start") or meta.get("page_number")
            sources.append({
                'source_id': i,
                'source_name': chunk.source or get_source_name(meta) or f"Document {i}",
                'page': page,
            })
        return sources

class DomainPromptAdapter:
    """Creates lightweight, in-context adapters to steer local LLM responses."""

    @staticmethod
    def build_adapter(profile_context: Dict[str, Any], query: str) -> str:
        if not profile_context:
            return ""

        keywords = profile_context.get("keywords") or []
        hints = profile_context.get("hints") or []
        sampled = profile_context.get("sampled_chunks", 0)

        adapter_lines = []
        if keywords:
            adapter_lines.append(
                f"Domain keywords to preserve and prefer in wording: {', '.join(keywords[:12])}."
            )
        if hints:
            adapter_lines.append(f"Relevant documents/sections to anchor on: {', '.join(hints[:6])}.")
        if sampled:
            adapter_lines.append(f"Context built from {sampled} profile chunks; avoid generic responses.")

        adapter_lines.append(
            "Favor the terminology above when answering; align synonyms in the question to these domain terms."
        )
        adapter_lines.append(
            f"If the user query is vague ('{query}'), proactively ground the answer in the domain cues above."
        )
        return "\n".join(adapter_lines)

class AnswerabilityDetector:
    """Detects if a question can be answered from provided context."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def check_answerability(self, query: str, context: str, has_chunks: bool = False) -> Tuple[bool, str]:
        """
        Check if the query can be answered from the context.

        If we already have retrieved chunks, bias toward answering to avoid premature
        "cannot answer" responses when relevant evidence exists.
        """
        if has_chunks and context.strip():
            return True, "Context present from retrieved chunks"
        if _is_gemini_backend(self.llm_client):
            return True, "Gemini disabled for answerability checks"

        prompt = f"""You are an answerability classifier. Determine if the USER QUESTION can be answered using ONLY the information in the DOCUMENT CONTEXT.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

Respond with ONLY one of these formats:
- If answerable: "ANSWERABLE: <brief reason>"
- If not answerable: "NOT_ANSWERABLE: <what information is missing>"

Your response:"""

        try:
            response = self.llm_client.generate(prompt, system=build_system_prompt(), max_retries=2)
            response = response.strip()

            if response.startswith("ANSWERABLE"):
                return True, response.replace("ANSWERABLE:", "").strip()
            elif response.startswith("NOT_ANSWERABLE"):
                return False, response.replace("NOT_ANSWERABLE:", "").strip()
            else:
                return True, "Classification unclear"

        except Exception as e:
            logger.warning(f"Answerability check failed: {e}")
            return True, "Check failed"

class MetricsTracker:
    """Lightweight metrics sink for usage and quality signals."""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.retention_days = 120  # keep ~4 months of daily metrics

    @staticmethod
    def _day(ts: Optional[float] = None) -> str:
        return datetime.utcfromtimestamp(ts or time.time()).strftime("%Y-%m-%d")

    def record(
            self,
            model_name: str,
            subscription_id: str,
            profile_id: str,
            query_type: str,
            context_found: bool,
            grounded: bool,
            cached: bool,
            processing_time: float,
            retrieval_stats: Optional[Dict[str, Any]] = None
    ):
        if not self.redis:
            return

        day = self._day()
        base_key = f"rag:metrics:daily:{day}"
        model_key = f"rag:metrics:model:{day}"
        profile_key = f"rag:metrics:profile:{day}"

        stats = retrieval_stats or {}
        hits = float(stats.get("initial_retrieved", 0))
        final_ctx = float(stats.get("final_context", 0))

        pipe = self.redis.pipeline()
        pipe.hincrby(base_key, "total", 1)
        pipe.hincrby(base_key, f"type:{query_type}", 1)
        if context_found:
            pipe.hincrby(base_key, "context_found", 1)
        if grounded:
            pipe.hincrby(base_key, "grounded", 1)
        if cached:
            pipe.hincrby(base_key, "cache_hits", 1)
        pipe.hincrbyfloat(base_key, "sum_latency", float(processing_time or 0.0))
        pipe.hincrbyfloat(base_key, "sum_hits", hits)
        pipe.hincrbyfloat(base_key, "sum_final_ctx", final_ctx)
        pipe.hincrby(model_key, model_name or "unknown", 1)
        pipe.hincrby(profile_key, _slug(profile_id), 1)
        pipe.expire(base_key, self.retention_days * 86400)
        pipe.expire(model_key, self.retention_days * 86400)
        pipe.expire(profile_key, self.retention_days * 86400)
        try:
            pipe.execute()
        except Exception as exc:
            logger.debug("Metrics write failed: %s", exc)

    def summary(self, days: int = 7) -> Dict[str, Any]:
        if not self.redis:
            return {"available": False, "reason": "redis_unavailable"}

        days = max(1, min(days, 90))
        total = Counter()
        model_usage: Counter = Counter()
        profile_usage: Counter = Counter()

        for offset in range(days):
            day = self._day(time.time() - offset * 86400)
            base_key = f"rag:metrics:daily:{day}"
            model_key = f"rag:metrics:model:{day}"
            profile_key = f"rag:metrics:profile:{day}"

            data = self.redis.hgetall(base_key) or {}
            for k, v in data.items():
                try:
                    if k.startswith("sum_"):
                        total[k] += float(v)
                    else:
                        total[k] += int(v)
                except Exception as exc:
                    logger.debug("Failed to parse metrics value for key %s", k, exc_info=True)
                    continue

            for k, v in (self.redis.hgetall(model_key) or {}).items():
                try:
                    model_usage[k] += int(v)
                except Exception as exc:
                    logger.debug("Failed to parse model usage value for key %s", k, exc_info=True)
                    continue
            for k, v in (self.redis.hgetall(profile_key) or {}).items():
                try:
                    profile_usage[k] += int(v)
                except Exception as exc:
                    logger.debug("Failed to parse profile usage value for key %s", k, exc_info=True)
                    continue

        grand_total = int(total.get("total", 0))
        grounded = int(total.get("grounded", 0))
        context_found = int(total.get("context_found", 0))
        cache_hits = int(total.get("cache_hits", 0))
        not_answerable = int(total.get("type:not_answerable", 0))
        no_results = int(total.get("type:no_results", 0))

        # Derived rates
        def _pct(num: float, denom: float) -> float:
            return round((num / denom) * 100, 2) if denom else 0.0

        avg_latency = round(total.get("sum_latency", 0.0) / grand_total, 3) if grand_total else 0.0
        avg_hits = round(total.get("sum_hits", 0.0) / grand_total, 3) if grand_total else 0.0
        avg_ctx = round(total.get("sum_final_ctx", 0.0) / max(grounded, 1), 3)

        return {
            "available": True,
            "window_days": days,
            "totals": {
                "total_queries": grand_total,
                "grounded_answers": grounded,
                "context_found": context_found,
                "cache_hits": cache_hits,
                "not_answerable": not_answerable,
                "no_results": no_results,
            },
            "rates": {
                "accuracy_pct": _pct(grounded, grand_total),
                "context_found_pct": _pct(context_found, grand_total),
                "cache_hit_pct": _pct(cache_hits, grand_total),
                "not_answerable_pct": _pct(not_answerable, grand_total),
                "no_results_pct": _pct(no_results, grand_total),
            },
            "averages": {
                "latency_seconds": avg_latency,
                "retrieval_hits": avg_hits,
                "context_chunks_used": avg_ctx,
            },
            "usage": {
                "models": dict(model_usage.most_common(10)),
                "profiles": dict(profile_usage.most_common(10)),
            }
        }

class PromptBuilder:
    """Builds structured prompts with strict grounding and citation requirements."""

    # In dw_newron.py, replace PromptBuilder.build_qa_prompt method (around line 1700)

    @staticmethod
    def build_qa_prompt(
            query: str,
            context: str,
            persona: str,
            conversation_summary: str = "",
            domain_guidance: str = "",
            feedback_memory: str = "",
            retrieval_brief: str = "",
            kg_evidence: str = "",
            profile_id: Optional[str] = None,
            subscription_id: Optional[str] = None,
            redis_client: Optional[Any] = None,
    ) -> str:
        """Build a structured QA prompt with grounding and citation requirements."""
        convo_block = f"\nPRIOR CONVERSATION SUMMARY:\n{conversation_summary}\n" if conversation_summary else ""
        domain_block = f"\nDOMAIN ADAPTER:\n{domain_guidance}\n" if domain_guidance else ""
        feedback_block = f"\nRECENT FEEDBACK:\n{feedback_memory}\n" if feedback_memory else ""
        retrieval_block = f"\nRETRIEVAL NOTES:\n{retrieval_brief}\n" if retrieval_brief else ""
        kg_block = f"\nKG EVIDENCE PACK (use only if supported by Document Context):\n{kg_evidence}\n" if kg_evidence else ""
        extraction_mode = bool(_extract_requested_fields(query))

        output_shape_block = (
            "\nOUTPUT SHAPE (MANDATORY):\n"
            "1) Understanding & Scope (1-2 lines): intent + scope + files used.\n"
            "2) Answer: use domain-specific sections.\n"
            "3) Evidence & Gaps: what is missing + where searched.\n"
            "4) Optional next-step hint (no questions; only helpful suggestions).\n"
        )
        extraction_block = ""
        if extraction_mode:
            extraction_block = (
                "\nFIELD REQUESTS:\n"
                "- Use the requested field names inside the Answer section.\n"
                "- If a field is missing, write: Not found in the current profile documents.\n"
                "- Avoid raw extraction dumps and filler.\n"
            )

        prompt = f"""You are DocWain-Agent, a document intelligence model. Follow the system instructions exactly.
Use ONLY the retrieved document context and never add DocWain product intro unless the user asks about DocWain.
When information is missing, say so briefly and proceed with what is available.
{output_shape_block}{extraction_block}

DOCUMENT CONTEXT:
{context}
{kg_block}
{convo_block}
{domain_block}
{retrieval_block}
{feedback_block}

USER QUESTION: {query}

GROUNDING & CITATION RULES (MANDATORY):
1) Use ONLY the document context above; if something is missing, say so plainly.
2) EVERY factual claim must cite immediately with [SOURCE-X]; multiple sources -> [SOURCE-1, SOURCE-3].
3) Prefer quoting exact figures, names, dates, section titles, and short snippets with citations.
4) If sources disagree, briefly note the discrepancy with citations.
5) If you use a KG fact, it MUST appear in the Document Context and be cited with [SOURCE-X].
6) Do not invent, pad, or generalize beyond the provided text.

ANSWER STYLE (MANDATORY):
- Follow the output shape above exactly.
- No static filler like "Working on it".
- No raw extraction dumps like "items:", "amounts:", "terms:".
- If the question is about an entity not present, say so and list files searched, then stop.

Provide the answer now with citations inline after each claim."""

        from src.prompting.prompt_builder import inject_persona_prompt

        return inject_persona_prompt(
            prompt,
            persona,
            profile_id=profile_id,
            subscription_id=subscription_id,
            redis_client=redis_client,
        )

class ConversationSummarizer:
    """Summarizes the last few turns to keep context tight."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def summarize(self, conversation_text: str) -> str:
        if not conversation_text:
            return ""
        if _is_gemini_backend(self.llm_client):
            return ""
        prompt = f"""Summarize the following conversation turns into 3-5 concise bullets capturing user intent and assistant answers. Do NOT invent details.

CONVERSATION:
{conversation_text}

SUMMARY:"""
        try:
            summary = self.llm_client.generate(prompt, system=build_system_prompt(), max_retries=2, backoff=0.5)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Conversation summarization failed: {e}")
            return ""

class EnterpriseRAGSystem:
    """
    Enhanced RAG system with query reformulation, conversational context,
    cross-encoder reranking, and grounding.
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            profile_id: Optional[str] = None,
            backend_override: Optional[str] = None,
            model_path: Optional[str] = None,
            llm_client: Optional[Any] = None,
            qdrant_client: Optional[QdrantClient] = None,
            embedder: Optional[SentenceTransformer] = None,
            cross_encoder: Optional[Any] = None,
            redis_client: Optional[redis.Redis] = None,
    ):
        """Initialize the RAG system with lazy-loaded components."""
        try:
            # Initialize LLM client (Ollama by default, Gemini when requested)
            self.llm_client = llm_client or create_llm_client(
                model_name,
                backend_override=backend_override,
                model_path=model_path
            )
            self.model_name = getattr(self.llm_client, "model_name", model_name or "default")

            # Initialize other components
            qdrant_client = qdrant_client or get_qdrant_client()
            model = embedder or get_model()
            cross_encoder = cross_encoder if cross_encoder is not None else get_cross_encoder()

            self.model = model
            self.client = qdrant_client
            self.retriever = QdrantRetriever(qdrant_client, model)
            self.reranker = HybridReranker(alpha=0.7, cross_encoder=cross_encoder)
            self.context_builder = ContextBuilder()
            self.intelligent_context_builder = IntelligentContextBuilder(
                max_context_chunks=getattr(Config.Retrieval, "MAX_CONTEXT_CHUNKS", 16)
            )
            self.prompt_builder = PromptBuilder()
            self.greeting_handler = GreetingHandler()
            self.query_reformulator = QueryReformulator(self.llm_client)
            self.answerability_detector = AnswerabilityDetector(self.llm_client)
            self.retrieval_planner = RetrievalPlanner(self.llm_client)
            self.evidence_synthesizer = EvidenceSynthesizer(self.llm_client)
            # Initialize Redis client for storing conversation history and feedback.
            redis_client = redis_client or get_redis_client()
            self.redis_client = redis_client

            # Initialize conversation history backed by Redis. Avoid instantiating
            # a non-redis version first since it would immediately be overwritten.
            self.conversation_history = ConversationHistory(max_turns=3, redis_client=redis_client)
            self.conversation_summarizer = ConversationSummarizer(self.llm_client)
            self.feedback_memory = ChatFeedbackMemory(max_items=12, redis_client=redis_client)
            self.conversation_state = ConversationState(
                conversation_history=self.conversation_history,
                redis_client=redis_client,
            )
            self.progressive_summarizer = ProgressiveSummarizer(
                llm_client=self.llm_client,
            )
            self._warm_up_llm()

            neo4j_store = None
            if getattr(Config.KnowledgeGraph, "ENABLED", False):
                try:
                    neo4j_store = Neo4jStore()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Neo4j unavailable, KG disabled: %s", exc)
            self.graph_augmenter = GraphAugmenter(
                neo4j_store=neo4j_store,
                enabled=bool(neo4j_store),
            )
            self.graph_support_scorer = GraphSupportScorer(
                alpha=float(getattr(Config.KnowledgeGraph, "GRAPH_SCORE_ALPHA", 0.7))
            )

            logger.info("EnterpriseRAGSystem initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EnterpriseRAGSystem: {e}")
            raise

    # In dw_newron.py, line ~1800 in preprocess_query method

    def preprocess_query(
            self,
            query: str,
            user_id: str,
            namespace: str,
            use_reformulation: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Preprocess query with normalization and optional reformulation."""
        metadata = {
            'original_query': query,
            'corrections': [],
            'reformulated': False,
            'expanded': False,
            'intent': self._detect_intent_llm(query)
        }

        processed_query = re.sub(r"\s+", " ", query or "").strip()
        metadata['normalized'] = processed_query

        if use_reformulation and self._is_query_vague(processed_query):
            conv_context = self.conversation_history.get_context(namespace, user_id, max_turns=2)
            reformulated = self.query_reformulator.reformulate(processed_query, conv_context)
            if reformulated and reformulated != processed_query:
                processed_query = reformulated
                metadata['reformulated'] = True
            elif conv_context:
                processed_query = f"{processed_query} ; context: {conv_context}"

        expanded = self._expand_query_terms(processed_query)
        if expanded and expanded != processed_query:
            processed_query = expanded
            metadata['expanded'] = True

        return processed_query, metadata

    @staticmethod
    def _is_retrieval_sufficient(chunks: List[RetrievedChunk], min_hits: int = 3, min_score: float = 0.18) -> bool:
        """Decide if current retrieval is good enough to stop trying fallbacks."""
        if not chunks:
            return False
        if len(chunks) >= min_hits:
            return True
        top_score = float(chunks[0].score)

        return top_score >= min_score

    @staticmethod
    def _is_query_vague(query: str) -> bool:
        """Heuristic to detect short or underspecified questions."""
        tokens = query.split()
        if len(tokens) <= 3:
            return True
        meaningful_tokens = [t for t in tokens if len(t) > 3]
        return len(meaningful_tokens) <= 2

    @staticmethod
    def _detect_intent(query: str) -> str:
        q = query.lower()
        if any(word in q for word in ["how", "steps", "procedure", "configure", "setup"]):
            return "procedural"
        if any(word in q for word in ["error", "issue", "fail", "troubleshoot", "bug"]):
            return "troubleshooting"
        if any(word in q for word in ["difference", "compare", "vs", "versus", "comparison"]):
            return "comparison"
        return "factual"

    def _detect_intent_llm(self, query: str) -> str:
        parsed = parse_intent(query=query, llm_client=self.llm_client, redis_client=getattr(self, "redis_client", None))
        if parsed and parsed.intent:
            mapping = {
                "compare": "comparison",
                "summarize": "factual",
                "rank": "factual",
                "list": "factual",
                "extract": "factual",
                "contact": "factual",
                "qa": "factual",
                "generate": "procedural",
            }
            return mapping.get(parsed.intent, "factual")
        return self._detect_intent(query)

    @staticmethod
    def _expand_query_terms(query: str) -> str:
        synonyms = {
            "install": ["setup", "configure"],
            "error": ["failure", "issue", "exception"],
            "policy": ["guideline", "rule", "standard"],
            "process": ["procedure", "workflow"],
        }
        additions = []
        lower_q = query.lower()
        for term, syns in synonyms.items():
            if term in lower_q:
                additions.extend(syns)
        if additions:
            return query + " " + " ".join(additions)
        return query

    @staticmethod
    def _contextualize_query(query: str, profile_context: ProfileContextSnapshot) -> Tuple[str, Dict[str, Any]]:
        """Blend the user query with profile-specific hints to guide retrieval."""
        if not profile_context or not (profile_context.top_keywords or profile_context.document_hints):
            return query, {}

        keywords = profile_context.top_keywords[:8]
        hints = profile_context.document_hints[:3]
        extras = []
        if hints:
            extras.append("related to " + ", ".join(hints))
        if keywords:
            extras.append("keywords: " + ", ".join(keywords))

        contextual_query = " ; ".join([query] + extras)
        return contextual_query, {"profile_keywords_used": keywords, "profile_hints_used": hints}

    def extract_person_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract person's name from query to filter by specific document.
        """
        import re

        if not query:
            return None

        query = query.strip()

        patterns = [
            # education/skills/experience of Name
            r'\b(?:education|skills?|experience|background|qualification|details?)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',

            # Name's education / skills
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[\'’]s\s+(?:education|skills?|experience)',

            # tell me about Name
            r'\btell\s+me\s+about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
            # about Name / for Name / regarding Name
            r'\b(?:about|for|regarding)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
            # patient/vendor Name
            r'\b(?:patient|vendor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def _normalize_plan_domain(value: Optional[str]) -> str:
        if not value:
            return "generic"
        cleaned = str(value).strip().lower().replace(" ", "_")
        aliases = {
            "bank": "bank_statement",
            "bankstatement": "bank_statement",
            "bank-statement": "bank_statement",
            "purchaseorder": "purchase_order",
            "purchase-order": "purchase_order",
            "po": "purchase_order",
            "cv": "resume",
        }
        cleaned = aliases.get(cleaned, cleaned)
        if cleaned in {"resume", "medical", "invoice", "tax", "bank_statement", "purchase_order", "generic"}:
            return cleaned
        return "generic"

    @staticmethod
    def _normalize_filename(value: str) -> str:
        base = os.path.basename(str(value or "")).lower()
        base = re.sub(r"\.[a-z0-9]{2,5}$", "", base)
        base = base.replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", base).strip()

    def _build_available_documents(
        self,
        subscription_id: str,
        profile_id: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        try:
            profile_index = build_profile_document_index(subscription_id, profile_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to build profile document index: %s", exc)
            return [], None

        documents: List[Dict[str, Any]] = []
        for entry in profile_index.documents.values():
            if not entry.source_name:
                continue
            domain = self._normalize_plan_domain(entry.document_type)
            doc_payload = {"file_name": entry.source_name}
            if domain != "generic":
                doc_payload["doc_domain"] = domain
            documents.append(doc_payload)
        return documents, profile_index

    def _match_target_files(self, query: str, available_documents: List[Dict[str, Any]]) -> List[str]:
        if not query or not available_documents:
            return []
        lowered = query.lower()
        matches: List[str] = []
        for doc in available_documents:
            name = doc.get("file_name") or ""
            if not name:
                continue
            base = os.path.basename(name).lower()
            base_no_ext = re.sub(r"\.[a-z0-9]{2,5}$", "", base)
            if base and base in lowered:
                matches.append(os.path.basename(name))
                continue
            if base_no_ext and base_no_ext in lowered:
                matches.append(os.path.basename(name))
                continue
        seen = set()
        return [m for m in matches if not (m in seen or seen.add(m))]

    def _resolve_doc_ids_for_files(self, profile_index: Optional[Any], target_files: List[str]) -> List[str]:
        if not profile_index or not target_files:
            return []
        mapping: Dict[str, List[str]] = {}
        for doc_id, entry in profile_index.documents.items():
            name = entry.source_name or ""
            if not name:
                continue
            base = os.path.basename(name).lower()
            base_no_ext = re.sub(r"\.[a-z0-9]{2,5}$", "", base)
            mapping.setdefault(base, []).append(str(doc_id))
            if base_no_ext:
                mapping.setdefault(base_no_ext, []).append(str(doc_id))

        doc_ids: List[str] = []
        for file_name in target_files:
            base = os.path.basename(str(file_name)).lower()
            base_no_ext = re.sub(r"\.[a-z0-9]{2,5}$", "", base)
            for key in (base, base_no_ext):
                doc_ids.extend(mapping.get(key, []))

        seen = set()
        return [doc_id for doc_id in doc_ids if not (doc_id in seen or seen.add(doc_id))]

    def _filter_chunks_by_plan(
        self,
        chunks: List[RetrievedChunk],
        *,
        plan_domain: Optional[str],
        target_files: Optional[List[str]],
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        domain = self._normalize_plan_domain(plan_domain)
        target_files = [os.path.basename(f) for f in (target_files or []) if f]
        target_bases = {self._normalize_filename(f) for f in target_files}
        filtered: List[RetrievedChunk] = []
        for chunk in chunks:
            meta = getattr(chunk, "metadata", {}) or {}
            if domain and domain != "generic":
                chunk_domain = self._normalize_plan_domain(
                    meta.get("doc_domain")
                    or meta.get("document_type")
                    or meta.get("doc_type")
                    or (meta.get("document") or {}).get("type")
                )
                if chunk_domain not in {"generic", domain}:
                    continue
            if target_files:
                source = get_source_name(meta) or chunk.source or ""
                source_base = os.path.basename(str(source)) if source else ""
                source_norm = self._normalize_filename(source_base)
                if source_base not in target_files and source_norm not in target_bases:
                    continue
            filtered.append(chunk)
        return filtered

    def find_document_ids_by_name(
            self,
            collection_name: str,
            person_name: str,
            profile_id: str,
            subscription_id: str,
    ) -> List[str]:
        try:
            #  Encode name
            query_vector = self.model.encode(
                person_name,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()

            #  Build filter (CORRECT way)
            qdrant_filter = build_qdrant_filter(
                subscription_id=str(subscription_id),
                profile_id=str(profile_id),
            )

            #  Call Qdrant
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="content_vector",  # ✅ REQUIRED
                query_filter=qdrant_filter,
                limit=10,
                with_payload=True,
                with_vectors=False
            )

            #  Validate response
            if not results or not results.points:
                return None

            #  Extract matching document_id(s)
            doc_ids: List[str] = []
            for pt in results.points:
                payload = pt.payload or {}
                text = (get_content_text(payload) or get_embedding_text(payload)).lower()
                source = (get_source_name(payload) or "").lower()

                for part in person_name.lower().split():
                    if part in text or part in source:
                        doc_id = payload.get("document_id")
                        if doc_id:
                            doc_ids.append(str(doc_id))
                        break

            seen = set()
            return [doc_id for doc_id in doc_ids if not (doc_id in seen or seen.add(doc_id))]

        except Exception as e:
            logger.error(f"Error finding document by name '{person_name}': {e}")
            return []

    def retrieve_with_priorities(
            self,
            query: str,
            user_id: str,
            profile_id: str,
            subscription_id: str,
            collection_name: str,
            namespace: str,
            top_k_retrieval: int = 50
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval with mandatory profile filters, query rewriting, and intent detection.
        """
        if not profile_id:
            raise ValueError("profile_id is required for retrieval")

        primary_query, primary_metadata = self.preprocess_query(query, user_id, namespace, use_reformulation=True)
        is_vague = self._is_query_vague(primary_query)
        attempt_records = []
        graph_hints = GraphHints()
        retrieval_query = primary_query
        if self.graph_augmenter:
            graph_hints = self.graph_augmenter.augment(
                primary_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                agent_mode=False,
            )
            if graph_hints.query_expansion_terms:
                expansion = " ".join(graph_hints.query_expansion_terms)
                retrieval_query = f"{primary_query} {expansion}"

        profile_context = self.retriever.get_profile_context(collection_name, subscription_id, profile_id)
        profile_context_data = {
            "keywords": profile_context.top_keywords[:12],
            "hints": profile_context.document_hints[:6],
            "sampled_chunks": profile_context.total_chunks
        }

        available_documents, profile_index = self._build_available_documents(subscription_id, profile_id)
        explicit_target_files = self._match_target_files(primary_query, available_documents)
        target_document_name = explicit_target_files[0] if len(explicit_target_files) == 1 else None
        retrieval_plan = self.retrieval_planner.plan(
            user_query=primary_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            target_document_name=target_document_name,
            available_documents=available_documents,
        )

        plan_scope = retrieval_plan.get("scope") or {}
        plan_target_files = list(plan_scope.get("target_files") or [])
        if explicit_target_files:
            plan_target_files = list(dict.fromkeys(plan_target_files + explicit_target_files))
        plan_target_files = [f for f in plan_target_files if f]
        plan_domain = self._normalize_plan_domain(retrieval_plan.get("domain"))
        plan_filters = (retrieval_plan.get("retrieval") or {}).get("filters") or {}
        raw_doc_domain = plan_filters.get("doc_domain")
        if isinstance(raw_doc_domain, list):
            raw_doc_domain = raw_doc_domain[0] if raw_doc_domain else None
        plan_doc_domain = self._normalize_plan_domain(raw_doc_domain)
        doc_domain_filter = None
        if plan_doc_domain and plan_doc_domain != "generic":
            doc_domain_filter = [plan_doc_domain]
        elif plan_domain and plan_domain != "generic":
            doc_domain_filter = [plan_domain]

        plan_top_k = (retrieval_plan.get("retrieval") or {}).get("top_k")
        if isinstance(plan_top_k, int) and plan_top_k > 0:
            top_k_retrieval = min(top_k_retrieval, plan_top_k)

        plan_name_query = retrieval_plan.get("name_query") or {}
        planner_name = plan_name_query.get("name")
        person_name = planner_name or self.extract_person_name_from_query(primary_query)
        name_scope_enabled = bool(plan_name_query.get("enabled")) or bool(person_name)

        doc_ids_from_files = self._resolve_doc_ids_for_files(profile_index, plan_target_files)
        target_document_ids: List[str] = []
        document_filter_ids: Optional[List[str]] = doc_ids_from_files or None

        if name_scope_enabled and person_name:
            logger.info(" Detected person-specific query for: '%s'", person_name)
            target_document_ids = self.find_document_ids_by_name(
                collection_name, person_name, profile_id, subscription_id
            )

            if target_document_ids:
                if document_filter_ids:
                    document_filter_ids = [doc_id for doc_id in target_document_ids if doc_id in document_filter_ids]
                else:
                    document_filter_ids = target_document_ids
                logger.info(" Will filter results to documents: %s", document_filter_ids)
            else:
                logger.warning(
                    " Could not find documents for '%s' - enforcing name scope with no matches",
                    person_name,
                )
        elif doc_ids_from_files:
            target_document_ids = doc_ids_from_files

        primary_metadata["vague_query"] = is_vague
        primary_metadata["profile_context"] = profile_context_data
        primary_metadata["person_name"] = person_name
        primary_metadata["target_document_ids"] = document_filter_ids or []
        primary_metadata["target_files"] = plan_target_files
        primary_metadata["retrieval_plan"] = retrieval_plan

        if name_scope_enabled and person_name and not document_filter_ids:
            attempt_records.append(
                {
                    "label": "name_lookup",
                    "query": person_name,
                    "hits": 0,
                    "top_score": 0.0,
                    "document_filter": [],
                }
            )
            return {
                "chunks": [],
                "query": primary_query,
                "metadata": primary_metadata,
                "attempts": attempt_records,
                "selected_strategy": "name_lookup",
                "profile_context": profile_context_data,
                "graph_hints": graph_hints,
            }

        must_not_fallback = bool(
            (retrieval_plan.get("retrieval") or {}).get("must_not_fallback_to_unfiltered", True)
        )
        candidate_doc_ids = graph_hints.candidate_filters.get("document_ids") if graph_hints else None
        if candidate_doc_ids and document_filter_ids:
            candidate_doc_ids = [doc_id for doc_id in candidate_doc_ids if doc_id in document_filter_ids]
        kg_chunks: List[RetrievedChunk] = []
        if candidate_doc_ids:
            kg_chunks = self.retriever.hybrid_retrieve(
                collection_name=collection_name,
                query=retrieval_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                top_k=min(top_k_retrieval, 60),
                document_ids=candidate_doc_ids,
                source_files=plan_target_files or None,
                doc_domains=doc_domain_filter,
            )
        chunks = self.retriever.hybrid_retrieve(
            collection_name=collection_name,
            query=retrieval_query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            top_k=top_k_retrieval,
            document_ids=document_filter_ids,
            source_files=plan_target_files or None,
            doc_domains=doc_domain_filter,
        )
        if kg_chunks:
            chunks = self.retriever._rrf_merge(kg_chunks, chunks)
            chunks = chunks[:top_k_retrieval]
        if doc_domain_filter or plan_target_files:
            chunks = self._filter_chunks_by_plan(
                chunks,
                plan_domain=doc_domain_filter[0] if doc_domain_filter else plan_domain,
                target_files=plan_target_files,
            )
        try:
            intent_analysis = QueryAnalyzer().analyze(primary_query)
            required_attrs = extract_required_attributes(primary_query, intent_analysis.intent_type)
            chunks = filter_chunks_by_intent(
                chunks,
                required_attrs,
                intent_analysis.entities,
                intent_analysis.intent_type,
            )
            primary_metadata["intent_filter"] = {
                "intent_type": intent_analysis.intent_type,
                "required_attributes": required_attrs,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Intent filter skipped: %s", exc)
        attempt_records.append(
            {
                "label": "hybrid",
                "query": retrieval_query,
                "hits": len(chunks),
                "top_score": round(float(chunks[0].score), 4) if chunks else 0.0,
                "document_filter": document_filter_ids,
            }
        )

        if not chunks and document_filter_ids and not person_name and not must_not_fallback:
            chunks = self.retriever.hybrid_retrieve(
                collection_name=collection_name,
                query=retrieval_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                top_k=top_k_retrieval,
                document_ids=None,
                source_files=None,
                doc_domains=doc_domain_filter,
            )
            attempt_records.append(
                {
                    "label": "hybrid_no_doc_filter",
                    "query": retrieval_query,
                    "hits": len(chunks),
                    "top_score": round(float(chunks[0].score), 4) if chunks else 0.0,
                    "document_filter": None,
                }
            )

        return {
            "chunks": chunks,
            "query": primary_query,
            "metadata": primary_metadata,
            "attempts": attempt_records,
            "selected_strategy": "hybrid",
            "profile_context": profile_context_data,
            "graph_hints": graph_hints,
        }

    def _warm_up_llm(self):
        """Warm LLM backend so first user calls do not fail cold."""
        warm_fn = getattr(self.llm_client, "warm_up", None)
        if callable(warm_fn):
            try:
                warm_fn()
            except Exception as e:
                logger.warning(f"LLM warm-up skipped due to error: {e}")

    def answer_question(
            self,
            query: str,
            profile_id: str,
            subscription_id: str,
            user_id: str,
            persona: str = "professional document analysis assistant",
            top_k_retrieval: int = 80,
            top_k_rerank: int = 25,
            final_k: int = 12,
            session_id: Optional[str] = None,
            new_session: bool = False,
            disable_answer_cache: bool = False,
            force_refresh: bool = False,
            request_id: Optional[str] = None,
            index_version: Optional[str] = None,
            tools: Optional[List[str]] = None,
            use_tools: bool = False,
            tool_inputs: Optional[Dict[str, Any]] = None,
            enable_internet: bool = False,
    ) -> Dict[str, Any]:
        """Main method to answer questions using enhanced RAG pipeline."""
        start_time = time.time()
        telemetry = telemetry_store() if METRICS_V2_ENABLED else None
        metrics_store = get_metrics_store()
        tool_list = tools or []
        use_tooling = bool(use_tools or tool_list)
        global _FIRST_METRICS_REQUEST
        is_cold_start = False
        if _FIRST_METRICS_REQUEST:
            is_cold_start = True
            _FIRST_METRICS_REQUEST = False

        try:
            if not profile_id:
                raise ValueError("profile_id is required for retrieval")
            collection_name = build_collection_name(subscription_id)
            logger.info(
                "Query from user=%s subscription=%s profile=%s collection=%s | q=%s",
                user_id, subscription_id, profile_id, collection_name,
                (query or "")[:120],
            )
            try:
                _ensure_qdrant_indexes(self.client, collection_name)
            except RetrievalFilterError as exc:
                # Index bootstrap is best-effort — indexes are created at startup
                # and during document ingestion. Don't block queries if the check
                # fails (e.g., transient Qdrant network issue). The actual query
                # will fail more specifically if indexes are truly missing.
                logger.warning(
                    "Qdrant index bootstrap check failed (non-blocking): %s | subscription=%s collection=%s",
                    exc, subscription_id, collection_name,
                )
            namespace = _build_namespace(subscription_id, profile_id, self.model_name, session_id)
            metrics = get_metrics_tracker()

            effective_new_session = new_session or force_refresh
            if effective_new_session:
                logger.info("Resetting conversation context for new session: %s", session_id or "default")
                self.conversation_history.clear_history(namespace, user_id)
                self.conversation_state.clear(namespace, user_id)
                try:
                    self.feedback_memory.clear(namespace, user_id)
                except Exception as feedback_exc:
                    logger.debug("Feedback memory clear failed: %s", feedback_exc)

            # Quick collection diagnostics — profile-filtered for accurate count
            try:
                _count_cache_key = f"{collection_name}:{profile_id}"
                cached = _COLLECTION_COUNT_CACHE.get(_count_cache_key)
                now = time.time()
                if cached and (now - cached[0]) < _COLLECTION_COUNT_TTL_SEC:
                    total_points = cached[1]
                else:
                    _profile_filter = build_qdrant_filter(
                        subscription_id=str(subscription_id),
                        profile_id=str(profile_id),
                    )
                    stats = self.client.count(
                        collection_name=collection_name,
                        count_filter=_profile_filter,
                        exact=False,
                    )
                    total_points = int(getattr(stats, "count", 0) or 0)
                    _COLLECTION_COUNT_CACHE[_count_cache_key] = (now, total_points)
                logger.info(f"Collection '{collection_name}' profile='{profile_id}' point count: {total_points}")
                if total_points == 0:
                    logger.debug(f"Collection '{collection_name}' is empty for profile '{profile_id}'; retrieval will return no results")
            except Exception as diag_exc:
                logger.warning(f"Could not count collection '{collection_name}': {diag_exc}")

            if self.greeting_handler.is_positive_feedback(query):
                feedback_response = "You're welcome! If you want me to dig into another document or topic, just let me know."
                # Template is fast and sufficient; skip LLM call to avoid
                # model-swap latency.
                return {
                    "response": feedback_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "positive_feedback",
                    "grounded": True
                }

            if self.greeting_handler.is_greeting(query):
                from src.intelligence.redis_intel_cache import RedisIntelCache
                from src.intelligence.response_composer import build_greeting_response

                catalog = {}
                if self.redis_client:
                    cache = RedisIntelCache(self.redis_client)
                    catalog = cache.get_json(cache.catalog_key(subscription_id, profile_id)) or {}
                greeting_response = build_greeting_response(catalog)
                # Template is fast and sufficient; skip LLM call to avoid
                # model-swap latency (200s+ when Ollama has to reload models).
                self.conversation_history.add_turn(namespace, user_id, query, greeting_response)

                return {
                    "response": greeting_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "greeting",
                    "grounded": True
                }

            if self.greeting_handler.is_farewell(query):
                farewell_response = "Thanks for chatting. If you need anything else, come back anytime."
                # Template is fast and sufficient; skip LLM call to avoid
                # model-swap latency.
                self.conversation_history.clear_history(namespace, user_id)

                return {
                    "response": farewell_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": True,
                    "query_type": "farewell",
                    "grounded": True
                }

            # General conversational intercept (document discovery, identity, etc.)
            # Skip conversational intercept when internet is enabled — let the query
            # reach the RAG pipeline where web search fallback can trigger.
            if not enable_internet:
                try:
                    from src.intelligence.conversational_nlp import generate_conversational_response
                    _catalog = {}
                    if self.redis_client:
                        from src.intelligence.redis_intel_cache import RedisIntelCache
                        _cache = RedisIntelCache(self.redis_client)
                        _catalog = _cache.get_json(_cache.catalog_key(subscription_id, profile_id)) or {}
                    _conv_resp = generate_conversational_response(
                        query,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        collection_point_count=total_points,
                        catalog=_catalog,
                    )
                    if _conv_resp and _conv_resp.text:
                        return {
                            "response": _conv_resp.text,
                            "sources": [],
                            "user_id": user_id,
                            "collection": collection_name,
                            "request_id": request_id,
                            "index_version": index_version,
                            "context_found": False,
                            "query_type": _conv_resp.intent.lower(),
                            "grounded": False,
                        }
                except Exception as exc:
                    logger.debug("Conversation-driven response failed", exc_info=True)

            if getattr(Config, "RAGV3", None) and getattr(Config.RAGV3, "ENABLED", False):
                try:
                    from src.rag_v3.pipeline import run_docwain_rag_v3

                    # Multi-turn context: inject conversation history for follow-up queries
                    _v3_conv_context = None
                    try:
                        _v3_conv = self.conversation_history.get_context(
                            namespace, user_id, max_turns=3, max_chars=1200,
                        )
                        if _v3_conv and _v3_conv.strip():
                            # Only inject for follow-up queries (pronouns, short queries, references)
                            _lower_q = (query or "").lower().strip()
                            _is_followup = (
                                any(p in _lower_q.split() for p in ("it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those", "he", "she", "his", "her", "same", "above", "previous", "prior", "earlier", "aforementioned", "similar", "other", "another", "former", "latter", "said", "such", "both"))
                                or len(_lower_q.split()) <= 4
                                or any(w in _lower_q for w in ("the same", "more about", "what about", "how about", "also", "and what", "tell me more"))
                            )
                            if _is_followup:
                                _v3_conv_context = _v3_conv
                                logger.info(
                                    "Multi-turn context injected: %d chars for follow-up query",
                                    len(_v3_conv_context),
                                )
                    except Exception as exc:
                        logger.debug("Failed to load conversation history for multi-turn context", exc_info=True)

                    v3_answer = run_docwain_rag_v3(
                        query=query,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        session_id=session_id,
                        user_id=user_id,
                        request_id=request_id,
                        llm_client=self.llm_client,
                        qdrant_client=self.client,
                        redis_client=self.redis_client,
                        embedder=self.model,
                        cross_encoder=getattr(self.reranker, "cross_encoder", None),
                        tools=tool_list if use_tooling else None,
                        tool_inputs=tool_inputs if use_tooling else None,
                        enable_internet=enable_internet,
                        conversation_context=_v3_conv_context,
                    )
                    if v3_answer:
                        return v3_answer
                except Exception as exc:  # noqa: BLE001
                    exc_lower = str(exc).lower()
                    stage = "generate"
                    if "rerank" in exc_lower:
                        stage = "rerank"
                    elif "retrieve" in exc_lower or "qdrant" in exc_lower:
                        stage = "retrieve"
                    logger.warning(
                        "DocWain RAG v3 failed; falling back: %s",
                        exc,
                        extra={
                            "stage": stage,
                            "correlation_id": request_id,
                            "session_id": session_id,
                            "provider": "rag_v3",
                        },
                        exc_info=True,
                    )

            if getattr(Config, "RAGV2", None) and getattr(Config.RAGV2, "ENABLED", False):
                try:
                    if not hasattr(self.model, "encode"):
                        raise RuntimeError("Embedder missing encode")
                    from src.ask.pipeline import run_docwain_rag_v2

                    v2_answer = run_docwain_rag_v2(
                        query=query,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        session_id=session_id,
                        user_id=user_id,
                        request_id=request_id,
                        llm_client=self.llm_client,
                        qdrant_client=self.client,
                        redis_client=self.redis_client,
                        embedder=self.model,
                        cross_encoder=getattr(self.reranker, "cross_encoder", None),
                    )
                    if v2_answer:
                        if v2_answer.get("context_found") or (v2_answer.get("sources") or []):
                            return v2_answer
                        if (v2_answer.get("metadata") or {}).get("intent") == "greet":
                            return v2_answer
                except Exception as exc:  # noqa: BLE001
                    exc_lower = str(exc).lower()
                    stage = "generate"
                    if isinstance(exc, RerankShapeError) or "rerank" in exc_lower:
                        stage = "rerank"
                    elif "retrieve" in exc_lower or "qdrant" in exc_lower:
                        stage = "retrieve"
                    logger.warning(
                        "DocWain RAG v2 failed; falling back: %s",
                        exc,
                        extra={
                            "stage": stage,
                            "correlation_id": request_id,
                            "session_id": session_id,
                            "provider": "rag_v2",
                        },
                        exc_info=True,
                    )

            if getattr(Config.Intelligence, "ENABLED", True):
                try:
                    intelligent_answer = run_intelligent_pipeline(
                        query=query,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        session_id=session_id,
                        user_id=user_id,
                        request_id=request_id,
                        redis_client=self.redis_client,
                        qdrant_client=self.client,
                        embedder=self.model,
                    )
                    if intelligent_answer:
                        return intelligent_answer
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Intelligent pipeline failed, falling back: %s", exc)

            logger.info(
                "[RAG_QUERY] query=%r collection=%s profile=%s subscription=%s user=%s",
                query[:100], collection_name, profile_id, subscription_id, user_id,
            )

            resolved_query = self.conversation_state.resolve_query(query, namespace, user_id)
            if resolved_query != query:
                logger.info("Context resolution: '%s' -> '%s'", query, resolved_query)
                query = resolved_query

            retrieval_start = time.time()
            try:
                retrieval_plan = self.retrieve_with_priorities(
                    query=query,
                    user_id=user_id,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    collection_name=collection_name,
                    namespace=namespace,
                    top_k_retrieval=top_k_retrieval
                )
            except RetrievalFilterError as exc:
                logger.error("Retrieval filter failure: %s", exc)
                documents_searched = _collect_profile_documents(
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    redis_client=self.redis_client,
                )
                return _build_retrieval_filter_error_response(
                    query=query,
                    user_id=user_id,
                    collection_name=collection_name,
                    request_id=request_id,
                    index_version=index_version,
                    details=getattr(exc, "details", str(exc)),
                    error_code=getattr(exc, "code", "RETRIEVAL_FILTER_FAILED"),
                    documents_searched=documents_searched,
                )

            # Use the retrieval plan output directly to avoid diverging queries and
            # ensure reranking operates on the best available candidates.
            processed_query = retrieval_plan.get("query", query)
            preprocessing_metadata = retrieval_plan.get("metadata", {})
            retrieved_chunks = retrieval_plan.get("chunks") or []
            graph_hints = retrieval_plan.get("graph_hints")

            # --- Profile isolation audit ---
            if retrieved_chunks:
                scoped = []
                dropped = 0
                _leaked_profiles = set()
                for chunk in retrieved_chunks:
                    meta = getattr(chunk, "metadata", None) or {}
                    chunk_profile = str(meta.get("profile_id") or "")
                    if chunk_profile == str(profile_id):
                        scoped.append(chunk)
                    else:
                        dropped += 1
                        _leaked_profiles.add(chunk_profile)
                if dropped:
                    logger.error(
                        "[PROFILE_ISOLATION_VIOLATION] Dropped %s chunks from foreign profiles %s "
                        "(expected profile=%s, query=%r)",
                        dropped, _leaked_profiles, profile_id, processed_query[:80],
                    )
                retrieved_chunks = scoped

            # Log retrieval summary with source document details
            _source_docs = set()
            for _ch in retrieved_chunks:
                _m = getattr(_ch, "metadata", None) or {}
                _src = _m.get("source_name") or _m.get("document_id") or "unknown"
                _source_docs.add(str(_src))
            logger.info(
                "[RAG_RETRIEVAL] profile=%s retrieved=%d strategy=%s sources=%s",
                profile_id, len(retrieved_chunks),
                retrieval_plan.get("selected_strategy", "?"),
                list(_source_docs)[:10],
            )

            retrieval_attempts = retrieval_plan.get("attempts", [])
            selected_strategy = retrieval_plan.get("selected_strategy", "direct_qdrant")
            profile_context_data = retrieval_plan.get("profile_context", {})
            documents_seen = _collect_documents_seen(retrieved_chunks)
            min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 50))
            min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 10))
            valid_chunks, invalid_samples = _filter_invalid_retrieved_chunks(
                retrieved_chunks,
                min_chars=min_chars,
                min_tokens=min_tokens,
            )
            dropped_invalid = max(0, len(retrieved_chunks) - len(valid_chunks))
            retrieval_empty_text = bool(retrieved_chunks) and not valid_chunks
            retrieval_diag = {
                "retrieved_count": len(retrieved_chunks),
                "dropped_invalid_count": dropped_invalid,
                "invalid_samples": invalid_samples[:3],
            }
            retrieved_chunks = valid_chunks
            mrr_score = 0.0
            target_doc_ids = preprocessing_metadata.get("target_document_ids") or []
            if not isinstance(target_doc_ids, list):
                target_doc_ids = [target_doc_ids] if target_doc_ids else []
            if target_doc_ids:
                for rank, chunk in enumerate(retrieved_chunks, start=1):
                    doc_id = (chunk.metadata or {}).get("document_id")
                    if doc_id and str(doc_id) in {str(v) for v in target_doc_ids}:
                        mrr_score = 1.0 / rank
                        break
            retrieval_hits = 1.0 if retrieved_chunks else 0.0

            def _record_request_metrics(
                *,
                query_type: str,
                answer_text: str,
                context_text: str,
                context_found: bool,
                grounded: bool,
                has_citations: bool,
                processing_seconds: float,
                prompt_text: Optional[str] = None,
                tool_successes: int = 0,
                document_ids: Optional[List[str]] = None,
            ) -> None:
                if not metrics_store.available:
                    return
                prompt_tokens = _approx_token_count(prompt_text or query)
                completion_tokens = _approx_token_count(answer_text)
                # Embedding similarity is sampled; token overlap is the low-cost fallback.
                sentence_similarity = _semantic_similarity(query, processed_query or query)
                semantic_preservation = _semantic_similarity(context_text, answer_text)
                answer_relevance = _semantic_similarity(query, answer_text)
                # Grounding blends retrieval context presence with citation signals.
                context_grounding = 0.5 * float(bool(context_found)) + 0.5 * float(bool(has_citations))
                # Hallucination risk inversely tracks preservation + grounding.
                hallucination_risk = max(0.0, min(1.0, 1.0 - ((semantic_preservation + context_grounding) / 2.0)))
                response_consistency = 1.0 if has_citations else (0.6 if grounded else 0.2)
                temperature = float(getattr(Config.LLM, "TEMPERATURE", 0.2))
                temperature_sensitivity = max(0.0, min(1.0, 1.0 - temperature))
                latency_ms = processing_seconds * 1000

                counters = {
                    "requests_total": 1,
                    "retrieval_total": 1,
                    "retrieval_hits": retrieval_hits,
                    "retrieval_grounded": 1 if grounded else 0,
                }
                if query_type != "error":
                    counters["requests_success"] = 1
                if use_tooling:
                    counters["requests_with_tools"] = 1
                    if tool_successes == 0:
                        counters["llm_without_tool_fallback"] = 1

                distributions = {}
                if query_type in {"error", "no_results", "not_answerable"}:
                    distributions["failure_type"] = {query_type: 1}

                values = {
                    "prompt_tokens": float(prompt_tokens),
                    "completion_tokens": float(completion_tokens),
                    "sentence_transformation_similarity_score": sentence_similarity,
                    "semantic_preservation_score": semantic_preservation,
                    "answer_faithfulness_score": semantic_preservation,
                    "answer_relevance_score": answer_relevance,
                    "context_grounding_score": context_grounding,
                    "hallucination_risk_score": hallucination_risk,
                    "response_consistency_score": response_consistency,
                    "temperature_sensitivity_score": temperature_sensitivity,
                    "mean_reciprocal_rank": mrr_score,
                    "request_latency_ms": latency_ms,
                }
                if is_cold_start:
                    values["cold_start_latency_ms"] = latency_ms
                else:
                    values["warm_start_latency_ms"] = latency_ms

                metrics_store.record(
                    counters=counters,
                    values=values,
                    distributions=distributions,
                    histograms={
                        "request_latency_ms": latency_ms,
                        "cold_start_latency_ms": latency_ms if is_cold_start else None,
                        "warm_start_latency_ms": latency_ms if not is_cold_start else None,
                    },
                    model_id=self.model_name,
                )
                if document_ids:
                    for doc_id in set(document_ids):
                        metrics_store.record(
                            counters=counters,
                            values=values,
                            distributions=distributions,
                            histograms={"request_latency_ms": latency_ms},
                            document_id=str(doc_id),
                            model_id=self.model_name,
                        )
            if telemetry:
                retrieval_latency_ms = (time.time() - retrieval_start) * 1000
                telemetry.increment("retrieval_requests_count")
                telemetry.observe("retrieval_latency_ms", retrieval_latency_ms)
                telemetry.observe("retrieval_topk", len(retrieved_chunks))
                approx_tokens = 0
                doc_ids = []
                for chunk in retrieved_chunks:
                    text = getattr(chunk, "text", "") or ""
                    approx_tokens += len(str(text).split())
                    meta = getattr(chunk, "metadata", {}) or {}
                    doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("docId")
                    if doc_id:
                        doc_ids.append(str(doc_id))
                telemetry.observe("retrieval_context_tokens", approx_tokens)
                if retrieved_chunks:
                    telemetry.increment("retrieval_hits")
                telemetry.set_gauge("last_retrieval_time", time.time())
                for doc_id in set(doc_ids):
                    telemetry.record_doc_metric(doc_id, "last_retrieval_time", time.time())

            if retrieval_empty_text:
                retrieval_response = _build_retrieval_empty_text_response(
                    query=query,
                    user_id=user_id,
                    collection_name=collection_name,
                    request_id=request_id,
                    index_version=index_version,
                    preprocessing_metadata=preprocessing_metadata,
                    retrieval_attempts=retrieval_attempts,
                    selected_strategy=selected_strategy,
                    profile_context_data=profile_context_data,
                    documents_seen=documents_seen,
                    processing_time=time.time() - start_time,
                )
                self.conversation_history.add_turn(namespace, user_id, query, retrieval_response["response"])

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="retrieval_empty_text",
                        context_found=False,
                        grounded=False,
                        cached=False,
                        processing_time=time.time() - start_time,
                        retrieval_stats={"initial_retrieved": len(retrieval_plan.get("chunks") or []), "final_context": 0},
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (retrieval_empty_text) failed: %s", metric_exc)

                _record_request_metrics(
                    query_type="retrieval_empty_text",
                    answer_text=retrieval_response["response"],
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query,
                )

                return retrieval_response

            # Cache lookup (per subscription/profile/query + recent memory fingerprint)
            cache = get_redis_client() if (ENABLE_ANSWER_CACHE and not disable_answer_cache) else None
            cache_key = None
            conversation_context_for_cache = self.conversation_history.get_context(namespace, user_id, max_turns=2)
            memory_fingerprint = "noctx"
            if conversation_context_for_cache:
                memory_fingerprint = hashlib.md5(conversation_context_for_cache.encode("utf-8")).hexdigest()
            session_fingerprint = _slug(session_id) if session_id else "default"
            profile_context_fingerprint = "noprof"
            try:
                profile_context_fingerprint = hashlib.md5(
                    json.dumps(profile_context_data or {}, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            except Exception as fp_exc:
                logger.debug("Failed to fingerprint profile context: %s", fp_exc)
            if cache:
                cache_key = (
                    f"rag:{ANSWER_CACHE_VERSION}:"
                    f"sub={_slug(subscription_id)}:"  # ✅ Added subscription
                    f"col={_slug(collection_name)}:"
                    f"prof={_slug(profile_id)}:"
                    f"usr={hashlib.md5(user_id.encode('utf-8')).hexdigest()[:8]}:"  # ✅ Added user
                    f"sess={session_fingerprint}:"
                    f"model={_slug(self.model_name)}:"
                    f"strat={selected_strategy}:"
                    f"mem={memory_fingerprint[:8]}:"  # ✅ Shortened to avoid key bloat
                    f"q={hashlib.md5(processed_query.encode('utf-8')).hexdigest()}"
                )
                cached = cache.get(cache_key)
                if cached:
                    try:
                        cached_obj = json.loads(cached)
                        if cached_obj.get("cache_version") and cached_obj["cache_version"] != ANSWER_CACHE_VERSION:
                            raise ValueError("Cache version mismatch")
                        logger.info("Cache hit for query; ignoring because answer cache is disabled")
                    except Exception:
                        logger.warning("Failed to parse cached answer; ignoring")
            logger.info(f"Preprocessed query: {processed_query}")

            if not retrieved_chunks and not use_tooling:
                target_doc_ids = preprocessing_metadata.get("target_document_ids") or []
                documents_searched = _collect_profile_documents(
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    redis_client=self.redis_client,
                    target_doc_ids=target_doc_ids,
                )
                preview = ", ".join(documents_searched[:8])
                if documents_searched and len(documents_searched) > 8:
                    preview += "..."
                searched_line = f"\nDocuments searched: {preview}" if documents_searched else ""
                no_results_response = (
                    f"I couldn't find anything in your documents that answers: '{query}'. "
                    "Try rephrasing or tell me which document or section to focus on."
                    + searched_line
                )
                self.conversation_history.add_turn(namespace, user_id, query, no_results_response)

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="no_results",
                        context_found=False,
                        grounded=False,
                        cached=False,
                        processing_time=time.time() - start_time,
                        retrieval_stats={"initial_retrieved": 0, "final_context": 0}
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (no_results) failed: %s", metric_exc)

                _record_request_metrics(
                    query_type="no_results",
                    answer_text=no_results_response,
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query,
                )

                return {
                    "response": no_results_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,
                    "query_type": "no_results",
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "documents_searched": documents_searched,
                    "grounded": True,
                    "processing_time": time.time() - start_time
                }

            # Boost relevance for recently cited documents to honor session context
            recent_docs = set(self.conversation_history.get_recent_doc_ids(namespace, user_id))
            if recent_docs:
                for chunk in retrieved_chunks:
                    doc_id = (chunk.metadata or {}).get("document_id")
                    if doc_id and doc_id in recent_docs:
                        chunk.score = float(chunk.score) + 0.1

            reranked_chunks = self.reranker.rerank(
                chunks=retrieved_chunks,
                query=processed_query,
                top_k=top_k_rerank,
                use_cross_encoder=True,
                diagnostics=retrieval_diag,
            )
            reranked_chunks = self.graph_support_scorer.score_chunks(reranked_chunks, graph_hints)

            if getattr(Config.Retrieval, "USE_ADJACENT_EXPANSION", False):
                neighbor_window = getattr(Config.Retrieval, "NEIGHBOR_WINDOW", 2) or 2
                neighbor_max_new = getattr(Config.Retrieval, "NEIGHBOR_MAX_NEW", 10) or 10
                reranked_chunks = self.retriever.expand_with_neighbors(
                    collection_name=collection_name,
                    seed_chunks=reranked_chunks,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    window=int(neighbor_window),
                    max_new=int(neighbor_max_new)
                )
                reranked_chunks = sorted(reranked_chunks, key=lambda c: to_py_scalar(c.score), reverse=True)

            config_context_limit = getattr(Config.Retrieval, "MAX_CONTEXT_CHUNKS", 16)
            context_chunk_limit = max(final_k or 12, config_context_limit)
            context_chunk_limit = min(context_chunk_limit, len(reranked_chunks))
            extraction_mode = bool(_extract_requested_fields(processed_query))
            if extraction_mode and len(reranked_chunks) > context_chunk_limit:
                # Field extraction benefits from broader coverage (names/skills often in short chunks)
                expanded_limit = max(context_chunk_limit + 4, int(context_chunk_limit * 2))
                context_chunk_limit = min(len(reranked_chunks), expanded_limit)
            final_chunks = reranked_chunks[:context_chunk_limit]

            context_sources: List[Dict[str, Any]] = []
            context = ""
            if final_chunks:
                try:
                    chunk_dicts = [
                        {
                            "text": chunk.text,
                            "score": float(chunk.score),
                            "metadata": chunk.metadata or {}
                        }
                        for chunk in final_chunks
                    ]
                    context, context_sources = self.intelligent_context_builder.build_context(
                        chunks=chunk_dicts,
                        query=processed_query,
                        include_metadata=True
                    )
                except Exception as ctx_exc:
                    logger.debug(f"Enhanced context builder failed; falling back: {ctx_exc}")
                    context = self.context_builder.build_context(
                        chunks=final_chunks,
                        max_chunks=context_chunk_limit
                    )
                    context_sources = self.context_builder.extract_sources(final_chunks)

            _ctx_source_names = set()
            for _fch in final_chunks:
                _fm = getattr(_fch, "metadata", None) or {}
                _ctx_source_names.add(str(_fm.get("source_name") or _fm.get("document_id") or "?"))
            logger.info(
                "[RAG_CONTEXT] profile=%s chunks=%d context_chars=%d sources=%s",
                profile_id, len(final_chunks), len(context), list(_ctx_source_names)[:10],
            )

            if not context.strip():
                retrieval_response = _build_retrieval_empty_text_response(
                    query=query,
                    user_id=user_id,
                    collection_name=collection_name,
                    request_id=request_id,
                    index_version=index_version,
                    preprocessing_metadata=preprocessing_metadata,
                    retrieval_attempts=retrieval_attempts,
                    selected_strategy=selected_strategy,
                    profile_context_data=profile_context_data,
                    documents_seen=documents_seen,
                    processing_time=time.time() - start_time,
                )
                self.conversation_history.add_turn(namespace, user_id, query, retrieval_response["response"])

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="retrieval_empty_text",
                        context_found=False,
                        grounded=False,
                        cached=False,
                        processing_time=time.time() - start_time,
                        retrieval_stats={"initial_retrieved": len(retrieval_plan.get("chunks") or []), "final_context": 0},
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (retrieval_empty_text) failed: %s", metric_exc)

                _record_request_metrics(
                    query_type="retrieval_empty_text",
                    answer_text=retrieval_response["response"],
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query,
                )

                return retrieval_response

            kg_evidence_pack = ""
            if graph_hints and graph_hints.graph_snippets:
                allowed_chunk_ids = {
                    (chunk.metadata or {}).get("chunk_id") for chunk in final_chunks if chunk.metadata
                }
                snippets = [
                    snippet for snippet in graph_hints.graph_snippets if snippet.chunk_id in allowed_chunk_ids
                ]
                if snippets:
                    lines = []
                    max_snippets = int(getattr(Config.KnowledgeGraph, "MAX_GRAPH_SNIPPETS", 10))
                    for idx, snippet in enumerate(snippets[:max_snippets]):
                        doc_label = snippet.doc_name or snippet.doc_id
                        lines.append(
                            f"{idx + 1}. {snippet.text} (Doc: {doc_label}, Chunk: {snippet.chunk_id})"
                        )
                    kg_evidence_pack = "\n".join(lines)

            tool_successes = 0
            tool_failures = 0
            if use_tooling and tool_list:
                try:
                    import asyncio
                    from src.tools.base import registry
                except Exception as tool_import_exc:  # noqa: BLE001
                    logger.debug("Tool registry not available: %s", tool_import_exc)
                    tool_list = []
                tool_chunks: List[str] = []
                for tool_name in tool_list:
                    payload = {
                        "input": {"query": query, "context": context},
                        "context": {"profile_id": profile_id, "subscription_id": subscription_id},
                        "options": {"requested_by": "rag_pipeline"},
                    }
                    extra_input = (tool_inputs or {}).get(tool_name) if tool_inputs else None
                    if isinstance(extra_input, dict):
                        payload["input"].update(extra_input)
                    elif extra_input is not None:
                        payload["input"]["value"] = extra_input
                    tool_start = time.time()
                    try:
                        tool_resp = asyncio.run(
                            registry.invoke(tool_name, payload, correlation_id=request_id)
                        )
                        if tool_resp.get("status") == "success":
                            tool_successes += 1
                            tool_result = tool_resp.get("result") or {}
                            snippet = json.dumps(tool_result, default=str)
                            tool_chunks.append(f"[{tool_name}] {snippet[:800]}")
                            for src in tool_resp.get("sources") or []:
                                meta = src.get("metadata") or {}
                                meta["tool_output"] = True
                                src["metadata"] = meta
                                context_sources.append(src)
                        else:
                            logger.warning("Tool %s returned status=%s", tool_name, tool_resp.get("status"))
                            tool_failures += 1
                        if metrics_store.available:
                            tool_latency_ms = (time.time() - tool_start) * 1000
                            metrics_store.record(
                                counters={
                                    "tool_calls": 1,
                                    "tool_success": 1 if tool_resp.get("status") == "success" else 0,
                                    "tool_failure": 0 if tool_resp.get("status") == "success" else 1,
                                },
                                values={"tool_latency_ms": tool_latency_ms},
                                histograms={"tool_latency_ms": tool_latency_ms},
                                distributions={"tool_usage": {tool_name: 1}},
                                model_id=self.model_name,
                                tool=tool_name,
                            )
                    except Exception as tool_exc:  # noqa: BLE001
                        logger.warning("Tool %s failed: %s", tool_name, tool_exc)
                        tool_failures += 1
                        if metrics_store.available:
                            tool_latency_ms = (time.time() - tool_start) * 1000
                            metrics_store.record(
                                counters={
                                    "tool_calls": 1,
                                    "tool_failure": 1,
                                },
                                values={"tool_latency_ms": tool_latency_ms},
                                histograms={"tool_latency_ms": tool_latency_ms},
                                distributions={"tool_usage": {tool_name: 1}},
                                model_id=self.model_name,
                                tool=tool_name,
                            )
                if tool_chunks:
                    context = (context or "") + "\n\n".join(tool_chunks)

            conversation_context = self.conversation_history.get_context(namespace, user_id, max_turns=3)
            if self.conversation_state.enriched_turns:
                last_turn = self.conversation_state.enriched_turns[-1]
                conversation_summary = self.progressive_summarizer.update(
                    last_turn, self.progressive_summarizer.get_summary()
                )
            else:
                conversation_summary = self.conversation_summarizer.summarize(conversation_context)
            adapter_text = DomainPromptAdapter.build_adapter(profile_context_data, query)
            feedback_text = self.feedback_memory.build_feedback_context(namespace, user_id, limit=5)
            evidence_synthesis = None
            if getattr(Config.Retrieval, "EVIDENCE_SYNTHESIZER_ENABLED", False):
                try:
                    plan_json = preprocessing_metadata.get("retrieval_plan") or {}
                    evidence_packets = _build_evidence_packets_from_chunks(final_chunks)
                    if evidence_packets:
                        evidence_synthesis = self.evidence_synthesizer.synthesize(
                            user_query=query,
                            plan_json=plan_json,
                            evidence_packets=evidence_packets,
                        )
                except Exception as synth_exc:  # noqa: BLE001
                    logger.debug("Evidence synthesis skipped: %s", synth_exc)

            # ✅ FIXED: Skip answerability check when we have retrieved chunks
            if final_chunks and context.strip():
                # We have context - let LLM decide if it can answer
                is_answerable = True
                answerability_reason = "Context available from retrieved chunks"
                logger.info("✅ Skipping answerability check - have valid context")
            else:
                # Only check when no chunks retrieved
                is_answerable, answerability_reason = self.answerability_detector.check_answerability(
                    query, context, has_chunks=bool(final_chunks)
                )

            # ✅ FIXED: Only block if we truly have NO context at all
            if not is_answerable and not final_chunks:
                not_answerable_response = (
                    f"I don't have enough detail in the documents to answer that. "
                    "If you can point me to a specific document or section, I can check again."
                )
                self.conversation_history.add_turn(namespace, user_id, query, not_answerable_response)

                try:
                    metrics.record(
                        model_name=self.model_name,
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        query_type="not_answerable",
                        context_found=False,  # ✅ Changed to False (no chunks)
                        grounded=False,
                        cached=False,
                        processing_time=time.time() - start_time,
                        retrieval_stats={
                            "initial_retrieved": len(retrieved_chunks),
                            "final_context": 0  # ✅ Changed to 0 (no final chunks)
                        }
                    )
                except Exception as metric_exc:
                    logger.debug("Metrics record (not_answerable) failed: %s", metric_exc)

                _record_request_metrics(
                    query_type="not_answerable",
                    answer_text=not_answerable_response,
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query,
                )

                return {
                    "response": not_answerable_response,
                    "sources": [],  # ✅ Changed to empty (no sources)
                    "user_id": user_id,
                    "collection": collection_name,
                    "request_id": request_id,
                    "index_version": index_version,
                    "context_found": False,  # ✅ Changed to False
                    "query_type": "not_answerable",
                    "answerability_reason": answerability_reason,
                    "preprocessing": preprocessing_metadata,
                    "retrieval_attempts": retrieval_attempts,
                    "selected_strategy": selected_strategy,
                    "profile_context": profile_context_data,
                    "grounded": False,  # ✅ Changed to False (couldn't answer)
                    "processing_time": time.time() - start_time
                }

            retrieval_brief = (
                f"strategy={selected_strategy}; processed_query=\"{processed_query}\"; "
                f"context_chunks={len(final_chunks)}; attempts={len(retrieval_attempts)}"
            )
            if context_sources:
                top_sources = ", ".join(str(src.get('source_name')) for src in context_sources[:3])
                retrieval_brief += f"; top_sources={top_sources}"
            if getattr(Config.Retrieval, "USE_ADJACENT_EXPANSION", False):
                retrieval_brief += "; adjacent_expansion=on"
                if conversation_summary:
                    retrieval_brief += "; convo_summary=on"

            prompt = self.prompt_builder.build_qa_prompt(
                query=query,
                context=context,
                persona=persona,
                conversation_summary=conversation_summary,
                domain_guidance=adapter_text,
                feedback_memory=feedback_text,
                retrieval_brief=retrieval_brief,
                kg_evidence=kg_evidence_pack,
                profile_id=profile_id,
                subscription_id=subscription_id,
                redis_client=self.redis_client,
            )

            gemini_backend = _is_gemini_backend(self.llm_client)
            gemini_used = False

            def _generate_with_metadata(prompt_text: str, options: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
                nonlocal gemini_used
                if gemini_backend:
                    if gemini_used:
                        raise RuntimeError("Gemini call budget exhausted for request")
                    gemini_used = True
                _sys = build_system_prompt()
                if hasattr(self.llm_client, "generate_with_metadata"):
                    return self.llm_client.generate_with_metadata(prompt_text, system=_sys, options=options)
                text = self.llm_client.generate(prompt_text, system=_sys)
                return text, {"response": text}

            temperature = float(getattr(Config.LLM, "TEMPERATURE", 0.2))
            top_p = float(getattr(Config.LLM, "TOP_P", 0.85))
            num_predict = int(getattr(Config.LLM, "MAX_TOKENS", 2048))
            base_options = {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            }
            try:
                answer, raw_response = _generate_with_metadata(prompt, options=base_options)
            except Exception as gen_exc:  # noqa: BLE001
                logger.debug("Generation failed; falling back to evidence summary: %s", gen_exc)
                raw_response = {"done_reason": "error", "error": str(gen_exc)}
                if final_chunks:
                    ledger = _build_evidence_ledger(final_chunks)
                    answer = _format_evidence_fallback(query, ledger)
                else:
                    answer = "I don’t have enough context to answer that from this profile."
            # ── Grounding verification: check answer is supported by context ──
            if answer and context and not _is_generation_empty(answer):
                try:
                    _answer_sentences = [s.strip() for s in answer.replace('\n', ' ').split('.') if len(s.strip()) > 20]
                    _context_lower = context.lower()
                    _ungrounded = []
                    for _sent in _answer_sentences[:10]:  # Check first 10 sentences
                        _sent_words = set(_sent.lower().split())
                        _key_words = {w for w in _sent_words if len(w) > 4}  # Meaningful words only
                        if not _key_words:
                            continue
                        _overlap = sum(1 for w in _key_words if w in _context_lower)
                        _ratio = _overlap / len(_key_words) if _key_words else 1.0
                        if _ratio < 0.3:  # Less than 30% of key words found in context
                            _ungrounded.append(_sent)
                    if _ungrounded and len(_ungrounded) > len(_answer_sentences) * 0.5:
                        logger.warning(
                            "Grounding check: %d/%d sentences appear ungrounded",
                            len(_ungrounded), len(_answer_sentences),
                        )
                        answer += "\n\n---\n*Note: Some details in this response may not be fully supported by the available documents. Please verify critical information against the source documents.*"
                except Exception:
                    pass  # Never block response for verification failure

            done_reason = raw_response.get("done_reason")
            eval_count = raw_response.get("eval_count")
            recovery_path_taken = "none"

            if _is_generation_empty(answer):
                if metrics_store.available:
                    metrics_store.record(counters={"generation_empty": 1}, model_id=self.model_name)
                if gemini_backend:
                    recovery_path_taken = "evidence_fallback"
                    if final_chunks:
                        ledger = _build_evidence_ledger(final_chunks)
                        answer = _format_evidence_fallback(query, ledger)
                    else:
                        answer = "I don’t have enough context or information to build a response from the documents."
                elif final_chunks:
                    recovery_path_taken = "retry"
                    retry_chunks = final_chunks
                    retry_context = context
                    retry_sources = context_sources
                    if _requires_detailed_summary(query) and len(reranked_chunks) > len(final_chunks):
                        expanded_limit = min(len(reranked_chunks), max(context_chunk_limit, context_chunk_limit * 2))
                        if expanded_limit > len(final_chunks):
                            retry_chunks = reranked_chunks[:expanded_limit]
                            try:
                                retry_dicts = [
                                    {
                                        "text": chunk.text,
                                        "score": float(chunk.score),
                                        "metadata": chunk.metadata or {},
                                    }
                                    for chunk in retry_chunks
                                ]
                                retry_context, retry_sources = self.intelligent_context_builder.build_context(
                                    chunks=retry_dicts,
                                    query=processed_query,
                                    include_metadata=True,
                                )
                            except Exception as ctx_exc:
                                logger.debug("Retry context rebuild failed; using fallback: %s", ctx_exc)
                                retry_context = self.context_builder.build_context(
                                    chunks=retry_chunks,
                                    max_chunks=expanded_limit,
                                )
                                retry_sources = self.context_builder.extract_sources(retry_chunks)
                            final_chunks = retry_chunks
                            context_sources = retry_sources
                            context = retry_context
                            prompt = self.prompt_builder.build_qa_prompt(
                                query=query,
                                context=retry_context,
                                persona=persona,
                                conversation_summary=conversation_summary,
                                domain_guidance=adapter_text,
                                feedback_memory=feedback_text,
                                retrieval_brief=retrieval_brief,
                                kg_evidence=kg_evidence_pack,
                                profile_id=profile_id,
                                subscription_id=subscription_id,
                                redis_client=self.redis_client,
                            )

                    retry_prompt = (
                        f"{prompt}\n\n"
                        "End your response with a single line starting with 'Takeaway:' "
                        "and then output END_OF_ANSWER."
                    )
                    retry_options = {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": -1,
                        "stop": ["END_OF_ANSWER"],
                    }
                    answer, raw_response = _generate_with_metadata(retry_prompt, options=retry_options)
                    temperature = float(retry_options["temperature"])
                    top_p = float(retry_options["top_p"])
                    num_predict = int(retry_options["num_predict"])
                    done_reason = raw_response.get("done_reason")
                    eval_count = raw_response.get("eval_count")

                if _is_generation_empty(answer):
                    if final_chunks:
                        recovery_path_taken = "evidence_fallback"
                        ledger = _build_evidence_ledger(final_chunks)
                        answer = _format_evidence_fallback(query, ledger)
                    else:
                        recovery_path_taken = "evidence_fallback"
                        answer = (
                            "I don’t have enough context or information to build a response from the documents."
                        )

            if answer and answer.strip().lower().startswith(query.strip().lower()):
                trimmed = answer[len(query):].lstrip(" :.-\n\t")
                if trimmed:
                    answer = trimmed

            sources = context_sources or self.context_builder.extract_sources(final_chunks)
            answer = render_source_citations(answer, sources)

            self.conversation_state.record_turn(
                namespace, user_id, query, answer,
                resolved_query=resolved_query if 'resolved_query' in dir() and resolved_query != query else None,
            )
            final_doc_ids = [
                (chunk.metadata or {}).get("document_id") for chunk in final_chunks
            ]
            final_doc_ids = [d for d in final_doc_ids if d]
            if final_doc_ids:
                self.conversation_history.add_sources(namespace, user_id, final_doc_ids)
            self.feedback_memory.add_feedback(namespace, user_id, query, answer, sources)

            has_citations = bool(re.search(r'\[SOURCE-\d+\]', answer))

            processing_time = time.time() - start_time

            # Construct the full response object up front so it can be cached
            response_obj = {
                "response": answer,
                "sources": sources,
                "user_id": user_id,
                "collection": collection_name,
                "request_id": request_id,
                "index_version": index_version,
                "context_found": True,
                "query_type": "document_qa",
                "num_sources": len(sources),
                "source_doc_ids": final_doc_ids,
                "preprocessing": preprocessing_metadata,
                "processed_query": processed_query,
                "evidence_synthesis": evidence_synthesis,
                "answerability": {
                    "is_answerable": is_answerable,
                    "reason": answerability_reason
                },
                "grounded": True,
                "has_citations": has_citations,
                "retrieval_attempts": retrieval_attempts,
                "selected_strategy": selected_strategy,
                "profile_context": profile_context_data,
                "processing_time": round(processing_time, 2),
                "retrieval_notes": retrieval_brief,
                "model_name": self.model_name,
                "persona": persona,
                "generation": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": num_predict,
                    "done_reason": done_reason,
                    "eval_count": eval_count,
                    "recovery_path_taken": recovery_path_taken,
                },
                "recovery_path_taken": recovery_path_taken,
                "cache_version": ANSWER_CACHE_VERSION,
                "context_fingerprint": memory_fingerprint,
                "profile_context_fingerprint": profile_context_fingerprint,
                "retrieval_stats": {
                    "initial_retrieved": len(retrieved_chunks),
                    "after_rerank": len(reranked_chunks),
                    "final_context": len(final_chunks)
                },
                "force_refresh": force_refresh,
                "disable_answer_cache": disable_answer_cache,
            }

            rerank_alpha = self.reranker.adjust_alpha(processed_query)
            logger.info(
                "[ASK] ctx_chunks=%s ctx_chars=%s top_k=%s rerank_alpha=%.2f model=%s temperature=%.2f num_predict=%s "
                "done_reason=%s eval_count=%s recovery_path_taken=%s",
                len(final_chunks),
                len(context),
                top_k_retrieval,
                rerank_alpha,
                self.model_name,
                temperature,
                num_predict,
                done_reason,
                eval_count,
                recovery_path_taken,
            )

            try:
                metrics.record(
                    model_name=self.model_name,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    query_type=response_obj.get("query_type", "document_qa"),
                    context_found=True,
                    grounded=bool(response_obj.get("grounded", True)),
                    cached=False,
                    processing_time=processing_time,
                    retrieval_stats=response_obj.get("retrieval_stats") or {}
                )
            except Exception as metric_exc:
                logger.debug("Metrics record (success) failed: %s", metric_exc)

            _record_request_metrics(
                query_type=response_obj.get("query_type", "document_qa"),
                answer_text=answer,
                context_text=context or "",
                context_found=True,
                grounded=bool(response_obj.get("grounded", True)),
                has_citations=has_citations,
                processing_seconds=processing_time,
                prompt_text=prompt,
                tool_successes=tool_successes,
                document_ids=final_doc_ids,
            )

            # Persist the response in Redis before returning. Only cache successful
            # answers (document_qa) to improve subsequent response accuracy.
            if cache and cache_key:
                try:
                    cache.setex(cache_key, ANSWER_CACHE_TTL, json.dumps(response_obj))
                except Exception as cache_exc:
                    logger.warning(f"Failed to cache answer: {cache_exc}")

            # Log response summary
            logger.info(
                "[RAG_RESPONSE] profile=%s grounded=%s context_found=%s "
                "response_len=%d sources=%s query=%r",
                profile_id,
                response_obj.get("grounded"),
                response_obj.get("context_found"),
                len(str(response_obj.get("response", ""))),
                [s.get("name", s.get("document_id", "?")) for s in response_obj.get("sources", [])[:5]],
                processed_query[:80],
            )
            # Return the constructed response
            return response_obj

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            try:
                if telemetry:
                    telemetry.increment("retrieval_failures_count")
            except Exception as exc:
                logger.debug("Failed to increment retrieval_failures_count telemetry", exc_info=True)

            error_response = "Sorry, something went wrong on my side. Please try again, and let me know if it keeps happening."

            try:
                metrics = get_metrics_tracker()
                metrics.record(
                    model_name=getattr(self, "model_name", "unknown"),
                    subscription_id=subscription_id if 'subscription_id' in locals() else "unknown",
                    profile_id=profile_id if 'profile_id' in locals() else "unknown",
                    query_type="error",
                    context_found=False,
                    grounded=False,
                    cached=False,
                    processing_time=time.time() - start_time,
                    retrieval_stats={}
                )
            except Exception as metric_exc:
                logger.debug("Metrics record (error) failed: %s", metric_exc)

            if "_record_request_metrics" in locals():
                _record_request_metrics(
                    query_type="error",
                    answer_text=error_response,
                    context_text="",
                    context_found=False,
                    grounded=False,
                    has_citations=False,
                    processing_seconds=time.time() - start_time,
                    prompt_text=processed_query if "processed_query" in locals() else query,
                )

            return {
                "response": error_response,
                "sources": [],
                "user_id": user_id,
                "collection": collection_name if 'collection_name' in locals() else profile_id,
                "request_id": request_id,
                "index_version": index_version,
                "context_found": False,
                "query_type": "error",
                "error": str(e),
                "grounded": False,
                "processing_time": time.time() - start_time
            }

# Global RAG system instance (lazy initialization)
_RAG_SYSTEM = None
_RAG_MODEL = None
_RAG_PROFILE = None
_RAG_BACKEND = None
_RAG_MODEL_PATH = None
_LLM_CLIENTS: dict[tuple[str, str, str | None], Any] = {}
_LLM_SEMAPHORE: Optional[threading.Semaphore] = None

class _LLMClientWrapper:
    """Wrap LLM clients to enforce concurrency limits without changing callers."""

    def __init__(self, client, semaphore: threading.Semaphore):
        self._client = client
        self._semaphore = semaphore

    def __getattr__(self, item):
        return getattr(self._client, item)

    def generate(self, *args, **kwargs):
        with self._semaphore:
            return self._client.generate(*args, **kwargs)

class ResilientLLMClient:
    """Fallback to a secondary client when the primary fails with rate/timeout errors."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.model_name = getattr(primary, "model_name", None) or getattr(fallback, "model_name", None)
        self.backend = getattr(primary, "backend", None) or "gemini"

    @staticmethod
    def _should_fallback(exc: Exception) -> bool:
        status = getattr(exc, "code", None) or getattr(exc, "status", None)
        if status in {408, 429, 500, 502, 503, 504}:
            return True
        msg = str(exc).lower()
        return "timeout" in msg or "timed out" in msg

    def _primary_in_cooldown(self) -> bool:
        primary = self.primary
        if not primary:
            return False
        checker = getattr(primary, "in_cooldown", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception as exc:
                logger.debug("Failed to check primary LLM cooldown status", exc_info=True)
                return False
        return False

    def generate(self, *args, **kwargs):
        if self._primary_in_cooldown() and self.fallback:
            logger.warning(
                "Primary LLM in cooldown; using fallback",
                extra={"stage": "generate", "provider": getattr(self.primary, "backend", "gemini")},
            )
            return self.fallback.generate(*args, **kwargs)
        try:
            return self.primary.generate(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if self._should_fallback(exc) and self.fallback:
                logger.warning(
                    "Primary LLM failed; falling back to local generation: %s",
                    exc,
                    extra={"stage": "generate", "provider": getattr(self.primary, "backend", "gemini")},
                )
                return self.fallback.generate(*args, **kwargs)
            raise

    def generate_with_metadata(self, *args, **kwargs):
        if self._primary_in_cooldown() and self.fallback:
            logger.warning(
                "Primary LLM in cooldown; using fallback",
                extra={"stage": "generate", "provider": getattr(self.primary, "backend", "gemini")},
            )
            if hasattr(self.fallback, "generate_with_metadata"):
                return self.fallback.generate_with_metadata(*args, **kwargs)
            text = self.fallback.generate(*args, **kwargs)
            return text, {"response": text}
        try:
            if hasattr(self.primary, "generate_with_metadata"):
                return self.primary.generate_with_metadata(*args, **kwargs)
            text = self.primary.generate(*args, **kwargs)
            return text, {"response": text}
        except Exception as exc:  # noqa: BLE001
            if self._should_fallback(exc) and self.fallback:
                logger.warning(
                    "Primary LLM failed; falling back to local generation: %s",
                    exc,
                    extra={"stage": "generate", "provider": getattr(self.primary, "backend", "gemini")},
                )
                if hasattr(self.fallback, "generate_with_metadata"):
                    return self.fallback.generate_with_metadata(*args, **kwargs)
                text = self.fallback.generate(*args, **kwargs)
                return text, {"response": text}
            raise

    def warm_up(self):
        warm = getattr(self.primary, "warm_up", None)
        if callable(warm):
            try:
                warm()
            except Exception as exc:
                logger.debug("Primary LLM warm-up failed", exc_info=True)
        warm_fb = getattr(self.fallback, "warm_up", None)
        if callable(warm_fb):
            try:
                warm_fb()
            except Exception as exc:
                logger.debug("Fallback LLM warm-up failed", exc_info=True)

def create_llm_client(
        model_name: Optional[str] = None,
        backend_override: Optional[str] = None,
        model_path: Optional[str] = None
):
    """Factory to select LLM backend based on requested model name or env.

    Delegates to src.llm.gateway.create_llm_gateway() when possible.
    Falls back to direct client creation for special backends (unsloth).
    """
    global _LLM_SEMAPHORE

    # For unsloth/finetuned models, keep direct creation (not in gateway)
    resolved_backend = (backend_override or os.getenv("LLM_BACKEND", "")).lower().strip()
    if resolved_backend == "unsloth" or model_path:
        if _LLM_SEMAPHORE is None:
            _LLM_SEMAPHORE = threading.Semaphore(getattr(Config.LLM, "MAX_CONCURRENCY", 2))
        from src.finetune.llm_client import UnslothLLMClient
        target = model_path or model_name
        client = UnslothLLMClient(target)
        wrapped = _LLMClientWrapper(client, _LLM_SEMAPHORE)
        setattr(wrapped, "backend", "unsloth")
        return wrapped

    # Check if gateway singleton already exists
    try:
        from src.llm.gateway import get_llm_gateway
        gateway = get_llm_gateway()
        if gateway is not None:
            return gateway
    except Exception as exc:
        logger.debug("Failed to get existing LLM gateway singleton", exc_info=True)

    # Fallback: create gateway directly
    try:
        from src.llm.gateway import create_llm_gateway
        return create_llm_gateway(model_name=model_name, backend_override=backend_override)
    except Exception as exc:
        logger.debug("Gateway creation failed, falling back to direct Ollama: %s", exc)

    # Ultimate fallback: bare Ollama
    model_name = _resolve_model_alias(model_name)
    return OllamaClient(model_name)

def get_metrics_tracker() -> MetricsTracker:
    """Singleton metrics tracker."""
    global _METRICS_TRACKER
    if _METRICS_TRACKER is None:
        _METRICS_TRACKER = MetricsTracker(get_redis_client())
    return _METRICS_TRACKER

def get_rag_system(
        model_name: Optional[str] = None,
        profile_id: Optional[str] = None,
        backend_override: Optional[str] = None,
        model_path: Optional[str] = None
) -> EnterpriseRAGSystem:
    """Get or create the RAG system instance (singleton with lazy loading)."""
    global _RAG_SYSTEM, _RAG_MODEL, _RAG_PROFILE, _RAG_BACKEND, _RAG_MODEL_PATH
    try:
        from src.api.rag_state import get_app_state

        app_state = get_app_state()
        if app_state and app_state.rag_system:
            return app_state.rag_system
    except Exception as exc:
        logger.debug("Failed to get RAG system from app state", exc_info=True)

    if _RAG_SYSTEM is not None:
        if (model_name and model_name != _RAG_MODEL) or (_RAG_BACKEND != backend_override) or (_RAG_MODEL_PATH != model_path):
            logger.error(
                "RAG system already initialized (model=%s); refusing reinit request model=%s backend=%s",
                _RAG_MODEL,
                model_name,
                backend_override,
            )
        return _RAG_SYSTEM

    try:
        _RAG_SYSTEM = EnterpriseRAGSystem(
            model_name=model_name,
            profile_id=profile_id,
            backend_override=backend_override,
            model_path=model_path
        )
        _RAG_MODEL = model_name
        _RAG_PROFILE = profile_id
        _RAG_BACKEND = backend_override
        _RAG_MODEL_PATH = model_path
        logger.info("RAG system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    return _RAG_SYSTEM

def answer_question(
        query: str,
        user_id: str,
        profile_id: str,
        subscription_id: str = "default",
        model_name: str = "DHS/DocWain",
        persona: str = "professional document analysis assistant",
        session_id: Optional[str] = None,
        new_session: bool = False,
        disable_answer_cache: bool = False,
        force_refresh: bool = False,
        request_id: Optional[str] = None,
        index_version: Optional[str] = None,
        tools: Optional[List[str]] = None,
        use_tools: bool = False,
        tool_inputs: Optional[Dict[str, Any]] = None,
        enable_internet: bool = False,
        rag_system: Optional[EnterpriseRAGSystem] = None,
) -> Dict[str, Any]:
    """
    Main entry point for answering questions with enhanced NLU.

    Args:
        query: User's question
        user_id: User identifier
        profile_id: Department/collection name ('finance', 'banking', 'technical')
        model_name: LLM model name (for compatibility)
        persona: Assistant persona

    Returns:
        Response dictionary with enhanced metadata
    """
    resolved_model = resolve_model_for_profile(profile_id, model_name)
    effective_model = resolved_model.model_name or model_name
    rag_system = rag_system or get_rag_system(
        effective_model,
        profile_id=profile_id,
        backend_override=resolved_model.backend,
        model_path=resolved_model.model_path
    )
    return rag_system.answer_question(
        query=query,
        profile_id=profile_id,
        subscription_id=subscription_id,
        user_id=user_id,
        persona=persona,
        session_id=session_id,
        new_session=new_session,
        disable_answer_cache=disable_answer_cache,
        force_refresh=force_refresh,
        request_id=request_id,
        index_version=index_version,
        tools=tools,
        use_tools=use_tools,
        tool_inputs=tool_inputs,
        enable_internet=enable_internet,
    )

def debug_collection(profile_id: str, subscription_id: str = "default") -> Dict[str, Any]:
    """
    Debug utility to check collection status with defensive error handling.

    Args:
        profile_id: Profile/department identifier
        subscription_id: Tenant/subscription identifier

    Returns:
        Collection statistics
    """
    try:
        collection_name = build_collection_name(subscription_id)
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(collection_name)

        # Scope scroll by profile_id + subscription_id to prevent cross-tenant leaks
        from src.api.vector_store import build_qdrant_filter as _build_isolation_filter
        scroll_filter = _build_isolation_filter(subscription_id, profile_id)
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=3,
            with_payload=True,
            with_vectors=False
        )

        sample_points = []
        if isinstance(scroll_result, tuple) and len(scroll_result) > 0:
            points_list = scroll_result[0]
            if points_list:
                sample_points = [
                    {
                        "id": str(p.id),
                        "text_preview": (get_content_text(p.payload or {}) or get_embedding_text(p.payload or {}))[:200] if p.payload else 'No text',
                        "source": p.payload.get('source', 'unknown') if p.payload else 'unknown'
                    }
                    for p in points_list
                ]

        return {
            "collection_name": collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance),
            "sample_points": sample_points,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error debugging collection '{collection_name if 'collection_name' in locals() else profile_id}': {e}", exc_info=True)
        return {
            "collection_name": collection_name if 'collection_name' in locals() else profile_id,
            "error": str(e),
            "status": "error"
        }

def add_domain_terms(terms: List[str]):
    """
    Legacy hook for domain terms; kept for API compatibility.

    Args:
        terms: List of domain-specific terms
    """
    try:
        logger.info("Spell checking is disabled; ignoring %d domain terms", len(terms))
    except Exception as e:
        logger.error(f"Error adding domain terms: {e}")

def clear_conversation_history(
        user_id: str,
        subscription_id: str = "default",
        profile_id: str = "default",
        model_name: str = "",
        session_id: Optional[str] = None
):
    """
    Clear conversation history for a specific user.

    Args:
        user_id: User identifier
        subscription_id: Tenant/subscription identifier
        profile_id: Profile identifier
        model_name: Optional model name to namespace cache keys
    """
    try:
        rag_system = get_rag_system()
        ns = _build_namespace(
            subscription_id,
            profile_id,
            model_name or getattr(rag_system, "model_name", ""),
            session_id=session_id
        )
        rag_system.conversation_history.clear_history(ns, user_id)
        logger.info(f"Cleared conversation history for user {user_id}")
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")

def metrics_summary(days: int = 7) -> Dict[str, Any]:
    """
    Aggregate metrics for the requested window (defaults to 7 days).
    """
    tracker = get_metrics_tracker()
    return tracker.summary(days=days)

class RAGEvaluator:
    """Evaluation utilities for monitoring RAG performance."""

    @staticmethod
    def evaluate_retrieval(
            queries: List[str],
            profile_id: str,
            ground_truth_docs: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using MRR and Precision@K.

        Args:
            queries: List of test queries
            profile_id: Collection name
            ground_truth_docs: List of lists of relevant document IDs for each query

        Returns:
            Dictionary of evaluation metrics
        """
        if len(queries) != len(ground_truth_docs):
            raise ValueError("Queries and ground truth must have same length")

        rag_system = get_rag_system()
        mrr_scores = []
        p_at_1 = []
        p_at_3 = []
        p_at_5 = []

        for query, relevant_docs in zip(queries, ground_truth_docs):
            try:
                chunks = rag_system.retriever.retrieve(
                    collection_name=profile_id,
                    query=query,
                    subscription_id="default",
                    top_k=10
                )

                retrieved_ids = [chunk.id for chunk in chunks]

                for rank, doc_id in enumerate(retrieved_ids, 1):
                    if doc_id in relevant_docs:
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)

                p_at_1.append(1.0 if retrieved_ids[:1] and retrieved_ids[0] in relevant_docs else 0.0)

                if len(retrieved_ids) >= 3:
                    p_at_3.append(
                        len(set(retrieved_ids[:3]) & set(relevant_docs)) / 3.0
                    )

                if len(retrieved_ids) >= 5:
                    p_at_5.append(
                        len(set(retrieved_ids[:5]) & set(relevant_docs)) / 5.0
                    )

            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")

        return {
            "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
            "precision_at_1": np.mean(p_at_1) if p_at_1 else 0.0,
            "precision_at_3": np.mean(p_at_3) if p_at_3 else 0.0,
            "precision_at_5": np.mean(p_at_5) if p_at_5 else 0.0,
            "num_queries": len(queries)
        }

__all__ = [
    'answer_question',
    'debug_collection',
    'add_domain_terms',
    'clear_conversation_history',
    'RAGEvaluator',
    'get_rag_system',
    'metrics_summary'
]
