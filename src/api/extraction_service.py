from src.utils.logging_utils import get_logger
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.api.content_store import save_extracted_pickle
from src.api.blob_store import blob_storage_configured
from src.api.structured_extraction import get_extraction_engine

# Import intelligence layer
try:
    from src.intelligence.integration import (
        DocumentIntelligenceProcessor,
        process_document_intelligence,
    )
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False
    DocumentIntelligenceProcessor = None
    process_document_intelligence = None
try:
    from src.visual_intelligence.orchestrator import get_visual_orchestrator
    VISUAL_INTELLIGENCE_AVAILABLE = True
except ImportError:
    VISUAL_INTELLIGENCE_AVAILABLE = False
    get_visual_orchestrator = None
try:
    from src.api.dataHandler import (
        extract_document_info,
        decrypt_data,
        fileProcessor,
        get_azure_docs,
        get_s3_client,
        get_subscription_pii_setting,
        mask_document_content,
        read_s3_file,
        resolve_subscription_id,
        update_extraction_metadata,
        update_layout_graph_metadata,
        update_pii_stats,
    )
except Exception as _datahandler_exc:  # noqa: BLE001
    def _datahandler_unavailable(*_args, **_kwargs):
        raise RuntimeError("dataHandler unavailable") from _datahandler_exc

    extract_document_info = _datahandler_unavailable
    decrypt_data = _datahandler_unavailable
    fileProcessor = _datahandler_unavailable
    get_azure_docs = _datahandler_unavailable
    get_s3_client = _datahandler_unavailable
    get_subscription_pii_setting = _datahandler_unavailable
    mask_document_content = _datahandler_unavailable
    read_s3_file = _datahandler_unavailable
    resolve_subscription_id = _datahandler_unavailable
    update_extraction_metadata = _datahandler_unavailable
    update_layout_graph_metadata = _datahandler_unavailable
    update_pii_stats = _datahandler_unavailable
from src.api.layout_graph_store import save_layout_graph, save_layout_graph_local
from src.api.document_status import emit_progress, get_documents_collection, init_document_record, set_error, update_document_fields, update_stage
from src.api.pipeline_models import ExtractedDocument
from src.api.statuses import (
    STATUS_DELETED,
    STATUS_EXTRACTION_COMPLETED,
    STATUS_EXTRACTION_FAILED,
    STATUS_SCREENING_COMPLETED,
    STATUS_UNDER_REVIEW,
)
from src.embedding.pipeline.payload_normalizer import normalize_chunk_metadata
from src.storage.azure_blob_client import BlobDownloadError, CredentialError, normalize_blob_name
from src.utils.idempotency import acquire_lock, release_lock
from src.embedding.layout_graph import build_layout_graph

logger = get_logger(__name__)

# Semaphore to limit concurrent deep_analyze() calls — prevents CPU starvation
# when multiple documents are uploaded simultaneously.
_DOC_PROCESSING_SEMAPHORE = threading.Semaphore(
    int(os.getenv("DOC_PROCESSING_MAX_CONCURRENT", "2"))
)

def _mark_intelligence_ready(document_id: str) -> None:
    """Set intelligence_ready=true and persist intelligence subdocument to MongoDB.

    The Core Agent reads ``intelligence.summary``, ``intelligence.entities``,
    ``intelligence.answerable_topics``, ``intelligence.key_facts``, and
    ``intelligence.document_type`` from MongoDB.  This function builds that
    subdocument from the extraction pickle so the agent has what it needs.
    """
    try:
        from src.api.document_status import update_document_fields
        fields: Dict[str, Any] = {
            "intelligence_ready": True,
            "intelligence_completed_at": time.time(),
        }

        # Build intelligence subdocument from pickle for Core Agent consumption
        try:
            from src.api.content_store import load_extracted_pickle
            pkl = load_extracted_pickle(document_id)
            if isinstance(pkl, dict):
                understanding = pkl.get("understanding") or {}
                intel_raw = pkl.get("intelligence") or {}
                deep = pkl.get("deep_analysis") or {}
                classification = pkl.get("document_classification") or {}
                domain_info = pkl.get("document_domain") or {}

                # Build summary from understanding
                summary = understanding.get("document_summary", "")

                # Build document type
                doc_type = (
                    understanding.get("document_type")
                    or classification.get("document_type")
                    or ""
                )

                # Build entities list (flatten from structured format)
                entities: List[str] = []
                raw_entities = intel_raw.get("entities") or {}
                if isinstance(raw_entities, dict):
                    for category, items in raw_entities.items():
                        if isinstance(items, list):
                            for item in items[:20]:
                                name = item.get("value") or item.get("name") or str(item) if isinstance(item, dict) else str(item)
                                if name and len(name) < 100:
                                    entities.append(name)
                elif isinstance(raw_entities, list):
                    for item in raw_entities[:50]:
                        name = item.get("value") or item.get("name") or str(item) if isinstance(item, dict) else str(item)
                        if name and len(name) < 100:
                            entities.append(name)
                # Also include key_entities from understanding
                for ent in understanding.get("key_entities") or []:
                    text = ent.get("text", "") if isinstance(ent, dict) else str(ent)
                    if text and text not in entities and len(text) < 100:
                        entities.append(text)

                # Build key_facts
                key_facts = understanding.get("key_facts") or []

                # Build answerable_topics from section summaries and intent tags
                answerable_topics: List[str] = []
                section_summaries = understanding.get("section_summaries") or {}
                if isinstance(section_summaries, dict):
                    for title, _summ in section_summaries.items():
                        if title and len(title) < 200:
                            answerable_topics.append(title)
                for tag in understanding.get("intent_tags") or []:
                    if tag and tag not in answerable_topics:
                        answerable_topics.append(tag)

                # Build domain
                domain = domain_info.get("domain", "") if isinstance(domain_info, dict) else ""

                # Extract language detection and translation metadata from raw docs
                raw_docs = pkl.get("raw") or {}
                detected_language = "en"
                has_translation = False
                for _fn, _content in raw_docs.items():
                    if isinstance(_content, dict):
                        detected_language = _content.get("detected_language", "en") or "en"
                        has_translation = bool(_content.get("translated_text"))
                        break

                intelligence = {
                    "summary": str(summary)[:3000] if summary else "",
                    "document_type": doc_type,
                    "entities": entities[:100],
                    "key_facts": key_facts[:50],
                    "answerable_topics": answerable_topics[:30],
                    "domain": domain,
                    "quality_grade": deep.get("quality_grade"),
                    "quality_score": deep.get("quality_score"),
                    "detected_language": detected_language,
                    "has_translation": has_translation,
                }
                fields["intelligence"] = intelligence
                fields["doc_type"] = doc_type
                fields["detected_language"] = detected_language
        except Exception as exc:
            logger.warning("Failed to build intelligence subdocument for %s: %s", document_id, exc)

        update_document_fields(document_id, fields)
        logger.info("Document %s marked intelligence_ready", document_id)
    except Exception as exc:
        logger.warning("Failed to mark intelligence_ready for %s: %s", document_id, exc)

def _ingest_to_knowledge_graph(
    document_id: str,
    subscription_id: str,
    profile_id: str,
    source_name: str,
    payload_to_save: Dict[str, Any],
    redis_client: Any = None,
    deep_result: Any = None,
) -> None:
    """Non-blocking KG ingestion from extraction results.

    Builds a graph payload from the extraction output and enqueues it
    for async processing.  KG failure must never block extraction.

    Parameters
    ----------
    deep_result:
        Optional :class:`~src.doc_understanding.deep_analyzer.DeepAnalysisResult`
        produced during extraction.  When provided, ``entities``,
        ``typed_relationships``, and ``temporal_spans`` from the deep analysis
        are forwarded to the KG ingest layer to activate the previously-dead
        ``create_entity_relationship()`` and ``create_timeline_node()`` store
        methods.  If *None*, the existing behaviour is preserved.
    """
    try:
        from src.kg.ingest import build_graph_payload, get_graph_ingest_queue

        # Build an embeddings_payload-shaped dict from the extraction pickle
        texts: List[str] = []
        chunk_metadata: List[Dict[str, Any]] = []
        structured = payload_to_save.get("structured") or {}
        for fname, content in structured.items():
            if isinstance(content, dict):
                full_text = str(content.get("full_text") or content.get("text") or "")
                if full_text:
                    texts.append(full_text)
                    chunk_metadata.append({
                        "chunk_id": f"{document_id}::extraction::{fname}",
                        "source_name": fname,
                    })

        if not texts:
            return

        # Extract deep analysis artifacts when available
        deep_entities: Optional[List[Dict[str, Any]]] = None
        typed_relationships: Optional[List[Dict[str, Any]]] = None
        temporal_spans: Optional[List[Dict[str, Any]]] = None
        if deep_result is not None:
            try:
                deep_entities = [e.to_dict() for e in (deep_result.entities or [])]
            except Exception as exc:
                logger.debug("Failed to extract deep entities from analysis result", exc_info=True)
                deep_entities = None
            try:
                typed_relationships = list(deep_result.typed_relationships or [])
            except Exception as exc:
                logger.debug("Failed to extract typed relationships from analysis result", exc_info=True)
                typed_relationships = None
            try:
                temporal_spans = list(deep_result.temporal_spans or [])
            except Exception as exc:
                logger.debug("Failed to extract temporal spans from analysis result", exc_info=True)
                temporal_spans = None

        doc_classification = payload_to_save.get("document_classification") or {}
        graph_payload = build_graph_payload(
            embeddings_payload={
                "texts": texts,
                "chunk_metadata": chunk_metadata,
                "doc_metadata": {
                    "document_type": doc_classification.get("document_type", "generic"),
                    "doc_type": doc_classification.get("domain", "generic"),
                },
            },
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
            document_id=str(document_id),
            doc_name=source_name,
            deep_entities=deep_entities,
            typed_relationships=typed_relationships,
        )
        if graph_payload:
            # Attach temporal spans so ingest_graph_payload can create timeline nodes
            if temporal_spans:
                graph_payload.temporal_spans = temporal_spans
            queue = get_graph_ingest_queue(redis_client)
            queue.enqueue(graph_payload)
            logger.info("KG ingestion enqueued for document %s", document_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("KG ingestion skipped for %s: %s", document_id, exc)

def _persist_layout_graph(
    *,
    document_id: str,
    subscription_id: Optional[str],
    profile_id: Optional[str],
    extracted_docs: Any,
) -> Optional[Dict[str, Any]]:
    try:
        file_name = "document"
        content = extracted_docs
        if isinstance(extracted_docs, dict) and extracted_docs:
            file_name, content = next(iter(extracted_docs.items()))
        layout_graph = build_layout_graph(content, document_id=document_id, file_name=file_name)
        if blob_storage_configured():
            info = save_layout_graph(
                document_id=document_id,
                layout_graph=layout_graph,
                subscription_id=subscription_id,
                profile_id=profile_id,
            )
        else:
            info = save_layout_graph_local(document_id=document_id, layout_graph=layout_graph)
        update_layout_graph_metadata(
            document_id,
            layout_latest_path=info.get("latest_path"),
            layout_versioned_path=info.get("versioned_path"),
            layout_hash=info.get("sha256"),
        )
        return info
    except Exception as exc:  # noqa: BLE001
        logger.warning("LayoutGraph persistence skipped for %s: %s", document_id, exc)
        return None

def _build_extraction_summary(extracted_obj: Any) -> Dict[str, Any]:
    total_chars = 0
    total_pages = 0
    total_chunks = 0

    def handle_extracted(value: Any) -> None:
        nonlocal total_chars, total_pages, total_chunks
        if isinstance(value, ExtractedDocument):
            text_val = value.full_text or ""
            if not text_val:
                text_val = "\n".join([sec.text for sec in value.sections if sec.text])
            total_chars += len(text_val)
            total_chunks += len(value.sections)
            for sec in value.sections:
                total_pages = max(total_pages, sec.end_page or sec.start_page or 0)
        elif isinstance(value, str):
            total_chars += len(value)
            total_chunks += 1
        elif isinstance(value, dict):
            inner_text = value.get("text") or value.get("content")
            if isinstance(inner_text, str):
                total_chars += len(inner_text)
                total_chunks += 1
            elif isinstance(inner_text, list):
                total_chunks += len(inner_text)
        elif isinstance(value, list):
            total_chunks += len(value)

    if isinstance(extracted_obj, dict):
        for entry in extracted_obj.values():
            handle_extracted(entry)
    else:
        handle_extracted(extracted_obj)

    return {
        "pages": total_pages or None,
        "chunks": total_chunks or None,
        "chars": total_chars,
        "language": None,
    }

def _update_understanding_fields(document_id: str, understanding: Dict[str, Any]) -> None:
    """Persist document understanding / deep-analysis fields to MongoDB."""
    from src.api.config import Config
    from src.api.dataHandler import db
    from bson import ObjectId

    if db is None:
        logger.warning("_update_understanding_fields: MongoDB unavailable for %s", document_id)
        return
    coll = db[Config.MongoDB.DOCUMENTS]
    if ObjectId.is_valid(str(document_id)):
        filt = {"_id": ObjectId(str(document_id))}
    else:
        filt = {"_id": str(document_id)}

    # --- understanding fields ---
    update_fields: Dict[str, Any] = {}
    if understanding.get("document_type"):
        update_fields["understanding_doc_type"] = understanding["document_type"]
    if understanding.get("document_summary"):
        update_fields["document_summary"] = str(understanding["document_summary"])[:2000]
    if understanding.get("key_entities"):
        update_fields["key_entities"] = understanding["key_entities"][:50]
    if understanding.get("key_facts"):
        update_fields["key_facts"] = understanding["key_facts"][:50]
    if understanding.get("intent_tags"):
        update_fields["doc_intent_tags"] = understanding["intent_tags"][:20]

    # --- deep-analysis fields ---
    if understanding.get("quality_grade"):
        update_fields["quality_grade"] = understanding["quality_grade"]
    if understanding.get("quality_score") is not None:
        update_fields["quality_score"] = understanding["quality_score"]
    if understanding.get("complexity_score") is not None:
        update_fields["complexity_score"] = understanding["complexity_score"]
    if understanding.get("entity_mentions"):
        update_fields["entity_mentions"] = understanding["entity_mentions"][:100]
    if understanding.get("chronological_span"):
        update_fields["chronological_span"] = understanding["chronological_span"][:20]
    if understanding.get("section_roles"):
        update_fields["section_roles"] = understanding["section_roles"]
    if understanding.get("domain_signals"):
        update_fields["domain_signals"] = understanding["domain_signals"]

    if not update_fields:
        logger.warning("_update_understanding_fields: no fields to write for %s (keys=%s)",
                       document_id, list(understanding.keys()))
        return

    result = coll.update_one(filt, {"$set": update_fields}, upsert=False)
    if result.matched_count == 0:
        logger.warning("_update_understanding_fields: no MongoDB doc matched for %s (filter=%s), retrying with upsert",
                       document_id, filt)
        coll.update_one(filt, {"$set": update_fields}, upsert=True)
    logger.info("Persisted %d understanding fields for document %s", len(update_fields), document_id)

def _process_document_intelligence(
    document_id: str,
    extracted_docs: Dict[str, Any],
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process document through intelligence layer for entity extraction and Q&A generation.

    Args:
        document_id: Document ID.
        extracted_docs: Extracted document content.
        filename: Original filename.
        subscription_id: Optional subscription ID.
        profile_id: Optional profile ID.

    Returns:
        Intelligence result dictionary or None on failure.
    """
    if not INTELLIGENCE_AVAILABLE:
        logger.debug("Intelligence layer not available, skipping")
        return None

    try:
        # Get text content from extracted docs
        raw_text = ""
        file_size = 0

        if isinstance(extracted_docs, dict):
            for fname, content in extracted_docs.items():
                if isinstance(content, dict):
                    text = content.get("full_text") or content.get("text") or ""
                    if not text and content.get("sections"):
                        text = "\n".join(
                            sec.get("text", "") for sec in content.get("sections", [])
                            if sec.get("text")
                        )
                    raw_text += text
                    file_size = content.get("file_size", 0) or len(raw_text)
                elif isinstance(content, str):
                    raw_text += content
                    file_size = len(raw_text)

        if not raw_text:
            logger.debug("No text content for intelligence processing")
            return None

        # Get Redis client if available
        redis_client = None
        try:
            from src.api.dataHandler import get_redis_client
            redis_client = get_redis_client()
        except Exception as exc:
            logger.debug("Failed to get Redis client for intelligence processing", exc_info=True)

        # Process through intelligence layer
        result = process_document_intelligence(
            document_id=document_id,
            content=raw_text,
            filename=filename,
            subscription_id=subscription_id,
            profile_id=profile_id,
            file_size=file_size,
            redis_client=redis_client,
        )

        logger.info(
            "Intelligence processing complete for %s: domain=%s, entities=%d, qa_pairs=%d",
            document_id,
            result.domain,
            len(result.entities.get_all_searchable_terms()) if result.entities else 0,
            len(result.qa_pairs),
        )

        return result.to_dict()

    except Exception as exc:
        logger.warning("Intelligence processing failed for %s: %s", document_id, exc)
        return None

def _normalize_extracted_metadata(extracted_docs: Any, *, document_id: str) -> Any:
    if isinstance(extracted_docs, dict):
        normalized: Dict[str, Any] = {}
        for name, content in extracted_docs.items():
            if isinstance(content, dict):
                content = dict(content)
                if isinstance(content.get("chunk_metadata"), list):
                    content["chunk_metadata"] = normalize_chunk_metadata(
                        content.get("chunk_metadata") or [],
                        document_id=str(document_id),
                    )
                if content.get("document") is not None:
                    content["document"] = _normalize_extracted_metadata(
                        content.get("document"),
                        document_id=document_id,
                    )
            normalized[name] = content
        return normalized
    return extracted_docs

def _extract_classification_from_structured(structured_docs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract document type/classification from structured extraction results.

    Falls back to the structured extraction metadata.
    """
    # ── Structured extraction metadata ───────────────────────────────
    for fname, value in (structured_docs or {}).items():
        if isinstance(value, dict):
            return {
                "document_type": value.get("document_type", "GENERIC"),
                "domain": (value.get("document_classification") or {}).get("domain", "generic"),
                "confidence": (value.get("document_classification") or {}).get("confidence", 0.0),
                "filename": fname,
            }
        if hasattr(value, "document_type"):
            doc_cls = getattr(value, "document_classification", None)
            domain = doc_cls.get("domain", "generic") if isinstance(doc_cls, dict) else "generic"
            confidence = doc_cls.get("confidence", 0.0) if isinstance(doc_cls, dict) else 0.0
            return {
                "document_type": getattr(value, "document_type", "GENERIC"),
                "domain": domain,
                "confidence": confidence,
                "filename": fname,
            }
    return {"document_type": "GENERIC", "domain": "generic", "confidence": 0.0}

# ── Ingestion-time entity metadata extraction ──────────────────────────

import re as _re

_EMAIL_RE = _re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = _re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
)
_LINKEDIN_RE = _re.compile(r"linkedin\.com/in/[\w\-]+", _re.IGNORECASE)

def _extract_entity_metadata(raw_docs: Any, filename: str = "") -> Dict[str, Any]:
    """Extract entity name, email, and phone from raw text at ingestion time.

    Uses the first 500 chars of raw text plus filename-based heuristics.
    Returns a dict suitable for storing as ``entity_metadata`` in the pickle.
    """
    text = ""
    if isinstance(raw_docs, dict):
        for _fname, content in raw_docs.items():
            if isinstance(content, dict):
                text = str(content.get("full_text") or content.get("text") or "")[:500]
            elif isinstance(content, str):
                text = content[:500]
            if text:
                break
    elif isinstance(raw_docs, str):
        text = raw_docs[:500]

    result: Dict[str, Any] = {}

    # Email
    email_match = _EMAIL_RE.search(text)
    if email_match:
        result["entity_email"] = email_match.group()

    # Phone
    phone_match = _PHONE_RE.search(text)
    if phone_match:
        candidate = phone_match.group().strip()
        # Only accept if it has enough digits (avoids matching year ranges)
        digits = _re.sub(r"\D", "", candidate)
        if len(digits) >= 7:
            result["entity_phone"] = candidate

    # LinkedIn
    linkedin_match = _LINKEDIN_RE.search(text)
    if linkedin_match:
        result["entity_linkedin"] = linkedin_match.group()

    # Name — try filename first (most reliable), then first line of text
    name = ""
    if filename:
        try:
            from src.rag_v3.extract import _name_from_filename
            name = _name_from_filename(filename) or ""
        except ImportError as exc:
            logger.debug("_name_from_filename not available from rag_v3.extract", exc_info=True)
    if not name and text:
        # Use first non-empty line that looks like a name (2-4 words, no special chars)
        _TITLE_KEYWORDS = {
            "resume", "curriculum", "invoice", "page", "date", "engineer", "analyst",
            "developer", "manager", "director", "specialist", "consultant", "architect",
            "designer", "scientist", "intern", "associate", "officer", "coordinator",
            "administrator", "assistant", "lead", "senior", "junior", "trainee",
            "executive", "supervisor", "data", "software", "project", "product",
            "business", "marketing", "sales", "finance", "human", "resource",
            "full-stack", "full stack", "front-end", "back-end", "devops", "qa",
        }
        for line in text.split("\n")[:10]:
            line = line.strip()
            if not line or len(line) > 60 or "/" in line:
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                lower = line.lower()
                if not any(kw in lower for kw in _TITLE_KEYWORDS):
                    name = line
                    break
    if name:
        result["entity_name"] = name

    return result

def _validate_extraction_fields(
    doc_classification: Dict[str, Any],
    raw_docs: Any,
    filename: str,
) -> None:
    """Log warnings for missing required fields per document type.

    No hard failures — just structured logging for observability.
    """
    doc_type = (doc_classification.get("document_type") or "").lower()
    domain = (doc_classification.get("domain") or "").lower()

    # Build a quick text sample for checking
    text = ""
    if isinstance(raw_docs, dict):
        for _fname, content in raw_docs.items():
            if isinstance(content, dict):
                text = str(content.get("full_text") or "")[:2000]
            elif isinstance(content, str):
                text = content[:2000]
            if text:
                break
    elif isinstance(raw_docs, str):
        text = raw_docs[:2000]

    text_lower = text.lower()

    if domain in ("resume", "hr") or "resume" in doc_type or "cv" in doc_type:
        # Check for name (at least one capitalized word pair in first 300 chars)
        has_name = bool(_re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", text[:300]))
        has_skills = any(kw in text_lower for kw in ("skill", "python", "java", "sql", "experience"))
        has_experience = any(kw in text_lower for kw in ("experience", "worked", "years", "company"))
        if not has_name:
            logger.warning("extraction_validation: resume missing entity name — doc=%s file=%s", filename, doc_type)
        if not has_skills:
            logger.warning("extraction_validation: resume missing skills — doc=%s file=%s", filename, doc_type)
        if not has_experience:
            logger.warning("extraction_validation: resume missing experience — doc=%s file=%s", filename, doc_type)

    elif domain == "invoice" or "invoice" in doc_type:
        has_number = any(kw in text_lower for kw in ("invoice number", "invoice no", "inv #", "inv no"))
        has_date = any(kw in text_lower for kw in ("invoice date", "date:", "due date"))
        has_items = any(kw in text_lower for kw in ("line item", "description", "qty", "quantity", "total"))
        if not has_number and not has_date:
            logger.warning("extraction_validation: invoice missing number/date — doc=%s file=%s", filename, doc_type)
        if not has_items:
            logger.warning("extraction_validation: invoice missing line items/total — doc=%s file=%s", filename, doc_type)

# ---------------------------------------------------------------------------
# Authoritative per-document domain assignment
# ---------------------------------------------------------------------------

_DOC_TYPE_TO_DOMAIN = {
    "resume": "hr", "invoice": "invoice", "purchase_order": "invoice",
    "contract": "legal", "policy": "legal", "statement": "financial",
    "report": "generic", "brochure": "generic", "presentation": "generic",
    "other": "generic",
}

def _resolve_authoritative_domain(
    domain_signals: Dict[str, float],
    doc_type: str,
    structured_domain: str = "",
    doc_type_confidence: float = 0.0,
) -> Dict[str, Any]:
    """Pick authoritative domain from signals.

    Priority:
    1. LLM doc_type classification when confidence >= 0.80 (most accurate — full context)
    2. Highest domain_signal if score >= 0.35 AND clearly dominant (gap >= 0.10 over runner-up)
    3. Fallback to structured_domain
    4. Fallback to doc_type mapped via _DOC_TYPE_TO_DOMAIN
    5. Fallback to "generic"
    """
    result: Dict[str, Any] = {"domain": "generic", "confidence": 0.0, "source": "fallback"}

    # Signal 1: LLM document type classification (most reliable — reads full text)
    doc_type_lower = (doc_type or "").lower()
    mapped = _DOC_TYPE_TO_DOMAIN.get(doc_type_lower)
    if mapped and mapped != "generic" and doc_type_confidence >= 0.80:
        logger.info("Domain resolved via LLM doc_type: %s → %s (confidence=%.2f)", doc_type, mapped, doc_type_confidence)
        return {"domain": mapped, "confidence": doc_type_confidence, "source": "llm_doc_type"}

    # Signal 2: deep analysis domain signals (only if clearly dominant)
    if domain_signals:
        sorted_domains = sorted(domain_signals.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_score = sorted_domains[0]
        runner_up_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0.0
        gap = best_score - runner_up_score
        if best_score >= 0.35 and gap >= 0.10:
            # Cross-check: if LLM doc_type disagrees, trust LLM for known types
            if mapped and mapped != "generic" and mapped != best_domain:
                logger.info("Domain signal %s (%.2f) overridden by LLM doc_type %s → %s",
                            best_domain, best_score, doc_type, mapped)
                return {"domain": mapped, "confidence": max(doc_type_confidence, 0.5), "source": "llm_doc_type_override"}
            result = {"domain": best_domain, "confidence": best_score, "source": "deep_analysis"}
            return result

    # Signal 3: structured extraction domain
    if structured_domain and structured_domain != "generic":
        result = {"domain": structured_domain, "confidence": 0.6, "source": "structured_extraction"}
        return result

    # Signal 4: document type mapping (lower confidence LLM or heuristic)
    if mapped and mapped != "generic":
        result = {"domain": mapped, "confidence": max(doc_type_confidence, 0.5), "source": "doc_type_mapping"}
        return result

    # Normalize final domain through canonical map
    try:
        from src.intelligence.domain_classifier import normalize_domain
        result["domain"] = normalize_domain(result["domain"])
    except ImportError as exc:
        logger.debug("normalize_domain not available from domain_classifier", exc_info=True)

    return result

def _persist_document_domain(document_id: str, domain_result: Dict[str, Any]) -> None:
    """Persist authoritative document domain to MongoDB."""
    try:
        from bson import ObjectId
        from src.api.dataHandler import db
        if db is None:
            return
        coll = db[Config.MongoDB.DOCUMENTS]
        filt = {"_id": ObjectId(str(document_id))} if ObjectId.is_valid(str(document_id)) else {"_id": str(document_id)}
        coll.update_one(filt, {"$set": {
            "document_domain": domain_result.get("domain", "generic"),
            "domain_confidence": domain_result.get("confidence", 0.0),
            "domain_source": domain_result.get("source", "fallback"),
        }}, upsert=False)
    except Exception as exc:
        logger.warning("Failed to persist document domain for %s: %s", document_id, exc)

def _extract_text_from_extracted_document(doc: "ExtractedDocument") -> str:
    """Extract usable text from an ExtractedDocument, trying all available fields.

    Priority: full_text → sections → chunk_candidates → tables.
    Never falls back to str(doc) which produces garbage repr strings.
    """
    text = (doc.full_text or "").strip()
    if text:
        return text

    # Try sections
    section_texts = [sec.text.strip() for sec in (doc.sections or []) if (sec.text or "").strip()]
    if section_texts:
        return "\n\n".join(section_texts)

    # Try chunk_candidates
    candidate_texts = [cand.text.strip() for cand in (doc.chunk_candidates or []) if (cand.text or "").strip()]
    if candidate_texts:
        return "\n\n".join(candidate_texts)

    # Try tables
    table_texts = [t.text.strip() for t in (doc.tables or []) if (t.text or "").strip()]
    if table_texts:
        return "\n\n".join(table_texts)

    # Try canonical_json
    canonical = getattr(doc, "canonical_json", None) or {}
    if isinstance(canonical, dict):
        text_parts = []
        for k, v in canonical.items():
            if isinstance(v, str) and v.strip():
                text_parts.append(f"{k}: {v.strip()}")
        if text_parts:
            return "\n".join(text_parts)

    logger.warning("ExtractedDocument has no usable text content")
    return ""

def _sanitize_raw_text_fields(docs: Any) -> Any:
    """Ensure no stringified repr or dict garbage in text fields."""
    if not isinstance(docs, dict):
        return docs
    from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage

    for fname, content in list(docs.items()):
        # Convert ExtractedDocument objects to dict for downstream compatibility
        if isinstance(content, ExtractedDocument):
            text = _extract_text_from_extracted_document(content)
            if text:
                docs[fname] = {
                    "full_text": text,
                    "sections": [{"text": sec.text, "start_page": sec.start_page, "end_page": sec.end_page, "title": sec.title} for sec in (content.sections or []) if sec.text],
                    "pages": max(1, max((sec.end_page for sec in content.sections if sec.end_page), default=1)) if content.sections else 1,
                    "texts": [text],
                    "doc_type": content.doc_type,
                }
                logger.info("Converted ExtractedDocument to dict for %s (%d chars)", fname, len(text))
            else:
                logger.warning("ExtractedDocument for %s has no extractable text", fname)
            continue
        if not isinstance(content, dict):
            continue
        # Sanitize full_text
        full_text = content.get("full_text")
        if isinstance(full_text, str) and _is_metadata_garbage(full_text):
            section_texts = []
            for sec in content.get("sections") or []:
                if isinstance(sec, dict):
                    t = sec.get("text") or sec.get("content") or ""
                    if isinstance(t, str) and t.strip() and not _is_metadata_garbage(t):
                        section_texts.append(t.strip())
            if section_texts:
                content["full_text"] = "\n\n".join(section_texts)
                logger.info("Recovered full_text from sections for %s", fname)
            else:
                logger.warning("full_text is garbage for %s and no clean sections found", fname)

        # Sanitize texts list
        texts = content.get("texts")
        if isinstance(texts, list):
            clean_texts = []
            for t in texts:
                if isinstance(t, str) and t.strip() and not _is_metadata_garbage(t):
                    clean_texts.append(t)
                elif isinstance(t, dict):
                    extracted = t.get("text") or t.get("content") or ""
                    if isinstance(extracted, str) and extracted.strip():
                        clean_texts.append(extracted)
            content["texts"] = clean_texts
    return docs

def _set_document_status(
    document_id: str,
    status: str,
    error_msg: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    fields: Dict[str, Any] = {"status": status, "updated_at": time.time()}
    # Sync pipeline_status with legacy status for new pipeline compatibility
    from src.api.statuses import (
        STATUS_EXTRACTION_COMPLETED as _SEC, STATUS_SCREENING_COMPLETED as _SSC,
        STATUS_TRAINING_SUCCEEDED as _STS, STATUS_TRAINING_FAILED as _STF,
        PIPELINE_EXTRACTION_COMPLETED, PIPELINE_SCREENING_COMPLETED,
        PIPELINE_TRAINING_COMPLETED, PIPELINE_EMBEDDING_FAILED,
    )
    _pipeline_map = {
        _SEC: PIPELINE_EXTRACTION_COMPLETED,
        _SSC: PIPELINE_SCREENING_COMPLETED,
        _STS: PIPELINE_TRAINING_COMPLETED,
        _STF: PIPELINE_EMBEDDING_FAILED,
    }
    if status in _pipeline_map:
        fields["pipeline_status"] = _pipeline_map[status]
    if error_msg:
        fields["training_error"] = error_msg
        fields["training_failed_at"] = time.time()
    else:
        fields["training_error"] = None
    if extra_fields:
        fields.update(extra_fields)
    update_document_fields(document_id, fields)
    logger.info("Document %s status updated to %s", document_id, status)

def _run_auto_screening(document_id: str, doc_type: Optional[str] = None) -> None:
    """Run screening automatically after extraction completes.

    When auto_attach_on_ingest is enabled, runs the screening engine and
    transitions status to SCREENING_COMPLETED (or TRAINING_BLOCKED_SECURITY).
    When disabled, bypasses screening and directly sets SCREENING_COMPLETED.
    On any failure, sets SCREENING_COMPLETED so embedding is not blocked.
    """
    try:
        from src.screening.config import ScreeningConfig
        cfg = ScreeningConfig.load()
    except Exception:  # noqa: BLE001
        logger.warning("Screening config unavailable for %s; bypassing screening", document_id)
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)
        return

    if not cfg.auto_attach_on_ingest:
        logger.info("Auto-screening disabled; bypassing screening for %s", document_id)
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)
        return

    try:
        from src.screening.engine import ScreeningEngine
        from src.api.screening_service import apply_security_result

        engine = ScreeningEngine(config=cfg)
        report = engine.run_all(document_id, doc_type=doc_type)
        report_dict = report.to_dict()
        apply_security_result(document_id, report_dict)
        logger.info(
            "Auto-screening completed for %s: risk=%s score=%.1f",
            document_id,
            report.risk_level,
            report.overall_score_0_100,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Auto-screening failed for %s: %s; setting SCREENING_COMPLETED to unblock pipeline",
            document_id,
            exc,
        )
        _set_document_status(document_id, STATUS_SCREENING_COMPLETED)

_DEBOUNCE_TTL_SECONDS = 5

def _debounce_extraction(subscription_id: str, doc_id: str) -> bool:
    """Return True if this extraction request is a duplicate within the debounce window."""
    try:
        from src.api.config import Config
        redis_client = getattr(Config.Redis, "get_client", lambda: None)()
        if redis_client is None:
            from src.api.rag_state import get_app_state
            state = get_app_state()
            redis_client = getattr(state, "redis_client", None) if state else None
        if redis_client is None:
            return False
        debounce_key = f"extract:debounce:{subscription_id}:{doc_id}"
        if redis_client.get(debounce_key):
            return True
        redis_client.setex(debounce_key, _DEBOUNCE_TTL_SECONDS, "1")
        return False
    except Exception as exc:
        logger.debug("Failed to check extraction debounce in Redis", exc_info=True)
        return False

# ── Extraction Auto-Subagents ─────────────────────────────────────────────────
# These run automatically during extraction to enrich documents before
# structured extraction / intelligence processing.  They are pre-screening
# enrichments and do NOT affect the HITL pipeline stages.


def _detect_document_language(text: str) -> str:
    """Detect the primary language of *text*.  Returns ISO 639-1 code."""
    if not text or len(text.strip()) < 20:
        return "en"
    try:
        from src.utils.language import detect_language
        lang, confidence = detect_language(text)
        if lang and lang not in ("unknown", "mixed", "non_en") and confidence >= 0.5:
            return lang
        if lang == "non_en":
            # langdetect unavailable but text is clearly non-English
            return "non_en"
    except Exception:
        pass
    return "en"


def _translate_content(text: str, source_lang: str) -> Optional[str]:
    """Translate *text* from *source_lang* to English using the translator tool's 3-tier backend.

    Returns the translated English text, or ``None`` on failure.
    """
    if not text or source_lang == "en":
        return None
    try:
        from src.tools.translator import TranslateRequest, _translate_text
        request = TranslateRequest(text=text[:30000], target_lang="en", source_lang=source_lang)
        result = _translate_text(request)
        translated = result.get("translated_text", "")
        backend = result.get("backend", "unknown")
        if translated and backend != "fallback":
            logger.info("Auto-translated %d chars from %s to en via %s", len(text), source_lang, backend)
            return translated
    except Exception as exc:
        logger.warning("Auto-translation failed for lang=%s: %s", source_lang, exc)
    return None


def _run_ocr_enhancement(doc_id: str, content: Any) -> Optional[str]:
    """Run enhanced Vision OCR on scanned document content.

    Loads the glm-ocr model on demand and unloads it after processing.
    Returns enhanced OCR text or ``None`` if not applicable.
    """
    if not isinstance(content, dict):
        return None

    # Check if document appears to be scanned
    full_text = content.get("full_text") or content.get("text") or ""
    sections = content.get("sections") or []

    # Heuristic: scanned docs have short/garbled text relative to page count
    page_count = 0
    for sec in sections:
        if isinstance(sec, dict):
            page_count = max(page_count, sec.get("end_page") or sec.get("start_page") or 0)
    page_count = max(page_count, 1)

    # If text is very sparse relative to pages, likely scanned
    chars_per_page = len(full_text) / max(page_count, 1)
    is_scanned = chars_per_page < 200 and page_count > 0

    if not is_scanned:
        return None

    logger.info("Document %s appears scanned (%d chars/%d pages); running Vision OCR enhancement", doc_id, len(full_text), page_count)

    try:
        from src.llm.vision_ocr import get_vision_ocr_client
        client = get_vision_ocr_client()
        if client is None or not client.is_available():
            logger.debug("Vision OCR client unavailable for %s", doc_id)
            return None

        # OCR each page image if available in sections
        enhanced_texts: List[str] = []
        for sec in sections:
            if isinstance(sec, dict):
                img_data = sec.get("page_image") or sec.get("image_bytes")
                if img_data:
                    ocr_text, confidence = client.ocr_page_image(img_data)
                    if ocr_text and len(ocr_text) > len(sec.get("text", "")):
                        enhanced_texts.append(ocr_text)
                    else:
                        enhanced_texts.append(sec.get("text", ""))
                else:
                    enhanced_texts.append(sec.get("text", ""))

        result = None
        if enhanced_texts:
            enhanced_full = "\n\n".join(enhanced_texts)
            if len(enhanced_full) > len(full_text) * 1.2:
                logger.info("OCR enhancement improved text from %d to %d chars for %s", len(full_text), len(enhanced_full), doc_id)
                result = enhanced_full

        # Unload model after processing (on-demand only)
        try:
            import ollama
            ollama.generate(model=client.model_name, prompt="", keep_alive=0)
            logger.debug("Unloaded Vision OCR model %s after processing %s", client.model_name, doc_id)
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.warning("OCR enhancement failed for %s: %s", doc_id, exc)
        # Still try to unload model on error
        try:
            from src.llm.vision_ocr import get_vision_ocr_client
            import ollama
            _client = get_vision_ocr_client()
            if _client:
                ollama.generate(model=_client.model_name, prompt="", keep_alive=0)
        except Exception:
            pass

    return None


def _run_extraction_subagents(doc_id: str, masked_docs: Dict[str, Any]) -> Dict[str, Any]:
    """Run auto-subagents on extracted content: language detection, translation, OCR enhancement.

    Modifies *masked_docs* in-place by adding translated text and enhanced OCR text.
    Returns the (possibly enriched) masked_docs.
    """
    for fname, content in masked_docs.items():
        if not isinstance(content, dict):
            continue

        full_text = content.get("full_text") or content.get("text") or ""
        if not full_text:
            continue

        # ── Subagent 1: Language Detection ──
        detected_lang = _detect_document_language(full_text)
        content["detected_language"] = detected_lang
        logger.info("Document %s [%s]: detected language = %s", doc_id, fname, detected_lang)

        # ── Subagent 2: Auto-Translation (non-English → English) ──
        if detected_lang != "en":
            translated_text = _translate_content(full_text, detected_lang)
            if translated_text:
                content["translated_text"] = translated_text
                content["original_language"] = detected_lang
                # Also translate section texts for chunk-level embedding
                for sec in content.get("sections") or []:
                    if isinstance(sec, dict) and sec.get("text"):
                        sec_translated = _translate_content(sec["text"], detected_lang)
                        if sec_translated:
                            sec["translated_text"] = sec_translated
                logger.info("Document %s [%s]: auto-translated from %s to en (%d chars)", doc_id, fname, detected_lang, len(translated_text))

        # ── Subagent 3: OCR Enhancement (scanned documents) ──
        enhanced_text = _run_ocr_enhancement(doc_id, content)
        if enhanced_text:
            content["ocr_enhanced_text"] = enhanced_text
            # If OCR produced significantly better text, use it as primary
            if len(enhanced_text) > len(full_text) * 1.5:
                content["original_text"] = full_text
                content["full_text"] = enhanced_text
                if "text" in content:
                    content["text"] = enhanced_text
                logger.info("Document %s [%s]: OCR-enhanced text promoted to primary", doc_id, fname)

                # Re-detect language on enhanced text and translate if needed
                enhanced_lang = _detect_document_language(enhanced_text)
                content["detected_language"] = enhanced_lang
                if enhanced_lang != "en" and "translated_text" not in content:
                    translated_text = _translate_content(enhanced_text, enhanced_lang)
                    if translated_text:
                        content["translated_text"] = translated_text
                        content["original_language"] = enhanced_lang

        break  # Process first document only (consistent with rest of pipeline)

    return masked_docs


def _extract_from_connector(doc_id: str, doc_data: Dict[str, Any], conn_data: Dict[str, Any]) -> Dict[str, Any]:
    from src.utils.logging_utils import set_pipeline_profile, clear_pipeline_profile
    profile_id = str(doc_data.get("profile")) if doc_data.get("profile") else None
    if profile_id:
        set_pipeline_profile(profile_id)
    subscription_candidate = (
        doc_data.get("subscriptionId")
        or doc_data.get("subscription_id")
        or doc_data.get("subscription")
        or (conn_data.get("subscriptionId") if isinstance(conn_data, dict) else None)
        or (conn_data.get("subscription") if isinstance(conn_data, dict) else None)
    )

    try:
        subscription_id = resolve_subscription_id(doc_id, subscription_candidate)
    except Exception as exc:  # noqa: BLE001
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, str(exc))
        return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

    if _debounce_extraction(subscription_id, doc_id):
        logger.info("Extraction debounced for %s (subscription %s); duplicate within %ds", doc_id, subscription_id, _DEBOUNCE_TTL_SECONDS)
        return {"document_id": doc_id, "status": "CONFLICT", "reason": "debounced_duplicate"}

    lock = acquire_lock(stage="extraction", document_id=doc_id, subscription_id=subscription_id)
    if not lock.acquired:
        logger.info("Extraction already in progress for %s; skipping duplicate trigger.", doc_id)
        return {"document_id": doc_id, "status": "CONFLICT", "reason": "duplicate_extraction_in_progress"}

    pii_masking_enabled = get_subscription_pii_setting(subscription_id)
    logger.info("Document %s (subscription %s): PII masking=%s", doc_id, subscription_id, pii_masking_enabled)

    update_stage(doc_id, "extraction", {"status": "IN_PROGRESS", "started_at": time.time(), "error": None})
    emit_progress(doc_id, "extraction", 0.05, "Starting document extraction")

    try:
        all_extracted_docs: Dict[str, Any] = {}
        _raw_file_bytes: Optional[bytes] = None  # captured for visual intelligence
        try:
            if doc_data.get("type") == "S3":
                bk_name = conn_data["s3_details"]["bucketName"]
                region = conn_data["s3_details"]["region"]
                ak = decrypt_data(conn_data["s3_details"]["accessKey"]).split("\x0c")[0].strip()
                sk = decrypt_data(conn_data["s3_details"]["secretKey"]).split("\x08")[0].strip()
                s3 = get_s3_client(ak, sk, region)
                if not s3:
                    raise ValueError("Failed to create S3 client")

                objs = s3.list_objects_v2(Bucket=bk_name)
                file_keys = [obj["Key"] for obj in objs.get("Contents", []) if obj["Key"] == doc_data.get("name")]
                if not file_keys:
                    raise ValueError("File not found in S3")

                doc_content = read_s3_file(s3, bk_name, file_keys[0])
                _raw_file_bytes = doc_content if isinstance(doc_content, bytes) else None
                if doc_content is None:
                    raise ValueError("Failed to read S3 file")

                extracted_doc = fileProcessor(doc_content, file_keys[0])
                if not extracted_doc:
                    raise ValueError("Content extraction failed")

                all_extracted_docs.update(extracted_doc)
            elif doc_data.get("type") == "LOCAL":
                doc_name = doc_data.get("name", "")
                if not doc_name:
                    raise ValueError("No filename specified")

                all_connector_files = conn_data.get("locations", [])
                matching_files: List[str] = []
                for file_path in all_connector_files:
                    file_name_only = file_path.split("/")[-1] if "/" in file_path else file_path
                    if file_name_only == doc_name:
                        matching_files.append(file_path)
                        break

                if not matching_files:
                    raise ValueError(f"Exact file match not found: {doc_name}")
                if len(matching_files) > 1:
                    raise ValueError("Multiple file matches")

                file_path = matching_files[0]
                file_key = normalize_blob_name(file_path, container_name=Config.AzureBlob.DOCUMENT_CONTAINER_NAME)
                doc_content = get_azure_docs(file_key, document_id=doc_id)
                if doc_content is None:
                    raise ValueError(f"Failed to read file {file_key}")
                _raw_file_bytes = doc_content if isinstance(doc_content, bytes) else None

                extracted_doc = fileProcessor(doc_content, file_path)
                if not extracted_doc:
                    raise ValueError("Content extraction failed")

                all_extracted_docs.update(extracted_doc)
            else:
                raise ValueError(f"Unsupported connector type: {doc_data.get('type')}")
        except CredentialError as exc:
            set_error(doc_id, "extraction", exc)
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"CredentialError: {exc}")
            raise
        except BlobDownloadError as exc:
            set_error(doc_id, "extraction", exc)
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"{exc.__class__.__name__}: {exc}")
            return {
                "document_id": doc_id,
                "status": STATUS_EXTRACTION_FAILED,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            set_error(doc_id, "extraction", exc)
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, str(exc))
            return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

        if not all_extracted_docs:
            _no_content_exc = ValueError("No content extracted")
            set_error(doc_id, "extraction", _no_content_exc)
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, "No content extracted")
            return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": "No content extracted"}

        masked_docs = all_extracted_docs
        pii_count = 0
        pii_items: List[Any] = []
        if pii_masking_enabled:
            masked_docs, pii_count, _high_conf, pii_items = mask_document_content(all_extracted_docs)
            update_pii_stats(doc_id, pii_count, False, pii_items)
        else:
            update_pii_stats(doc_id, 0, False, [])

        if pii_masking_enabled:
            for fname, content in masked_docs.items():
                if isinstance(content, dict) and "texts" in content:
                    texts = content.get("texts") or []
                    if texts:
                        from src.api.dataHandler import encode_with_fallback

                        content["embeddings"] = encode_with_fallback(
                            texts,
                            convert_to_numpy=True,
                            normalize_embeddings=False,
                        )
                    masked_docs[fname] = content

        masked_docs = _normalize_extracted_metadata(masked_docs, document_id=doc_id)
        masked_docs = _sanitize_raw_text_fields(masked_docs)

        _persist_layout_graph(
            document_id=doc_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            extracted_docs=masked_docs,
        )

        emit_progress(doc_id, "extraction", 0.08, "Text extracted from document")

        # ── Auto-subagents: Language Detection + Translation + OCR Enhancement ──
        masked_docs = _run_extraction_subagents(doc_id, masked_docs)

        # ── LLM Deep Knowledge Extraction → KG + Redis hot cache ──
        try:
            from src.intelligence.knowledge_extractor import get_knowledge_extractor
            from src.intelligence.evidence_verifier import verify_knowledge_result
            from src.intelligence.hot_cache import cache_document_knowledge, recompute_profile_domain

            extractor = get_knowledge_extractor()
            for fname, content in masked_docs.items():
                full_text = ""
                if isinstance(content, dict):
                    full_text = content.get("translated_text") or content.get("full_text", "")
                elif isinstance(content, str):
                    full_text = content
                if not full_text or len(full_text.strip()) < 100:
                    continue

                # Extract knowledge from sections or full text
                sections_data = []
                if isinstance(content, dict) and content.get("sections"):
                    for sec in content["sections"]:
                        if isinstance(sec, dict) and sec.get("text"):
                            sections_data.append(sec)
                if not sections_data:
                    # Split into ~2000 char chunks for extraction
                    chunk_size = 2000
                    for i in range(0, len(full_text), chunk_size):
                        sections_data.append({
                            "text": full_text[i:i + chunk_size],
                            "page": (i // chunk_size) + 1,
                            "section_title": f"Section {(i // chunk_size) + 1}",
                        })

                all_entities = []
                all_facts = []
                all_relationships = []
                all_claims = []

                # Parallel KG extraction across sections using ThreadPoolExecutor
                capped_sections = sections_data[:15]  # Cap at 15 sections
                _kg_max_workers = min(len(capped_sections), int(os.getenv("KG_EXTRACTION_MAX_WORKERS", "4")))

                def _extract_and_verify(sec):
                    """Extract knowledge from a single section and verify."""
                    result = extractor.extract_section(
                        text=sec.get("text", ""),
                        page=sec.get("start_page", sec.get("page", 1)),
                        section=sec.get("title", sec.get("section_title", "unknown")),
                    )
                    source_text = sec.get("text", "")
                    return verify_knowledge_result(result, source_text)

                from concurrent.futures import ThreadPoolExecutor, as_completed
                section_results = []
                with ThreadPoolExecutor(max_workers=_kg_max_workers) as kg_executor:
                    future_to_sec = {
                        kg_executor.submit(_extract_and_verify, sec): sec
                        for sec in capped_sections
                    }
                    for future in as_completed(future_to_sec):
                        sec = future_to_sec[future]
                        try:
                            result = future.result()
                            section_results.append(result)
                        except Exception as sec_err:
                            sec_title = sec.get("title", sec.get("section_title", "unknown"))
                            logger.warning(
                                "[EXTRACTION] KG section extraction failed for %s section=%s: %s",
                                doc_id, sec_title, sec_err,
                            )

                for result in section_results:
                    for ent in result.entities:
                        all_entities.append({
                            "name": ent.name, "type": ent.type,
                            "context": ent.context, "evidence": ent.evidence,
                            "confidence": ent.confidence, "location": ent.location,
                        })
                    for fact in result.facts:
                        all_facts.append({
                            "statement": fact.statement, "evidence": fact.evidence,
                            "confidence": fact.confidence, "location": fact.location,
                        })
                    for rel in result.relationships:
                        all_relationships.append({
                            "subject": rel.subject, "object": rel.object,
                            "relation": rel.relation, "evidence": rel.evidence,
                            "confidence": rel.confidence,
                        })
                    for claim in result.claims:
                        all_claims.append({
                            "claim": claim.claim, "evidence": claim.evidence,
                            "confidence": claim.confidence,
                        })

                # Generate document summary
                doc_summary = extractor.generate_document_summary(full_text)

                # Store in content for downstream KG ingestion
                if isinstance(content, dict):
                    content["kg_entities"] = all_entities
                    content["kg_facts"] = all_facts
                    content["kg_relationships"] = all_relationships
                    content["kg_claims"] = all_claims
                    content["kg_summary"] = doc_summary
                    content["detected_domain"] = doc_summary.get("domain", "general")

                # Cache in Redis hot cache
                try:
                    from src.api.rag_state import get_app_state
                    app_state = get_app_state()
                    redis_client = getattr(app_state, "redis_client", None) if app_state else None
                    if redis_client:
                        cache_document_knowledge(
                            redis_client=redis_client,
                            profile_id=profile_id,
                            doc_id=doc_id,
                            entities=all_entities,
                            facts=all_facts,
                            claims=all_claims,
                            relationships=all_relationships,
                            domain=doc_summary.get("domain", "general"),
                            summary=doc_summary.get("summary", ""),
                        )
                        recompute_profile_domain(redis_client, profile_id)
                except Exception as cache_err:
                    logger.debug("[EXTRACTION] Redis cache write failed (non-fatal): %s", cache_err)

                logger.info(
                    "[EXTRACTION] KG extraction for %s: entities=%d facts=%d rels=%d claims=%d domain=%s",
                    doc_id, len(all_entities), len(all_facts),
                    len(all_relationships), len(all_claims),
                    doc_summary.get("domain", "general"),
                )

            emit_progress(doc_id, "extraction", 0.09, "Knowledge graph enriched")
        except ImportError:
            logger.debug("[EXTRACTION] Knowledge extractor not available — skipping KG enrichment")
        except Exception as kg_err:
            logger.warning("[EXTRACTION] KG enrichment failed (non-fatal): %s", kg_err)

        # Structured extraction: build structured JSON from masked/raw text and persist alongside raw extraction
        try:
            engine = get_extraction_engine()
            # For each file in masked_docs, create structured doc; if multiple files, pick the first as primary
            # Compose a combined structured representation keyed by filename
            structured_docs = {}
            for fname, content in masked_docs.items():
                try:
                    # Determine raw text to feed into structured extractor
                    raw_text = ""
                    if isinstance(content, dict):
                        raw_text = content.get("full_text") or content.get("text") or "\n".join(
                            [sec.get("text") for sec in (content.get("sections") or []) if sec.get("text")]
                        )
                        page_count = content.get("pages") or (len(content.get("sections") or []) or 1)
                    elif isinstance(content, ExtractedDocument):
                        raw_text = _extract_text_from_extracted_document(content)
                        page_count = max(1, max((sec.end_page for sec in content.sections if sec.end_page), default=1)) if content.sections else 1
                    else:
                        raw_text = str(content) if not hasattr(content, "full_text") else (getattr(content, "full_text", "") or "")
                        page_count = 1

                    structured = engine.extract_document(
                        document_id=doc_id,
                        text=raw_text,
                        filename=fname,
                        metadata={"source": "extraction", "orig_filename": fname},
                        page_count=page_count,
                    )
                    structured_docs[fname] = structured
                except Exception as sexc:  # noqa: BLE001
                    logger.warning("Structured extraction failed for %s: %s", fname, sexc)
                    # fallback: store raw content under structured as minimal
                    structured_docs[fname] = {
                        "document_id": doc_id,
                        "original_filename": fname,
                        "document_type": "GENERIC",
                        "sections": [{"section_id": "section_0", "section_type": "content", "content": raw_text}],
                        "extraction_quality_score": 0.0,
                    }

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to run structured extraction engine for %s: %s", doc_id, exc)
            structured_docs = {}

        # --- Visual Intelligence Enrichment (second pass) ---
        if VISUAL_INTELLIGENCE_AVAILABLE and _raw_file_bytes:
            try:
                import asyncio
                _vi_orch = get_visual_orchestrator()
                for _vi_fname, _vi_content in (masked_docs or {}).items():
                    try:
                        _vi_enriched = asyncio.run(
                            _vi_orch.enrich(doc_id, _vi_content, _raw_file_bytes)
                        )
                        if _vi_enriched is not _vi_content:
                            masked_docs[_vi_fname] = _vi_enriched
                            logger.info("Visual intelligence enriched document %s (%s)", doc_id, _vi_fname)
                    except Exception as _vi_page_exc:
                        logger.warning("Visual intelligence enrichment failed for %s/%s: %s", doc_id, _vi_fname, _vi_page_exc)
                    break  # Process first document only
            except Exception as _vi_exc:
                logger.warning("Visual intelligence enrichment skipped: %s", _vi_exc)

        # Intelligence layer processing: entity extraction, Q&A generation
        intelligence_result = _process_document_intelligence(
            document_id=doc_id,
            extracted_docs=masked_docs,
            filename=doc_data.get("name", "document"),
            subscription_id=subscription_id,
            profile_id=profile_id,
        )

        # Document understanding (summary, entities, facts) — runs inline for pickle enrichment
        understanding_result = None
        try:
            from src.doc_understanding import identify_document, understand_document, build_content_map
            for _fname, _content in (masked_docs or {}).items():
                identification = identify_document(extracted=_content, filename=_fname)
                content_map = build_content_map(_content)
                understanding = understand_document(extracted=_content, doc_type=identification.document_type)
                understanding_result = {
                    "document_type": identification.document_type,
                    "doc_type_confidence": identification.confidence,
                    "document_summary": understanding.get("document_summary"),
                    "section_summaries": understanding.get("section_summaries"),
                    "key_entities": understanding.get("key_entities"),
                    "key_facts": understanding.get("key_facts"),
                    "intent_tags": understanding.get("intent_tags"),
                    "content_map": content_map,
                }
                # Store in MongoDB for fast lookups during retrieval
                try:
                    update_extraction_metadata(doc_id, subscription_id, None, None)
                    _update_understanding_fields(doc_id, understanding_result)
                except Exception as exc:
                    logger.warning("Failed to persist understanding for connector %s: %s", doc_id, exc)
                break  # Process first document only
        except Exception as exc:
            logger.warning("Document understanding skipped for %s: %s", doc_id, exc)

        # Deep document analysis (entity extraction, quality grading, temporal analysis)
        deep_result = None
        try:
            from src.api.config import Config as _Cfg
            if getattr(_Cfg, "DeepAnalysis", None) and getattr(_Cfg.DeepAnalysis, "ENABLED", True):
                from src.doc_understanding.deep_analyzer import deep_analyze
                for _fname, _content in (masked_docs or {}).items():
                    with _DOC_PROCESSING_SEMAPHORE:
                        deep_result = deep_analyze(
                            _content,
                            identification=understanding_result,
                            content_map=understanding_result.get("content_map") if understanding_result else None,
                        )
                    # Store deep analysis fields in MongoDB
                    try:
                        _deep_fields = {
                            "quality_grade": deep_result.quality_grade,
                            "quality_score": deep_result.quality_score,
                            "complexity_score": deep_result.complexity_score,
                            "entity_mentions": [e.to_dict() for e in deep_result.entities[:100]],
                            "chronological_span": deep_result.temporal_spans[:20],
                            "section_roles": deep_result.section_roles,
                            "domain_signals": deep_result.domain_signals,
                        }
                        _update_understanding_fields(doc_id, _deep_fields)
                    except Exception as exc:
                        logger.warning("Failed to persist deep analysis for connector %s: %s", doc_id, exc)
                    # Enqueue background analysis for heavy processing
                    try:
                        from src.doc_understanding.background_analyzer import get_background_analyzer
                        bg = get_background_analyzer()
                        if bg and getattr(_Cfg.DeepAnalysis, "BACKGROUND_ENABLED", True):
                            bg.enqueue(doc_id, subscription_id, profile_id)
                    except Exception as exc:
                        logger.debug("Failed to enqueue background analysis for connector %s", doc_id, exc_info=True)
                    break
        except Exception as exc:
            logger.warning("Deep analysis skipped for %s: %s", doc_id, exc)

        try:
            # Persist raw, structured extraction and intelligence in the pickle for source-of-truth
            doc_classification = _extract_classification_from_structured(structured_docs)
            entity_meta = _extract_entity_metadata(masked_docs, doc_data.get("name", ""))
            _validate_extraction_fields(doc_classification, masked_docs, doc_data.get("name", ""))

            # Resolve authoritative document domain
            document_domain = _resolve_authoritative_domain(
                domain_signals=deep_result.domain_signals if deep_result else {},
                doc_type=(understanding_result or {}).get("document_type", ""),
                structured_domain=doc_classification.get("domain", ""),
                doc_type_confidence=(understanding_result or {}).get("doc_type_confidence", 0.0),
            )
            _persist_document_domain(doc_id, document_domain)

            payload_to_save = {
                "raw": masked_docs,
                "structured": structured_docs,
                "intelligence": intelligence_result,
                "document_classification": doc_classification,
                "entity_metadata": entity_meta,
                "understanding": understanding_result,
                "document_domain": document_domain,
                "deep_analysis": {
                    "quality_grade": deep_result.quality_grade if deep_result else None,
                    "quality_score": deep_result.quality_score if deep_result else None,
                    "complexity_score": deep_result.complexity_score if deep_result else None,
                    "domain_signals": deep_result.domain_signals if deep_result else None,
                } if deep_result else None,
            }
            save_info = save_extracted_pickle(doc_id, payload_to_save)
            update_extraction_metadata(doc_id, subscription_id, save_info.get("path"), save_info.get("sha256"))
        except Exception as exc:  # noqa: BLE001
            set_error(doc_id, "extraction", exc)
            _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"Failed to persist extracted content: {exc}")
            return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

        extra_fields = {
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "extracted_pickle_path": save_info.get("path"),
            "extracted_hash": save_info.get("sha256"),
        }
        _set_document_status(doc_id, STATUS_EXTRACTION_COMPLETED, extra_fields=extra_fields)
        update_stage(doc_id, "extraction", {"status": "COMPLETED", "completed_at": time.time(), "error": None})

        # KG ingestion (async, non-blocking — run in daemon thread to avoid stalling extraction)
        _kg_async = getattr(getattr(Config, "DocumentProcessing", None), "KG_INGEST_ASYNC", True)
        if _kg_async:
            threading.Thread(
                target=_ingest_to_knowledge_graph,
                kwargs=dict(
                    document_id=doc_id,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    source_name=doc_data.get("name", "Unknown"),
                    payload_to_save=payload_to_save,
                    deep_result=deep_result,
                ),
                daemon=True,
            ).start()
        else:
            _ingest_to_knowledge_graph(
                document_id=doc_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
                source_name=doc_data.get("name", "Unknown"),
                payload_to_save=payload_to_save,
                deep_result=deep_result,
            )

        # HITL: No auto-embedding. Document stays at EXTRACTION_COMPLETED.
        # User must manually trigger screening then embedding.
        _mark_intelligence_ready(doc_id)
        logger.info("Extraction completed for %s; awaiting manual screening/embedding", doc_id)

        # Refresh profile-level domain tag
        try:
            from src.profiles.profile_domain_tagger import refresh_profile_domain_on_document_change
            if subscription_id and profile_id:
                refresh_profile_domain_on_document_change(subscription_id, profile_id)
        except Exception as exc:
            logger.debug("Failed to refresh profile domain tag after connector extraction", exc_info=True)

        summary = _build_extraction_summary(masked_docs)
        return {
            "document_id": doc_id,
            "status": STATUS_EXTRACTION_COMPLETED,
            "pickle_path": save_info.get("path"),
            "summary": summary,
            "doc_name": doc_data.get("name", "Unknown"),
        }
    finally:
        clear_pipeline_profile()
        if lock.acquired:
            release_lock(lock)

def _get_current_doc_status(doc_id: str) -> Optional[str]:
    """Query MongoDB for the current status of a single document."""
    try:
        from bson import ObjectId
        collection = get_documents_collection()
        # Try ObjectId first, then string fallback
        doc = None
        if ObjectId.is_valid(str(doc_id)):
            doc = collection.find_one({"_id": ObjectId(str(doc_id))}, {"status": 1})
        if not doc:
            doc = collection.find_one({"_id": str(doc_id)}, {"status": 1})
        return doc.get("status") if doc else None
    except Exception:
        return None


# Subscription-level batch lock to prevent concurrent extraction runs
_BATCH_LOCK_TTL_SECONDS = 1800  # 30 minutes max

def _acquire_batch_lock(subscription_id: str) -> Optional[str]:
    """Acquire a subscription-level batch extraction lock. Returns lock key if acquired, None otherwise."""
    lock_key = f"docwain:batch_extraction:{subscription_id}"
    try:
        from src.api.dw_newron import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            acquired = bool(redis_client.set(lock_key, "1", nx=True, ex=_BATCH_LOCK_TTL_SECONDS))
            return lock_key if acquired else None
    except Exception:
        pass
    # Fallback: use in-memory lock
    from src.utils.idempotency import _MEMORY_LOCKS, _MEMORY_LOCK
    now = time.time()
    with _MEMORY_LOCK:
        existing = _MEMORY_LOCKS.get(lock_key)
        if existing and existing > now:
            return None
        _MEMORY_LOCKS[lock_key] = now + _BATCH_LOCK_TTL_SECONDS
        return lock_key

def _release_batch_lock(lock_key: str) -> None:
    """Release the subscription-level batch extraction lock."""
    try:
        from src.api.dw_newron import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            redis_client.delete(lock_key)
            return
    except Exception:
        pass
    from src.utils.idempotency import _MEMORY_LOCKS, _MEMORY_LOCK
    with _MEMORY_LOCK:
        _MEMORY_LOCKS.pop(lock_key, None)


def _emit_batch_progress(subscription_id: str, completed: int, total: int,
                          current_doc: str = "", stage: str = "") -> None:
    """Publish batch extraction progress to Redis for frontend polling."""
    try:
        from src.api.dw_newron import get_redis_client
        import json as _json
        client = get_redis_client()
        if not client:
            return
        progress = round(completed / total, 3) if total > 0 else 0.0
        event = {
            "subscription_id": subscription_id,
            "completed": completed,
            "total": total,
            "progress": progress,
            "current_document": current_doc,
            "stage": stage,
            "timestamp": time.time(),
        }
        payload = _json.dumps(event)
        client.setex(f"dw:extraction:batch_progress:{subscription_id}", 3600, payload)
        client.publish("dw:extraction:progress", payload)
    except Exception:
        pass


def get_batch_extraction_progress(subscription_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest batch extraction progress for a subscription (for polling)."""
    try:
        import json as _json
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return None
        raw = client.get(f"dw:extraction:batch_progress:{subscription_id}")
        return _json.loads(raw) if raw else None
    except Exception:
        return None


def extract_documents(subscription_id: Optional[str] = None) -> Dict[str, Any]:
    batch_start = time.time()

    # --- Subscription-level lock: prevent concurrent batch extractions ---
    effective_sub = str(subscription_id) if subscription_id else "global"
    batch_lock_key = _acquire_batch_lock(effective_sub)
    if not batch_lock_key:
        logger.info(
            "Batch extraction already in progress for subscription %s; rejecting duplicate request",
            effective_sub,
        )
        return {
            "status": "already_running",
            "message": f"Extraction is already running for subscription {effective_sub}. "
                       "Please wait for the current batch to complete.",
            "documents": [],
        }

    try:
        doc_coll = extract_document_info()
        if not doc_coll:
            return {"status": "no_documents", "message": "No documents found for extraction", "documents": []}

        allowed_statuses = {STATUS_UNDER_REVIEW, STATUS_EXTRACTION_FAILED}
        eligible_docs = {
            doc_id: doc_info
            for doc_id, doc_info in doc_coll.items()
            if doc_info.get("dataDict", {}).get("status") in allowed_statuses
        }

        # Scope by subscription_id when provided
        if subscription_id:
            sub_str = str(subscription_id)
            eligible_docs = {
                doc_id: doc_info
                for doc_id, doc_info in eligible_docs.items()
                if str(doc_info.get("dataDict", {}).get("subscription_id")
                       or doc_info.get("dataDict", {}).get("subscription")
                       or doc_info.get("dataDict", {}).get("subscriptionId")
                       or "") == sub_str
            }

        if not eligible_docs:
            # Report why no documents are eligible
            total_with_connectors = len(doc_coll)
            all_statuses = {}
            for _did, _di in doc_coll.items():
                s = _di.get("dataDict", {}).get("status", "unknown")
                all_statuses[s] = all_statuses.get(s, 0) + 1
            return {
                "status": "no_documents",
                "message": (
                    f"No documents eligible for extraction. "
                    f"Found {total_with_connectors} documents with valid connectors. "
                    f"Status breakdown: {all_statuses}. "
                    f"Only UNDER_REVIEW or EXTRACTION_FAILED documents can be extracted."
                ),
                "documents": [],
            }

        total = len(eligible_docs)
        logger.info(
            "╔══════════════════════════════════════════════════════════════╗"
        )
        logger.info(
            "║  BATCH EXTRACTION START: %d documents queued (sub=%s)  ║",
            total, effective_sub,
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════════╝"
        )
        _emit_batch_progress(effective_sub, 0, total, stage="starting")

        documents = []
        completed_count = 0
        skipped_count = 0
        for idx, (doc_id, doc_info) in enumerate(eligible_docs.items(), 1):
            doc_name = doc_info.get("dataDict", {}).get("name", "Unknown")

            # --- FRESH STATUS CHECK: re-query MongoDB before processing ---
            current_status = _get_current_doc_status(doc_id)
            if current_status and current_status not in allowed_statuses:
                logger.info(
                    "[EXTRACTION %d/%d] Skipped: doc=%s name=%s reason=already_%s",
                    idx, total, doc_id, doc_name, current_status,
                )
                skipped_count += 1
                documents.append({
                    "document_id": doc_id,
                    "doc_name": doc_name,
                    "status": current_status,
                    "reason": f"already_{current_status}",
                    "index": f"{idx}/{total}",
                    "elapsed_seconds": 0,
                })
                _emit_batch_progress(effective_sub, completed_count, total,
                                      current_doc=doc_name,
                                      stage=f"skipped {idx}/{total} (already {current_status})")
                continue

            doc_start = time.time()
            logger.info(
                "[EXTRACTION %d/%d] ▶ Starting: doc=%s name=%s",
                idx, total, doc_id, doc_name,
            )
            _emit_batch_progress(effective_sub, completed_count, total,
                                  current_doc=doc_name,
                                  stage=f"extracting {idx}/{total}")
            # Emit per-document status log and progress for batch extraction
            try:
                from src.api.document_status import emit_status_log as _esl, clear_status_logs as _csl
                _csl(doc_id)
                _esl(doc_id, "extraction", "batch_extraction_start",
                     f"Batch extraction started ({idx}/{total})",
                     extra={"batch_index": idx, "batch_total": total, "doc_name": doc_name})
                emit_progress(doc_id, "extraction", 0.05, f"Extraction starting ({idx}/{total})")
            except Exception:
                pass
            try:
                res = _extract_from_connector(doc_id, doc_info.get("dataDict", {}), doc_info.get("connDict", {}))
            except CredentialError as exc:
                logger.error("Credential error during extraction; failing batch: %s", exc)
                return {"status": "error", "message": f"CredentialError: {exc}", "documents": documents}
            elapsed = round(time.time() - doc_start, 1)
            res["elapsed_seconds"] = elapsed
            res["doc_name"] = doc_name
            res["index"] = f"{idx}/{total}"
            status = res.get("status", "")
            if status == STATUS_EXTRACTION_COMPLETED:
                completed_count += 1
                logger.info(
                    "[EXTRACTION %d/%d] ✓ Completed: doc=%s name=%s in %.1fs",
                    idx, total, doc_id, doc_name, elapsed,
                )
                try:
                    _esl(doc_id, "extraction", "extraction_completed",
                         f"Extraction completed in {elapsed}s",
                         extra={"elapsed_seconds": elapsed})
                    emit_progress(doc_id, "extraction", 0.20, f"Extraction completed in {elapsed}s")
                except Exception:
                    pass
            elif status == "CONFLICT":
                skipped_count += 1
                logger.info(
                    "[EXTRACTION %d/%d] ⊘ Skipped: doc=%s reason=%s",
                    idx, total, doc_id, res.get("reason", "conflict"),
                )
                try:
                    _esl(doc_id, "extraction", "extraction_skipped",
                         f"Skipped: {res.get('reason', 'conflict')}")
                except Exception:
                    pass
            else:
                logger.warning(
                    "[EXTRACTION %d/%d] ✗ Failed: doc=%s name=%s in %.1fs error=%s",
                    idx, total, doc_id, doc_name, elapsed, res.get("error", str(status)),
                )
                try:
                    _esl(doc_id, "extraction", "extraction_failed",
                         f"Extraction failed after {elapsed}s: {res.get('error', str(status))}",
                         extra={"elapsed_seconds": elapsed, "error": res.get("error")})
                    emit_progress(doc_id, "failed", 0.0, f"Extraction failed: {res.get('error', str(status))}")
                except Exception:
                    pass
            documents.append(res)
            _emit_batch_progress(effective_sub, completed_count, total,
                                  current_doc=doc_name,
                                  stage=f"processed {idx}/{total}")

        successful = [d for d in documents if d.get("status") == STATUS_EXTRACTION_COMPLETED]
        failed = [d for d in documents if d.get("status") not in (STATUS_EXTRACTION_COMPLETED, "CONFLICT") and not d.get("reason", "").startswith("already_")]
        batch_elapsed = round(time.time() - batch_start, 1)
        logger.info(
            "╔══════════════════════════════════════════════════════════════╗"
        )
        logger.info(
            "║  BATCH EXTRACTION DONE: %d succeeded, %d skipped, %d failed (%.1fs)  ║",
            len(successful), skipped_count, len(failed), batch_elapsed,
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════════╝"
        )
        _emit_batch_progress(effective_sub, len(successful), total,
                              stage="completed")
        return {
            "status": "completed",
            "total": total,
            "successful_count": len(successful),
            "skipped_count": skipped_count,
            "failed_count": len(failed),
            "elapsed_seconds": batch_elapsed,
            "documents": documents,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Extraction process failed: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc), "documents": []}
    finally:
        _release_batch_lock(batch_lock_key)

def extract_single_document(doc_id: str) -> Dict[str, Any]:
    doc_coll = extract_document_info()
    if doc_coll and doc_id in doc_coll:
        doc_info = doc_coll[doc_id]
        return _extract_from_connector(doc_id, doc_info.get("dataDict", {}), doc_info.get("connDict", {}))

    # Fallback: profile-uploaded document (content in Azure Blob)
    from src.api.document_status import get_document_record
    record = get_document_record(doc_id)
    if not record:
        return {"status": "not_found", "message": f"Document {doc_id} not found"}

    # Resolve blob location: try blob_url (new pipeline) or location (UI connector)
    blob_url = record.get("blob_url") or record.get("location", "")
    source_file = record.get("source_file") or record.get("name", "document")
    subscription_id = record.get("subscription_id") or record.get("subscription")
    profile_id = record.get("profile_id") or record.get("profile")

    if blob_url:
        try:
            if blob_url.startswith("az://"):
                # UI connector format: az://container/path
                from src.storage.azure_blob_client import get_blob_service_client
                svc = get_blob_service_client()
                parts = blob_url.replace("az://", "", 1).split("/", 1)
                container_name, blob_name = parts[0], parts[1] if len(parts) > 1 else ""
                container = svc.get_container_client(container_name)
                blob_client = container.get_blob_client(blob_name)
            else:
                from src.api.blob_content_store import get_blob_client
                container = get_blob_client()
                blob_name = f"raw/{doc_id}/{source_file}"
                blob_client = container.get_blob_client(blob_name)

            file_bytes = blob_client.download_blob().readall()
            logger.info("Loaded %d bytes from blob for doc=%s", len(file_bytes), doc_id)
        except Exception as exc:
            logger.error("Failed to download from blob for doc=%s: %s", doc_id, exc)
            return {"status": "error", "message": f"Blob download failed: {exc}"}
    else:
        return {"status": "error", "message": f"Document {doc_id} has no connector or blob_url"}

    return extract_uploaded_document(
        document_id=doc_id,
        file_bytes=file_bytes,
        filename=source_file,
        subscription_id=subscription_id,
        profile_id=profile_id,
        doc_type=record.get("doc_type") or record.get("file_type") or record.get("type"),
        content_type=record.get("content_type"),
        content_size=len(file_bytes),
    )

def extract_uploaded_document(
    *,
    document_id: str,
    file_bytes: bytes,
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    profile_name: Optional[str] = None,
    doc_type: Optional[str] = None,
    content_type: Optional[str] = None,
    content_size: Optional[int] = None,
) -> Dict[str, Any]:
    from src.utils.logging_utils import set_pipeline_profile, clear_pipeline_profile
    if profile_id:
        set_pipeline_profile(profile_id)
    lock = acquire_lock(stage="extraction", document_id=document_id, subscription_id=subscription_id)
    if not lock.acquired:
        logger.info("Extraction already in progress for %s; skipping duplicate upload.", document_id)
        return {"status": "skipped", "reason": "duplicate_extraction_in_progress", "document_id": document_id}
    init_document_record(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        doc_type=doc_type,
        filename=filename,
        content_type=content_type,
        size=content_size,
    )
    if profile_name:
        update_document_fields(document_id, {"profile_name": profile_name, "metadata.profile_name": profile_name})

    update_stage(document_id, "extraction", {"status": "IN_PROGRESS", "started_at": time.time(), "error": None})
    emit_progress(document_id, "extraction", 0.05, "Starting document extraction")
    from src.api.document_status import emit_status_log, clear_status_logs
    from src.utils.logging_utils import clear_live_logs
    clear_status_logs(document_id)
    if profile_id:
        clear_live_logs(profile_id)
    emit_status_log(document_id, "extraction", "pipeline_start", "Extraction pipeline started",
                    extra={"filename": filename, "content_type": content_type, "content_size": content_size})

    try:
        extracted = fileProcessor(file_bytes, filename, content_type=content_type or "")
        if not extracted:
            raise ValueError("No content extracted from file")
    except Exception as exc:  # noqa: BLE001
        set_error(document_id, "extraction", exc)
        _set_document_status(document_id, STATUS_EXTRACTION_FAILED, str(exc))
        emit_status_log(document_id, "extraction", "text_extraction_failed", f"Text extraction failed: {exc}")
        clear_pipeline_profile()
        raise

    emit_progress(document_id, "extraction", 0.08, "Text extracted from document")
    emit_status_log(document_id, "extraction", "text_extracted", "Raw text extracted from document",
                    extra={"filename": filename})

    extracted = _normalize_extracted_metadata(extracted, document_id=document_id)
    extracted = _sanitize_raw_text_fields(extracted)
    emit_status_log(document_id, "extraction", "content_normalized", "Extraction content normalized and sanitized")

    _persist_layout_graph(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        extracted_docs=extracted,
    )

    emit_status_log(document_id, "extraction", "layout_graph_persisted", "Layout graph built and persisted")

    # Structured extraction for uploaded document
    try:
        engine = get_extraction_engine()
        structured_docs = {}
        for fname, content in extracted.items() if isinstance(extracted, dict) else [(filename, extracted)]:
            try:
                raw_text = ""
                if isinstance(content, dict):
                    raw_text = content.get("full_text") or content.get("text") or "\n".join(
                        [sec.get("text") for sec in (content.get("sections") or []) if sec.get("text")]
                    )
                    page_count = content.get("pages") or (len(content.get("sections") or []) or 1)
                elif isinstance(content, ExtractedDocument):
                    raw_text = _extract_text_from_extracted_document(content)
                    page_count = max(1, max((sec.end_page for sec in content.sections if sec.end_page), default=1)) if content.sections else 1
                else:
                    raw_text = str(content) if not hasattr(content, "full_text") else (getattr(content, "full_text", "") or "")
                    page_count = 1

                structured = engine.extract_document(
                    document_id=document_id,
                    text=raw_text,
                    filename=fname,
                    metadata={"source": "upload", "orig_filename": fname},
                    page_count=page_count,
                )
                structured_docs[fname] = structured
            except Exception as sexc:  # noqa: BLE001
                logger.warning("Structured extraction failed for upload %s: %s", fname, sexc)
                structured_docs[fname] = {"document_id": document_id, "original_filename": fname, "document_type": "GENERIC", "sections": [{"section_id": "section_0", "section_type": "content", "content": raw_text}], "extraction_quality_score": 0.0}
    except Exception as exc:
        logger.warning("Failed to build structured documents for upload", exc_info=True)
        structured_docs = {}

    emit_status_log(document_id, "extraction", "structured_extraction_done",
                    f"Structured extraction completed ({len(structured_docs)} files processed)",
                    extra={"files_processed": len(structured_docs)})

    # --- Visual Intelligence Enrichment (second pass) ---
    if VISUAL_INTELLIGENCE_AVAILABLE:
        try:
            import asyncio
            _vi_orch = get_visual_orchestrator()
            _vi_docs = extracted if isinstance(extracted, dict) else {filename: extracted}
            for _vi_fname, _vi_content in _vi_docs.items():
                try:
                    _vi_enriched = asyncio.run(
                        _vi_orch.enrich(document_id, _vi_content, file_bytes)
                    )
                    if _vi_enriched is not _vi_content:
                        if isinstance(extracted, dict):
                            extracted[_vi_fname] = _vi_enriched
                        else:
                            extracted = _vi_enriched
                        logger.info("Visual intelligence enriched document %s (%s)", document_id, _vi_fname)
                except Exception as _vi_page_exc:
                    logger.warning("Visual intelligence enrichment failed for %s/%s: %s", document_id, _vi_fname, _vi_page_exc)
                break  # Process first document only
        except Exception as _vi_exc:
            logger.warning("Visual intelligence enrichment skipped: %s", _vi_exc)

    # Intelligence layer processing: entity extraction, Q&A generation
    intelligence_result = _process_document_intelligence(
        document_id=document_id,
        extracted_docs=extracted if isinstance(extracted, dict) else {filename: extracted},
        filename=filename,
        subscription_id=subscription_id,
        profile_id=profile_id,
    )

    emit_status_log(document_id, "extraction", "intelligence_processing_done",
                    "Intelligence layer processing completed (entity extraction, Q&A generation)")

    # Document understanding (summary, entities, facts) — runs inline for pickle enrichment
    understanding_result = None
    try:
        from src.doc_understanding import identify_document, understand_document, build_content_map
        _docs_for_understanding = extracted if isinstance(extracted, dict) else {filename: extracted}
        for _fname, _content in _docs_for_understanding.items():
            identification = identify_document(extracted=_content, filename=_fname)
            content_map = build_content_map(_content)
            understanding = understand_document(extracted=_content, doc_type=identification.document_type)
            understanding_result = {
                "document_type": identification.document_type,
                "doc_type_confidence": identification.confidence,
                "document_summary": understanding.get("document_summary"),
                "section_summaries": understanding.get("section_summaries"),
                "key_entities": understanding.get("key_entities"),
                "key_facts": understanding.get("key_facts"),
                "intent_tags": understanding.get("intent_tags"),
                "content_map": content_map,
            }
            try:
                _update_understanding_fields(document_id, understanding_result)
            except Exception as exc:
                logger.warning("Failed to persist understanding for upload %s: %s", document_id, exc)
            break
    except Exception as exc:
        logger.warning("Document understanding skipped for %s: %s", document_id, exc)

    if understanding_result:
        emit_status_log(document_id, "extraction", "understanding_done",
                        f"Document understanding completed: type={understanding_result.get('document_type', 'unknown')}",
                        extra={"document_type": understanding_result.get("document_type"),
                               "doc_type_confidence": understanding_result.get("doc_type_confidence"),
                               "entities_count": len(understanding_result.get("key_entities") or []),
                               "facts_count": len(understanding_result.get("key_facts") or [])})
    else:
        emit_status_log(document_id, "extraction", "understanding_skipped", "Document understanding skipped")

    # Deep document analysis (upload path)
    deep_result = None
    try:
        from src.api.config import Config as _Cfg
        if getattr(_Cfg, "DeepAnalysis", None) and getattr(_Cfg.DeepAnalysis, "ENABLED", True):
            from src.doc_understanding.deep_analyzer import deep_analyze
            _docs_for_deep = extracted if isinstance(extracted, dict) else {filename: extracted}
            for _fname, _content in _docs_for_deep.items():
                with _DOC_PROCESSING_SEMAPHORE:
                    deep_result = deep_analyze(
                        _content,
                        identification=understanding_result,
                        content_map=understanding_result.get("content_map") if understanding_result else None,
                    )
                try:
                    _deep_fields = {
                        "quality_grade": deep_result.quality_grade,
                        "quality_score": deep_result.quality_score,
                        "complexity_score": deep_result.complexity_score,
                        "entity_mentions": [e.to_dict() for e in deep_result.entities[:100]],
                        "chronological_span": deep_result.temporal_spans[:20],
                        "section_roles": deep_result.section_roles,
                        "domain_signals": deep_result.domain_signals,
                    }
                    _update_understanding_fields(document_id, _deep_fields)
                except Exception as exc:
                    logger.warning("Failed to persist deep analysis for upload %s: %s", document_id, exc)
                try:
                    from src.doc_understanding.background_analyzer import get_background_analyzer
                    bg = get_background_analyzer()
                    if bg and getattr(_Cfg.DeepAnalysis, "BACKGROUND_ENABLED", True):
                        bg.enqueue(document_id, subscription_id, profile_id)
                except Exception as exc:
                    logger.debug("Failed to enqueue background analysis for upload %s", document_id, exc_info=True)
                break
    except Exception as exc:
        logger.warning("Deep analysis skipped for upload %s: %s", document_id, exc)

    if deep_result:
        emit_status_log(document_id, "extraction", "deep_analysis_done",
                        f"Deep analysis completed: quality={deep_result.quality_grade}, complexity={deep_result.complexity_score:.2f}",
                        extra={"quality_grade": deep_result.quality_grade,
                               "quality_score": deep_result.quality_score,
                               "complexity_score": deep_result.complexity_score,
                               "entities_found": len(deep_result.entities) if deep_result.entities else 0})
    else:
        emit_status_log(document_id, "extraction", "deep_analysis_skipped", "Deep analysis skipped")

    try:
        # Persist uploaded file extraction (raw, structured, intelligence, classification) in the pickle
        doc_classification = _extract_classification_from_structured(structured_docs)
        entity_meta = _extract_entity_metadata(extracted, filename)
        _validate_extraction_fields(doc_classification, extracted, filename)

        # Resolve authoritative document domain
        document_domain = _resolve_authoritative_domain(
            domain_signals=deep_result.domain_signals if deep_result else {},
            doc_type=(understanding_result or {}).get("document_type", ""),
            structured_domain=doc_classification.get("domain", ""),
            doc_type_confidence=(understanding_result or {}).get("doc_type_confidence", 0.0),
        )
        _persist_document_domain(document_id, document_domain)

        payload_to_save = {
            "raw": extracted,
            "structured": structured_docs,
            "intelligence": intelligence_result,
            "document_classification": doc_classification,
            "entity_metadata": entity_meta,
            "understanding": understanding_result,
            "document_domain": document_domain,
            "deep_analysis": {
                "quality_grade": deep_result.quality_grade if deep_result else None,
                "quality_score": deep_result.quality_score if deep_result else None,
                "complexity_score": deep_result.complexity_score if deep_result else None,
                "domain_signals": deep_result.domain_signals if deep_result else None,
            } if deep_result else None,
        }
        save_info = save_extracted_pickle(document_id, payload_to_save)
        update_extraction_metadata(document_id, subscription_id, save_info.get("path"), save_info.get("sha256"))
    except Exception as exc:  # noqa: BLE001
        _set_document_status(document_id, STATUS_EXTRACTION_FAILED, f"Failed to persist extracted content: {exc}")
        emit_status_log(document_id, "extraction", "persistence_failed", f"Failed to persist extracted content: {exc}")
        clear_pipeline_profile()
        return {"document_id": document_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

    extra_fields = {
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "extracted_pickle_path": save_info.get("path"),
        "extracted_hash": save_info.get("sha256"),
    }
    emit_status_log(document_id, "extraction", "data_persisted", "Extraction data persisted to storage",
                    extra={"pickle_path": save_info.get("path")})

    _set_document_status(document_id, STATUS_EXTRACTION_COMPLETED, extra_fields=extra_fields)
    update_stage(document_id, "extraction", {"status": "COMPLETED", "completed_at": time.time(), "error": None})

    # KG ingestion (async, non-blocking — run in daemon thread to avoid stalling extraction)
    emit_status_log(document_id, "extraction", "kg_ingestion_started", "Knowledge graph ingestion triggered (async)")
    _kg_async = getattr(getattr(Config, "DocumentProcessing", None), "KG_INGEST_ASYNC", True)
    if _kg_async:
        threading.Thread(
            target=_ingest_to_knowledge_graph,
            kwargs=dict(
                document_id=document_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
                source_name=filename,
                payload_to_save=payload_to_save,
                deep_result=deep_result,
            ),
            daemon=True,
        ).start()
    else:
        _ingest_to_knowledge_graph(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            source_name=filename,
            payload_to_save=payload_to_save,
            deep_result=deep_result,
        )

    # HITL: No auto-embedding. Document stays at EXTRACTION_COMPLETED.
    # User must manually trigger screening then embedding.
    _mark_intelligence_ready(document_id)
    emit_status_log(document_id, "extraction", "extraction_completed",
                    "Extraction pipeline completed. Awaiting screening and embedding.",
                    extra={"document_domain": document_domain,
                           "status": STATUS_EXTRACTION_COMPLETED})
    logger.info("Extraction completed for %s; awaiting manual screening/embedding", document_id)

    # Refresh profile-level domain tag
    try:
        from src.profiles.profile_domain_tagger import refresh_profile_domain_on_document_change
        if subscription_id and profile_id:
            refresh_profile_domain_on_document_change(subscription_id, profile_id)
    except Exception as exc:
        logger.debug("Failed to refresh profile domain tag after upload extraction", exc_info=True)

    summary = _build_extraction_summary(extracted)
    clear_pipeline_profile()
    return {
        "document_id": document_id,
        "status": STATUS_EXTRACTION_COMPLETED,
        "pickle_path": save_info.get("path"),
        "summary": summary,
        "doc_name": filename,
        "intelligence": intelligence_result is not None,
    }

