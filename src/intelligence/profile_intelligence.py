"""Profile Intelligence — auto-generated document insights per profile.

Triggers after each document completes embedding. Generates:
1. Document brief (key facts, entities, insights)
2. Cross-document analysis (comparisons, trends, anomalies)
3. Profile overview (summary, metrics, overall insights)
"""

import json
import threading
import time
import logging
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_ANALYST_PREAMBLE = (
    "You are a senior analyst providing actionable intelligence. "
    "Go beyond stating facts — explain significance, flag risks, recommend actions. "
    "Every insight must answer: 'So what? Why does this matter to the user?'"
)

_BRIEF_PROMPT = """{preamble}

Analyze this {doc_type} document:
Name: {doc_name}
Content:
{content}

Return valid JSON only:
{{
  "brief": "2-3 sentence summary explaining what matters and why",
  "key_facts": [{{"label": "...", "value": "..."}}],
  "entities": ["important entities"],
  "insights": ["actionable observations — each answers 'so what?'"]
}}

Rules:
- Every fact must come from the document text. Never fabricate.
- Quantify findings: percentages, comparisons, totals.
- Flag anomalies or items needing attention.
"""

_CROSS_DOC_PROMPT = """{preamble}

Analyze {n} documents in this profile.
Dominant type: {dominant_domain}

Document briefs:
{formatted_briefs}

Provide cross-document analysis as valid JSON only:
{{
  "summary": "What the comparison reveals — key narrative",
  "comparisons": [{{"aspect": "...", "findings": "..."}}],
  "trends": ["patterns with quantified evidence"],
  "anomalies": ["outliers with why they matter"],
  "rankings": [{{"category": "...", "ranked": ["doc1", "doc2"], "reason": "..."}}]
}}

Rules:
- Only compare fields present in ALL documents.
- Quantify differences: percentages, deltas, ratios.
- Explain why each finding matters to the user.
"""

_OVERVIEW_PROMPT = """{preamble}

Summarize this document profile containing {n} documents.
Document types: {doc_types}

Document briefs:
{formatted_briefs}

Provide a profile overview as valid JSON only:
{{
  "summary": "What this collection is about and its significance — help a new user understand in 10 seconds",
  "key_metrics": {{"metric_name": "value"}},
  "overall_insights": ["high-level strategic intelligence — not repetition of individual facts"]
}}
"""


# ---------------------------------------------------------------------------
# Core Generation
# ---------------------------------------------------------------------------

def generate_profile_intelligence(
    document_id: str,
    profile_id: str,
    subscription_id: str,
) -> None:
    """Generate/update profile intelligence after a document completes embedding.

    Called as a background thread from embedding_service.
    """
    try:
        logger.info("Profile intelligence: starting for doc=%s profile=%s", document_id, profile_id)
        start = time.time()

        # 1. Load document extraction data
        doc_record, extraction_data = _load_document_data(document_id)
        if not extraction_data:
            logger.warning("Profile intelligence: no extraction data for doc=%s", document_id)
            return

        doc_name = doc_record.get("name") or doc_record.get("source_file") or "document"
        doc_type = doc_record.get("doc_domain") or doc_record.get("doc_type") or doc_record.get("type") or "document"

        # 2. Generate document brief
        content = _extract_text(extraction_data)
        brief = _generate_document_brief(document_id, doc_name, doc_type, content)
        if not brief:
            logger.warning("Profile intelligence: brief generation failed for doc=%s", document_id)
            return

        brief["document_id"] = document_id
        brief["name"] = doc_name
        brief["doc_type"] = doc_type
        brief["generated_at"] = time.time()

        # 3. Upsert document brief into profile_intelligence
        report = _upsert_document_brief(profile_id, subscription_id, brief)

        # 4. Load all briefs for cross-document analysis
        all_briefs = report.get("document_briefs", [])
        doc_count = len(all_briefs)

        # 5. Generate cross-document analysis (2+ docs)
        cross_doc = None
        if doc_count >= 2:
            cross_doc = _generate_cross_document_analysis(all_briefs, doc_type)

        # 6. Generate profile overview
        overview = _generate_profile_overview(all_briefs)

        # 7. Update full report
        _update_report(profile_id, subscription_id, overview, cross_doc, document_id)

        elapsed = round(time.time() - start, 1)
        logger.info(
            "Profile intelligence: completed for doc=%s profile=%s docs=%d elapsed=%.1fs",
            document_id, profile_id, doc_count, elapsed,
        )

    except Exception:
        logger.exception("Profile intelligence: failed for doc=%s profile=%s", document_id, profile_id)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _load_document_data(document_id: str):
    """Load document record and content from best available source."""
    from src.api.document_status import get_document_record
    record = get_document_record(document_id) or {}

    # Try pickle first (available if embedding just completed)
    try:
        from src.api.content_store import load_extracted_pickle
        extraction = load_extracted_pickle(document_id)
        return record, extraction
    except Exception:
        pass

    # Pickle purged — build content from MongoDB record + Qdrant chunks
    extraction = {"from_fallback": True}

    # Get document summary and entities from MongoDB record
    summary = record.get("document_summary", "")
    entities = record.get("key_entities", [])
    doc_domain = record.get("document_domain") or record.get("doc_domain", "")

    # Get actual text from Qdrant chunks
    chunks_text = _load_chunks_from_qdrant(
        document_id,
        str(record.get("subscription_id") or record.get("subscription") or ""),
        str(record.get("profile_id") or record.get("profile") or ""),
    )

    # Get facts from doc_facts collection
    try:
        from pymongo import MongoClient
        from src.api.config import Config
        client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
        db = client[Config.MongoDB.DB]
        facts_doc = db["doc_facts"].find_one({"document_id": document_id})
        facts = facts_doc.get("facts_json", []) if facts_doc else []
    except Exception:
        facts = []

    extraction["chunks_text"] = chunks_text
    extraction["summary"] = summary
    extraction["entities"] = entities
    extraction["facts"] = facts
    extraction["doc_domain"] = doc_domain

    return record, extraction


def _load_chunks_from_qdrant(document_id: str, subscription_id: str, profile_id: str) -> str:
    """Load canonical text from Qdrant chunks for a document."""
    try:
        from qdrant_client import QdrantClient
        from src.api.config import Config
        qc = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)

        filters = {"must": [{"key": "document_id", "match": {"value": document_id}}]}
        if profile_id:
            filters["must"].append({"key": "profile_id", "match": {"value": profile_id}})

        points, _ = qc.scroll(
            collection_name=subscription_id,
            scroll_filter=filters,
            limit=50,
            with_payload=True,
        )

        texts = []
        for p in sorted(points, key=lambda x: x.payload.get("chunk_index", 0)):
            text = (p.payload.get("canonical_text")
                    or p.payload.get("embedding_text")
                    or p.payload.get("text")
                    or "")
            if text.strip():
                texts.append(text.strip())

        return "\n\n".join(texts)
    except Exception as e:
        logger.debug("Qdrant chunk load failed for %s: %s", document_id, e)
        return ""


def _extract_text(extraction_data: dict) -> str:
    """Get clean text from extraction data — best source wins."""
    if not extraction_data:
        return ""

    # From pickle (raw field)
    raw = extraction_data.get("raw")
    if raw:
        if isinstance(raw, dict):
            text = raw.get("full_text") or raw.get("text") or raw.get("content") or ""
            if text:
                return text
        if hasattr(raw, "full_text") and raw.full_text:
            return raw.full_text
        if isinstance(raw, str) and raw.strip():
            return raw

    # From Qdrant chunks (fallback after pickle purge)
    chunks = extraction_data.get("chunks_text", "")
    if chunks and len(chunks) > 50:
        # Enrich with summary and entities if available
        parts = []
        summary = extraction_data.get("summary", "")
        if summary:
            parts.append(f"Document Summary: {summary}")
        entities = extraction_data.get("entities", [])
        if entities:
            parts.append(f"Key Entities: {', '.join(str(e) for e in entities[:20])}")
        parts.append(chunks)
        return "\n\n".join(parts)

    # From facts store
    facts = extraction_data.get("facts", [])
    if facts:
        return "\n".join(
            f.get("statement", "") + " " + f.get("evidence", "")
            for f in facts if isinstance(f, dict)
        )

    return ""


# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------

def _get_llm():
    """Get local LLM client for document processing."""
    from src.llm.clients import get_local_client
    return get_local_client()


def _llm_generate_json(prompt: str, max_tokens: int = 2048) -> Optional[dict]:
    """Call LLM and parse JSON response."""
    try:
        llm = _get_llm()
        raw, _meta = llm.generate_with_metadata(
            prompt,
            options={"temperature": 0.1, "num_predict": max_tokens},
        )

        # Parse JSON
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        return None

    except Exception as e:
        logger.warning("LLM JSON generation failed: %s", e)
        return None


def _generate_document_brief(document_id: str, doc_name: str, doc_type: str, content: str) -> Optional[dict]:
    """Generate intelligent brief for a single document."""
    if not content or len(content.strip()) < 50:
        return {"brief": "Document has insufficient content for analysis.", "key_facts": [], "entities": [], "insights": []}

    # Truncate to fit context window
    truncated = content[:6000]

    prompt = _BRIEF_PROMPT.format(
        preamble=_ANALYST_PREAMBLE,
        doc_type=doc_type,
        doc_name=doc_name,
        content=truncated,
    )
    return _llm_generate_json(prompt, max_tokens=2048)


def _generate_cross_document_analysis(briefs: list, dominant_domain: str) -> Optional[dict]:
    """Generate comparative analysis across all documents."""
    formatted = _format_briefs(briefs)
    prompt = _CROSS_DOC_PROMPT.format(
        preamble=_ANALYST_PREAMBLE,
        n=len(briefs),
        dominant_domain=dominant_domain,
        formatted_briefs=formatted,
    )
    return _llm_generate_json(prompt, max_tokens=3072)


def _generate_profile_overview(briefs: list) -> Optional[dict]:
    """Generate high-level profile overview."""
    doc_types = list(set(b.get("doc_type", "document") for b in briefs))
    formatted = _format_briefs(briefs)
    prompt = _OVERVIEW_PROMPT.format(
        preamble=_ANALYST_PREAMBLE,
        n=len(briefs),
        doc_types=", ".join(doc_types),
        formatted_briefs=formatted,
    )
    return _llm_generate_json(prompt, max_tokens=2048)


def _format_briefs(briefs: list) -> str:
    """Format document briefs for inclusion in prompts."""
    parts = []
    for b in briefs:
        name = b.get("name", "Unknown")
        brief_text = b.get("brief", "")
        facts = b.get("key_facts", [])
        facts_str = ", ".join(f"{f['label']}: {f['value']}" for f in facts if isinstance(f, dict))
        parts.append(f"- **{name}**: {brief_text} Key facts: {facts_str}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MongoDB Operations
# ---------------------------------------------------------------------------

def _get_collection():
    """Get the profile_intelligence MongoDB collection."""
    from pymongo import MongoClient
    from src.api.config import Config
    client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
    db = client[Config.MongoDB.DB]
    return db["profile_intelligence"]


def _upsert_document_brief(profile_id: str, subscription_id: str, brief: dict) -> dict:
    """Add or update a document brief in the profile intelligence report."""
    collection = _get_collection()

    # Atomic: find existing report or create new
    report = collection.find_one({"profile_id": profile_id, "subscription_id": subscription_id})

    if not report:
        report = {
            "profile_id": profile_id,
            "subscription_id": subscription_id,
            "profile_overview": None,
            "document_briefs": [],
            "cross_document_analysis": None,
            "generated_at": time.time(),
            "version": 1,
        }

    # Replace or add document brief
    briefs = report.get("document_briefs", [])
    briefs = [b for b in briefs if b.get("document_id") != brief["document_id"]]
    briefs.append(brief)
    report["document_briefs"] = briefs
    report["last_document_id"] = brief["document_id"]
    report["generated_at"] = time.time()

    collection.update_one(
        {"profile_id": profile_id, "subscription_id": subscription_id},
        {"$set": report},
        upsert=True,
    )
    return report


def _update_report(profile_id: str, subscription_id: str, overview: Optional[dict], cross_doc: Optional[dict], last_doc_id: str):
    """Update profile overview and cross-document analysis."""
    updates = {
        "last_document_id": last_doc_id,
        "generated_at": time.time(),
    }
    if overview:
        updates["profile_overview"] = overview
    if cross_doc:
        updates["cross_document_analysis"] = cross_doc

    collection = _get_collection()
    collection.update_one(
        {"profile_id": profile_id, "subscription_id": subscription_id},
        {"$set": updates},
    )
