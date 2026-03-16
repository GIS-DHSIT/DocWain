# Profile Intelligence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-generate intelligent, actionable profile-level insights as each document completes embedding — profile overview, per-document briefs, and cross-document analysis.

**Architecture:** Background thread triggers after TRAINING_COMPLETED in embedding_service. Three focused LLM calls (qwen3:14b local) generate document brief, cross-document analysis, and profile overview. Stored in MongoDB `profile_intelligence` collection. Served via GET endpoint.

**Tech Stack:** FastAPI, MongoDB, Ollama local (qwen3:14b), threading

---

### Task 1: Create Profile Intelligence Module

**Files:**
- Create: `src/intelligence/profile_intelligence.py`

**Step 1: Create the module**

```python
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
    """Load document record and extraction pickle."""
    from src.api.document_status import get_document_record
    record = get_document_record(document_id) or {}

    try:
        from src.api.content_store import load_extracted_pickle
        extraction = load_extracted_pickle(document_id)
    except Exception:
        # Pickle may have been purged after embedding — use doc_facts instead
        from pymongo import MongoClient
        from src.api.config import Config
        client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
        db = client[Config.MongoDB.DB]
        facts = db["doc_facts"].find_one({"document_id": document_id})
        if facts:
            extraction = {"facts": facts.get("facts_json", []), "from_facts_store": True}
        else:
            extraction = None

    return record, extraction


def _extract_text(extraction_data: dict) -> str:
    """Get clean text from extraction data."""
    if not extraction_data:
        return ""

    # From pickle
    raw = extraction_data.get("raw")
    if raw:
        if isinstance(raw, dict):
            return raw.get("full_text") or raw.get("text") or raw.get("content") or ""
        if hasattr(raw, "full_text"):
            return raw.full_text or ""
        if isinstance(raw, str):
            return raw

    # From facts store fallback
    facts = extraction_data.get("facts", [])
    if facts:
        return "\n".join(
            f.get("statement", "") + " " + f.get("evidence", "")
            for f in facts if isinstance(f, dict)
        )

    return str(extraction_data)[:3000]


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
```

**Step 2: Verify import**

```bash
python -c "from src.intelligence.profile_intelligence import generate_profile_intelligence; print('OK')"
```

**Step 3: Commit**

```bash
git add src/intelligence/profile_intelligence.py
git commit -m "feat: profile intelligence module — LLM-driven document insights"
```

---

### Task 2: Wire Trigger into Embedding Service

**Files:**
- Modify: `src/api/embedding_service.py` (after TRAINING_COMPLETED lines ~2627 and ~3286)

**Step 1: Add trigger after TRAINING_COMPLETED**

Find every place where `_set_document_status(doc_id, STATUS_TRAINING_COMPLETED, ...)` is called. After each, add:

```python
# Trigger profile intelligence generation (background, non-blocking)
try:
    from src.intelligence.profile_intelligence import generate_profile_intelligence
    _sub = subscription_id
    _prof = profile_id
    _did = document_id  # or doc_id depending on scope
    threading.Thread(
        target=generate_profile_intelligence,
        args=(_did, _prof, _sub),
        daemon=True,
        name=f"profile-intel-{_did[:12]}",
    ).start()
except Exception:
    logger.debug("Profile intelligence trigger skipped", exc_info=True)
```

There are two TRAINING_COMPLETED trigger points:
1. `_process_blob()` around line 2627 — blob-based embedding path
2. `_process_local_document()` around line 3286 — local pickle path

Add the trigger after BOTH. The `subscription_id` and `profile_id` variables are already in scope at both locations.

**Step 2: Verify server starts**

```bash
python -c "from src.api.embedding_service import embed_documents; print('OK')"
```

**Step 3: Commit**

```bash
git add src/api/embedding_service.py
git commit -m "feat: trigger profile intelligence after embedding completes"
```

---

### Task 3: Create API Endpoint

**Files:**
- Create: `src/api/profile_intelligence_api.py`
- Modify: `src/main.py` (add router)

**Step 1: Create the endpoint**

```python
"""Profile Intelligence API — serves auto-generated document insights."""

from fastapi import APIRouter, HTTPException

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

profile_intelligence_router = APIRouter(prefix="/profiles", tags=["Profile Intelligence"])


@profile_intelligence_router.get("/{profile_id}/intelligence", summary="Get profile intelligence report")
async def get_profile_intelligence(profile_id: str):
    """Return the auto-generated intelligence report for a profile.

    Contains:
    - Profile overview (summary, key metrics, overall insights)
    - Per-document briefs (key facts, entities, insights per document)
    - Cross-document analysis (comparisons, trends, anomalies, rankings)
    """
    from pymongo import MongoClient
    from src.api.config import Config

    client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
    db = client[Config.MongoDB.DB]
    report = db["profile_intelligence"].find_one(
        {"profile_id": profile_id},
        {"_id": 0},
    )

    if not report:
        return {
            "profile_id": profile_id,
            "status": "pending",
            "message": "No intelligence report yet. Reports generate automatically after documents complete processing.",
            "profile_overview": None,
            "document_briefs": [],
            "cross_document_analysis": None,
        }

    return report
```

**Step 2: Register router in main.py**

Add at line ~181 (after model_router):
```python
from src.api.profile_intelligence_api import profile_intelligence_router
```

And in the router block:
```python
api_router.include_router(profile_intelligence_router)
```

**Step 3: Verify**

```bash
python -c "from src.api.profile_intelligence_api import profile_intelligence_router; print('Routes:', len(profile_intelligence_router.routes))"
```

**Step 4: Commit**

```bash
git add src/api/profile_intelligence_api.py src/main.py
git commit -m "feat: GET /api/profiles/{id}/intelligence endpoint"
```

---

### Task 4: Integration Test

**Files:**
- Create: `tests/test_profile_intelligence.py`

**Step 1: Write tests**

```python
"""Tests for profile intelligence module."""

import pytest


class TestProfileIntelligenceModule:
    def test_imports(self):
        from src.intelligence.profile_intelligence import generate_profile_intelligence
        from src.intelligence.profile_intelligence import _format_briefs

    def test_format_briefs(self):
        from src.intelligence.profile_intelligence import _format_briefs
        briefs = [
            {"name": "invoice1.pdf", "brief": "An invoice for $500", "key_facts": [{"label": "Total", "value": "$500"}]},
            {"name": "invoice2.pdf", "brief": "An invoice for $700", "key_facts": [{"label": "Total", "value": "$700"}]},
        ]
        result = _format_briefs(briefs)
        assert "invoice1.pdf" in result
        assert "invoice2.pdf" in result
        assert "$500" in result

    def test_extract_text_from_dict(self):
        from src.intelligence.profile_intelligence import _extract_text
        data = {"raw": {"full_text": "Hello world content"}}
        assert _extract_text(data) == "Hello world content"

    def test_extract_text_from_facts(self):
        from src.intelligence.profile_intelligence import _extract_text
        data = {"facts": [{"statement": "Fact one", "evidence": "Evidence one"}]}
        result = _extract_text(data)
        assert "Fact one" in result
        assert "Evidence one" in result

    def test_extract_text_empty(self):
        from src.intelligence.profile_intelligence import _extract_text
        assert _extract_text({}) == ""
        assert _extract_text(None) == ""


class TestProfileIntelligenceAPI:
    def test_router_imports(self):
        from src.api.profile_intelligence_api import profile_intelligence_router
        paths = [r.path for r in profile_intelligence_router.routes]
        assert "/{profile_id}/intelligence" in paths
```

**Step 2: Run tests**

```bash
pytest tests/test_profile_intelligence.py -v
```

**Step 3: Commit**

```bash
git add tests/test_profile_intelligence.py
git commit -m "test: profile intelligence module and API tests"
```
