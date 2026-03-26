# Document Intelligence Overhaul — Design Document

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Fix hallucinated/incomplete responses by overhauling extraction fidelity and retrieval coverage. Two changes: richer full-section chunks that preserve complete document content, and document-level LLM intelligence extraction that gives the LLM awareness of all documents in a profile.

## Problem

Despite multi-stage document intelligence pipeline, responses are hallucinated and missing key details:

1. **Chunks are too small** — Average 223 characters per chunk. Entire contracts reduced to one-sentence summaries. The LLM sees fragments, not content.
2. **Retrieval misses most documents** — Top-k limits (3-8 evidence chunks) mean only 6-10 of 33 documents are represented. 23+ documents are invisible to the LLM.
3. **LLM doesn't know what documents exist** — No document inventory. When asked "list all suppliers across 33 documents", the LLM fabricates because it only sees 3-8 chunks from a handful of docs.

## Architecture

```
INGESTION (enhanced):
  Document Upload → Existing Extraction (unchanged)
                          ↓
                    Full-Section Chunking (replaces current 223-char chunks)
                          ↓
                    Document-Level LLM Intelligence Extraction
                          ↓
                    Qdrant Storage:
                      ├─ 1 "doc_index" point (compact one-liner, ~50 tokens)
                      ├─ 1 "doc_intelligence" point (full structured summary)
                      └─ N "section" points (full-section text, one per section)

QUERY (enhanced):
  User Query → UNDERSTAND (intent analysis)
                    ↓
              Retrieve:
                ├─ ALWAYS inject doc_index points (compact, all docs)
                ├─ For broad queries: also inject doc_intelligence points
                └─ Vector search for relevant section chunks
                    ↓
              Rerank + Build Evidence
                    ↓
              REASON (LLM generates answer with full context)
```

## Full-Section Chunking

Replaces current chunking that produces 223-char fragments. Each document section becomes one chunk preserving complete text.

### For digital PDFs (text-selectable)

```
PyMuPDF extraction → Section detection (heading patterns, font size changes)
    ↓
Section 1: "1. SCOPE OF SUPPLY" → full text (500-3000 chars)
Section 2: "2. PAYMENT TERMS" → full text (200-1500 chars)
Section 3: "3. SERVICE LEVEL AGREEMENT" → full text (300-2000 chars)
```

### For scanned PDFs (OCR-dependent)

```
Page images → DiT layout detection (Visual Intelligence Layer)
    ↓
Layout regions grouped into logical sections:
  - DiT "title" regions mark section boundaries
  - Adjacent "text" regions merged into the section above
  - "table" regions preserved intact within their section
    ↓
Per-section OCR:
  - Tesseract/EasyOCR (primary)
  - TrOCR enhancement for low-confidence regions
    ↓
Same output: Section N → full text
```

### Size guardrails

- Minimum section size: 100 characters (below this, merge with adjacent section)
- Maximum section size: 4000 characters (~1000 tokens). If exceeded, split at paragraph boundaries with 2-sentence overlap
- Sections without detectable headings: split by page boundaries

### Qdrant payload changes

- `resolution: "section"` (was `"chunk"`)
- `chunk_kind: "section_full"` (new, distinguishes from old fragments)
- `canonical_text`: complete section text (was 223-char summary)
- All other payload fields unchanged

## Document-Level LLM Intelligence Extraction

One LLM call per document during ingestion. Extracts structured metadata dynamically — no predefined schema. Works for any document type.

### Universal extraction prompt

```
System: You are a document analyst. Extract ALL structured information
from this document. Output valid JSON only.

User:
--- DOCUMENT ---
{full document text}
--- END DOCUMENT ---

Extract everything that is factually present in this document:

1. document_type: what kind of document this is (e.g., contract, invoice,
   resume, report, policy, statement, presentation, letter, form, other)
2. parties: all named people, organizations, or entities with their roles
   as described in the document
3. key_dates: all dates mentioned with their context
4. key_values: all monetary amounts, quantities, percentages, durations,
   limits, scores, or measurable values with context
5. key_sections: list each section/heading with a one-line summary of
   what it contains
6. key_facts: important statements, terms, conditions, findings, or
   conclusions — whatever the document asserts
7. one_line_summary: single sentence describing this document

Rules:
- Extract ONLY what is explicitly stated. Never infer or fabricate.
- Adapt to the document — a resume has different structure than an invoice.
- If a field has no data in the document, omit it entirely.
- Return valid JSON only.
```

### Qdrant storage — two dedicated points per document

**Point 1 — doc_index (compact, ~50 tokens):**
```json
{
  "resolution": "doc_index",
  "canonical_text": "US_Healthcare_Contract_5.pdf | Healthcare Equipment Agreement | Leica Biosystems Inc. (Supplier) ↔ Prime Diagnostics LLC (Customer) | Jan 2026 | Net 30 days | 100% liability cap",
  "source_name": "US_Healthcare_Contract_5.pdf",
  "document_id": "...",
  "profile_id": "...",
  "doc_intelligence": { ... full JSON ... }
}
```

**Point 2 — doc_intelligence (full structured summary, ~200-500 tokens):**
```json
{
  "resolution": "doc_intelligence",
  "canonical_text": "Healthcare Equipment Agreement between Leica Biosystems Inc. (Supplier) and Prime Diagnostics LLC (Customer). Effective January 1, 2026. Scope: diagnostic medical equipment, installation, maintenance. Payment: Net 30 days. SLA: 90% uptime. Liability: 100% of contract value...",
  "doc_intelligence": {
    "document_type": "healthcare_equipment_agreement",
    "parties": [...],
    "key_dates": [...],
    "key_values": [...],
    "key_sections": [...],
    "key_facts": [...],
    "one_line_summary": "..."
  }
}
```

### Processing time

- Current: ~25.6s × N sections per doc (KnowledgeExtractor per-section calls)
- New: ~15-20s × 1 call per doc (single document-level extraction)
- Net: faster than current pipeline for most documents

## Enhanced Retrieval — Document Index Injection

### Step 1: Always fetch document index (every query)

```python
doc_index_points = qdrant.scroll(
    filter=Filter(must=[
        FieldCondition(key="profile_id", match=profile_id),
        FieldCondition(key="resolution", match="doc_index"),
    ]),
    limit=200,
)
```

Token budget: 33 docs × 50 tokens = 1,650 tokens. Under 5% of context.

### Step 2: For broad queries, also fetch full intelligence

```python
BROAD_TASK_TYPES = {"summarize", "overview", "list", "aggregate", "compare"}

if task_type in BROAD_TASK_TYPES:
    intel_points = qdrant.scroll(
        filter=Filter(must=[
            FieldCondition(key="profile_id", match=profile_id),
            FieldCondition(key="resolution", match="doc_intelligence"),
        ]),
        limit=200,
    )
```

### Step 3: Enhanced vector search

- Increase default top_k from 30 → 50
- Increase evidence_top_k per task type:
  - lookup: 4 → 8
  - extract: 6 → 12
  - list: 8 → 20
  - summarize: 8 → 20
  - overview: 10 → 25
- Filter out doc_index/doc_intelligence from vector search (fetched separately)

### LLM prompt structure

```
--- DOCUMENT INDEX (33 documents in this profile) ---
1. US_Healthcare_Contract_5.pdf | Healthcare Equipment Agreement | Leica Biosystems ↔ Prime Diagnostics | Jan 2026
2. stellar_retail_invoices.pdf | Invoice Bundle | Stellar Retail | 12 invoices | $45,230 total
3. ...
--- END DOCUMENT INDEX ---

--- DOCUMENT INTELLIGENCE (for relevant documents) ---
[Full structured summaries — broad queries only]
--- END DOCUMENT INTELLIGENCE ---

--- EVIDENCE ---
[Full-section chunks from vector search]
--- END EVIDENCE ---

QUESTION: {user query}
```

## Integration Points

### Ingestion pipeline (3 files)

| File | Change |
|------|--------|
| `src/extraction/document_intelligence.py` | **NEW** — Universal LLM extraction, produces doc_index + doc_intelligence |
| `src/embedding/chunking/section_chunker.py` | **MODIFY** — Full-section preservation, 4000-char max, 100-char min merge |
| `src/api/embedding_service.py` | **MODIFY** — Upsert doc_index + doc_intelligence points after section chunks |

### Query pipeline (3 files)

| File | Change |
|------|--------|
| `src/agent/core_agent.py` | **MODIFY** — Fetch doc_index always, doc_intelligence for broad queries, increased evidence_top_k |
| `src/generation/prompts.py` | **MODIFY** — Add DOCUMENT INDEX and DOCUMENT INTELLIGENCE blocks to reason prompt |
| `src/retrieval/retriever.py` | **MODIFY** — Increase top_k to 50, filter out doc_index/doc_intelligence from vector search |

### OCR path (already built)

```
src/visual_intelligence/  [EXISTING - Phase 1-3]
  - DiT provides section boundaries for scanned PDFs
  - TrOCR improves text quality for low-confidence regions
  - Feed into same full-section chunking pipeline
```

**Total: 1 new file, 5 modified files.** No new storage, no MongoDB changes, no new dependencies.

## Re-ingestion Strategy

Existing documents re-processed to get new format:

```
For each document in a profile:
  1. Load extraction pickle from Azure Blob (already exists)
  2. Re-chunk using full-section strategy
  3. Run document-level LLM intelligence extraction
  4. Delete old Qdrant points for this document_id
  5. Upsert new points:
     - N section chunks (resolution="section")
     - 1 doc_index point (resolution="doc_index")
     - 1 doc_intelligence point (resolution="doc_intelligence")
  6. Update document status
```

**Triggered via:** Existing `/api/documents/embed` endpoint with `force_reprocess=true` flag.

**Backward compatibility:**
- Old chunks: `resolution: "chunk"`, `chunk_kind: "section_summary"/"section_text"`
- New chunks: `resolution: "section"`, `chunk_kind: "section_full"`
- Retrieval works with both during migration — no big-bang cutover
- doc_index/doc_intelligence points simply don't exist for old docs — retrieval gracefully handles empty results

## Constraints

- No customer data stored in MongoDB — all structured metadata lives in Qdrant payloads
- No predefined document type schemas — LLM dynamically extracts what's relevant
- Best-effort extraction — full-section chunks are the ground truth safety net
- Works for all document types: contracts, invoices, resumes, reports, policies, presentations, forms
