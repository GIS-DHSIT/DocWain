# Profile Intelligence — Auto-Generated Document Insights

**Date:** 2026-03-16
**Status:** Approved
**Goal:** Automatically generate intelligent, actionable insights at the profile level as each document completes processing.

---

## Overview

When a document completes embedding, DocWain automatically analyzes its content and generates/updates a profile-level intelligence report. The report contains three sections:

1. **Profile Overview** — high-level analysis of the entire collection
2. **Per-Document Briefs** — each document's key facts, entities, and insights
3. **Cross-Document Analysis** — comparisons, trends, rankings, anomalies

The analysis is LLM-driven (not template-based), adapting to any document type. Intelligence quality targets actionable insights, not data summaries.

---

## Report Structure (MongoDB `profile_intelligence`)

```javascript
{
  profile_id: "prof_uuid",
  subscription_id: "sub_uuid",

  profile_overview: {
    summary: "High-level narrative about this profile",
    doc_count: 4,
    doc_types: ["invoice"],
    dominant_domain: "invoice",
    key_metrics: {"total_value": "$35,425.00", "vendors": 1},
    overall_insights: ["actionable insight 1", "actionable insight 2"]
  },

  document_briefs: [
    {
      document_id: "doc_uuid",
      name: "filename.pdf",
      doc_type: "invoice",
      brief: "2-3 sentence intelligent summary",
      key_facts: [{"label": "Total", "value": "$9,000.00"}],
      entities: ["Super Widget Industries", "Jessica Jones"],
      insights: ["notable observations"],
      confidence: "high",
      generated_at: ISODate
    }
  ],

  cross_document_analysis: {
    summary: "Comparative narrative",
    comparisons: [{"aspect": "...", "findings": "..."}],
    trends: ["patterns across documents"],
    anomalies: ["outliers or inconsistencies"],
    rankings: [{"category": "...", "ranked": ["doc1", "doc2"]}]
  },

  generated_at: ISODate,
  last_document_id: "doc_uuid",
  version: 1
}
```

---

## Generation Flow

```
Document embedding completes (TRAINING_COMPLETED)
        │
        ▼
Background thread: generate_profile_intelligence()
        │
        ├─ 1. Load extraction data from pickle (Azure Blob)
        ├─ 2. Generate document brief via LLM (qwen3:14b local)
        ├─ 3. Upsert document brief into profile_intelligence
        ├─ 4. Load ALL existing briefs for the profile
        ├─ 5. If 2+ docs: generate cross-document analysis via LLM
        ├─ 6. Generate/update profile overview via LLM
        └─ 7. Upsert full report to MongoDB
```

**Trigger point:** Inside embedding_service.py, after TRAINING_COMPLETED status is set. Fire-and-forget background thread — never blocks embedding.

**Concurrency:** MongoDB atomic lock prevents simultaneous cross-document regeneration. Each worker generates its own document brief independently.

---

## LLM Prompts

### Prompt 1 — Document Brief
```
You are a senior analyst providing actionable intelligence. Go beyond stating
facts — explain significance, flag risks, and recommend actions.

Analyze this {doc_type} document:
Name: {doc_name}
Content: {clean_text}

Return JSON:
{
  "brief": "2-3 sentence summary explaining what matters and why",
  "key_facts": [{"label": "...", "value": "..."}],
  "entities": ["important entities"],
  "insights": ["actionable observations — answer 'so what?' for each"]
}

Rules:
- Every fact must come from the document text. Never fabricate.
- Quantify findings: percentages, comparisons, totals.
- Flag anomalies or items that may need attention.
- Use domain-appropriate analysis lens.
```

### Prompt 2 — Cross-Document Analysis (2+ docs)
```
You are a senior analyst comparing {n} {dominant_domain} documents.

Document briefs:
{formatted_briefs}

Provide cross-document analysis as JSON:
{
  "summary": "What the comparison reveals — key narrative",
  "comparisons": [{"aspect": "...", "findings": "..."}],
  "trends": ["patterns with quantified evidence"],
  "anomalies": ["outliers with why they matter"],
  "rankings": [{"category": "...", "ranked": ["doc_name1", "doc_name2"], "reason": "..."}]
}

Rules:
- Only compare fields that exist in ALL documents.
- Quantify differences: percentages, deltas, ratios.
- Explain why each finding matters to the user.
- Flag items that may need action or review.
```

### Prompt 3 — Profile Overview
```
You are a senior analyst summarizing a document collection.

This profile contains {n} documents of type: {doc_types}.
Document briefs:
{formatted_briefs}

Provide a profile overview as JSON:
{
  "summary": "What this collection is about and its overall significance",
  "key_metrics": {"metric_name": "value"},
  "overall_insights": ["high-level actionable intelligence"]
}

Rules:
- The summary should help a new user understand this profile in 10 seconds.
- Key metrics should be the numbers that matter most for this domain.
- Insights should be strategic, not repetitive of individual document facts.
```

---

## Accuracy Safeguards

1. **Evidence-only** — LLM receives actual extraction text, not summaries of summaries
2. **Fact-checking** — cross-check LLM key_facts against doc_facts in MongoDB
3. **No hallucinated comparisons** — only compare fields present in ALL documents
4. **Confidence tagging** — high (direct text), medium (inferred), low (pattern-based)
5. **Source tracing** — every fact links to document_id for UI source navigation

---

## API Endpoint

```
GET /api/profiles/{profile_id}/intelligence
```

Returns the full profile_intelligence document. The UI renders it as a dashboard.

---

## Integration

- **Trigger:** background thread after TRAINING_COMPLETED in embedding_service
- **LLM:** qwen3:14b local (accuracy over speed, post-embedding timing)
- **Storage:** MongoDB `profile_intelligence` collection
- **Pipeline impact:** NONE — fire-and-forget, no changes to extract/screen/embed flow
- **Failure handling:** if generation fails, embedding status is unaffected. Report updates on next successful document.

---

## Intelligence Quality Standard

| Avoid (shallow) | Target (intelligent) |
|---|---|
| "Total is $9,000" | "Total of **$9,000** is 16.7% higher than Aug invoice, driven by 3x coffee bean quantity increase" |
| "4 invoices uploaded" | "4 invoices from single vendor show escalating costs — each revision increases balance due" |
| "Candidate has 5 years" | "Strongest Python candidate (5 yrs), but no cloud exposure — may need AWS onboarding" |
| "Contract expires Dec" | "Contract auto-renews in 90 days — action required by Nov 1 to renegotiate" |

Every insight answers: **"So what? Why does this matter to the user?"**
