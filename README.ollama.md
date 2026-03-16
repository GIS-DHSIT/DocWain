# DocWain — Intelligent Document Analysis Model

[![Ollama](https://img.shields.io/badge/Ollama-MuthuSubramanian%2FDocWain-blue)](https://ollama.com/MuthuSubramanian/DocWain)

## Overview

DocWain is a fine-tuned language model purpose-built for enterprise document intelligence. It processes any document type — invoices, contracts, policies, reports, specifications, research papers — and delivers structured, evidence-grounded insights.

## Quick Start

```bash
ollama pull MuthuSubramanian/DocWain
ollama run MuthuSubramanian/DocWain
```

## What Makes DocWain Different

| Capability | Description |
|-----------|-------------|
| **Evidence Grounding** | Every claim traces to the source document. No hallucination. |
| **Structured Output** | Responses use proper markdown: headers, tables, bold values, bullets |
| **Cross-Document Analysis** | Finds patterns and relationships across multiple documents |
| **Domain Adaptive** | Works with any document type — adapts analysis to the domain |
| **Visible Reasoning** | Shows its logic: "Since X states Y, this means Z" |

## Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen3 8B |
| Quantization | Q4_K_M |
| Context Window | 16,384 tokens |
| Max Output | 8,192 tokens |
| Size | ~5.2 GB |
| Thinking Mode | Native support |
| JSON Output | Yes |

## Response Style

DocWain produces human-like structured responses:

```
Invoice 0522 from Super Widget Industries totals **$9,000.00**.

## Vendor Details
- **Vendor:** Super Widget Industries
- **Address:** 123 Mill St., Main, AK 213546
- **Billed To:** Jessica Jones (jessicajones@defenders.com)

## Line Items

| Service | Description | Amount |
|---------|-------------|--------|
| Website Design | New design, 5 mockups | **$720.00** |
| Kitchen Construction | Marble Countertops | **$3,000.00** |
| Computer Repair | Batcave Super Computer | **$400.00** |
```

## Use Cases

- **Document Q&A** — Ask questions, get precise answers from your documents
- **Data Extraction** — Pull structured data from invoices, contracts, forms
- **Compliance Analysis** — Check documents against regulatory requirements
- **Cross-Document Comparison** — Compare terms, values, and clauses across documents
- **Summarization** — Get structured overviews of document collections
- **Risk Assessment** — Identify anomalies, missing clauses, and compliance gaps

## Continuous Improvement

DocWain improves daily through an automated feedback loop:

1. **Every query** records confidence, grounding, and task type signals
2. **Daily evaluation** checks quality metrics per document profile
3. **Auto fine-tuning** triggers when quality drops below thresholds
4. **Updated model** is rebuilt and pushed to the registry

Training data sources:
- User query patterns and feedback corrections
- Document Q&A pairs generated during extraction (20+ per document)
- Entity relationships from knowledge graph
- Answerability index (32 query type taxonomy)

## API Integration

DocWain powers the [DocWain Platform](https://docwain.ai) — a full document intelligence system with:

- Multi-stage document processing pipeline (Extract → Screen → Embed)
- HITL (Human-in-the-Loop) quality gates
- Enterprise multi-tenancy (subscription + profile isolation)
- Qdrant vector search + Neo4j knowledge graph + MongoDB metadata

## Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| temperature | 0.3 | Balanced creativity and precision |
| top_p | 0.85 | Diverse but focused token selection |
| top_k | 40 | Vocabulary constraint |
| repeat_penalty | 1.1 | Prevents repetitive output |
| num_ctx | 16384 | Context window for long documents |
| num_predict | 8192 | Max response length |

## License

DocWain - Intelligent Document Analysis Platform
Copyright (c) 2026 DHS IT Solutions. All rights reserved.
