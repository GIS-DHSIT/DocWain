# DocWain V2 — Unified Vision + Tool-Calling Model Design

**Date:** 2026-03-22
**Status:** Approved
**Author:** Claude Code + Muthu

## Overview

Graft a vision encoder (SigLIP-SO400M) onto the existing DHS/DocWain V1 (Qwen3-14B), add native function-calling for 9 core document intelligence tools, and fine-tune on public datasets + distilled OCR data. The result is one unified model that can see documents, understand them deeply, and call tools natively.

## Architecture

```
DHS/DocWain:v2 (ONE MODEL)

  SigLIP-SO400M (400M, frozen)
       │
  Projection MLP (50M, trained)
       │
       ▼
  Qwen3-14B V1 backbone (frozen + LoRA)
  ├─ All existing V1 knowledge preserved
  ├─ Vision awareness via LoRA adapters
  └─ Native <tool_call> output format

Total: ~15.25B params | Trainable: ~94M | GGUF: ~9.5 GB Q4_K_M
```

## Versioning

```
DHS/DocWain:v1     ← current Qwen3-14B (preserved forever)
DHS/DocWain:v2     ← V1 + vision + tools + doc intelligence
DHS/DocWain:latest ← points to v2 after gate passes
```

## Core Document Intelligence Tools (9)

### Extraction (auto-invoked)
- **ocr_extract** — Vision-based text extraction from images/scans. Auto-invoked on image input.
- **layout_extract** — Document structure detection (headers, sections, tables, lists). Auto-invoked on new document access.
- **extract_table** — Structured table extraction to rows/columns. Auto-invoked when query references table data.
- **extract_entities** — NER for people, dates, amounts, clauses. Auto-invoked when query asks about specific data.

### Understanding (auto-invoked)
- **context_understand** — Deep document comprehension with confidence scores. Auto-invoked on EVERY query for grounding.
- **cross_reference** — Multi-section/document linking. Auto-invoked when query spans multiple sections.
- **search_documents** — Semantic search across corpus. Auto-invoked when context_understand lacks evidence.

### Action (model-decided)
- **summarize_section** — Targeted summarization at requested granularity.
- **visualize_data** — Chart/graph generation via VIZ directive.

## Function-Calling Format

```xml
<!-- Model outputs tool calls -->
<tool_call>
{"name": "extract_table", "arguments": {"page": 3}}
</tool_call>

<!-- Server returns results -->
<tool_response>
{"rows": [["Q1", "$1.2M"], ["Q2", "$1.5M"]], "cols": ["Quarter", "Revenue"]}
</tool_response>

<!-- Model uses results in response -->
Based on the financial table on page 3:
| Quarter | Revenue |
|---------|---------|
| Q1      | $1.2M   |
| Q2      | $1.5M   |
```

Parallel tool calls: multiple `<tool_call>` blocks in one response → server executes in parallel.

## Hybrid Tool Chaining

Model outputs a plan with parallel-safe tool calls batched together. Server executes batch, returns results. If follow-up needed, model plans next batch.

## Training Pipeline — 4 Phases

### Phase 1: Vision Grafting (Projection Training)
- **Duration:** ~2-3 hours on A100
- **Trainable:** Projection MLP only (~50M params)
- **Data:** LLaVA-Pretrain (558K image-caption pairs)
- **Task:** Image → caption (teaches projection to produce useful visual tokens)
- **Gate:** BLEU >= 30, CIDEr >= 80

### Phase 2: Document Intelligence Fine-Tuning
- **Duration:** ~4-6 hours on A100
- **Trainable:** Projection MLP + LoRA on Qwen3-14B

**Public datasets:**
| Dataset | Size | Purpose |
|---------|------|---------|
| DocVQA | 12K | Document visual QA |
| ChartQA | 32K | Chart understanding |
| InfographicsVQA | 5K | Infographic QA |
| PubTabNet | 516K (sampled) | Table structure |
| FinTabNet | 112K (sampled) | Financial tables |
| WikiTableQuestions | 22K | Table QA |
| DocLayNet | 80K | Layout annotations |
| PubLayNet | 360K (sampled) | Layout segments |
| MP-DocVQA | multi-page | Multi-page QA |
| DUDE | doc understanding | Extraction benchmark |

**glm-ocr distillation:**
- Run glm-ocr on 5K diverse document images
- Capture OCR outputs as training targets
- Create SFT pairs: image → OCR-quality text

**Training:** LoRA r=16, 2 epochs, mixed dataset (~50K curated)
**DPO:** Accurate extraction vs hallucinated, layout-aware vs ignoring
**Gate:** DocVQA >= 75%, Table F1 >= 80%, Layout mAP >= 70%

### Phase 3: Tool-Calling Specialization
- **Duration:** ~2-3 hours
- **Trainable:** LoRA (continued from phase 2)

**Data sources:**
| Source | Size | Purpose |
|--------|------|---------|
| Claude Code generated | 3500+ | DocWain-specific tool scenarios |
| ToolBench (filtered) | ~5K | General tool-use trajectories |
| Gorilla API Bench | ~1K | Function-calling benchmark |
| NexusRaven | ~2K | Function-calling fine-tune |
| Parallel planning | 500+ | Multi-tool batch calls |
| Auto-invocation | 500+ | Pre-filled context examples |
| No-tool-needed | 200+ | Prevents over-calling |

**DPO:** Right vs wrong tool, correct vs malformed args, parallel vs unnecessary sequential
**Gate:** Tool accuracy >= 85%, Arg correctness >= 90%, False positive <= 10%

### Phase 4: Merge, Test & Promote
1. Merge projection MLP + LoRA into Qwen3-14B
2. Export GGUF Q4_K_M (~9.5 GB)
3. V1 regression test (persona >= 90%, RAG accuracy >= V1, formatting >= V1)
4. New capabilities test (vision >= 75%, OCR >= 90%, tools >= 85%)
5. Promote: `DHS/DocWain:v2` → `DHS/DocWain:latest`
6. Evolving pipeline (`/finetune`) targets v2 going forward

## Auto-Invocation Rules

| Tool | Trigger | Priority |
|------|---------|----------|
| ocr_extract | Image input detected | Immediate |
| layout_extract | New document or first query on doc | Immediate |
| context_understand | Every query | Always |
| extract_table | Query references table data | High |
| extract_entities | Query asks about specific data points | High |
| cross_reference | Query spans multiple sections | Medium |
| search_documents | context_understand lacks evidence | Medium |
| summarize_section | Model decides | Model |
| visualize_data | Model decides | Model |

## V2 Modelfile

```
FROM gguf/DocWain-V2.Q4_K_M.gguf

PARAMETER temperature 0.3
PARAMETER num_ctx 16384
PARAMETER num_predict 8192

SYSTEM """
You are DocWain — Document Wise AI Node.
[...existing V1 persona preserved...]

You have vision capabilities. When given document images, you can directly
read text, understand layout, extract tables, and interpret visual elements.

You have access to tools. When a task requires action beyond your direct
knowledge, call the appropriate tool using <tool_call> format. You can call
multiple tools in parallel when they are independent.

Auto-available context:
- Document layout structure (from layout_extract)
- Relevant passages (from context_understand)
- OCR text for image inputs (from your vision encoder)
"""
```

## Disk Management

Clean intermediate artifacts between phases:
- Phase 1 complete → delete LLaVA-Pretrain raw data
- Phase 2 complete → delete raw dataset downloads, keep only processed JSONL
- Phase 3 complete → delete intermediate checkpoints, keep only final adapter
- Phase 4 complete → delete merged HF weights after GGUF export

## Integration with Evolving Pipeline

After V2 promotion, the existing `/finetune` pipeline (iteration 2+) targets V2:
- Observer sends vision prompts (document images) to V2
- Harvester collects tool-calling accuracy signals
- Teacher generates improved tool-call + vision examples
- Trainer fine-tunes V2's LoRA adapters
- Gate evaluates both vision and tool accuracy
