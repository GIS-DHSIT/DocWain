# Visual Intelligence Layer — Design Document

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Add a second-pass visual intelligence layer to DocWain's document processing pipeline using best-in-class ML models for layout analysis, table extraction, OCR enhancement, and document understanding.

## Problem

DocWain's current document extraction relies on heuristic-based layout analysis (column width thresholds), traditional OCR (Tesseract + EasyOCR), and basic table extraction (CSV via pdfplumber). This fails on:

- Complex multi-column layouts, sidebars, captions
- Scanned/image-heavy PDFs with degraded text
- Tables with merged cells, nested headers, spanning rows
- Forms with checkboxes, multi-column fields, handwritten annotations
- Mixed-content documents (charts, diagrams, photos interspersed with text)

## Architecture

### Core Principle: Layered Enrichment

The existing extraction pipeline stays untouched as the "fast path." A new Visual Intelligence Layer runs as a second pass on original page images, enriching the existing `ExtractedDocument` with deeper structural understanding.

```
Existing Pipeline (unchanged):
  Upload → Extraction → Understanding → Intelligence → Chunking → Embedding

New Visual Intelligence Layer (second pass):
  Upload → Existing Extraction
                ↓
           Complexity Scorer (decides: skip / light / full)
                ↓
           ┌─────────────────────────────────┐
           │  Visual Intelligence Layer       │
           │  ├─ Page Renderer (PDF → images) │
           │  ├─ Layout Detector (DiT)        │
           │  ├─ Table Extractor (Table Transformer) │
           │  ├─ OCR Enhancer (TrOCR)         │
           │  └─ Doc Understanding (LayoutLMv3)│
           └─────────────────────────────────┘
                ↓
           Enrichment Merger (merges visual results into existing ExtractedDocument)
                ↓
           Understanding → Intelligence → Chunking → Embedding (continues as normal)
```

### KG Integration: Parallel Consumer

The Knowledge Graph is updated in parallel with the Understanding stage, not downstream:

```
Existing Pipeline → ExtractedDocument → Enrichment Merger
                                              │
                                    ┌─────────┼─────────┐
                                    ▼         ▼         ▼
                              Understanding  KG Writer  Chunking
                              (existing)     (parallel) (existing)
                                    │         │         │
                                    ▼         ▼         ▼
                              Intelligence   Neo4j    Embedding
```

Why parallel:
- Visual layer produces layout regions, table structures, and spatial key-value pairs — rich graph-ready entities that don't need Understanding to interpret first
- DiT region labels → KG nodes with spatial relationships
- Table Transformer cell grids → KG table nodes with row/column edges
- LayoutLMv3 key-value pairs → KG entity-attribute edges directly
- Non-blocking, non-fatal (same fire-and-forget pattern as existing KG writes)

## Complexity Scorer — The Adaptive Gate

Each page is assigned a processing tier based on existing extraction signals:

```
Tier 0 — SKIP (simple text-only pages)
  Signals: >95% text blocks, no images, no tables detected,
           single column, high OCR confidence (>85%)
  Action:  Use existing extraction as-is. Zero overhead.

Tier 1 — LIGHT (moderate complexity)
  Signals: Tables present but simple, some images,
           multi-column layout, OCR confidence 70-85%
  Action:  Run Layout Detector + Table Extractor only.
  Budget:  ~3-5 sec/page (CPU-friendly models)

Tier 2 — FULL (complex pages)
  Signals: Low OCR confidence (<70%), scanned/image-heavy,
           forms with checkboxes, merged table cells,
           charts/diagrams, handwritten content
  Action:  Run full visual pipeline (all 4 models).
  Budget:  ~10-20 sec/page (GPU models)
```

Scoring inputs (all available from existing extraction):
- OCR confidence scores (already tracked per image)
- Image-to-text ratio per page
- Table detection flags from pdfplumber
- Block type distribution from PyMuPDF
- Layout confidence from current heuristics

## Model Selection

### GPU Models (heavy, run on Tier 1/2 pages)

**1. DiT (Document Image Transformer) — Layout Detection**
- Model: `microsoft/dit-large-finetuned-publaynet` (~350MB)
- Why: Self-supervised pre-trained on 42M document images, outperforms LayoutParser on PubLayNet/DocLayNet
- Output: Bounding boxes with labels (text, title, table, figure, list, header, footer)
- VRAM: ~2GB

**2. Table Transformer (DETR-based) — Table Structure Recognition**
- Models: `microsoft/table-transformer-detection` + `microsoft/table-transformer-structure-recognition` (~220MB combined)
- Why: Purpose-built for table detection AND cell-level structure extraction
- Output: Table bounding boxes → cell grid with row/column spans → structured data
- VRAM: ~1.5GB combined

### CPU-Friendly Models

**3. TrOCR — Enhanced OCR**
- Models: `microsoft/trocr-base-printed` + `microsoft/trocr-base-handwritten` (~660MB combined)
- Why: Transformer-based, SOTA on both printed and handwritten text
- Usage: Only on regions where existing OCR confidence <70%
- Can run on CPU with ~3 sec/region

**4. LayoutLMv3 — Document Understanding Backbone**
- Model: `microsoft/layoutlmv3-base` (~500MB)
- Why: Best multi-modal model for KIE, form understanding, and classification simultaneously
- Usage: Tier 2 only — extracts key-value pairs, classifies regions semantically
- VRAM: ~1.5GB (or CPU at ~8 sec/page)

**Total GPU footprint: ~5GB for all models loaded simultaneously.**

## Pipeline Flow Per Page

```
PageImage (300 DPI, rendered via PyMuPDF)
    │
    ├─► DiT Layout Detector
    │     Output: List[Region(bbox, label, confidence)]
    │
    ├─► For each region labeled "table":
    │     Table Transformer Detection → confirms table bounds
    │     Table Transformer Structure → cell grid extraction
    │     Output: StructuredTable(rows, cols, cells, spans, content)
    │
    ├─► For each region with low OCR confidence:
    │     Crop region from page image
    │     TrOCR (printed or handwritten, auto-detected)
    │     Output: enhanced text replacing low-confidence OCR
    │
    └─► For Tier 2 pages only:
          LayoutLMv3 processes full page
          Input: page image + OCR text + bounding boxes
          Output: semantic labels per token, key-value pairs, form field associations
```

Parallelism:
- DiT and TrOCR are independent — run concurrently per page
- Table Transformer depends on DiT regions — sequential
- LayoutLMv3 can run in parallel with DiT
- Multiple pages processed concurrently (configurable, default: 4 pages)

## Enrichment Merger

Confidence-based arbitration between existing and visual results:

- **Layout**: DiT regions replace heuristic section boundaries where DiT confidence > 0.7. Preserves existing sections otherwise. Adds newly found regions (sidebars, captions).
- **Tables**: Existing CSV tables kept as fallback. Structured representation (headers, rows, spans, cell_types) added from Table Transformer.
- **Text**: Character-level alignment between existing OCR and TrOCR. Replace only where TrOCR confidence > existing + 0.15 threshold. All replacements logged.
- **Key-Value Pairs**: Union of regex-extracted and LayoutLMv3-extracted pairs, deduplicated by key similarity. LayoutLMv3 pairs get spatial_confidence field.

Key principles:
- **Never delete** — visual layer only adds or upgrades
- **Confidence wins** — higher confidence version kept when both extract the same element
- **Provenance tracked** — every enrichment tagged with `source: "visual_intelligence"`
- **Audit trail** — merger logs diff summary per page

## KG Enricher

Fires async after Enrichment Merger completes. Writes directly to Neo4j (non-fatal):

```
Nodes created:
  - LayoutRegion(type, bbox, page, confidence)
  - StructuredTable(rows, cols, page)
  - FormField(key, value, spatial_confidence)

Edges created:
  - Document -[HAS_REGION]→ LayoutRegion
  - LayoutRegion -[CONTAINS]→ StructuredTable
  - FormField -[FOUND_IN]→ LayoutRegion
  - StructuredTable -[HAS_CELL]→ Cell
```

## Model Management

### Three-Stage Loading: Cache → Auto-Install → Warn & Skip

```python
class ModelPool:
    def load_model(self, model_key):
        # Stage 1: Try loading from HuggingFace cache
        # Stage 2: Auto-download if not cached
        # Stage 3: Warn and skip — pipeline continues without this model
```

Per-model graceful degradation:
- DiT unavailable → skip layout enrichment, use existing heuristics
- Table Transformer unavailable → keep existing CSV tables
- TrOCR unavailable → keep Tesseract/EasyOCR output
- LayoutLMv3 unavailable → skip semantic KIE, keep regex form extraction
- ALL unavailable → existing pipeline runs as today, single warning logged

If dependencies (torch, transformers) are not installed, attempt auto-install. If that fails, log warning and proceed with existing pipeline. Zero breakage.

## File Structure

```
src/visual_intelligence/
├── __init__.py
├── complexity_scorer.py      # Tier 0/1/2 gating per page
├── page_renderer.py          # PDF → page images (PyMuPDF)
├── models/
│   ├── __init__.py
│   ├── dit_layout.py         # DiT layout detection wrapper
│   ├── table_transformer.py  # Table detection + structure recognition
│   ├── trocr_enhancer.py     # TrOCR for low-confidence OCR regions
│   └── layoutlmv3.py         # LayoutLMv3 KIE + form understanding
├── enrichment_merger.py      # Confidence-based result reconciliation
├── kg_enricher.py            # Parallel KG writes for visual entities
├── model_pool.py             # Lazy loading, warm-up, GPU/CPU routing
└── orchestrator.py           # Coordinates the full visual pipeline
```

### Integration Points (single existing file modified)

```python
# extraction_service.py — single call added after existing extraction:
enriched = await visual_orchestrator.enrich(
    doc_id=doc_id,
    extracted_doc=extracted,
    file_bytes=raw_bytes
)
```

### Configuration

```python
# src/api/config.py
class Config:
    class VisualIntelligence:
        enabled: bool = True
        gpu_device: str = "cuda:0"
        cpu_fallback: bool = True
        max_concurrent_pages: int = 4
        tier1_models: list = ["dit"]
        tier2_models: list = ["dit", "table_transformer", "trocr", "layoutlmv3"]
```

## Dependencies

```
# Additions to requirements.txt
transformers>=4.35.0
timm>=0.9.0
torchvision>=0.16.0
Pillow>=10.0.0
```

No new inference servers needed — all models run via HuggingFace transformers directly.

## Phasing

### Phase 1: Foundation + Layout (Week 1-2)
- Complexity Scorer + Page Renderer + DiT Layout Detector + Enrichment Merger
- Improves: multi-column docs, figure/header/footer labeling, sidebar/caption detection
- Models: DiT only (~2GB VRAM)
- Risk: Low — purely additive

### Phase 2: Tables + OCR (Week 3-4)
- Table Transformer + TrOCR + KG Enricher
- Improves: complex tables, scanned documents, handwritten text, KG table entities
- Models: +Table Transformer +TrOCR (~3GB more VRAM)
- Risk: Medium — table structure is new capability

### Phase 3: Deep Understanding (Week 5-6)
- LayoutLMv3 integration + form field extraction + batch inference optimization
- Improves: semantic layout understanding, form checkboxes, cross-region relationships
- Models: +LayoutLMv3 (~1.5GB more VRAM)
- Risk: Medium — may need fine-tuning for specific doc types
