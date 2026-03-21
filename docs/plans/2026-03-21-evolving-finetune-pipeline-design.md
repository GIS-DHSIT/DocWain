# Evolving Fine-Tune Pipeline — Design Document

**Date:** 2026-03-21
**Status:** Approved
**Author:** Claude Code + Muthu

## Overview

An end-to-end, iterative fine-tuning pipeline that continuously improves the DHS/DocWain model. Claude Code acts as both teacher and judge — observing DocWain's weaknesses, generating training data, and evaluating improvements. Multiple student models train in parallel, and the best performer becomes `DocWain:latest` through a tournament system with hybrid distillation.

## Core Principles

- **No document content in training** — model learns understanding patterns and interaction quality, not facts
- **Claude Code as teacher** — no external LLM API cost for the teach stage; Azure GPT-4.1 available as optional cross-validator/fallback
- **Interactive pipeline** — user steers at every checkpoint (observe, harvest, teach, train, gate)
- **Pluggable models** — any HuggingFace-compatible model can be added as a student
- **Hybrid trigger** — scheduled daily baseline + threshold-triggered burst runs

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                CLAUDE CODE SESSION                    │
│                                                      │
│  /finetune                                           │
│                                                      │
│  OBSERVE → HARVEST → TEACH → TRAIN → GATE           │
│     │         │         │       │       │            │
│   [review]  [review]  [review] [auto] [approve]      │
│                                                      │
│  Claude generates signals by probing DocWain         │
│  Claude generates SFT + DPO training pairs           │
│  Claude evaluates trained models on 5 criteria       │
│  Winner → DocWain:latest, others → alternatives      │
└─────────────────────────────────────────────────────┘
```

## Stage 1: Observe

Claude sends ~50 evaluation prompts to DocWain's Ollama endpoint, covering:

- Table extraction and reasoning
- Layout parsing (headers, lists, nested structures)
- Cross-reference resolution
- Section hierarchy detection
- Multi-page reasoning
- Uncertainty handling
- Adaptive tone matching

Claude analyzes responses against the 5 evaluation criteria, identifies systematic weaknesses, and generates targeted follow-up probes. Results presented to user for review.

**Output:** Weakness report with category scores.

## Stage 2: Harvest

Combines two signal sources:

### Document Understanding Signals
- Table extraction failures/successes
- Layout parsing accuracy
- Cross-reference resolution hits/misses
- Section hierarchy detection quality
- Multi-page reasoning patterns

### User Interaction Signals
- Explicit feedback (thumbs up/down + correction text)
- Low-confidence queries (model uncertain)
- Follow-up questions (signal of incomplete answer)
- Query reformulations (user retried)
- Session length patterns (stuck users)

### Signal Quality Filtering
- Dedup via Simhash64 (≥0.92 threshold)
- Minimum richness — feedback corrections >50 chars
- Recency weighting — newer signals scored higher
- Category balance — ensures coverage across all understanding types

**Signal format:**
```json
{
  "signal_type": "table_extraction_miss",
  "query": "...",
  "model_response": "...",
  "context_chunks": ["chunk_id_1", "chunk_id_2"],
  "category": "document_understanding",
  "subcategory": "table_extraction",
  "timestamp": "2026-03-21T...",
  "confidence_score": 0.34
}
```

**Output:** `signals/{iteration_N}/` with understanding_signals.jsonl, interaction_signals.jsonl, signal_summary.json.

## Stage 3: Teach

Claude operates in two modes:

### Teacher Mode (SFT Data Generation)
- Input: signal + context chunk metadata
- Task: generate the ideal DocWain response
- Output: `{instruction, ideal_output, metadata}`
- Quality: teacher confidence threshold ≥0.8

### Judge Mode (DPO Pair Generation)
- Input: signal + DocWain's actual response
- Task: score on 5 criteria, generate better response
- Output: `{chosen, rejected, scores, reasoning}`
- Quality: cross-validate — teacher output must score ≥0.7 when re-judged

### Quality Filters
- Reject self-contradicting outputs
- Reject outputs referencing document content (must teach patterns, not facts)
- Cross-validation between teacher and judge modes

**Output:** `teach_output/{iteration_N}/` with sft_pairs.jsonl, dpo_pairs.jsonl, judge_scores.json, teach_summary.json.

## Stage 4: Train

### Phase 1 — Parallel SFT (per model)
All configured student models train independently on the same SFT dataset using Unsloth LoRA.

### Phase 2 — DPO Alignment (per model)
Each SFT output goes through DPO training on judge-generated preference pairs.

### Phase 3 — Tournament
All trained models answer 200 eval prompts. Claude scores each response on weighted composite:

| Criterion | Weight |
|-----------|--------|
| Accuracy | 0.30 |
| Groundedness | 0.25 |
| Reasoning | 0.20 |
| Formatting | 0.15 |
| Adaptive Tone | 0.10 |

### Phase 4 — Hybrid Distillation (every 3rd iteration)
Cherry-pick best responses per category from ALL trained models. Create distillation SFT dataset. Retrain primary DocWain model on combined best outputs.

**Result:** Highest-scoring model → `DocWain:latest`. Others listed as alternatives (e.g., `DocWain:llama`, `DocWain:model-n`).

## Stage 5: Quality Gate

Promotion criteria:
- **Composite score ≥ 80%**
- **No single criterion below 60%**
- **Must beat previous DocWain:latest score**

If gate fails: nothing changes in production. Artifacts preserved for analysis.

User approves promotion before Ollama push.

## Model Registry

```yaml
# registry.yaml
models:
  DocWain:latest:
    base: qwen3-8b
    iteration: 12
    composite_score: 84.2
    scores: {accuracy: 87, groundedness: 83, reasoning: 82, formatting: 81, tone: 79}
    promoted_at: 2026-03-21T14:00:00
    artifact: finetune_artifacts/iter_12/
    status: production

  DocWain:llama:
    base: llama-3.1-8b
    iteration: 12
    composite_score: 79.1
    status: available

  DocWain:previous:
    base: qwen3-8b
    iteration: 11
    composite_score: 81.7
    status: rollback_ready
```

**Rollback:** `DocWain:previous` always kept. Instant rollback via Ollama tag swap.

## Deployment

- **Now:** Ollama — local inference, GGUF export, register via Ollama API
- **Later:** vLLM for production with subscription-based API keys and token metering

### Deployment Flow
```
Merge LoRA → GGUF Export → Ollama Register
                              ↓
                    DocWain:latest (production)
                    DocWain:llama (available)
                    DocWain:previous (rollback)
```

## Hybrid Distillation Detail

```
Student Models (each fine-tuned independently)
       ↓
Response Tournament (all answer same eval prompts)
       ↓
GPT-4.1 / Claude judges each response
       ↓
Best responses per category selected
       ↓
Distillation SFT dataset
       ↓
DocWain PRIMARY retrained on cherry-picked best
       ↓
DocWain absorbs strengths of all architectures
```

## Trigger Logic

- **Scheduled:** Daily baseline scan at configured interval
- **Threshold:** 50+ new feedback entries OR 100+ new doc understanding events
- **Hybrid:** Scheduled checks + threshold burst runs when signal quality is high

## Interactive Flow (`/finetune`)

1. **OBSERVE** — Claude probes DocWain, reports weak areas → user reviews
2. **HARVEST** — Claude collects signals, shows summary → user approves/adjusts focus
3. **TEACH** — Claude generates training pairs, shows 5 samples → user approves/adjusts
4. **TRAIN** — Claude runs Unsloth SFT → DPO, reports metrics
5. **GATE** — Claude shows tournament results → user approves promotion

User can steer at every step: skip, focus, adjust, or abort.

## File Structure

```
src/finetune/
├─ evolve/                        # Evolution pipeline
│  ├─ __init__.py
│  ├─ config.py                   # Pipeline config + model registry schema
│  ├─ observer.py                 # Eval prompts to DocWain, score responses
│  ├─ harvester.py                # Collect signals from observer + stored feedback
│  ├─ teacher.py                  # Claude-driven SFT + DPO pair generation
│  ├─ trainer.py                  # Multi-model parallel Unsloth training
│  ├─ tournament.py               # Run all models against eval set, rank
│  ├─ distiller.py                # Cherry-pick best responses, retrain primary
│  ├─ gate.py                     # Composite scoring + promotion logic
│  ├─ registry.py                 # Model registry CRUD (registry.yaml)
│  ├─ deployer.py                 # GGUF export + Ollama registration
│  └─ prompts/
│     ├─ observer_prompts.py      # Eval prompt templates by category
│     ├─ teacher_sft.py           # SFT generation prompt templates
│     └─ teacher_dpo.py           # DPO judgment prompt templates
│
├─ evolve_config.yaml             # User-editable pipeline config

signals/                          # Per-iteration signal storage
├─ iter_N/
│  ├─ understanding_signals.jsonl
│  ├─ interaction_signals.jsonl
│  └─ signal_summary.json

finetune_artifacts/               # Per-iteration artifacts
├─ iter_N/
│  ├─ models/
│  │  ├─ qwen3-8b/{sft_merged/, dpo_merged/, gguf/}
│  │  └─ llama-8b/{sft_merged/, dpo_merged/, gguf/}
│  ├─ distillation/               # Every 3rd iteration
│  ├─ tournament_results.json
│  ├─ scores.json
│  └─ registry_snapshot.yaml

registry.yaml                     # Model registry (root level)
```

## Configuration

```yaml
# evolve_config.yaml
pipeline:
  scheduled_interval_hours: 24
  signal_threshold: 50
  distillation_every_n: 3
  eval_prompt_count: 200

models:
  primary: "unsloth/Qwen3-8B-bnb-4bit"
  students:
    - name: "qwen3-8b"
      repo: "unsloth/Qwen3-8B-bnb-4bit"
      enabled: true
    - name: "llama-3.1-8b"
      repo: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
      enabled: true

training:
  sft:
    epochs: 3
    learning_rate: 2e-4
    lora_r: 16
    lora_alpha: 16
    batch_size: 4
  dpo:
    epochs: 1
    beta: 0.1
    learning_rate: 5e-5

gate:
  composite_minimum: 80.0
  criterion_floor: 60.0
  must_beat_previous: true
  weights:
    accuracy: 0.30
    groundedness: 0.25
    reasoning: 0.20
    formatting: 0.15
    tone: 0.10

deployment:
  target: ollama
  ollama_host: "http://localhost:11434"
  keep_previous: true
  max_stored_iterations: 10

docwain:
  endpoint: "http://localhost:11434"
  model_name: "DHS/DocWain"

azure_fallback:
  enabled: false
  endpoint: "${AZURE_AI_ENDPOINT}"
  api_key: "${AZURE_AI_API_KEY}"
  model: "gpt-4.1"
  deployment_date: "2025-04-14"
```

## Reuse from Existing Codebase

| Existing Module | Reused For |
|----------------|------------|
| `unsloth_trainer.py` | Core SFT training logic |
| `dpo_trainer.py` | DPO alignment phase |
| `feedback_collector.py` | Interaction signal mining |
| `evaluate_model.py` | Scoring framework |
| `dataset_builder.py` | Dedup + validation |
| `docwain_finetune.py` | GGUF export + Ollama registration |

New `evolve/` modules build on top — no rewrites of working code.
