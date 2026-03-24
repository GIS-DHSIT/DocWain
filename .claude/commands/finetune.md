# /finetune — Evolving Fine-Tune Pipeline

Run the DocWain evolving fine-tune pipeline interactively. Claude Code acts as both teacher and judge.

## Pipeline Stages

Execute the 5-stage evolution cycle with review checkpoints:

### Stage 1: OBSERVE
1. Load the pipeline: `from src.finetune.evolve import EvolvePipeline, EvolveConfig`
2. Initialize with config from `src/finetune/evolve_config.yaml`
3. Send eval prompts to DocWain's Ollama endpoint (`DHS/DocWain`)
4. Score responses on 5 criteria (accuracy, groundedness, reasoning, formatting, tone)
5. Identify weak areas — present findings to user for review
6. Wait for user approval before proceeding

### Stage 2: HARVEST
1. Load observation signals from Stage 1
2. Load interaction signals from `src/outputs/learning_signals/high_quality.jsonl`
3. Merge, deduplicate (Simhash64 >= 0.92), and balance categories
4. Present signal summary — wait for user approval

### Stage 3: TEACH
1. For each harvested signal, generate the IDEAL DocWain response (SFT pair)
2. For signals with existing model responses, generate improved versions (DPO pairs)
3. Filter out content-specific responses (must teach patterns, not facts)
4. Show 5 sample pairs to user for review
5. Save to `signals/{iter_N}/sft_pairs.jsonl` and `dpo_pairs.jsonl`

### Stage 4: TRAIN
1. For each enabled student model in `evolve_config.yaml`:
   - Run Unsloth LoRA SFT training on generated pairs
   - Run DPO alignment on preference pairs
2. Report training metrics (loss curves)
3. Run tournament: all models answer 200 eval prompts
4. Rank by weighted composite score

### Stage 5: GATE
1. Check winner against quality gate:
   - Composite score >= 80%
   - No single criterion below 60%
   - Must beat previous DocWain:latest
2. Present results to user
3. If approved: promote winner to `DocWain:latest`, others as alternatives
4. Update `registry.yaml`

## Key Rules
- NO document content in training data — only understanding patterns and interaction quality
- User reviews and approves at each checkpoint
- Azure GPT-4.1 available as fallback (endpoint in env vars)
- Weights: accuracy=0.30, groundedness=0.25, reasoning=0.20, formatting=0.15, tone=0.10

## Quick Start — Evolving Pipeline
```python
from src.finetune.evolve import EvolvePipeline, EvolveConfig
from pathlib import Path

config = EvolveConfig.load_default()
pipeline = EvolvePipeline(
    config=config,
    signals_dir=Path("signals"),
    artifact_dir=Path("finetune_artifacts"),
    registry_path=Path("registry.yaml"),
)
print(pipeline.status())
```

---

## V2 Pipeline — Vision + Tool-Calling

Build DocWain V2: graft SigLIP vision encoder onto Qwen3-14B V1, train document intelligence + native function-calling.

### Phase 1: Projection Pre-Training
- Train projection MLP on image-caption alignment (LLaVA-Pretrain)
- Only projection trainable, SigLIP + Qwen3 frozen
- Gate: BLEU >= 30, CIDEr >= 80

### Phase 2: Document Intelligence
- SFT + DPO on DocVQA, ChartQA, PubTabNet, DocLayNet + glm-ocr distillation
- Projection + LoRA trainable
- Gate: DocVQA >= 75%, Table F1 >= 80%, Layout mAP >= 70%

### Phase 3: Tool-Calling
- SFT + DPO on synthetic tool-call data + ToolBench + Gorilla
- 9 core tools: ocr_extract, layout_extract, extract_table, extract_entities, context_understand, cross_reference, search_documents, summarize_section, visualize_data
- Gate: Tool accuracy >= 85%, Arg correctness >= 90%

### Phase 4: Merge & Promote
- Merge projection + LoRA into Qwen3-14B, export GGUF
- V1 regression test (persona, RAG, formatting must not regress)
- Promote: DHS/DocWain:v1 (backup) → DHS/DocWain:v2 → DHS/DocWain:latest

### Quick Start — V2 Pipeline
```python
from src.finetune.v2 import V2Pipeline
pipe = V2Pipeline()
print(pipe.status())
# Run phases interactively: phase1 → phase2 → phase3 → phase4
```
