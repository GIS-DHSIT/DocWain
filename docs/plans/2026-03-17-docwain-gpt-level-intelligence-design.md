# DocWain GPT-Level Intelligence Design

## Goal
Evolve DHS/DocWain from a system-prompt-enhanced 8B model to a GPT-competitive document intelligence model through base model scaling, synthetic training data, preference-based training, multi-stage reasoning architecture, and continuous improvement.

## Context
- **Current state:** qwen3:8b (5.2GB Q4_K_M) with enriched system prompt, hosted at ollama.com/DHS/DocWain
- **Target:** ~90-95% of GPT-4 quality on document intelligence tasks
- **Timeline:** 4-6 weeks aggressive
- **Hardware:** Start on T4 16GB, scale as needed
- **Training data:** Synthetic (Claude Code) for bootstrap, production signals for continuous improvement
- **Constraint:** No document content in training data — metadata reasoning patterns only

## Phase 1: Base Model Upgrade + Vision (Weeks 1-2)

### Model Progression
```
qwen3:8b (5.2GB) → qwen3:14b (9.3GB) → 32B vision-language model (~20GB Q4)
```

### Why 32B is the sweet spot
- 8B models follow instructions but lack reasoning depth for cross-document synthesis
- 32B achieves ~85-90% of GPT-4 quality on document tasks
- 72B is diminishing returns for the compute cost
- Vision capability is critical for charts, diagrams, scanned PDFs, signatures

### Candidate Models
| Model | Size | Vision | Reasoning | Notes |
|-------|------|--------|-----------|-------|
| Qwen3-14B | 9.3GB | No | Good | Already local, immediate upgrade |
| Qwen2.5-VL-32B | ~20GB Q4 | Yes | Strong | Ollama ready, target model |
| Llama-3.2-Vision-90B | ~55GB Q4 | Yes | Excellent | Needs A100 |

### Deliverables
- Update Modelfile FROM to qwen3:14b
- Rebuild and push DHS/DocWain on 14B base
- Benchmark 14B vs 8B on eval harness
- Evaluate 32B VL candidates for Phase 2 migration

## Phase 2: Synthetic Training Data Pipeline (Weeks 2-3)

### Data Generation Strategy
Claude Code generates gold-standard responses directly — no API costs, full codebase context, real-time validation.

### Target: 3,000+ Training Examples

| Category | Current | Target | Method |
|----------|---------|--------|--------|
| Identity & DHS knowledge | 28 | 50 | Hand-crafted (done) |
| Pipeline & meta questions | 20 | 60 | Synthetic expansion |
| Response formatting | 20 | 200 | Claude Code generates from raw evidence |
| Document extraction | 0 | 500 | Synthetic invoices/contracts/resumes + gold extraction |
| Cross-document reasoning | 0 | 300 | Multi-doc evidence sets with complex analysis |
| Gap handling & honesty | 16 | 150 | Intentionally incomplete evidence, train honest refusal |
| Image/chart description | 0 | 200 | Chart/table-as-image descriptions |
| Content generation | 0 | 300 | Cover letters, interview questions, summaries from evidence |
| Domain-specific (HR/Legal/Finance/Medical) | 12 | 500 | Domain expert scenarios with correct terminology |
| Multi-turn conversation | 15 | 200 | 3-5 turn dialogues with pronoun resolution |
| Intent understanding | 0 | 300 | Ambiguous queries mapped to correct task types |
| Edge cases & adversarial | 0 | 200 | Prompt injection, off-topic, manipulation attempts |

### Generation Pipeline
```
1. Claude Code generates gold responses for scenario templates
2. Write directly to JSONL with immediate quality validation
3. Automated checks (formatting, grounding, no hallucination)
4. Iterate weak categories with better examples
5. Split 90/10 train/eval
```

### Key Principle
Train on metadata reasoning patterns (how to analyze structure, detect domains, cross-reference entities) — NOT on specific document content.

## Phase 3: Advanced Training Techniques (Weeks 3-4)

### Training Progression
```
Stage 1: SFT (Supervised Fine-Tuning)
    → Teach the model WHAT good responses look like
    → 3,000+ examples across 12 categories

Stage 2: DPO (Direct Preference Optimization)
    → Teach the model WHY one response is better
    → Pairs of (chosen, rejected) for the same query
    → No reward model needed

Stage 3: ORPO (Odds Ratio Preference Optimization)
    → Combines SFT + preference in single training pass
    → More efficient, better for smaller datasets
```

### Why DPO Over RLHF
- RLHF needs separate reward model + PPO — complex, unstable, massive compute
- DPO achieves ~90% of RLHF quality with simple pairwise loss
- Perfect for our setup

### DPO Preference Pair Example
```json
{
  "prompt": "What is the total on this invoice?\n[EVIDENCE] Invoice #500: $47,250",
  "chosen": "The total on Invoice **#500** is **$47,250.00**.",
  "rejected": "Based on my analysis of the documents provided, the invoice appears to show a total amount. The total is $47,250. Let me know if you need anything else!"
}
```

### Preference Pair Categories (1,000+ pairs)
- Good formatting vs verbose/unstructured
- Grounded vs hallucinated
- Concise lookup vs over-explained
- Honest gap disclosure vs fabricated answer
- Proper isolation vs leaked internal data

### Production Signal Integration
- `POST /api/feedback/positive` → "chosen" examples
- `POST /api/feedback/negative` with correction → "rejected" examples
- Weekly auto-generation of DPO pairs from accumulated feedback

### Training Stack
- SFT: Unsloth + TRL SFTTrainer (current)
- DPO: Unsloth + TRL DPOTrainer
- ORPO: TRL ORPOTrainer (single-pass alternative)

## Phase 4: Reasoning Architecture (Weeks 4-5)

### Current Pipeline (Single-Pass)
```
Query → Retrieve → Generate → Response
```

### Target Pipeline (Multi-Stage Reasoning)
```
Query → UNDERSTAND → PLAN → RETRIEVE → VERIFY → GENERATE → VALIDATE → Response
```

### New Stages

| Stage | Purpose | Implementation |
|-------|---------|----------------|
| **UNDERSTAND** | Classify intent, detect entities, decompose multi-part queries | Extends existing intent classifier |
| **PLAN** | Decide retrieval strategy — which docs, how many chunks, what to look for | New: `src/reasoning/planner.py` |
| **RETRIEVE** | Vector search + reranking | Existing (already strong) |
| **VERIFY** | Check if evidence actually answers the question BEFORE generating | New: `src/reasoning/verifier.py` |
| **GENERATE** | Produce response with chain-of-thought reasoning | Enhanced with `<think>` blocks |
| **VALIDATE** | Post-generation grounding check, ID leak detection, format enforcement | New: `src/reasoning/validator.py` |

### Chain-of-Thought
Qwen3 natively supports `<think>` blocks:
```
<think>
User asks for invoice totals across 3 documents.
Found: Invoice_A ($12,500), Invoice_B ($8,750), Invoice_C (no total).
Aggregate A + B = $21,250, flag C as missing.
</think>

Total across invoices: **$21,250.00** (2 of 3 invoices)
```

### VALIDATE Gate (Rule-Based, No Extra LLM Call)
- Hallucinated values not in evidence
- Leaked internal IDs (subscription_id, chunk_id)
- Broken markdown formatting
- Response doesn't match detected task type

## Phase 5: Continuous Improvement Loop (Weeks 5-6)

### Automated Cycle
```
Users query DocWain
    ↓
Responses delivered + confidence logged
    ↓
Feedback collected (thumbs up/down, corrections)
    ↓
Weekly: Auto-generate DPO pairs from feedback
    ↓
Weekly: Evaluate model against benchmark suite
    ↓
If score drops OR new pairs > threshold → retrain
    ↓
If improved → deploy, if worse → revert
    ↓
Monthly: Claude Code reviews weak categories, generates targeted data
```

### Eval Harness
- 100+ golden test cases across all categories
- Automated scoring: identity (20%), formatting (20%), grounding (25%), reasoning (20%), isolation (15%)
- Pass/fail gates per category — no regression allowed
- Score tracking over time

## Expected Quality Progression
```
Week 0:  qwen3:8b + system prompt           → ~55% of GPT-4
Week 2:  qwen3:14b + system prompt           → ~65% of GPT-4
Week 3:  14b + SFT (3,000 examples)          → ~75% of GPT-4
Week 4:  14b + DPO preference training       → ~82% of GPT-4
Week 5:  + reasoning architecture            → ~88% of GPT-4
Week 6:  + continuous improvement            → ~90%+ and climbing
Future:  32B VL base + all phases            → ~93-95% of GPT-4
```

## Key Files

| File | Purpose |
|------|---------|
| `src/finetune/synthetic_data_generator.py` | Claude Code-powered data generation (12 categories) |
| `src/finetune/dpo_trainer.py` | DPO/ORPO preference training pipeline |
| `src/finetune/eval_harness.py` | 100+ test benchmark with automated scoring |
| `src/reasoning/planner.py` | Query decomposition and retrieval planning |
| `src/reasoning/verifier.py` | Pre-generation evidence sufficiency check |
| `src/reasoning/validator.py` | Post-generation grounding and format validation |
| `Modelfile` | Updated for each base model upgrade |

## What This Does NOT Require
- No cloud API dependency for inference (everything local)
- No document content in training data (metadata reasoning only)
- No RLHF complexity (DPO is simpler and nearly as effective)
- No custom pre-training (fine-tuning on strong base models)
