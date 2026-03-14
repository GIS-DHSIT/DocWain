"""Evaluation harness for TaskSpec fine-tuned model.

Scores the model on 5 dimensions against ground-truth labels:
- JSON parse rate (target: 95%)
- Intent accuracy (target: 85%)
- Domain accuracy (target: 90%)
- Entity recall (target: 80%)
- Constraint extraction (target: 70%)

Composite score = weighted average → pass/marginal/fail decision.

Usage::

    python -m src.finetune.evaluate_model
    python -m src.finetune.evaluate_model --model DocWain-Agent-v2:latest
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Scoring weights ──────────────────────────────────────────────────────────

WEIGHT_INTENT = 0.30
WEIGHT_DOMAIN = 0.25
WEIGHT_ENTITIES = 0.20
WEIGHT_CONSTRAINTS = 0.15
WEIGHT_PARSE = 0.10

# ── Thresholds ───────────────────────────────────────────────────────────────

PASS_COMPOSITE = 80.0
PASS_PARSE_RATE = 95.0
MARGINAL_COMPOSITE = 60.0

# ── JSON extraction pattern ──────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# ── Default eval set ─────────────────────────────────────────────────────────

_DEFAULT_EVAL_PATH = Path("finetune_data") / "taskspec_eval.jsonl"


def evaluate_model(
    model_name: str = "DocWain-Agent-v2:latest",
    eval_set_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full evaluation harness and return scores.

    Returns a dict with:
        json_parse_rate, intent_accuracy, domain_accuracy,
        entity_recall, constraint_accuracy, composite_score,
        verdict ("pass"|"marginal"|"fail"), per_example details,
        weak_categories
    """
    eval_path = eval_set_path or _DEFAULT_EVAL_PATH
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval set not found: {eval_path}")

    # Load evaluation examples
    examples = _load_eval_set(eval_path)
    log.info("Loaded %d evaluation examples from %s", len(examples), eval_path)

    # Score each example
    results: List[Dict[str, Any]] = []
    parse_ok = 0
    intent_ok = 0
    domain_ok = 0
    entity_recalls: List[float] = []
    constraint_scores: List[float] = []

    for i, example in enumerate(examples):
        query = example["query"]
        ground_truth = example["ground_truth"]

        prediction = _call_model(model_name, query)
        detail: Dict[str, Any] = {
            "query": query,
            "ground_truth": ground_truth,
            "prediction": prediction,
        }

        if prediction is None:
            detail["parsed"] = False
            results.append(detail)
            entity_recalls.append(0.0)
            constraint_scores.append(0.0)
            continue

        # JSON parse success
        detail["parsed"] = True
        parse_ok += 1

        # Intent accuracy
        gt_intent = ground_truth.get("intent", "")
        pred_intent = prediction.get("intent", "")
        intent_match = gt_intent == pred_intent
        detail["intent_match"] = intent_match
        if intent_match:
            intent_ok += 1

        # Domain accuracy
        gt_domain = ground_truth.get("domain", "")
        pred_domain = prediction.get("domain", "")
        domain_match = gt_domain == pred_domain
        detail["domain_match"] = domain_match
        if domain_match:
            domain_ok += 1

        # Entity recall
        gt_entities = set(e.lower() for e in ground_truth.get("entities", []))
        pred_entities = set(e.lower() for e in prediction.get("entities", []))
        if gt_entities:
            recall = len(gt_entities & pred_entities) / len(gt_entities)
        else:
            recall = 1.0 if not pred_entities else 0.5
        entity_recalls.append(recall)
        detail["entity_recall"] = recall

        # Constraint accuracy
        gt_constraints = ground_truth.get("constraints", {})
        pred_constraints = prediction.get("constraints", {})
        if gt_constraints:
            matched = sum(
                1 for k, v in gt_constraints.items()
                if str(pred_constraints.get(k, "")).lower() == str(v).lower()
            )
            c_score = matched / len(gt_constraints)
        else:
            c_score = 1.0 if not pred_constraints else 0.5
        constraint_scores.append(c_score)
        detail["constraint_score"] = c_score

        results.append(detail)

    n = max(len(examples), 1)

    json_parse_rate = (parse_ok / n) * 100
    intent_accuracy = (intent_ok / n) * 100
    domain_accuracy = (domain_ok / n) * 100
    avg_entity_recall = (sum(entity_recalls) / n) * 100
    avg_constraint = (sum(constraint_scores) / n) * 100

    composite = (
        WEIGHT_INTENT * intent_accuracy
        + WEIGHT_DOMAIN * domain_accuracy
        + WEIGHT_ENTITIES * avg_entity_recall
        + WEIGHT_CONSTRAINTS * avg_constraint
        + WEIGHT_PARSE * json_parse_rate
    )

    if composite >= PASS_COMPOSITE and json_parse_rate >= PASS_PARSE_RATE:
        verdict = "pass"
    elif composite >= MARGINAL_COMPOSITE:
        verdict = "marginal"
    else:
        verdict = "fail"

    # Identify weak categories
    weak = _identify_weak_from_results(results)

    eval_result = {
        "json_parse_rate": round(json_parse_rate, 1),
        "intent_accuracy": round(intent_accuracy, 1),
        "domain_accuracy": round(domain_accuracy, 1),
        "entity_recall": round(avg_entity_recall, 1),
        "constraint_accuracy": round(avg_constraint, 1),
        "composite_score": round(composite, 1),
        "verdict": verdict,
        "total_examples": n,
        "weak_categories": weak,
        "per_example": results,
    }

    log.info(
        "Evaluation: composite=%.1f%% parse=%.1f%% intent=%.1f%% "
        "domain=%.1f%% entity=%.1f%% constraint=%.1f%% → %s",
        composite, json_parse_rate, intent_accuracy,
        domain_accuracy, avg_entity_recall, avg_constraint, verdict,
    )

    return eval_result


def identify_weak_categories(eval_result: Dict[str, Any]) -> List[str]:
    """Extract the weak categories list from an evaluation result."""
    return eval_result.get("weak_categories", [])


def _identify_weak_from_results(results: List[Dict[str, Any]]) -> List[str]:
    """Identify intents and domains with below-average accuracy."""
    intent_counts: Dict[str, int] = {}
    intent_correct: Dict[str, int] = {}
    domain_counts: Dict[str, int] = {}
    domain_correct: Dict[str, int] = {}

    for r in results:
        gt = r.get("ground_truth", {})
        intent = gt.get("intent", "unknown")
        domain = gt.get("domain", "unknown")

        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if r.get("intent_match"):
            intent_correct[intent] = intent_correct.get(intent, 0) + 1
        if r.get("domain_match"):
            domain_correct[domain] = domain_correct.get(domain, 0) + 1

    weak: List[str] = []

    for intent, total in intent_counts.items():
        correct = intent_correct.get(intent, 0)
        if total >= 2 and (correct / total) < 0.7:
            weak.append(intent)

    for domain, total in domain_counts.items():
        correct = domain_correct.get(domain, 0)
        if total >= 2 and (correct / total) < 0.7:
            weak.append(domain)

    return weak


def _load_eval_set(path: Path) -> List[Dict[str, Any]]:
    """Load evaluation examples from JSONL.

    Each line is a chat-format example. We extract the user query and
    the assistant's JSON (ground truth TaskSpec).
    """
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])
                query = ""
                gt_json = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        query = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        gt_json = msg.get("content", "")
                if query and gt_json:
                    ground_truth = json.loads(gt_json)
                    examples.append({
                        "query": query,
                        "ground_truth": ground_truth,
                    })
            except (json.JSONDecodeError, KeyError) as exc:
                log.debug("Skipping malformed eval line: %s", exc)

    return examples


def _call_model(model_name: str, query: str) -> Optional[Dict[str, Any]]:
    """Call the fine-tuned model and parse its JSON output."""
    try:
        import ollama as _ollama
    except ImportError:
        log.error("ollama package not installed")
        return None

    from src.finetune.training_data_generator import SYSTEM_PROMPT
    system_prompt = SYSTEM_PROMPT

    try:
        response = _ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.1, "num_predict": 512},
        )
        raw = response.get("message", {}).get("content", "")

        match = _JSON_RE.search(raw)
        if not match:
            return None

        return json.loads(match.group())

    except json.JSONDecodeError:
        return None
    except Exception as exc:
        log.debug("Model call failed for %r: %s", query[:50], exc)
        return None


def generate_augmentation_data(
    weak_categories: List[str],
    count: int = 200,
) -> List[Dict[str, Any]]:
    """Generate targeted training examples for weak categories.

    Returns chat-format dicts ready to append to the training JSONL.
    """
    from src.finetune.training_data_generator import (
        _DOMAIN_TASK_SPECS,
        _SEED_PARAPHRASES,
        _chat_example,
        _expand_paraphrases_offline,
    )

    augmented: List[Dict[str, Any]] = []
    per_category = max(1, count // max(len(weak_categories), 1))

    for category in weak_categories:
        generated = 0
        for task_key, ts in _DOMAIN_TASK_SPECS.items():
            if generated >= per_category:
                break
            domain = ts.get("domain", "")
            intent = ts.get("intent", "")
            if category not in (domain, intent, task_key):
                continue
            seeds = _SEED_PARAPHRASES.get(task_key, [])
            ts_with_conf = {**ts, "confidence": 0.90}
            for seed in seeds:
                if generated >= per_category:
                    break
                expanded = _expand_paraphrases_offline(seed, count=10)
                for q in expanded:
                    augmented.append(_chat_example(q, ts_with_conf))
                    generated += 1
                    if generated >= per_category:
                        break

    return augmented[:count]


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DocWain-Agent-v2 TaskSpec model")
    parser.add_argument("--model", default="DocWain-Agent-v2:latest")
    parser.add_argument("--eval-set", type=str, default=str(_DEFAULT_EVAL_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = evaluate_model(
        model_name=args.model,
        eval_set_path=Path(args.eval_set),
    )

    print(f"\n{'═' * 50}")
    print(f"Verdict:    {result['verdict'].upper()}")
    print(f"Composite:  {result['composite_score']:.1f}%")
    print(f"Parse rate: {result['json_parse_rate']:.1f}%")
    print(f"Intent:     {result['intent_accuracy']:.1f}%")
    print(f"Domain:     {result['domain_accuracy']:.1f}%")
    print(f"Entity:     {result['entity_recall']:.1f}%")
    print(f"Constraint: {result['constraint_accuracy']:.1f}%")
    if result["weak_categories"]:
        print(f"Weak:       {', '.join(result['weak_categories'])}")
    print(f"{'═' * 50}")

    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
