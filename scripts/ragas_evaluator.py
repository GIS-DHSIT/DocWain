#!/usr/bin/env python3
"""RAGAS-aligned evaluation metrics for DocWain intensive test results.

Computes four metrics aligned with the RAGAS framework:
1. Answer Faithfulness: How well the answer is grounded in retrieved evidence
2. Hallucination Rate: Fraction of responses with content not in evidence
3. Context Recall: How well the retrieved chunks cover the expected answer
4. Grounding Bypass Rate: Fraction of responses that bypass grounding checks

Usage:
    python scripts/ragas_evaluator.py [results_path]

    results_path: Path to intensive test JSON results (default: /tmp/intensive_test_results.json)

Output:
    tests/ragas_metrics.json
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Banned phrases that indicate hallucination or template leakage ────
_HALLUCINATION_MARKERS = [
    "as an ai", "i don't have access", "i cannot", "i'm unable",
    "unfortunately, i", "i apologize", "as a language model",
    "MISSING_REASON", "section_id", "chunk_type", "page_start",
    "embedding_text", "canonical_text",
]

# ── Intelligence / grounding signals ──────────────────────────────────
_GROUNDING_SIGNALS = [
    "according to", "based on", "the document", "the resume",
    "the report", "the invoice", "the policy", "states that",
    "mentions", "indicates", "shows that", "from the",
]

_EVIDENCE_KEYWORDS = [
    "source", "document", "page", "section", "evidence",
    "found in", "cited", "referenced",
]


def _normalize(text: str) -> str:
    """Lowercase and strip for comparison."""
    return (text or "").lower().strip()


def compute_answer_faithfulness(result: Dict[str, Any]) -> float:
    """Measure how well the answer references evidence and avoids fabrication.

    Score 0-1:
    - 1.0: All claims traceable to sources, uses grounding language
    - 0.0: No grounding, contains hallucination markers
    """
    text = _normalize(result.get("preview", ""))
    if not text:
        return 0.0

    score = 0.0
    total_checks = 4

    # Check 1: No hallucination markers
    hallucinations = [m for m in _HALLUCINATION_MARKERS if m.lower() in text]
    if not hallucinations:
        score += 1.0

    # Check 2: Uses grounding language (references documents/evidence)
    grounding = [s for s in _GROUNDING_SIGNALS if s in text]
    if grounding:
        score += 1.0

    # Check 3: Has sources
    source_count = result.get("sources", 0)
    if source_count > 0:
        score += 1.0

    # Check 4: Response length indicates substance (not a refusal)
    length = result.get("length", 0)
    if length > 100:
        score += 1.0

    return round(score / total_checks, 3)


def compute_hallucination_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of responses containing hallucination markers."""
    if not results:
        return 0.0

    hallucinated = 0
    for r in results:
        text = _normalize(r.get("preview", ""))
        if any(m.lower() in text for m in _HALLUCINATION_MARKERS):
            hallucinated += 1

    return round(hallucinated / len(results), 3)


def compute_context_recall(result: Dict[str, Any]) -> float:
    """How many expected keywords were found in the response.

    This approximates RAGAS context recall — measures whether the retrieved
    context contained the information needed to answer correctly.
    """
    found = result.get("found", [])
    expected_count = len(result.get("found", [])) + len(result.get("missing", []))
    if expected_count == 0:
        return 1.0  # No expectations = trivially satisfied
    return round(len(found) / expected_count, 3)


def compute_grounding_bypass_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of responses that lack any evidence grounding.

    A response "bypasses grounding" when it has no sources, no grounding
    language, and contains banned template phrases.
    """
    if not results:
        return 0.0

    bypassed = 0
    for r in results:
        text = _normalize(r.get("preview", ""))
        has_sources = r.get("sources", 0) > 0
        has_grounding = any(s in text for s in _GROUNDING_SIGNALS)
        has_banned = bool(r.get("banned", []))

        if not has_sources and not has_grounding and has_banned:
            bypassed += 1

    return round(bypassed / len(results), 3)


def evaluate(results_path: str) -> Dict[str, Any]:
    """Run full RAGAS-aligned evaluation on intensive test results."""
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return {"error": "No results found in input file"}

    # Per-result metrics
    faithfulness_scores = []
    context_recall_scores = []

    per_result = []
    for r in results:
        faith = compute_answer_faithfulness(r)
        recall = compute_context_recall(r)
        faithfulness_scores.append(faith)
        context_recall_scores.append(recall)
        per_result.append({
            "id": r.get("id", "unknown"),
            "grade": r.get("grade", "?"),
            "faithfulness": faith,
            "context_recall": recall,
        })

    # Aggregate metrics
    avg_faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
    avg_context_recall = round(sum(context_recall_scores) / len(context_recall_scores), 3)
    hallucination_rate = compute_hallucination_rate(results)
    grounding_bypass_rate = compute_grounding_bypass_rate(results)

    # Low faithfulness items (for debugging)
    low_faith = [p for p in per_result if p["faithfulness"] < 0.5]
    low_recall = [p for p in per_result if p["context_recall"] < 0.5]

    metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": results_path,
        "total_queries": len(results),
        "aggregate": {
            "answer_faithfulness": avg_faithfulness,
            "hallucination_rate": hallucination_rate,
            "context_recall": avg_context_recall,
            "grounding_bypass_rate": grounding_bypass_rate,
        },
        "thresholds": {
            "answer_faithfulness_target": 0.85,
            "hallucination_rate_target": 0.05,
            "context_recall_target": 0.80,
            "grounding_bypass_rate_target": 0.02,
        },
        "pass": (
            avg_faithfulness >= 0.85
            and hallucination_rate <= 0.05
            and avg_context_recall >= 0.80
            and grounding_bypass_rate <= 0.02
        ),
        "low_faithfulness": low_faith[:10],
        "low_context_recall": low_recall[:10],
        "per_result": per_result,
    }

    return metrics


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/intensive_test_results.json"

    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        print("Run `python scripts/intensive_test.py` first to generate results.")
        sys.exit(1)

    print(f"Evaluating: {results_path}")
    metrics = evaluate(results_path)

    # Print summary
    agg = metrics["aggregate"]
    thresholds = metrics["thresholds"]
    print("\n" + "=" * 60)
    print("  RAGAS-ALIGNED EVALUATION METRICS")
    print("=" * 60)
    print(f"  Total queries:          {metrics['total_queries']}")
    print(f"  Answer Faithfulness:    {agg['answer_faithfulness']:.3f}  (target >= {thresholds['answer_faithfulness_target']})")
    print(f"  Hallucination Rate:     {agg['hallucination_rate']:.3f}  (target <= {thresholds['hallucination_rate_target']})")
    print(f"  Context Recall:         {agg['context_recall']:.3f}  (target >= {thresholds['context_recall_target']})")
    print(f"  Grounding Bypass Rate:  {agg['grounding_bypass_rate']:.3f}  (target <= {thresholds['grounding_bypass_rate_target']})")
    print(f"\n  Overall: {'PASS' if metrics['pass'] else 'FAIL'}")
    print("=" * 60)

    # Low scores
    if metrics.get("low_faithfulness"):
        print(f"\n  Low faithfulness ({len(metrics['low_faithfulness'])}):")
        for item in metrics["low_faithfulness"][:5]:
            print(f"    {item['id']}: faith={item['faithfulness']:.2f} grade={item['grade']}")

    if metrics.get("low_context_recall"):
        print(f"\n  Low context recall ({len(metrics['low_context_recall'])}):")
        for item in metrics["low_context_recall"][:5]:
            print(f"    {item['id']}: recall={item['context_recall']:.2f} grade={item['grade']}")

    # Save
    output_path = Path(__file__).parent.parent / "tests" / "ragas_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {output_path}")

    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
