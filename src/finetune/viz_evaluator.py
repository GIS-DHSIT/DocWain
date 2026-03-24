"""Visualization evaluation harness for DocWain.

Scores model responses on 5 dimensions:
1. Directive presence  (weight 0.30) -- Did the model emit <!--DOCWAIN_VIZ-->?
2. Chart type accuracy (weight 0.25) -- Correct chart for the data?
3. Data accuracy       (weight 0.20) -- Labels/values match expected?
4. Table formatting    (weight 0.15) -- Proper markdown table present?
5. Mode selection      (weight 0.10) -- Chart vs flow vs text-only correct?

Composite score = weighted average (0-100) -> pass/marginal/fail decision.

Usage::

    python -m src.finetune.viz_evaluator
    python -m src.finetune.viz_evaluator --model DocWain:latest
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# -- Scoring weights --------------------------------------------------------

WEIGHT_DIRECTIVE = 0.30
WEIGHT_CHART_TYPE = 0.25
WEIGHT_DATA = 0.20
WEIGHT_TABLE = 0.15
WEIGHT_MODE = 0.10

# -- Thresholds -------------------------------------------------------------

PASS_COMPOSITE = 80.0
MARGINAL_COMPOSITE = 60.0

# -- Regex patterns ---------------------------------------------------------

_VIZ_RE = re.compile(r"<!--DOCWAIN_VIZ\s*(\{.*?\})\s*-->", re.DOTALL)
_TABLE_RE = re.compile(r"^\|.+\|", re.MULTILINE)
_FLOW_RE = re.compile(r"\u2192|\u27f6|-->")

# -- Compatible chart type groups -------------------------------------------

_COMPATIBLE_GROUPS: List[frozenset] = [
    frozenset({"bar", "horizontal_bar"}),
    frozenset({"donut", "pie"}),
    frozenset({"line", "area"}),
    frozenset({"grouped_bar", "stacked_bar"}),
    frozenset({"line", "multi_line"}),
]


def _are_compatible_types(actual: str, expected: str) -> bool:
    """Check whether two chart types are in the same compatibility group."""
    if actual == expected:
        return True
    for group in _COMPATIBLE_GROUPS:
        if actual in group and expected in group:
            return True
    return False


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_viz_response(
    response: str,
    expected_chart_type: Optional[str],
    expects_chart: bool,
    expected_labels: Optional[List[str]] = None,
    expected_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Score a single model response on 5 visualization dimensions.

    Args:
        response: Raw model output text.
        expected_chart_type: The chart type the response should contain (or None).
        expects_chart: True if a VIZ directive is expected.
        expected_labels: Optional list of expected data labels.
        expected_values: Optional list of expected numeric values.

    Returns:
        Dict with: has_directive, parsed_spec, chart_type_correct, has_table,
        has_flow_analysis, data_accuracy, mode_correct, composite (0-100).
    """
    # Parse directive
    viz_match = _VIZ_RE.search(response)
    has_directive = viz_match is not None
    parsed_spec: Optional[Dict[str, Any]] = None

    if viz_match:
        try:
            parsed_spec = json.loads(viz_match.group(1))
            # Normalize field aliases — model may emit "type" instead of "chart_type"
            if parsed_spec and "chart_type" not in parsed_spec:
                for alias in ("type", "chart"):
                    if alias in parsed_spec:
                        parsed_spec["chart_type"] = parsed_spec.pop(alias)
                        break
        except json.JSONDecodeError:
            parsed_spec = None

    # Detect table and flow
    has_table = bool(_TABLE_RE.search(response))
    has_flow_analysis = bool(_FLOW_RE.search(response))

    # --- Dimension 1: Directive presence (0-100) ---
    if expects_chart:
        directive_score = 100.0 if has_directive else 0.0
    else:
        # No chart expected: directive absent is good, present is a penalty
        directive_score = 100.0 if not has_directive else 20.0

    # --- Dimension 2: Chart type accuracy (0-100) ---
    chart_type_correct = False
    if expects_chart and parsed_spec and expected_chart_type:
        actual_type = parsed_spec.get("chart_type", "")
        if actual_type == expected_chart_type:
            chart_type_correct = True
            chart_type_score = 100.0
        elif _are_compatible_types(actual_type, expected_chart_type):
            chart_type_score = 60.0  # partial credit
        else:
            chart_type_score = 0.0
    elif expects_chart and not has_directive:
        chart_type_score = 0.0
    elif not expects_chart:
        # No chart expected -- not applicable, give full marks
        chart_type_score = 100.0
    else:
        chart_type_score = 0.0

    # --- Dimension 3: Data accuracy (0-100) ---
    data_accuracy = _compute_data_accuracy(
        parsed_spec, expected_labels, expected_values, expects_chart,
    )

    # --- Dimension 4: Table formatting (0-100) ---
    if expects_chart:
        table_score = 100.0 if has_table else 30.0
    else:
        # Text-only or flow: table is a bonus but not required
        table_score = 100.0 if has_table else 70.0

    # --- Dimension 5: Mode selection (0-100) ---
    mode_correct = _check_mode(
        expects_chart, has_directive, has_flow_analysis,
        expected_chart_type,
    )
    mode_score = 100.0 if mode_correct else 0.0

    # --- Composite ---
    composite = (
        WEIGHT_DIRECTIVE * directive_score
        + WEIGHT_CHART_TYPE * chart_type_score
        + WEIGHT_DATA * data_accuracy
        + WEIGHT_TABLE * table_score
        + WEIGHT_MODE * mode_score
    )

    return {
        "has_directive": has_directive,
        "parsed_spec": parsed_spec,
        "chart_type_correct": chart_type_correct,
        "has_table": has_table,
        "has_flow_analysis": has_flow_analysis,
        "data_accuracy": round(data_accuracy, 1),
        "mode_correct": mode_correct,
        "composite": round(composite, 1),
    }


def _compute_data_accuracy(
    parsed_spec: Optional[Dict[str, Any]],
    expected_labels: Optional[List[str]],
    expected_values: Optional[List[float]],
    expects_chart: bool,
) -> float:
    """Compute data accuracy score (0-100)."""
    if not expects_chart:
        return 100.0  # not applicable for text-only

    if parsed_spec is None:
        return 0.0

    score = 0.0
    checks = 0

    # Label matching
    if expected_labels is not None:
        checks += 1
        actual_labels = parsed_spec.get("labels", [])
        expected_set = {l.lower() for l in expected_labels}
        actual_set = {l.lower() for l in actual_labels}
        if expected_set and actual_set:
            overlap = len(expected_set & actual_set)
            label_score = overlap / max(len(expected_set), len(actual_set))
            score += label_score
        elif not expected_set and not actual_set:
            score += 1.0

    # Value matching
    if expected_values is not None:
        checks += 1
        actual_values = parsed_spec.get("values", [])
        if expected_values and actual_values and len(expected_values) == len(actual_values):
            matches = sum(
                1 for a, e in zip(actual_values, expected_values)
                if abs(float(a) - float(e)) < 0.01 * max(abs(float(e)), 1)
            )
            score += matches / len(expected_values)
        elif not expected_values and not actual_values:
            score += 1.0

    if checks == 0:
        # No expected data provided -- check that parsed_spec has labels/values
        has_labels = bool(parsed_spec.get("labels"))
        has_values = bool(parsed_spec.get("values"))
        return 80.0 if (has_labels and has_values) else 40.0

    return (score / checks) * 100.0


def _check_mode(
    expects_chart: bool,
    has_directive: bool,
    has_flow_analysis: bool,
    expected_chart_type: Optional[str],
) -> bool:
    """Check whether the response mode (chart / flow / text) is correct."""
    if expects_chart:
        return has_directive
    # No chart expected
    if expected_chart_type == "flow":
        return has_flow_analysis and not has_directive
    # Text-only
    return not has_directive


# ---------------------------------------------------------------------------
# Evaluation set builder
# ---------------------------------------------------------------------------

def build_viz_eval_set() -> List[Dict[str, Any]]:
    """Build a synthetic evaluation set with 20+ items.

    Returns:
        List of dicts with: query, expects_chart, expected_chart_type,
        expected_labels, expected_values.
    """
    items: List[Dict[str, Any]] = []

    # ── Chart-expected queries (12+) ──────────────────────────────────────

    items.append({
        "query": "Show revenue by quarter.\n\n[EVIDENCE]\nQ1: $1.2M, Q2: $1.5M, Q3: $1.8M, Q4: $2.1M.",
        "expects_chart": True,
        "expected_chart_type": "bar",
        "expected_labels": ["Q1", "Q2", "Q3", "Q4"],
        "expected_values": [1.2, 1.5, 1.8, 2.1],
    })

    items.append({
        "query": "What is the expense breakdown?\n\n[EVIDENCE]\nSalaries: 55%, Cloud: 20%, Marketing: 12%, Office: 8%, Travel: 5%.",
        "expects_chart": True,
        "expected_chart_type": "donut",
        "expected_labels": ["Salaries", "Cloud", "Marketing", "Office", "Travel"],
        "expected_values": [55, 20, 12, 8, 5],
    })

    items.append({
        "query": "Show the monthly revenue trend.\n\n[EVIDENCE]\nJan $400K, Feb $420K, Mar $390K, Apr $450K, May $480K, Jun $510K.",
        "expects_chart": True,
        "expected_chart_type": "line",
        "expected_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "expected_values": [400, 420, 390, 450, 480, 510],
    })

    items.append({
        "query": "Compare Q1 and Q2 expenses by category.\n\n[EVIDENCE]\nSalaries Q1: $800K, Q2: $820K. Cloud Q1: $120K, Q2: $150K. Travel Q1: $45K, Q2: $60K.",
        "expects_chart": True,
        "expected_chart_type": "grouped_bar",
        "expected_labels": ["Salaries", "Cloud", "Travel"],
        "expected_values": [800, 120, 45],
    })

    items.append({
        "query": "Rate the vendor across criteria.\n\n[EVIDENCE]\nReliability: 9/10, Cost: 7/10, Support: 8/10, Innovation: 6/10, Compliance: 9/10.",
        "expects_chart": True,
        "expected_chart_type": "radar",
        "expected_labels": ["Reliability", "Cost", "Support", "Innovation", "Compliance"],
        "expected_values": [9, 7, 8, 6, 9],
    })

    items.append({
        "query": "Show department headcount.\n\n[EVIDENCE]\nEngineering: 85, Sales: 42, Marketing: 28, HR: 15, Finance: 22.",
        "expects_chart": True,
        "expected_chart_type": "horizontal_bar",
        "expected_labels": ["Engineering", "Sales", "Marketing", "HR", "Finance"],
        "expected_values": [85, 42, 28, 15, 22],
    })

    items.append({
        "query": "Show the profit and loss waterfall.\n\n[EVIDENCE]\nRevenue: $5.0M. COGS: -$2.0M. Gross Profit: $3.0M. Opex: -$1.5M. Net Profit: $1.1M.",
        "expects_chart": True,
        "expected_chart_type": "waterfall",
        "expected_labels": ["Revenue", "COGS", "Gross Profit", "Opex", "Net Profit"],
        "expected_values": [5.0, -2.0, 3.0, -1.5, 1.1],
    })

    items.append({
        "query": "Show cumulative spending over the year.\n\n[EVIDENCE]\nJan $50K, Feb $110K, Mar $175K, Apr $245K, May $320K, Jun $400K.",
        "expects_chart": True,
        "expected_chart_type": "area",
        "expected_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "expected_values": [50, 110, 175, 245, 320, 400],
    })

    items.append({
        "query": "Show budget allocation by department.\n\n[EVIDENCE]\nEngineering: $4.2M, Sales: $2.8M, Marketing: $1.5M, HR: $0.8M.",
        "expects_chart": True,
        "expected_chart_type": "treemap",
        "expected_labels": ["Engineering", "Sales", "Marketing", "HR"],
        "expected_values": [4.2, 2.8, 1.5, 0.8],
    })

    items.append({
        "query": "What is the SLA compliance rate?\n\n[EVIDENCE]\nCurrent SLA compliance: 97.3%. Target: 99.5%.",
        "expects_chart": True,
        "expected_chart_type": "gauge",
        "expected_labels": ["SLA Compliance"],
        "expected_values": [97.3],
    })

    items.append({
        "query": "Show document types distribution.\n\n[EVIDENCE]\nPDFs: 340, Word Docs: 180, Spreadsheets: 95, Images: 45, Other: 20.",
        "expects_chart": True,
        "expected_chart_type": "donut",
        "expected_labels": ["PDFs", "Word Docs", "Spreadsheets", "Images", "Other"],
        "expected_values": [340, 180, 95, 45, 20],
    })

    items.append({
        "query": "Compare revenue and costs over quarters.\n\n[EVIDENCE]\nQ1 revenue $1.2M cost $0.9M. Q2 revenue $1.5M cost $1.0M. Q3 revenue $1.8M cost $1.1M. Q4 revenue $2.1M cost $1.2M.",
        "expects_chart": True,
        "expected_chart_type": "multi_line",
        "expected_labels": ["Q1", "Q2", "Q3", "Q4"],
        "expected_values": [1.2, 1.5, 1.8, 2.1],
    })

    items.append({
        "query": "Plot deal size vs close time.\n\n[EVIDENCE]\nDeal A: $50K 15 days. Deal B: $120K 45 days. Deal C: $30K 8 days.",
        "expects_chart": True,
        "expected_chart_type": "scatter",
        "expected_labels": ["A", "B", "C"],
        "expected_values": [50, 120, 30],
    })

    items.append({
        "query": "Break down project costs by phase.\n\n[EVIDENCE]\nProject Alpha: Design $50K, Development $120K, Testing $30K. Project Beta: Design $40K, Development $90K, Testing $25K.",
        "expects_chart": True,
        "expected_chart_type": "stacked_bar",
        "expected_labels": ["Alpha", "Beta"],
        "expected_values": [50, 40],
    })

    # ── Text-only queries (6+) ────────────────────────────────────────────

    items.append({
        "query": "When does the contract expire?\n\n[EVIDENCE]\nContract effective January 1, 2025, term 24 months.",
        "expects_chart": False,
        "expected_chart_type": None,
        "expected_labels": None,
        "expected_values": None,
    })

    items.append({
        "query": "Who is the landlord?\n\n[EVIDENCE]\nLandlord is Riverside Properties LLC, 500 Harbor Drive, Miami FL.",
        "expects_chart": False,
        "expected_chart_type": None,
        "expected_labels": None,
        "expected_values": None,
    })

    items.append({
        "query": "What is the payment amount?\n\n[EVIDENCE]\nPayment: $15,000 to Johnson & Associates for consulting.",
        "expects_chart": False,
        "expected_chart_type": None,
        "expected_labels": None,
        "expected_values": None,
    })

    items.append({
        "query": "How many pages is the report?\n\n[EVIDENCE]\nCompliance_Report_2025.pdf, 63 pages.",
        "expects_chart": False,
        "expected_chart_type": None,
        "expected_labels": None,
        "expected_values": None,
    })

    items.append({
        "query": "What is the warranty period?\n\n[EVIDENCE]\nProduct warranty: 36 months from date of delivery.",
        "expects_chart": False,
        "expected_chart_type": None,
        "expected_labels": None,
        "expected_values": None,
    })

    items.append({
        "query": "Who signed the agreement?\n\n[EVIDENCE]\nSigned by Maria Torres (CEO) and James Park (VP Sales) on March 15, 2025.",
        "expects_chart": False,
        "expected_chart_type": None,
        "expected_labels": None,
        "expected_values": None,
    })

    # ── Flow analysis queries (2+) ────────────────────────────────────────

    items.append({
        "query": "Describe the invoice approval process.\n\n[EVIDENCE]\nInvoice submitted by vendor. Finance validates. Manager approves if under $10K. Director approval above $10K. Payment within Net 30.",
        "expects_chart": False,
        "expected_chart_type": "flow",
        "expected_labels": None,
        "expected_values": None,
    })

    items.append({
        "query": "How does the document onboarding pipeline work?\n\n[EVIDENCE]\nUser uploads file. System extracts text. Content chunked. Embeddings generated. Vectors stored.",
        "expects_chart": False,
        "expected_chart_type": "flow",
        "expected_labels": None,
        "expected_values": None,
    })

    items.append({
        "query": "What is the employee offboarding procedure?\n\n[EVIDENCE]\nManager initiates separation. HR schedules exit interview. IT revokes access. Final paycheck processed. Equipment returned.",
        "expects_chart": False,
        "expected_chart_type": "flow",
        "expected_labels": None,
        "expected_values": None,
    })

    return items


# ---------------------------------------------------------------------------
# Model call helper
# ---------------------------------------------------------------------------

def _call_model(model_name: str, query: str) -> Optional[str]:
    """Call the model via ollama and return raw response text."""
    try:
        import ollama as _ollama
    except ImportError:
        log.error("ollama package not installed")
        return None

    from src.finetune.viz_training_data import SYSTEM_PROMPT

    try:
        response = _ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.1, "num_predict": 1024},
        )
        return response.get("message", {}).get("content", "")
    except Exception as exc:
        log.debug("Model call failed for %r: %s", query[:50], exc)
        return None


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_viz_evaluation(
    model_name: str = "DocWain:latest",
    eval_set: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the visualization evaluation harness against a model.

    Args:
        model_name: Ollama model name to evaluate.
        eval_set: Optional custom eval set. Defaults to build_viz_eval_set().

    Returns:
        Dict with: composite_score, verdict (pass/marginal/fail),
        chart_avg, text_avg, per_example.
    """
    if eval_set is None:
        eval_set = build_viz_eval_set()

    per_example: List[Dict[str, Any]] = []
    chart_scores: List[float] = []
    text_scores: List[float] = []

    for item in eval_set:
        query = item["query"]
        response = _call_model(model_name, query)

        if response is None:
            result = {
                "has_directive": False,
                "parsed_spec": None,
                "chart_type_correct": False,
                "has_table": False,
                "has_flow_analysis": False,
                "data_accuracy": 0.0,
                "mode_correct": False,
                "composite": 0.0,
            }
        else:
            result = score_viz_response(
                response=response,
                expected_chart_type=item.get("expected_chart_type"),
                expects_chart=item["expects_chart"],
                expected_labels=item.get("expected_labels"),
                expected_values=item.get("expected_values"),
            )

        entry = {
            "query": query,
            "expects_chart": item["expects_chart"],
            "expected_chart_type": item.get("expected_chart_type"),
            "score": result,
        }
        per_example.append(entry)

        if item["expects_chart"]:
            chart_scores.append(result["composite"])
        else:
            text_scores.append(result["composite"])

    # Aggregate
    all_scores = [e["score"]["composite"] for e in per_example]
    composite = sum(all_scores) / max(len(all_scores), 1)
    chart_avg = sum(chart_scores) / max(len(chart_scores), 1) if chart_scores else 0.0
    text_avg = sum(text_scores) / max(len(text_scores), 1) if text_scores else 0.0

    if composite >= PASS_COMPOSITE:
        verdict = "pass"
    elif composite >= MARGINAL_COMPOSITE:
        verdict = "marginal"
    else:
        verdict = "fail"

    log.info(
        "Viz evaluation: composite=%.1f chart_avg=%.1f text_avg=%.1f -> %s",
        composite, chart_avg, text_avg, verdict,
    )

    return {
        "composite_score": round(composite, 1),
        "verdict": verdict,
        "chart_avg": round(chart_avg, 1),
        "text_avg": round(text_avg, 1),
        "total_examples": len(per_example),
        "per_example": per_example,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DocWain visualization fine-tuned model")
    parser.add_argument("--model", default="DocWain:latest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = run_viz_evaluation(model_name=args.model)

    print(f"\n{'=' * 50}")
    print(f"Verdict:     {result['verdict'].upper()}")
    print(f"Composite:   {result['composite_score']:.1f}")
    print(f"Chart avg:   {result['chart_avg']:.1f}")
    print(f"Text avg:    {result['text_avg']:.1f}")
    print(f"Examples:    {result['total_examples']}")
    print(f"{'=' * 50}")

    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
