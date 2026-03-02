#!/usr/bin/env python3
"""Recursive UAT runner.

Wraps uat_comprehensive.py in a loop, running up to MAX_ROUNDS until
the target quality threshold is met.

Usage:
    python scripts/recursive_uat.py [--max-rounds 5] [--target-score 95] [--target-pass-rate 100]
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recursive_uat")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UAT_SCRIPT = PROJECT_ROOT / "scripts" / "uat_comprehensive.py"
UAT_RESULTS_FILE = PROJECT_ROOT / "tests" / "uat_results.json"
RECURSIVE_REPORT_FILE = PROJECT_ROOT / "tests" / "recursive_uat_report.json"


def run_uat_round(round_num: int) -> Dict[str, Any]:
    """Run one UAT round and return parsed results."""
    logger.info("=" * 60)
    logger.info("ROUND %d: Starting UAT", round_num)
    logger.info("=" * 60)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(UAT_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )
        elapsed = time.time() - start
        logger.info("Round %d completed in %.1fs (exit code: %d)", round_num, elapsed, result.returncode)

        if result.returncode != 0:
            logger.warning("UAT script returned non-zero exit code: %d", result.returncode)
            if result.stderr:
                logger.warning("STDERR: %s", result.stderr[:500])
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        logger.error("Round %d timed out after 1800s", round_num)
        return {"round": round_num, "status": "timeout", "elapsed_s": round(elapsed, 1)}
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("Round %d failed: %s", round_num, exc)
        return {"round": round_num, "status": "error", "error": str(exc), "elapsed_s": round(elapsed, 1)}

    # Parse results from the JSON output file
    return parse_uat_results(round_num, elapsed)


def parse_uat_results(round_num: int, elapsed: float) -> Dict[str, Any]:
    """Parse UAT results from tests/uat_results.json.

    The UAT script writes a JSON file with this structure:
    {
        "timestamp": "...",
        "total": int,
        "passed": int,
        "failed": int,
        "grades": {"A": int, "B": int, ...},
        "results": [
            {"id": "...", "pass": bool, "grade": "A", "score": int, ...},
            ...
        ]
    }
    """
    report: Dict[str, Any] = {
        "round": round_num,
        "status": "completed",
        "elapsed_s": round(elapsed, 1),
    }

    if not UAT_RESULTS_FILE.exists():
        logger.warning("UAT results file not found: %s", UAT_RESULTS_FILE)
        report["status"] = "no_results_file"
        return report

    try:
        data = json.loads(UAT_RESULTS_FILE.read_text())
    except Exception as exc:
        logger.warning("Failed to parse UAT results: %s", exc)
        report["status"] = "parse_error"
        return report

    # Use pre-computed summary fields from the UAT output when available
    total = data.get("total", 0)
    passed = data.get("passed", 0)
    raw_grades = data.get("grades", {})
    results_list = data.get("results", [])

    # Normalize grades dict to always include A/B/C/D/F keys
    grades: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for g, count in raw_grades.items():
        key = g.upper()
        if key in grades:
            grades[key] = count

    # Compute scores from individual results (latency is not saved to JSON)
    scores: List[float] = []
    failures: List[Dict[str, str]] = []

    for r in results_list:
        if not isinstance(r, dict):
            continue

        score = r.get("score", 0)
        if isinstance(score, (int, float)):
            scores.append(float(score))

        if not r.get("pass", False):
            failures.append({
                "id": str(r.get("id", "unknown")),
                "grade": str(r.get("grade", "?")),
                "reason": str(r.get("reason", "low keyword match"))[:200],
            })

    # Recompute total/passed from results list if top-level fields are missing
    if total == 0 and results_list:
        total = len(results_list)
        passed = sum(1 for r in results_list if isinstance(r, dict) and r.get("pass", False))

    pass_rate = (passed / total * 100) if total > 0 else 0
    avg_score = (sum(scores) / len(scores)) if scores else 0

    report.update({
        "total": total,
        "passed": passed,
        "pass_rate": round(pass_rate, 1),
        "grades": grades,
        "avg_score": round(avg_score, 1),
        "failures": failures,
        "timestamp": data.get("timestamp", ""),
    })

    # Log summary
    logger.info("-" * 40)
    logger.info("ROUND %d RESULTS:", round_num)
    logger.info("  Pass: %d/%d (%.1f%%)", passed, total, pass_rate)
    logger.info("  Grades: A=%d B=%d C=%d D=%d F=%d",
                grades["A"], grades["B"], grades["C"], grades["D"], grades["F"])
    logger.info("  Avg Score: %.1f/100", avg_score)
    logger.info("  Elapsed: %.1fs", elapsed)
    if failures:
        logger.info("  Failures:")
        for f in failures:
            logger.info("    - %s (grade=%s): %s", f["id"], f["grade"], f["reason"][:80])
    logger.info("-" * 40)

    return report


def meets_target(report: Dict[str, Any], target_score: float, target_pass_rate: float) -> bool:
    """Check if this round meets the quality target."""
    if report.get("status") != "completed":
        return False
    return (
        report.get("avg_score", 0) >= target_score
        and report.get("pass_rate", 0) >= target_pass_rate
    )


def main():
    parser = argparse.ArgumentParser(description="Recursive UAT Runner")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum UAT rounds (default: 5)")
    parser.add_argument("--target-score", type=float, default=95.0, help="Target avg score (default: 95)")
    parser.add_argument("--target-pass-rate", type=float, default=100.0, help="Target pass rate %% (default: 100)")
    args = parser.parse_args()

    logger.info("Recursive UAT: max_rounds=%d, target_score=%.0f, target_pass_rate=%.0f%%",
                args.max_rounds, args.target_score, args.target_pass_rate)

    all_rounds: List[Dict[str, Any]] = []
    target_met = False

    for round_num in range(1, args.max_rounds + 1):
        report = run_uat_round(round_num)
        all_rounds.append(report)

        target_met = meets_target(report, args.target_score, args.target_pass_rate)

        # Save intermediate results after each round
        try:
            RECURSIVE_REPORT_FILE.write_text(json.dumps({
                "config": {
                    "max_rounds": args.max_rounds,
                    "target_score": args.target_score,
                    "target_pass_rate": args.target_pass_rate,
                },
                "rounds": all_rounds,
                "current_round": round_num,
                "target_met": target_met,
            }, indent=2))
        except Exception:
            pass

        if target_met:
            logger.info("TARGET MET in round %d! Score=%.1f, PassRate=%.1f%%",
                        round_num, report.get("avg_score", 0), report.get("pass_rate", 0))
            break
        else:
            logger.info("Target NOT met. Score=%.1f (need %.1f), PassRate=%.1f%% (need %.1f%%)",
                        report.get("avg_score", 0), args.target_score,
                        report.get("pass_rate", 0), args.target_pass_rate)
            if round_num < args.max_rounds:
                logger.info("Starting next round in 5s...")
                time.sleep(5)

    # Final summary
    logger.info("=" * 60)
    logger.info("RECURSIVE UAT COMPLETE: %d round(s)", len(all_rounds))
    logger.info("Target met: %s", target_met)
    for r in all_rounds:
        grades = r.get("grades", {})
        grade_str = " ".join(f"{g}={grades.get(g, 0)}" for g in ("A", "B", "C", "D", "F"))
        logger.info("  Round %d: %s | Pass=%d/%d (%.1f%%) | AvgScore=%.1f | %s | %.1fs",
                     r.get("round", 0), r.get("status", "?"),
                     r.get("passed", 0), r.get("total", 0), r.get("pass_rate", 0),
                     r.get("avg_score", 0), grade_str, r.get("elapsed_s", 0))
    logger.info("=" * 60)

    sys.exit(0 if target_met else 1)


if __name__ == "__main__":
    main()
