#!/usr/bin/env python3
"""Iterative Optimization Harness for DocWain RAG Pipeline.

Generates dynamic test cases, evaluates quality + speed, identifies gaps,
and tracks iteration-over-iteration progress. If 5+ iterations show no
improvement, signals to escalate to model fine-tuning.

Usage:
    python scripts/iterative_optimizer.py [--iteration N] [--max-iterations 10]
"""

import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ── Configuration ──────────────────────────────────────────────────────────
BASE_URL = os.getenv("DOCWAIN_URL", "http://localhost:8000")
TIMEOUT = 600.0

# Existing profiles with real data
PROFILES = {
    "hr": {
        "subscription_id": "67fde0754e36c00b14cea7f5",
        "profile_id": "69a6dddc23d47adc8e7ee7e4",
        "domain": "hr",
        "has_data": True,
    },
    "medical": {
        "subscription_id": "67fde0754e36c00b14cea7f5",
        "profile_id": "69a552ac23d47adc8e7ee27a",
        "domain": "medical",
        "has_data": True,
    },
    "invoice": {
        "subscription_id": "67e6920588f8ff4644d2dfb1",
        "profile_id": "69a6de9e23d47adc8e7ee90a",
        "domain": "invoice",
        "has_data": True,
    },
}

# ── Grading ─────────────────────────────────────────────────────────────────

BANNED_PATTERNS = [
    "not explicitly mentioned", "i don't have access", "i cannot",
    "i'm unable", "as an ai assistant", "as an ai language model",
    "MISSING_REASON", "Not enough information", "unfortunately, i",
    "tool:resumes", "tool:medical", "tool:insights",
    "section_id", "chunk_type", "page_start",
]

INTEL_SIGNALS = [
    "across", "average", "range", "total", "compared", "pattern",
    "common", "unique", "distribution", "highest", "lowest", "overview",
    "analyzed", "shared", "between", "versus", "whereas",
]


@dataclass
class TestCase:
    id: str
    query: str
    profile: str  # key into PROFILES
    category: str  # hr_basic, medical_analytical, cross_doc, etc.
    expect_keywords: List[str]
    expect_format: str = "any"  # any, table, bullets, numbered, json
    expect_min_length: int = 80
    tools: Optional[List[str]] = None
    enable_internet: bool = False
    agent_mode: bool = False
    max_latency_s: float = 120.0  # timeout expectation
    difficulty: str = "medium"  # easy, medium, hard, extreme


@dataclass
class TestResult:
    test_id: str
    grade: str  # A/B/C/D/F
    score: int
    latency_s: float
    response_length: int
    keywords_found: List[str]
    keywords_missing: List[str]
    banned_found: List[str]
    intel_signals: List[str]
    format_ok: bool
    timed_out: bool
    error: str = ""
    preview: str = ""


@dataclass
class IterationReport:
    iteration: int
    timestamp: str
    total_tests: int
    pass_count: int  # A+B
    pass_rate: float
    grade_dist: Dict[str, int]
    avg_latency_s: float
    avg_score: float
    timeout_count: int
    category_scores: Dict[str, float]
    category_latencies: Dict[str, float]
    failures: List[Dict]
    improvements_from_prev: Dict[str, Any] = field(default_factory=dict)


# ── API Client ──────────────────────────────────────────────────────────────

def ask(query: str, profile_key: str, tools=None, enable_internet=False, agent_mode=False) -> Dict:
    """Send a query to DocWain API and return the response."""
    prof = PROFILES[profile_key]
    payload = {
        "query": query,
        "profile_id": prof["profile_id"],
        "subscription_id": prof["subscription_id"],
        "user_id": "iterative_optimizer@docwain.ai",
        "enable_internet": enable_internet,
    }
    if tools:
        payload["tools"] = tools
        payload["use_tools"] = True
    if agent_mode:
        payload["agent_mode"] = agent_mode

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/api/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Grading ─────────────────────────────────────────────────────────────────

def grade_response(text: str, tc: TestCase, latency_s: float) -> TestResult:
    """Grade a response on multiple dimensions."""
    lower = text.lower()
    length = len(text)

    found = [k for k in tc.expect_keywords if k.lower() in lower]
    missing = [k for k in tc.expect_keywords if k.lower() not in lower]
    banned = [p for p in BANNED_PATTERNS if p.lower() in lower]
    intel = [s for s in INTEL_SIGNALS if s in lower]

    # Keyword coverage score
    ratio = len(found) / max(len(tc.expect_keywords), 1)
    score = 0
    if ratio >= 0.5:
        score += 30
    if ratio >= 1.0:
        score += 20
    if length > tc.expect_min_length:
        score += 15
    if length > 300:
        score += 5
    if not banned:
        score += 15
    if intel:
        score += 15

    # Format compliance check
    format_ok = True
    if tc.expect_format == "table":
        format_ok = "|" in text and "---" in text
        if format_ok:
            score += 5
    elif tc.expect_format == "bullets":
        format_ok = bool(re.search(r"^[\-\*•]", text, re.MULTILINE))
        if format_ok:
            score += 5
    elif tc.expect_format == "numbered":
        format_ok = bool(re.search(r"^\d+\.", text, re.MULTILINE))
        if format_ok:
            score += 5

    # Latency penalty for slow responses (over max_latency_s)
    timed_out = latency_s > tc.max_latency_s
    if timed_out:
        score = max(score - 10, 0)

    # Speed bonus for fast responses
    if latency_s < 30:
        score = min(score + 5, 100)

    grade = (
        "A" if score >= 90 else
        "B" if score >= 75 else
        "C" if score >= 60 else
        "D" if score >= 40 else
        "F"
    )

    return TestResult(
        test_id=tc.id,
        grade=grade,
        score=score,
        latency_s=latency_s,
        response_length=length,
        keywords_found=found,
        keywords_missing=missing,
        banned_found=banned,
        intel_signals=intel,
        format_ok=format_ok,
        timed_out=timed_out,
        preview=text[:200],
    )


# ── Dynamic Test Generation ────────────────────────────────────────────────

def generate_test_suite(iteration: int = 1, prev_failures: List[Dict] = None) -> List[TestCase]:
    """Generate a diverse test suite. Adapts based on previous failures."""
    tests: List[TestCase] = []
    _id = 0

    def _next_id(prefix: str) -> str:
        nonlocal _id
        _id += 1
        return f"I{iteration}-{prefix}-{_id:02d}"

    # ── TIER 1: Foundational Retrieval (any domain, any phrasing) ────
    tier1_hr = [
        ("List all candidates with their key skills", ["candidate", "skill"]),
        ("Who has Python experience?", ["python"]),
        ("Show me education background of candidates", ["education"]),
        ("What contact info is available for the candidates?", ["contact"]),
        ("Tell me about the candidates' work history", ["experience"]),
        ("Which candidate knows machine learning?", ["machine learning"]),
        ("Summarize candidate profiles briefly", ["candidate"]),
        ("What are the top skills across all resumes?", ["skill"]),
    ]
    for q, kw in tier1_hr:
        tests.append(TestCase(
            id=_next_id("HR"), query=q, profile="hr",
            category="hr_retrieval", expect_keywords=kw,
            tools=["resumes"], max_latency_s=90,
        ))

    tier1_med = [
        ("What diagnoses are recorded?", ["diagnos"]),
        ("List medications prescribed to patients", ["medicat"]),
        ("What lab tests were done?", ["lab"]),
        ("Summarize the patient medical records", ["patient"]),
        ("Any procedures or treatments documented?", ["treatment"]),
    ]
    for q, kw in tier1_med:
        tests.append(TestCase(
            id=_next_id("MED"), query=q, profile="medical",
            category="medical_retrieval", expect_keywords=kw,
            tools=["medical"], max_latency_s=90,
        ))

    tier1_inv = [
        ("What are the invoice totals?", ["invoice"]),
        ("List all vendors in the invoices", ["vendor"]),
        ("Show invoice dates and numbers", ["invoice"]),
        ("What items appear on the invoices?", ["item"]),
        ("Summarize invoice data", ["invoice"]),
    ]
    for q, kw in tier1_inv:
        tests.append(TestCase(
            id=_next_id("INV"), query=q, profile="invoice",
            category="invoice_retrieval", expect_keywords=kw,
            max_latency_s=120,
        ))

    # ── TIER 2: Analytical + Reasoning ───────────────────────────────
    tier2 = [
        ("Compare all candidates for a senior backend developer position",
         "hr", ["candidate"], ["resumes"], "comparison"),
        ("Rank candidates by relevance for a data science role",
         "hr", ["candidate"], ["resumes"], "ranking"),
        ("What patterns emerge across all candidate profiles?",
         "hr", ["pattern"], ["insights"], "analytics"),
        ("Analyze career progression of each candidate",
         "hr", ["experience"], ["resumes"], "reasoning"),
        ("What unique skills does each candidate bring?",
         "hr", ["skill"], ["resumes"], "reasoning"),
        ("Identify potential health risks from the medical records",
         "medical", ["risk"], ["medical"], "reasoning"),
        ("What spending patterns can you identify from invoices?",
         "invoice", ["pattern"], None, "analytics"),
    ]
    for q, prof, kw, tools, cat in tier2:
        tests.append(TestCase(
            id=_next_id("ANLY"), query=q, profile=prof,
            category=f"{prof}_{cat}", expect_keywords=kw,
            tools=tools, max_latency_s=180, difficulty="hard",
        ))

    # ── TIER 3: Cross-document Intelligence ──────────────────────────
    tier3 = [
        ("What skills appear in every single resume?", "hr", ["skill"]),
        ("What is the average experience level across candidates?", "hr", ["experience"]),
        ("Which candidates have the most overlapping skill sets?", "hr", ["skill"]),
        ("Key differences between the top two candidates?", "hr", ["candidate"]),
        ("Summarize total invoice amounts by vendor", "invoice", ["vendor"]),
    ]
    for q, prof, kw in tier3:
        tests.append(TestCase(
            id=_next_id("XDOC"), query=q, profile=prof,
            category="cross_document", expect_keywords=kw,
            max_latency_s=180, difficulty="hard",
        ))

    # ── TIER 4: Format Compliance ────────────────────────────────────
    format_tests = [
        ("Create a skills comparison table for all candidates", "hr",
         ["skill"], "table", ["resumes"]),
        ("List key findings from medical records as bullet points", "medical",
         ["patient"], "bullets", ["medical"]),
        ("Rank all candidates in numbered order by experience", "hr",
         ["candidate"], "numbered", ["resumes"]),
    ]
    for q, prof, kw, fmt, tools in format_tests:
        tests.append(TestCase(
            id=_next_id("FMT"), query=q, profile=prof,
            category="format_compliance", expect_keywords=kw,
            expect_format=fmt, tools=tools, max_latency_s=120,
        ))

    # ── TIER 5: Content Generation ───────────────────────────────────
    gen_tests = [
        ("Draft interview questions for the strongest candidate",
         "hr", ["question"], ["resumes"]),
        ("Write a summary email about the top candidates",
         "hr", ["candidate"], None),
        ("Generate a job description based on candidate skills found",
         "hr", ["skill"], None),
    ]
    for q, prof, kw, tools in gen_tests:
        tests.append(TestCase(
            id=_next_id("GEN"), query=q, profile=prof,
            category="content_generation", expect_keywords=kw,
            tools=tools, max_latency_s=120,
        ))

    # ── TIER 6: Novel Phrasings + Implicit Intents ───────────────────
    novel_tests = [
        ("Can you give me a quick rundown of everyone's backgrounds?",
         "hr", ["experience"], ["resumes"]),
        ("Who would you pick for a Python role and why?",
         "hr", ["python"], ["resumes"]),
        ("Break down the invoice data for me", "invoice", ["invoice"], None),
        ("I need to understand the patient's condition", "medical",
         ["patient"], ["medical"]),
        ("What should I know about these candidates?", "hr",
         ["candidate"], ["resumes"]),
        ("Give me the highlights from all documents", "hr",
         ["candidate"], None),
    ]
    for q, prof, kw, tools in novel_tests:
        tests.append(TestCase(
            id=_next_id("NOVEL"), query=q, profile=prof,
            category="novel_phrasing", expect_keywords=kw,
            tools=tools, max_latency_s=120,
        ))

    # ── TIER 7: Translation ──────────────────────────────────────────
    trans_tests = [
        ("Translate the candidate skills overview to Spanish",
         "hr", ["skill"]),
        ("Convert the medical summary to French",
         "medical", ["patient"]),
    ]
    for q, prof, kw in trans_tests:
        tests.append(TestCase(
            id=_next_id("TRANS"), query=q, profile=prof,
            category="translation", expect_keywords=kw,
            tools=["translator"], max_latency_s=240,
        ))

    # ── TIER 8: Extract (structured data retrieval) ──────────────────
    extract_tests = [
        ("Extract all email addresses and phone numbers from resumes",
         "hr", ["email"]),
        ("Extract all company names from the resumes", "hr", ["company"]),
        ("Extract all dates from the invoices", "invoice", ["date"]),
        ("List all certifications with candidate names", "hr", ["certification"]),
    ]
    for q, prof, kw in extract_tests:
        tests.append(TestCase(
            id=_next_id("EXT"), query=q, profile=prof,
            category="extraction", expect_keywords=kw,
            max_latency_s=120,
        ))

    # ── TIER 9: Latency-Critical Fast-Path Tests ─────────────────────
    fast_tests = [
        ("How many documents are in this profile?", "hr",
         ["document"], 30),
        ("Hello, what can you help me with?", "hr",
         ["docwain"], 10),
        ("What types of files can I upload?", "hr",
         ["file"], 10),
    ]
    for q, prof, kw, max_lat in fast_tests:
        tests.append(TestCase(
            id=_next_id("FAST"), query=q, profile=prof,
            category="fast_path", expect_keywords=kw,
            max_latency_s=max_lat, difficulty="easy",
        ))

    # ── TIER 10: Adaptive — harder tests based on previous failures ──
    if prev_failures:
        # Generate targeted tests for failing categories
        failing_categories = Counter(f.get("category", "") for f in prev_failures)
        for cat, count in failing_categories.most_common(3):
            if "hr" in cat:
                tests.append(TestCase(
                    id=_next_id("RETRY"), query="Give me a detailed analysis of all candidate qualifications and rank them",
                    profile="hr", category=f"retry_{cat}",
                    expect_keywords=["candidate", "skill"],
                    tools=["resumes"], max_latency_s=180, difficulty="hard",
                ))
            elif "medical" in cat:
                tests.append(TestCase(
                    id=_next_id("RETRY"), query="Provide a comprehensive patient health overview",
                    profile="medical", category=f"retry_{cat}",
                    expect_keywords=["patient", "diagnos"],
                    tools=["medical"], max_latency_s=180, difficulty="hard",
                ))
            elif "invoice" in cat:
                tests.append(TestCase(
                    id=_next_id("RETRY"), query="Analyze all invoice data and identify key financial patterns",
                    profile="invoice", category=f"retry_{cat}",
                    expect_keywords=["invoice"],
                    max_latency_s=180, difficulty="hard",
                ))

        # Generate tests for high-latency queries
        slow_cats = [f.get("category") for f in prev_failures if f.get("latency_s", 0) > 120]
        if slow_cats:
            # Add simpler versions of slow queries to test speed improvement
            tests.append(TestCase(
                id=_next_id("SPEED"), query="List candidate names only",
                profile="hr", category="speed_test",
                expect_keywords=["candidate"],
                tools=["resumes"], max_latency_s=45, difficulty="easy",
            ))

    random.shuffle(tests)
    return tests


# ── Run Tests ───────────────────────────────────────────────────────────────

def run_test_suite(tests: List[TestCase], iteration: int) -> IterationReport:
    """Run all tests and produce an iteration report."""
    results: List[TestResult] = []
    total_start = time.time()

    print(f"\n{'='*80}")
    print(f"  ITERATION {iteration} — {len(tests)} tests — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    for i, tc in enumerate(tests, 1):
        sys.stdout.write(f"  [{i}/{len(tests)}] {tc.id}: {tc.query[:55]}...")
        sys.stdout.flush()

        start = time.time()
        try:
            resp = ask(
                query=tc.query,
                profile_key=tc.profile,
                tools=tc.tools,
                enable_internet=tc.enable_internet,
                agent_mode=tc.agent_mode,
            )
            elapsed = time.time() - start
            text = _extract_text(resp)
            result = grade_response(text, tc, elapsed)
            sources = resp.get("answer", {}).get("sources", [])
            result_info = f" [{result.grade}] {elapsed:.1f}s | {result.response_length} chars | {len(sources)} src"
        except httpx.TimeoutException:
            elapsed = time.time() - start
            result = TestResult(
                test_id=tc.id, grade="F", score=0, latency_s=elapsed,
                response_length=0, keywords_found=[], keywords_missing=tc.expect_keywords,
                banned_found=[], intel_signals=[], format_ok=False, timed_out=True,
                error="TIMEOUT",
            )
            result_info = f" [F] TIMEOUT after {elapsed:.1f}s"
        except Exception as exc:
            elapsed = time.time() - start
            result = TestResult(
                test_id=tc.id, grade="F", score=0, latency_s=elapsed,
                response_length=0, keywords_found=[], keywords_missing=tc.expect_keywords,
                banned_found=[], intel_signals=[], format_ok=False, timed_out=False,
                error=str(exc)[:200],
            )
            result_info = f" [F] ERROR: {str(exc)[:60]}"

        print(result_info)
        results.append(result)

    # ── Build report ────────────────────────────────────────────────
    total_time = time.time() - total_start
    grade_dist = Counter(r.grade for r in results)
    pass_count = grade_dist.get("A", 0) + grade_dist.get("B", 0)
    pass_rate = pass_count / max(len(results), 1)
    avg_latency = sum(r.latency_s for r in results) / max(len(results), 1)
    avg_score = sum(r.score for r in results) / max(len(results), 1)
    timeout_count = sum(1 for r in results if r.timed_out)

    # Category breakdowns
    cat_scores: Dict[str, List[int]] = defaultdict(list)
    cat_lats: Dict[str, List[float]] = defaultdict(list)
    for tc, r in zip(tests, results):
        cat_scores[tc.category].append(r.score)
        cat_lats[tc.category].append(r.latency_s)

    cat_avg_scores = {c: sum(s)/len(s) for c, s in cat_scores.items()}
    cat_avg_lats = {c: sum(l)/len(l) for c, l in cat_lats.items()}

    failures = []
    for tc, r in zip(tests, results):
        if r.grade in ("D", "F"):
            failures.append({
                "test_id": r.test_id,
                "query": tc.query,
                "category": tc.category,
                "grade": r.grade,
                "score": r.score,
                "latency_s": r.latency_s,
                "missing": r.keywords_missing,
                "banned": r.banned_found,
                "timed_out": r.timed_out,
                "error": r.error,
                "preview": r.preview,
            })

    report = IterationReport(
        iteration=iteration,
        timestamp=datetime.now().isoformat(),
        total_tests=len(results),
        pass_count=pass_count,
        pass_rate=pass_rate,
        grade_dist=dict(grade_dist),
        avg_latency_s=round(avg_latency, 1),
        avg_score=round(avg_score, 1),
        timeout_count=timeout_count,
        category_scores=cat_avg_scores,
        category_latencies=cat_avg_lats,
        failures=failures,
    )

    # ── Print summary ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ITERATION {iteration} RESULTS")
    print(f"{'='*80}")
    print(f"  TOTAL: {len(results)} | PASS(A/B): {pass_count} ({pass_rate:.1%})")
    print(f"  GRADES: A={grade_dist.get('A',0)}, B={grade_dist.get('B',0)}, "
          f"C={grade_dist.get('C',0)}, D={grade_dist.get('D',0)}, F={grade_dist.get('F',0)}")
    print(f"  AVG SCORE: {avg_score:.1f}/100 | AVG LATENCY: {avg_latency:.1f}s | TIMEOUTS: {timeout_count}")
    print(f"  TOTAL TIME: {total_time:.0f}s")

    if cat_avg_scores:
        print(f"\n  Category breakdown:")
        for cat in sorted(cat_avg_scores, key=cat_avg_scores.get, reverse=True):
            print(f"    {cat:30s} score={cat_avg_scores[cat]:.0f}  lat={cat_avg_lats[cat]:.1f}s")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures[:10]:
            print(f"    {f['test_id']}: [{f['grade']}] {f.get('error') or 'Missing: ' + str(f['missing'][:3])}")
            if f['preview']:
                print(f"      Preview: {f['preview'][:120]}")

    return report


def _extract_text(resp: Dict) -> str:
    """Extract response text from API response."""
    answer = resp.get("answer", {})
    if isinstance(answer, dict):
        return answer.get("response", "") or ""
    return str(answer or "")


# ── Iteration Controller ────────────────────────────────────────────────────

def save_report(report: IterationReport, output_dir: Path):
    """Save iteration report to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"iteration_{report.iteration}.json"
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\n  Report saved: {path}")
    return path


def compare_iterations(current: IterationReport, prev: Optional[IterationReport]) -> Dict:
    """Compare current iteration against previous."""
    if prev is None:
        return {"is_first": True}

    improvements = {
        "pass_rate_delta": current.pass_rate - prev.pass_rate,
        "avg_score_delta": current.avg_score - prev.avg_score,
        "avg_latency_delta": current.avg_latency_s - prev.avg_latency_s,
        "timeout_delta": current.timeout_count - prev.timeout_count,
        "is_improving": (
            current.pass_rate >= prev.pass_rate
            and current.avg_score >= prev.avg_score
        ),
        "latency_improving": current.avg_latency_s < prev.avg_latency_s,
    }

    print(f"\n  Comparison vs Iteration {prev.iteration}:")
    print(f"    Pass rate:  {prev.pass_rate:.1%} → {current.pass_rate:.1%} ({improvements['pass_rate_delta']:+.1%})")
    print(f"    Avg score:  {prev.avg_score:.1f} → {current.avg_score:.1f} ({improvements['avg_score_delta']:+.1f})")
    print(f"    Avg latency: {prev.avg_latency_s:.1f}s → {current.avg_latency_s:.1f}s ({improvements['avg_latency_delta']:+.1f}s)")
    print(f"    Timeouts:   {prev.timeout_count} → {current.timeout_count} ({improvements['timeout_delta']:+d})")

    if improvements["is_improving"]:
        print(f"    ✓ Quality IMPROVING")
    else:
        print(f"    ✗ Quality NOT improving")

    if improvements["latency_improving"]:
        print(f"    ✓ Latency IMPROVING")
    else:
        print(f"    ✗ Latency NOT improving")

    return improvements


def analyze_gaps(report: IterationReport) -> Dict:
    """Analyze what's failing and why, to guide next iteration."""
    analysis = {
        "weak_categories": [],
        "latency_bottlenecks": [],
        "quality_issues": [],
        "recommendations": [],
    }

    # Find weak categories (below 70 avg score)
    for cat, score in report.category_scores.items():
        if score < 70:
            analysis["weak_categories"].append({
                "category": cat,
                "avg_score": score,
                "avg_latency": report.category_latencies.get(cat, 0),
            })

    # Find latency bottlenecks (categories > 90s average)
    for cat, lat in report.category_latencies.items():
        if lat > 90:
            analysis["latency_bottlenecks"].append({
                "category": cat,
                "avg_latency": lat,
                "avg_score": report.category_scores.get(cat, 0),
            })

    # Analyze failure patterns
    timeout_failures = [f for f in report.failures if f.get("timed_out")]
    keyword_failures = [f for f in report.failures if f.get("missing") and not f.get("timed_out")]
    banned_failures = [f for f in report.failures if f.get("banned")]

    if timeout_failures:
        analysis["quality_issues"].append(f"{len(timeout_failures)} queries timed out")
        analysis["recommendations"].append("LATENCY: Parallelize more pipeline stages or reduce context window")

    if keyword_failures:
        analysis["quality_issues"].append(f"{len(keyword_failures)} queries missing expected keywords")
        analysis["recommendations"].append("QUALITY: Improve evidence assembly or prompt engineering")

    if banned_failures:
        analysis["quality_issues"].append(f"{len(banned_failures)} queries contain banned patterns")
        analysis["recommendations"].append("QUALITY: Fix LLM prompt to avoid metadata/tool leakage")

    # Check if fine-tuning is needed
    if report.pass_rate < 0.7:
        analysis["recommendations"].append("CRITICAL: Pass rate < 70% — consider fine-tuning the model")

    print(f"\n  Gap Analysis:")
    if analysis["weak_categories"]:
        print(f"    Weak categories: {[w['category'] for w in analysis['weak_categories']]}")
    if analysis["latency_bottlenecks"]:
        print(f"    Latency bottlenecks: {[b['category'] for b in analysis['latency_bottlenecks']]}")
    for rec in analysis["recommendations"]:
        print(f"    → {rec}")

    return analysis


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DocWain Iterative Optimizer")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--output-dir", type=str, default="/tmp/docwain_iterations",
                        help="Directory for iteration reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    iteration = args.iteration

    # Load previous iteration report if exists
    prev_report = None
    prev_failures = None
    if iteration > 1:
        prev_path = output_dir / f"iteration_{iteration - 1}.json"
        if prev_path.exists():
            with open(prev_path) as f:
                prev_data = json.load(f)
                prev_report = IterationReport(**{
                    k: v for k, v in prev_data.items()
                    if k in IterationReport.__dataclass_fields__
                })
                prev_failures = prev_data.get("failures", [])

    # Generate test suite (adapts based on previous failures)
    tests = generate_test_suite(iteration=iteration, prev_failures=prev_failures)
    print(f"\n  Generated {len(tests)} dynamic test cases for iteration {iteration}")

    # Run tests
    report = run_test_suite(tests, iteration)

    # Compare with previous
    improvements = compare_iterations(report, prev_report)
    report.improvements_from_prev = improvements

    # Save report
    save_report(report, output_dir)

    # Analyze gaps
    analysis = analyze_gaps(report)

    # Decision logic
    print(f"\n{'='*80}")
    if report.pass_rate >= 0.95 and report.avg_latency_s < 50:
        print("  ✓ TARGETS MET — Quality and speed are optimal!")
        print(f"{'='*80}")
        return 0
    elif report.pass_rate >= 0.85 and report.avg_latency_s < 80:
        print("  ◐ GOOD — Meeting baseline targets. Further optimization possible.")
        print(f"{'='*80}")
        return 0
    else:
        stale_iterations = 0
        if iteration >= 2 and not improvements.get("is_improving", True):
            stale_iterations += 1

        if iteration >= 5 and stale_iterations >= 3:
            print("  ✗ 5+ ITERATIONS WITHOUT IMPROVEMENT — ESCALATE TO FINE-TUNING")
            print(f"{'='*80}")
            return 2  # Signal: fine-tuning needed

        print(f"  → Continue to iteration {iteration + 1}")
        print(f"  → Focus areas: {', '.join(analysis.get('recommendations', ['general optimization']))}")
        print(f"{'='*80}")
        return 1  # Signal: continue iterating


if __name__ == "__main__":
    sys.exit(main())
