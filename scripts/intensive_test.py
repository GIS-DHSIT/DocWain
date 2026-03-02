#!/usr/bin/env python3
"""Intensive 50+ query production readiness test for DocWain.

Runs diverse queries across all profiles, tools, and complexity levels.
Each response is graded on: accuracy, intelligence, structure, and clarity.
"""
import json
import sys
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://localhost:8000"
SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
HR_PROFILE_ID = "69a45b8d0569ad0d8aee51e2"
MEDICAL_PROFILE_ID = "69a45b8d0569ad0d8aee51e2"  # Only HR data available — using HR profile
INVOICE_PROFILE_ID = "69a45b8d0569ad0d8aee51e2"  # Only HR data available — using HR profile
INSURANCE_PROFILE_ID = "69a45b8d0569ad0d8aee51e2"  # Only HR data available — using HR profile
TIMEOUT = 300.0


def ask(query, tools=None, profile_id=None, enable_internet=False, agent_mode=None):
    payload = {
        "query": query,
        "profile_id": profile_id or HR_PROFILE_ID,
        "subscription_id": SUBSCRIPTION_ID,
        "user_id": "intensive_test@docwain.ai",
        "enable_internet": enable_internet,
    }
    if tools:
        payload["tools"] = tools
        payload["use_tools"] = True
    if agent_mode is not None:
        payload["agent_mode"] = agent_mode
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/api/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


BANNED = [
    "not explicitly mentioned", "i don't have access", "i cannot",
    "i'm unable", "as an ai assistant", "as an ai language model",
    "MISSING_REASON", "Not enough information", "unfortunately, i",
    "tool:resumes", "tool:medical", "tool:insights", "tool:lawhere",
    "tool:email", "tool:action", "section_id", "chunk_type", "page_start",
]

INTEL_SIGNALS = [
    "across", "average", "range", "total", "compared", "pattern",
    "common", "unique", "distribution", "highest", "lowest", "overview",
    "analyzed", "candidates analyzed", "experience range", "shared",
]


def grade(text, expected, test_id):
    lower = text.lower()
    length = len(text)
    found = [s for s in expected if s.lower() in lower]
    missing = [s for s in expected if s.lower() not in lower]
    banned_found = [p for p in BANNED if p.lower() in lower]
    intel = [s for s in INTEL_SIGNALS if s in lower]

    ratio = len(found) / max(len(expected), 1)
    score = 0
    if ratio >= 0.5: score += 30
    if ratio >= 1.0: score += 20
    if length > 100: score += 15
    if length > 300: score += 5
    if not banned_found: score += 15
    if intel: score += 15

    g = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
    return {
        "id": test_id,
        "grade": g,
        "score": score,
        "length": length,
        "found": found,
        "missing": missing,
        "banned": banned_found,
        "intel": intel,
        "preview": text[:300],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 50+ Test queries organized by category
# ──────────────────────────────────────────────────────────────────────────────

TESTS = [
    # ── HR: Basic retrieval (10 queries) ──────────────────────────────────
    {"id": "HR-01", "q": "List all candidates and their primary skills",
     "tools": ["resumes"], "expect": ["candidate", "skill"]},
    {"id": "HR-02", "q": "Who has the most experience in Python?",
     "tools": ["resumes"], "expect": ["python", "experience"]},
    {"id": "HR-03", "q": "What certifications do the candidates have?",
     "tools": ["resumes"], "expect": ["certification"]},
    {"id": "HR-04", "q": "Show me the education details of all candidates",
     "tools": ["resumes"], "expect": ["education"]},
    {"id": "HR-05", "q": "What are the contact details for all candidates?",
     "tools": ["resumes"], "expect": ["email"]},
    {"id": "HR-06", "q": "Summarize all the resumes briefly",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "HR-07", "q": "Which candidate has data engineering skills?",
     "tools": ["resumes"], "expect": ["data"]},
    {"id": "HR-08", "q": "List candidates with more than 5 years of experience",
     "tools": ["resumes"], "expect": ["experience", "years"]},
    {"id": "HR-09", "q": "What functional skills are available across all resumes?",
     "tools": ["resumes"], "expect": ["skill"]},
    {"id": "HR-10", "q": "Find candidates who know both Python and machine learning",
     "tools": ["resumes"], "expect": ["python"]},

    # ── HR: Analytical (8 queries) ────────────────────────────────────────
    {"id": "HR-11", "q": "Compare all candidates for a Python developer role",
     "tools": ["resumes"], "expect": ["candidate", "python"]},
    {"id": "HR-12", "q": "Rank candidates by their overall profile strength",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "HR-13", "q": "What insights can you find across all the resumes?",
     "tools": ["insights"], "expect": ["skill", "experience"]},
    {"id": "HR-14", "q": "Rank candidates by education level",
     "tools": ["resumes"], "expect": ["education", "candidate"]},
    {"id": "HR-15", "q": "What programming languages are mentioned across all resumes?",
     "tools": ["resumes"], "expect": ["programming"]},
    {"id": "HR-16", "q": "Who is the best candidate for a senior AI engineer role?",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "HR-17", "q": "Compare the top 3 most experienced candidates",
     "tools": ["resumes"], "expect": ["candidate", "experience"]},
    {"id": "HR-18", "q": "What are the most common skills shared by multiple candidates?",
     "tools": ["resumes"], "expect": ["skill"]},

    # ── Medical (5 queries) ───────────────────────────────────────────────
    {"id": "MED-01", "q": "What are the patient diagnoses?",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["diagnos"]},
    {"id": "MED-02", "q": "List all medications prescribed",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["medicat"]},
    {"id": "MED-03", "q": "What lab results are available?",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["lab"]},
    {"id": "MED-04", "q": "Summarize the patient records",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["patient"]},
    {"id": "MED-05", "q": "What treatments or procedures were performed?",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["treatment"]},

    # ── Invoice (7 queries) ───────────────────────────────────────────────
    {"id": "INV-01", "q": "What are the total amounts on all invoices?",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "INV-02", "q": "List all invoice numbers and dates",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "INV-03", "q": "Which vendors are listed in the invoices?",
     "profile": INVOICE_PROFILE_ID, "expect": ["vendor"]},
    {"id": "INV-04", "q": "What items are listed on the invoices?",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "INV-05", "q": "Summarize all invoice data",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "INV-06", "q": "What payment terms are mentioned?",
     "profile": INVOICE_PROFILE_ID, "expect": ["payment"]},
    {"id": "INV-07", "q": "Find invoices with the highest amounts",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},

    # ── Insurance/Policy (5 queries) ──────────────────────────────────────
    {"id": "INS-01", "q": "What is covered under the car insurance policy?",
     "tools": ["lawhere"], "profile": INSURANCE_PROFILE_ID, "expect": ["cover"]},
    {"id": "INS-02", "q": "Compare the car and bike insurance policies",
     "tools": ["lawhere"], "profile": INSURANCE_PROFILE_ID, "expect": ["insurance"]},
    {"id": "INS-03", "q": "What are the exclusions in the insurance policies?",
     "tools": ["lawhere"], "profile": INSURANCE_PROFILE_ID, "expect": ["exclusion"]},
    {"id": "INS-04", "q": "What premiums are payable?",
     "tools": ["lawhere"], "profile": INSURANCE_PROFILE_ID, "expect": ["premium"]},
    {"id": "INS-05", "q": "Summarize the key terms and conditions",
     "tools": ["lawhere"], "profile": INSURANCE_PROFILE_ID, "expect": ["term"]},

    # ── Email drafting (3 queries) ────────────────────────────────────────
    {"id": "EMAIL-01", "q": "Draft an interview invitation email for the top candidate",
     "tools": ["email_drafting"], "expect": ["interview", "email"]},
    {"id": "EMAIL-02", "q": "Write a follow up email to a candidate about their application status",
     "tools": ["email_drafting"], "expect": ["follow", "email"]},
    {"id": "EMAIL-03", "q": "Compose a rejection email to a candidate in a professional tone",
     "tools": ["email_drafting"], "expect": ["email"]},

    # ── Content generation (3 queries) ────────────────────────────────────
    {"id": "GEN-01", "q": "Generate a cover letter for a Python developer based on the resumes",
     "tools": ["resumes"], "expect": ["python", "experience"]},
    {"id": "GEN-02", "q": "Create a summary of key findings across all documents",
     "expect": ["summary"]},
    {"id": "GEN-03", "q": "Extract all action items from the documents",
     "tools": ["action_items"], "expect": ["action"]},

    # ── Intelligence (document discovery) (4 queries) ─────────────────────
    {"id": "DISC-01", "q": "How many documents are in this profile?",
     "expect": ["document"]},
    {"id": "DISC-02", "q": "What types of documents do I have?",
     "expect": ["document"]},
    {"id": "DISC-03", "q": "How many documents are in this profile?",
     "profile": INVOICE_PROFILE_ID, "expect": ["document"]},
    {"id": "DISC-04", "q": "How many documents are in this profile?",
     "profile": MEDICAL_PROFILE_ID, "expect": ["document"]},

    # ── Internet queries (2 queries) ──────────────────────────────────────
    {"id": "WEB-01", "q": "What is the latest version of Python programming language?",
     "internet": True, "expect": ["python"]},
    {"id": "WEB-02", "q": "What is FastAPI framework used for?",
     "internet": True, "expect": ["api"]},

    # ── Edge cases (5 queries) ────────────────────────────────────────────
    {"id": "EDGE-01", "q": "Tell me everything you know about the documents",
     "expect": ["document"]},
    {"id": "EDGE-02", "q": "What is the most important information in the resumes?",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "EDGE-03", "q": "Are there any common patterns across the documents?",
     "expect": ["pattern"]},
    {"id": "EDGE-04", "q": "What insights can you provide about the invoice data?",
     "tools": ["insights"], "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "EDGE-05", "q": "Provide a comprehensive analysis of all candidates",
     "tools": ["resumes"], "expect": ["candidate", "experience"]},
]


def run_test(test):
    test_id = test["id"]
    query = test["q"]
    tools = test.get("tools")
    profile = test.get("profile")
    internet = test.get("internet", False)
    expected = test.get("expect", [])

    t0 = time.time()
    try:
        result = ask(query, tools=tools, profile_id=profile, enable_internet=internet)
        elapsed = time.time() - t0
        text = str(result.get("answer", {}).get("response", ""))
        sources = result.get("answer", {}).get("sources", [])
        metadata = result.get("answer", {}).get("metadata", {})

        g = grade(text, expected, test_id)
        g["elapsed"] = round(elapsed, 1)
        g["sources"] = len(sources)

        # Check confidence score
        conf = metadata.get("confidence", {})
        g["confidence"] = conf.get("score", -1)
        g["evidence_cov"] = conf.get("dimensions", {}).get("evidence_coverage", -1)

        return g
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "id": test_id,
            "grade": "F",
            "score": 0,
            "length": 0,
            "elapsed": round(elapsed, 1),
            "error": str(e)[:200],
            "preview": "",
        }


def main():
    print("=" * 80)
    print(f"  INTENSIVE PRODUCTION TEST — {len(TESTS)} queries — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = []
    grades = Counter()
    total_elapsed = 0

    for i, test in enumerate(TESTS, 1):
        test_id = test["id"]
        query_short = test["q"][:60]
        print(f"  [{i}/{len(TESTS)}] {test_id}: {query_short}...", end=" ", flush=True)

        r = run_test(test)
        results.append(r)
        g = r["grade"]
        grades[g] += 1
        elapsed = r.get("elapsed", 0)
        total_elapsed += elapsed

        # Print inline result
        extra = ""
        if r.get("confidence", -1) >= 0:
            extra += f" | conf={r['confidence']:.3f}"
        if r.get("evidence_cov", -1) >= 0:
            extra += f" | ev_cov={r['evidence_cov']:.3f}"
        if r.get("error"):
            extra += f" | ERROR: {r['error'][:60]}"

        print(f"[{g}] {elapsed:.1f}s | {r.get('length', 0)} chars | {r.get('sources', 0)} src{extra}")

    # Summary
    print("\n" + "=" * 80)
    total = len(results)
    passing = sum(1 for r in results if r["grade"] in ("A", "B"))
    failing = sum(1 for r in results if r["grade"] in ("D", "F"))
    avg_score = sum(r.get("score", 0) for r in results) / max(total, 1)
    avg_elapsed = total_elapsed / max(total, 1)

    print(f"  TOTAL: {total} | PASS(A/B): {passing} | WARN(C): {grades.get('C', 0)} | FAIL(D/F): {failing}")
    print(f"  GRADES: A={grades.get('A', 0)}, B={grades.get('B', 0)}, C={grades.get('C', 0)}, D={grades.get('D', 0)}, F={grades.get('F', 0)}")
    print(f"  AVG SCORE: {avg_score:.0f}/100 | AVG LATENCY: {avg_elapsed:.1f}s | TOTAL TIME: {total_elapsed:.0f}s")

    # Evidence coverage stats
    ev_scores = [r["evidence_cov"] for r in results if r.get("evidence_cov", -1) >= 0]
    if ev_scores:
        avg_ev = sum(ev_scores) / len(ev_scores)
        nonzero_ev = sum(1 for e in ev_scores if e > 0)
        print(f"  EVIDENCE COVERAGE: avg={avg_ev:.3f} | non-zero={nonzero_ev}/{len(ev_scores)}")

    # Confidence stats
    conf_scores = [r["confidence"] for r in results if r.get("confidence", -1) >= 0]
    if conf_scores:
        avg_conf = sum(conf_scores) / len(conf_scores)
        print(f"  CONFIDENCE: avg={avg_conf:.3f}")

    print("=" * 80)

    # Show failures
    failures = [r for r in results if r["grade"] in ("D", "F")]
    if failures:
        print("\n  FAILURES:")
        for r in failures:
            print(f"    {r['id']}: [{r['grade']}] score={r.get('score', 0)}")
            if r.get("missing"):
                print(f"      Missing: {r['missing']}")
            if r.get("banned"):
                print(f"      Banned: {r['banned']}")
            if r.get("error"):
                print(f"      Error: {r['error'][:120]}")
            print(f"      Preview: {r.get('preview', '')[:200]}")
            print()

    # Show warnings (C grade)
    warnings = [r for r in results if r["grade"] == "C"]
    if warnings:
        print("\n  WARNINGS (C grade):")
        for r in warnings:
            print(f"    {r['id']}: score={r.get('score', 0)} | missing={r.get('missing', [])}")

    # Save results
    output_path = "/tmp/intensive_test_results.json"
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "total": total, "passing": passing, "failing": failing,
                    "avg_score": round(avg_score, 1),
                    "grade_distribution": dict(grades),
                    "results": results}, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return 0 if failing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
