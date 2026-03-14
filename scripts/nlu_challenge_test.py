#!/usr/bin/env python3
"""NLU Intelligence Challenge Test for DocWain.

Tests the model's ability to handle:
- Paraphrased / non-standard queries
- Indirect / implied intent
- Multi-step reasoning
- Ambiguous / vague queries that require inference
- Domain-switching / mixed-domain queries
- Novel vocabulary and phrasing
"""
import json
import sys
import time
import traceback
from collections import Counter
from typing import Any, Dict, List

import httpx

BASE_URL = "http://localhost:8000"
SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
HR_PROFILE_ID = "69a6dddc23d47adc8e7ee7e4"
MEDICAL_PROFILE_ID = "69a552ac23d47adc8e7ee27a"
INVOICE_SUBSCRIPTION_ID = "67e6920588f8ff4644d2dfb1"
INVOICE_PROFILE_ID = "69a6de9e23d47adc8e7ee90a"
TIMEOUT = 600.0


def ask(query, tools=None, profile_id=None, enable_internet=False, subscription_id=None):
    pid = profile_id or HR_PROFILE_ID
    sid = subscription_id or SUBSCRIPTION_ID
    if pid == INVOICE_PROFILE_ID:
        sid = INVOICE_SUBSCRIPTION_ID
    payload = {
        "query": query,
        "profile_id": pid,
        "subscription_id": sid,
        "user_id": "nlu_challenge@docwain.ai",
        "enable_internet": enable_internet,
    }
    if tools:
        payload["tools"] = tools
        payload["use_tools"] = True
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
# NLU CHALLENGE TESTS — novel phrasing, indirect intent, paraphrasing
# ──────────────────────────────────────────────────────────────────────────────

TESTS = [
    # ── Paraphrased queries (same intent as standard, different words) ──────
    {"id": "PARA-01", "q": "Show me the talent pool and what each person specializes in",
     "tools": ["resumes"], "expect": ["candidate", "skill"],
     "desc": "Paraphrase of 'list candidates with skills'"},
    {"id": "PARA-02", "q": "I need to know the academic backgrounds of the applicants",
     "tools": ["resumes"], "expect": ["education"],
     "desc": "Paraphrase of 'show education details'"},
    {"id": "PARA-03", "q": "Give me a rundown of what bills are outstanding",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"],
     "desc": "Paraphrase of 'list invoice totals'"},
    {"id": "PARA-04", "q": "I'd like to see who we could reach out to and how",
     "tools": ["resumes"], "expect": ["contact"],
     "desc": "Paraphrase of 'show contact details'"},
    {"id": "PARA-05", "q": "Help me understand what illnesses the patient is dealing with",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["diagnos"],
     "desc": "Paraphrase of 'what are the diagnoses'"},

    # ── Indirect / implied intent ──────────────────────────────────────────
    {"id": "IND-01", "q": "We have a senior backend position opening up next month",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Implied: find suitable candidates for backend role"},
    {"id": "IND-02", "q": "The client is asking about their outstanding balance",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"],
     "desc": "Implied: show invoice totals/amounts due"},
    {"id": "IND-03", "q": "I'm preparing for tomorrow's candidate screening call",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Implied: provide candidate overview for screening"},
    {"id": "IND-04", "q": "The doctor wants to review the patient's current treatment plan before the follow-up",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["patient"],
     "desc": "Implied: show medications and treatment details"},

    # ── Vague / open-ended queries requiring inference ─────────────────────
    {"id": "VAGUE-01", "q": "What should I know before the interview?",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Vague: needs to infer context = candidate prep"},
    {"id": "VAGUE-02", "q": "Give me the highlights",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Vague: needs to summarize whatever docs are available"},
    {"id": "VAGUE-03", "q": "What's interesting here?",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Vague: needs to identify notable information"},
    {"id": "VAGUE-04", "q": "Help me make a decision",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Vague: infer decision = candidate comparison/ranking"},

    # ── Complex multi-constraint queries ───────────────────────────────────
    {"id": "MULTI-01", "q": "Find someone with at least 5 years experience who knows both Python and cloud technologies and has relevant certifications",
     "tools": ["resumes"], "expect": ["experience", "python"],
     "desc": "Multi-constraint: experience + skills + certifications"},
    {"id": "MULTI-02", "q": "Show me invoices over $1000 from vendors we've used more than once",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"],
     "desc": "Multi-constraint: amount + vendor frequency"},
    {"id": "MULTI-03", "q": "Which candidate has the best combination of technical depth and varied industry exposure?",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Multi-constraint: technical skills + industry diversity"},

    # ── Conversational / natural phrasing ──────────────────────────────────
    {"id": "NAT-01", "q": "Hey, can you pull up the resumes for me? I want to see who stands out",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Natural greeting + request"},
    {"id": "NAT-02", "q": "Actually, I changed my mind — show me the medical records instead",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["patient"],
     "desc": "Context switch with natural language"},
    {"id": "NAT-03", "q": "Hmm, that's good but can you dig deeper into the skills comparison?",
     "tools": ["resumes"], "expect": ["skill"],
     "desc": "Follow-up with 'dig deeper' idiom"},
    {"id": "NAT-04", "q": "Just a quick question — what's the total we owe across all invoices?",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"],
     "desc": "Casual phrasing for financial query"},

    # ── Domain-specific vocabulary ─────────────────────────────────────────
    {"id": "VOCAB-01", "q": "What's the candidate's tech stack breadth?",
     "tools": ["resumes"], "expect": ["skill"],
     "desc": "Tech jargon: 'tech stack breadth' = range of technical skills"},
    {"id": "VOCAB-02", "q": "Show me the burn rate across these invoices",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"],
     "desc": "Finance jargon: 'burn rate' = spending rate"},
    {"id": "VOCAB-03", "q": "What comorbidities are present?",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["diagnos"],
     "desc": "Medical jargon: comorbidities = co-occurring conditions"},
    {"id": "VOCAB-04", "q": "Pull up the candidate pipeline metrics",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "HR jargon: 'pipeline metrics' = candidate pool stats"},

    # ── Negation and exclusion queries ─────────────────────────────────────
    {"id": "NEG-01", "q": "Which candidates do NOT have cloud experience?",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Negation: filter by absence of skill"},
    {"id": "NEG-02", "q": "Show me all invoices that aren't from recurring vendors",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"],
     "desc": "Negation: exclude by vendor frequency"},
    {"id": "NEG-03", "q": "What medications is the patient NOT currently taking?",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["medicat"],
     "desc": "Negation: absence of medications"},

    # ── Hypothetical / what-if queries ─────────────────────────────────────
    {"id": "HYPO-01", "q": "If I could only hire one person for a full-stack role, who would it be?",
     "tools": ["resumes"], "expect": ["candidate"],
     "desc": "Hypothetical: ranking for specific role"},
    {"id": "HYPO-02", "q": "What would happen if we combined the skills of the top two candidates?",
     "tools": ["resumes"], "expect": ["skill"],
     "desc": "Hypothetical: combined skill profile"},
    {"id": "HYPO-03", "q": "If the patient's condition worsens, what escalation options exist based on the records?",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["patient"],
     "desc": "Hypothetical: medical escalation planning"},

    # ── Quantity and aggregation ───────────────────────────────────────────
    {"id": "AGG-01", "q": "How many unique programming languages are mentioned across all resumes?",
     "tools": ["resumes"], "expect": ["programming"]},
    {"id": "AGG-02", "q": "What's the total dollar amount across all invoices combined?",
     "profile": INVOICE_PROFILE_ID, "expect": ["total"]},
    {"id": "AGG-03", "q": "Count the number of distinct certifications held by all candidates",
     "tools": ["resumes"], "expect": ["certif"]},
    {"id": "AGG-04", "q": "How many different vendors appear in the invoice data?",
     "profile": INVOICE_PROFILE_ID, "expect": ["vendor"]},

    # ── Comparative and superlative ────────────────────────────────────────
    {"id": "COMP-01", "q": "Who has more relevant experience — the data engineers or the ML engineers?",
     "tools": ["resumes"], "expect": ["experience"]},
    {"id": "COMP-02", "q": "Is the largest invoice from the same vendor as the smallest one?",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "COMP-03", "q": "Which candidate's education is most directly relevant to an AI research position?",
     "tools": ["resumes"], "expect": ["education"]},

    # ── Temporal queries ───────────────────────────────────────────────────
    {"id": "TIME-01", "q": "Which candidate was most recently employed and where?",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "TIME-02", "q": "Show me invoices from the most recent billing period",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "TIME-03", "q": "What's the career timeline for each candidate?",
     "tools": ["resumes"], "expect": ["experience"]},

    # ── Format-specific requests ───────────────────────────────────────────
    {"id": "FMT-01", "q": "Present the candidate comparison as a table",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "FMT-02", "q": "Give me bullet points of the key invoice data",
     "profile": INVOICE_PROFILE_ID, "expect": ["invoice"]},
    {"id": "FMT-03", "q": "Summarize the medical records in SOAP format",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["patient"]},

    # ── Mixed intent / multi-part queries ──────────────────────────────────
    {"id": "MIX-01", "q": "First summarize the resumes, then rank the top 3 candidates for a Python role",
     "tools": ["resumes"], "expect": ["candidate", "python"]},
    {"id": "MIX-02", "q": "List the vendors and then tell me which one we've spent the most with",
     "profile": INVOICE_PROFILE_ID, "expect": ["vendor"]},
    {"id": "MIX-03", "q": "Give me the patient diagnosis and then suggest follow-up questions I should ask",
     "tools": ["medical"], "profile": MEDICAL_PROFILE_ID, "expect": ["diagnos"]},
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
        g["desc"] = test.get("desc", "")
        g["query"] = query

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
            "desc": test.get("desc", ""),
            "query": query,
        }


def main():
    print("=" * 80)
    print(f"  NLU CHALLENGE TEST — {len(TESTS)} queries — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = []
    grades = Counter()
    total_elapsed = 0
    category_stats = {}

    for i, test in enumerate(TESTS, 1):
        test_id = test["id"]
        category = test_id.rsplit("-", 1)[0]
        query_short = test["q"][:55]
        print(f"  [{i}/{len(TESTS)}] {test_id}: {query_short}...", end=" ", flush=True)

        r = run_test(test)
        results.append(r)
        g = r["grade"]
        grades[g] += 1
        elapsed = r.get("elapsed", 0)
        total_elapsed += elapsed

        # Track per-category
        if category not in category_stats:
            category_stats[category] = {"pass": 0, "total": 0, "grades": Counter()}
        category_stats[category]["total"] += 1
        category_stats[category]["grades"][g] += 1
        if g in ("A", "B"):
            category_stats[category]["pass"] += 1

        extra = ""
        if r.get("confidence", -1) >= 0:
            extra += f" | conf={r['confidence']:.3f}"
        if r.get("error"):
            extra += f" | ERROR: {r['error'][:60]}"

        print(f"[{g}] {elapsed:.1f}s | {r.get('length', 0)} chars{extra}")

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
    print(f"  PASS RATE: {passing/total*100:.1f}%")

    # Per-category breakdown
    print("\n  PER-CATEGORY BREAKDOWN:")
    for cat, stats in sorted(category_stats.items()):
        p = stats["pass"]
        t = stats["total"]
        gs = stats["grades"]
        grade_str = " ".join(f"{g}={c}" for g, c in sorted(gs.items()))
        print(f"    {cat:10s}: {p}/{t} pass | {grade_str}")

    print("=" * 80)

    # Show failures
    failures = [r for r in results if r["grade"] in ("D", "F")]
    if failures:
        print("\n  FAILURES:")
        for r in failures:
            print(f"    {r['id']}: [{r['grade']}] {r.get('desc', '')}")
            print(f"      Query: {r.get('query', '')[:100]}")
            if r.get("missing"):
                print(f"      Missing: {r['missing']}")
            if r.get("banned"):
                print(f"      Banned: {r['banned']}")
            if r.get("error"):
                print(f"      Error: {r['error'][:120]}")
            print(f"      Preview: {r.get('preview', '')[:200]}")
            print()

    # Show C grades
    warnings = [r for r in results if r["grade"] == "C"]
    if warnings:
        print("\n  WARNINGS (C grade):")
        for r in warnings:
            print(f"    {r['id']}: {r.get('desc', '')} | missing={r.get('missing', [])}")

    # Save results
    output_path = "/tmp/nlu_challenge_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total": total, "passing": passing, "failing": failing,
            "avg_score": round(avg_score, 1),
            "pass_rate": round(passing / total * 100, 1),
            "grade_distribution": dict(grades),
            "category_stats": {k: {"pass": v["pass"], "total": v["total"],
                                    "grades": dict(v["grades"])}
                               for k, v in category_stats.items()},
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return 0 if failing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
