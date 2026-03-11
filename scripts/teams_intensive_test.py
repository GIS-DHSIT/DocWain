#!/usr/bin/env python3
"""Intensive test for DocWain's Microsoft Teams endpoint.

Sends diverse queries via the Teams message format and grades responses.
"""
import json
import sys
import time
from collections import Counter

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 600.0

# Teams uses conversation.id as subscription and from.aadObjectId as profile
# We use the existing HR profile's subscription/profile for testing
# When SESSION_AS_SUBSCRIPTION=true, conversation.id becomes subscription_id
# When PROFILE_PER_USER=true, from.aadObjectId becomes profile_id
TEAMS_USER_ID = "69a6dddc23d47adc8e7ee7e4"  # HR profile ID
TEAMS_CONV_ID = "67fde0754e36c00b14cea7f5"  # HR subscription ID


def teams_ask(query, conversation_id=None):
    """Send a message via the Teams endpoint."""
    activity = {
        "type": "message",
        "text": query,
        "from": {
            "id": TEAMS_USER_ID,
            "aadObjectId": TEAMS_USER_ID,
        },
        "conversation": {
            "id": conversation_id or TEAMS_CONV_ID,
        },
        "channelData": {},
        "serviceUrl": "https://smba.trafficmanager.net/uk/",
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/api/teams/messages", json=activity)
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "")


BANNED = [
    "MISSING_REASON", "tool:resumes", "tool:medical", "tool:insights",
    "section_id", "chunk_type", "page_start",
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
    if ratio >= 0.5:
        score += 30
    if ratio >= 1.0:
        score += 20
    if length > 100:
        score += 15
    if length > 300:
        score += 5
    if not banned_found:
        score += 15
    if intel:
        score += 15

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


TESTS = [
    # ── Document queries via Teams (queries against real HR data) ──────
    {"id": "T-HR-01", "q": "List all candidates and their skills",
     "expect": ["candidate", "skill"]},
    {"id": "T-HR-02", "q": "Who has the most experience in Python?",
     "expect": ["python", "experience"]},
    {"id": "T-HR-03", "q": "Compare all candidates for a data science role",
     "expect": ["candidate"]},
    {"id": "T-HR-04", "q": "What certifications do the candidates have?",
     "expect": ["certif"]},
    {"id": "T-HR-05", "q": "Rank candidates by education level",
     "expect": ["candidate", "education"]},
    {"id": "T-HR-06", "q": "What programming languages are mentioned across all resumes?",
     "expect": ["programming"]},
    {"id": "T-HR-07", "q": "Generate interview questions for the top candidate",
     "expect": ["question"]},
    {"id": "T-HR-08", "q": "Summarize all the resumes briefly",
     "expect": ["experience"]},

    # ── Analytical HR queries via Teams ────────────────────────────────
    {"id": "T-ANAL-01", "q": "What are the most common skills shared by candidates?",
     "expect": ["skill"]},
    {"id": "T-ANAL-02", "q": "Which candidate would be the best fit for a senior engineer role?",
     "expect": ["candidate"]},
    {"id": "T-ANAL-03", "q": "Extract all email addresses from the resumes",
     "expect": ["email"]},

    # ── Conversational (3 queries — these depend on conversational handler) ──
    {"id": "T-CONV-01", "q": "How many documents are in this profile?",
     "expect": ["document"]},
    {"id": "T-CONV-02", "q": "What types of documents do I have?",
     "expect": ["document"]},
    {"id": "T-CONV-03", "q": "Thank you very much!",
     "expect": ["glad", "help"]},

    # ── Content generation via Teams ──────────────────────────────────
    {"id": "T-CGEN-01", "q": "Create a professional summary for the strongest candidate",
     "expect": ["experience"]},
    {"id": "T-CGEN-02", "q": "Write a skills comparison matrix for all candidates",
     "expect": ["skill"]},
    {"id": "T-CGEN-03", "q": "Generate interview questions for a data science candidate",
     "expect": ["question"]},

    # ── Cross-document intelligence via Teams ─────────────────────────
    {"id": "T-CROSS-01", "q": "What skills are mentioned in every single resume?",
     "expect": ["skill"]},
    {"id": "T-CROSS-02", "q": "What is the average years of experience across all candidates?",
     "expect": ["experience"]},
    {"id": "T-CROSS-03", "q": "Which candidates have overlapping skill sets?",
     "expect": ["candidate", "skill"]},

    # ── Complex reasoning via Teams ───────────────────────────────────
    {"id": "T-REASON-01", "q": "If I need someone who can build data pipelines and deploy ML models, which candidate is the best fit?",
     "expect": ["candidate"]},
    {"id": "T-REASON-02", "q": "What are the unique skills that each candidate brings that others don't?",
     "expect": ["skill", "candidate"]},
    {"id": "T-REASON-03", "q": "Analyze the career progression of each candidate",
     "expect": ["candidate", "experience"]},

    # ── Specific extraction via Teams ─────────────────────────────────
    {"id": "T-EXTRACT-01", "q": "Extract all email addresses and phone numbers from the resumes",
     "expect": ["email"]},
    {"id": "T-EXTRACT-02", "q": "List every company name mentioned across all resumes",
     "expect": ["candidate"]},
    {"id": "T-EXTRACT-03", "q": "List all technical certifications and who holds them",
     "expect": ["certif"]},
]


def run_test(test):
    test_id = test["id"]
    query = test["q"]
    expected = test.get("expect", [])

    t0 = time.time()
    try:
        text = teams_ask(query)
        elapsed = time.time() - t0

        g = grade(text, expected, test_id)
        g["elapsed"] = round(elapsed, 1)
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
    print(f"  TEAMS INTENSIVE TEST — {len(TESTS)} queries — {time.strftime('%Y-%m-%d %H:%M:%S')}")
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

        extra = ""
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
    print("=" * 80)

    # Show failures
    failures = [r for r in results if r["grade"] in ("D", "F")]
    if failures:
        print("\n  FAILURES:")
        for r in failures:
            print(f"    {r['id']}: [{r['grade']}] score={r.get('score', 0)}")
            if r.get("error"):
                print(f"      Error: {r['error'][:120]}")
            print(f"      Preview: {r.get('preview', '')[:200]}")
            print()

    # Save results
    output_path = "/tmp/teams_test_results.json"
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
