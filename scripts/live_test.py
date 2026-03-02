#!/usr/bin/env python3
"""Live server test harness for DocWain /api/ask endpoint.

Sends a batch of queries across domains and tools, captures responses,
and outputs a graded report.
"""
import json
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://localhost:8000"
# Collection 67fde0754e36c00b14cea7f5: HR profile (18 resumes, 100 chunks) + medical profile
SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
PROFILE_ID = "6994620d6034385742e45abe"  # HR profile with 18 resumes
MEDICAL_PROFILE_ID = "699467806034385742e45b9f"  # Medical + invoice profile
# Collection 67fde0754e36c00b14cea7f5: invoice profile (24 invoices: INV, PO, QUT)
INVOICE_SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
INVOICE_PROFILE_ID = "699c8f9c1aed615c51b0a4dc"  # 24 invoices, vendors: WADE, PERRY, DUNCAN
# Insurance profile
INSURANCE_PROFILE_ID = "6995a20b2f85f720e96fa486"  # 2 docs: car + bike insurance
TIMEOUT = 300.0

# --------------------------------------------------------------------------
# Query definitions: ordered by cost (light → heavy) to avoid LLM exhaustion
# --------------------------------------------------------------------------
QUERIES: List[Dict[str, Any]] = [
    # ── Fast conversational queries (no LLM needed) ───────────────
    {
        "id": "INTEL-1",
        "query": "How many documents are in this profile?",
        "tools": None,
        "expect": ["document"],
        "domain": "general",
    },
    {
        "id": "INTEL-2",
        "query": "What types of documents do I have?",
        "tools": None,
        "expect": ["resume", "document"],
        "domain": "general",
    },
    # ── HR / Resume queries (deterministic extraction, no LLM) ────
    {
        "id": "HR-1",
        "query": "List all candidates and their skills",
        "tools": ["resumes"],
        "expect": ["skills", "candidate"],
        "domain": "hr",
    },
    {
        "id": "HR-2",
        "query": "Who has the most experience?",
        "tools": ["resumes"],
        "expect": ["experience", "years"],
        "domain": "hr",
    },
    {
        "id": "HR-3",
        "query": "Compare all candidates for a Python developer role",
        "tools": ["resumes"],
        "expect": ["python", "candidate"],
        "domain": "hr",
    },
    {
        "id": "HR-4",
        "query": "What are the contact details for all candidates?",
        "tools": ["resumes"],
        "expect": ["email", "phone"],
        "domain": "hr",
    },
    {
        "id": "HR-6",
        "query": "Rank candidates by their education level",
        "tools": ["resumes"],
        "expect": ["education", "candidate"],
        "domain": "hr",
    },
    # ── Medical queries (small profile, fast) ─────────────────────
    {
        "id": "MED-1",
        "query": "What are the patient diagnoses?",
        "tools": ["medical"],
        "expect": ["patient", "diagnos"],
        "domain": "medical",
        "profile_id": MEDICAL_PROFILE_ID,
    },
    {
        "id": "MED-2",
        "query": "List all medications prescribed across all patient records",
        "tools": ["medical"],
        "expect": ["medication"],
        "domain": "medical",
        "profile_id": MEDICAL_PROFILE_ID,
    },
    # ── Invoice queries (small profile) ───────────────────────────
    {
        "id": "INV-1",
        "query": "What are the total amounts on all invoices?",
        "tools": None,
        "expect": ["total", "amount"],
        "domain": "invoice",
        "profile_id": INVOICE_PROFILE_ID,
        "subscription_id": INVOICE_SUBSCRIPTION_ID,
    },
    {
        "id": "INV-2",
        "query": "List all invoice numbers and their dates",
        "tools": None,
        "expect": ["invoice"],
        "domain": "invoice",
        "profile_id": INVOICE_PROFILE_ID,
        "subscription_id": INVOICE_SUBSCRIPTION_ID,
    },
    # ── Insurance/Policy queries ────────────────────────────────────
    {
        "id": "INS-POL-1",
        "query": "What is covered under the car insurance policy?",
        "tools": None,
        "expect": ["cover", "car"],
        "domain": "insurance",
        "profile_id": INSURANCE_PROFILE_ID,
    },
    {
        "id": "INS-POL-2",
        "query": "Compare the car and bike insurance policies",
        "tools": None,
        "expect": ["car", "bike"],
        "domain": "insurance",
        "profile_id": INSURANCE_PROFILE_ID,
    },
    # ── No-tool factual query ─────────────────────────────────────
    {
        "id": "FACT-1",
        "query": "What programming languages are mentioned across all resumes?",
        "tools": None,
        "expect": ["python", "c++"],
        "domain": "hr",
    },
    # ── Email drafting (moderate LLM) ─────────────────────────────
    {
        "id": "EMAIL-1",
        "query": "Draft an interview invitation email for the top candidate",
        "tools": ["email_drafting"],
        "expect": ["interview", "invitation"],
        "domain": "hr",
    },
    # ── Tool queries with LLM enhancement (heavier) ───────────────
    {
        "id": "INS-1",
        "query": "What insights can you find across all the resumes?",
        "tools": ["insights"],
        "expect": ["insight", "skill"],
        "domain": "hr",
    },
    {
        "id": "ACT-1",
        "query": "Extract all action items and tasks from the documents",
        "tools": ["action_items"],
        "expect": ["action", "task"],
        "domain": "general",
    },
    # ── Content generation (heavy LLM) ────────────────────────────
    {
        "id": "GEN-2",
        "query": "Create a summary of key findings across all documents",
        "tools": ["content_generate"],
        "expect": ["summary", "document"],
        "domain": "general",
    },
    {
        "id": "GEN-1",
        "query": "Generate a cover letter for a Python developer based on the best candidate",
        "tools": ["content_generate"],
        "expect": ["python", "candidate"],
        "domain": "hr",
    },
    # ── HR-5 (heaviest: 18 resume summarization via tool) ─────────
    {
        "id": "HR-5",
        "query": "Summarize all the resumes",
        "tools": ["resumes"],
        "expect": ["experience", "resume"],
        "domain": "hr",
    },
    # ── Web search (enable_internet, external network) ────────────
    {
        "id": "WEB-1",
        "query": "What is the latest version of Python programming language?",
        "tools": None,
        "enable_internet": True,
        "expect": ["python"],
        "domain": "web",
    },
]

# Banned phrases that indicate poor response quality
BANNED_PHRASES = [
    "not explicitly mentioned",
    "i don't have access",
    "i cannot",
    "i'm unable",
    "as an ai assistant",
    "as an ai language model",
    "as an ai model",
    "i apologize",
    "unfortunately, i",
    "MISSING_REASON",
    "Not enough information",
    "no relevant information",
]


def ask(
    query: str,
    tools: Optional[List[str]] = None,
    enable_internet: bool = False,
    profile_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a query to /api/ask and return the full response."""
    payload = {
        "query": query,
        "profile_id": profile_id or PROFILE_ID,
        "subscription_id": subscription_id or SUBSCRIPTION_ID,
        "user_id": "live_test@docwain.ai",
        "enable_internet": enable_internet,
    }
    if tools:
        payload["tools"] = tools
        payload["use_tools"] = True

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/api/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


def grade_response(query_def: Dict, response: Dict) -> Dict[str, Any]:
    """Grade a response for quality signals."""
    answer = response.get("answer", {})
    response_text = str(answer.get("response", ""))
    sources = answer.get("sources", [])
    metadata = answer.get("metadata", {})
    grounded = answer.get("grounded", False)
    context_found = answer.get("context_found", False)

    response_lower = response_text.lower()
    text_length = len(response_text)

    # Check expected signals
    signals_found = []
    signals_missing = []
    for sig in query_def.get("expect", []):
        if sig.lower() in response_lower:
            signals_found.append(sig)
        else:
            signals_missing.append(sig)

    # Check banned phrases
    banned_found = []
    for phrase in BANNED_PHRASES:
        if phrase.lower() in response_lower:
            banned_found.append(phrase)

    # Scoring
    signal_score = len(signals_found) / max(len(query_def.get("expect", ["x"])), 1)
    length_ok = text_length > 50
    no_banned = len(banned_found) == 0
    has_sources = len(sources) > 0 or query_def.get("enable_internet", False)

    # Overall grade
    score = 0
    if signal_score >= 0.5:
        score += 40
    if signal_score >= 1.0:
        score += 20
    if length_ok:
        score += 15
    if no_banned:
        score += 15
    if has_sources or context_found:
        score += 10

    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "id": query_def["id"],
        "query": query_def["query"],
        "grade": grade,
        "score": score,
        "response_length": text_length,
        "signals_found": signals_found,
        "signals_missing": signals_missing,
        "banned_found": banned_found,
        "sources_count": len(sources),
        "grounded": grounded,
        "context_found": context_found,
        "web_search": metadata.get("web_search", False),
        "response_preview": response_text[:300],
    }


def run_test_round(round_num: int = 1) -> List[Dict[str, Any]]:
    """Run a full test round and return graded results."""
    print(f"\n{'='*80}")
    print(f"  LIVE TEST ROUND {round_num} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    results = []
    for i, qdef in enumerate(QUERIES):
        qid = qdef["id"]
        query = qdef["query"]
        tools = qdef.get("tools")
        enable_internet = qdef.get("enable_internet", False)
        profile_id = qdef.get("profile_id")
        subscription_id = qdef.get("subscription_id")

        print(f"  [{i+1}/{len(QUERIES)}] {qid}: {query[:60]}...", end=" ", flush=True)

        try:
            start = time.time()
            response = ask(query, tools=tools, enable_internet=enable_internet,
                          profile_id=profile_id, subscription_id=subscription_id)
            elapsed = time.time() - start
            graded = grade_response(qdef, response)
            graded["elapsed_s"] = round(elapsed, 1)
            results.append(graded)
            print(f"[{graded['grade']}] {elapsed:.1f}s | len={graded['response_length']} | signals={len(graded['signals_found'])}/{len(graded['signals_found'])+len(graded['signals_missing'])}")
        except Exception as exc:
            print(f"[ERROR] {exc}")
            results.append({
                "id": qid,
                "query": query,
                "grade": "F",
                "score": 0,
                "error": str(exc)[:200],
            })

    # Summary
    grades = [r["grade"] for r in results]
    avg_score = sum(r.get("score", 0) for r in results) / max(len(results), 1)
    pass_count = sum(1 for g in grades if g in ("A", "B"))
    fail_count = sum(1 for g in grades if g in ("D", "F"))

    print(f"\n{'─'*80}")
    print(f"  ROUND {round_num} SUMMARY:")
    print(f"    Total: {len(results)} | Pass(A/B): {pass_count} | Fail(D/F): {fail_count}")
    print(f"    Average Score: {avg_score:.0f}/100")
    print(f"    Grade Distribution: {', '.join(f'{g}={grades.count(g)}' for g in ['A','B','C','D','F'] if grades.count(g))}")
    print(f"{'─'*80}\n")

    # Print failures in detail
    failures = [r for r in results if r.get("grade") in ("D", "F")]
    if failures:
        print("  FAILED QUERIES:")
        for f in failures:
            print(f"    {f['id']}: {f['query'][:50]}")
            if f.get("error"):
                print(f"      Error: {f['error'][:100]}")
            else:
                print(f"      Missing signals: {f.get('signals_missing', [])}")
                print(f"      Banned phrases: {f.get('banned_found', [])}")
                print(f"      Response: {f.get('response_preview', '')[:150]}")
            print()

    return results


if __name__ == "__main__":
    round_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    results = run_test_round(round_num)

    # Save results
    with open(f"/tmp/live_test_round_{round_num}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to /tmp/live_test_round_{round_num}.json")
