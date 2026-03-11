#!/usr/bin/env python3
"""Quick re-test of previously failing queries after fixes."""

import json
import time
import requests
import sys

BASE = "http://localhost:8000"
SUB_ID = "67fde0754e36c00b14cea7f5"
RESUME_P1 = "69a6e13623d47adc8e7eeadd"
RESUME_P2 = "69a6dddc23d47adc8e7ee7e4"
INVOICE_P = "69a6de9e23d47adc8e7ee90a"
MEDICAL_P = "69a552ac23d47adc8e7ee27a"


def ask(profile_id, query):
    payload = {
        "query": query, "profile_id": profile_id,
        "subscription_id": SUB_ID, "new_session": True, "debug": True,
    }
    start = time.time()
    try:
        resp = requests.post(f"{BASE}/api/ask", json=payload, timeout=300)
        lat = (time.time() - start) * 1000
        if resp.status_code == 200:
            return resp.json(), lat
        return {"error": f"HTTP {resp.status_code}"}, lat
    except Exception as e:
        return {"error": str(e)}, (time.time() - start) * 1000


TESTS = [
    # Previously misrouted to USAGE_HELP
    ("SKILLS", RESUME_P1, "What are Philip Simon Derock's skills?",
     ["skills", "Philip", "experience"], ["Here's what you can do", "DocWain"]),
    # Previously misrouted to PRIVACY
    ("CONTACT", RESUME_P1, "What is Rahul Deshbhratar's email and phone number?",
     ["email", "phone", "@"], ["privacy", "data safe", "DocWain keeps"]),
    # Previously misrouted to USAGE_HELP
    ("SUMMARY", RESUME_P1, "Give me a summary of all candidates",
     ["candidate", "experience", "skills"], ["Here's what you can do", "Summarization"]),
    # Previously misrouted to SECURITY
    ("TYPO", RESUME_P1, "Wht is Gokuls educaton?",
     ["education", "college", "university", "degree", "CGPA"], ["security", "data safe"]),
    # Previously returned "couldn't find"
    ("INVOICE", INVOICE_P, "What is the total amount on the invoices?",
     ["total", "amount", "$", "invoice"], ["couldn't find"]),
    # Previously returned resume data for irrelevant query
    ("NO_INFO", RESUME_P1, "What is the company's annual revenue?",
     ["couldn't find", "not available", "no information"], ["revenue uplift", "retention"]),
    # Check metadata leak fix
    ("METADATA", RESUME_P1, "Tell me about Gokul",
     ["Gokul"], ["id:", "person:", "organization:", "chunk_type"]),
]

print("=" * 70)
print("DocWain Re-Test — Critical Fix Validation")
print("=" * 70)

pass_count = 0
for name, profile, query, good_signs, bad_signs in TESTS:
    print(f"\n[{name}] {query[:55]}...")
    sys.stdout.flush()

    data, lat = ask(profile, query)
    ans = data.get("answer", data)
    text = ans.get("response", "") if isinstance(ans, dict) else str(ans)
    grounded = ans.get("grounded", False) if isinstance(ans, dict) else False

    issues = []
    # Check for bad patterns (should NOT appear)
    for bad in bad_signs:
        if bad.lower() in text.lower():
            issues.append(f"BAD_PATTERN: '{bad}' found in response")
    # Check for good patterns (at least one should appear)
    good_found = any(g.lower() in text.lower() for g in good_signs)
    if not good_found and text:
        issues.append(f"MISSING: none of {good_signs[:3]} found")

    status = "PASS" if not issues else "FAIL"
    if status == "PASS":
        pass_count += 1
    print(f"  [{status}] {lat/1000:.1f}s | grounded={grounded} | {len(text)} chars")
    if issues:
        for iss in issues:
            print(f"  ✗ {iss}")
    preview = text[:300].replace("\n", "\\n") if text else "(empty)"
    print(f"  → {preview}")

print(f"\n{'='*70}")
print(f"Result: {pass_count}/{len(TESTS)} passed")
print("=" * 70)
