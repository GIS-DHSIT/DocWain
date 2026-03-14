#!/usr/bin/env python3
"""Comprehensive quality audit — tests response quality, not just routing."""

import json
import time
import requests
import sys

BASE = "http://localhost:8000"
SUB_ID = "67fde0754e36c00b14cea7f5"
RESUME_P1 = "69a6e13623d47adc8e7eeadd"
RESUME_P2 = "69a6dddc23d47adc8e7ee7e4"
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
    # ── Entity Scoping (person-targeted queries) ──
    ("ENTITY_SINGLE", RESUME_P1, "What is Gokul's education?",
     {"must_have": ["education", "Karpagam", "B.Tech"],
      "must_not": ["Philip", "Rahul", "Prudhvi"],
      "category": "entity_scoping"}),

    ("ENTITY_CONTACT", RESUME_P1, "What is Philip Simon Derock's email?",
     {"must_have": ["@", "email", "philip"],
      "must_not": ["Rahul", "Gokul", "Prudhvi"],
      "category": "entity_scoping"}),

    # ── Comparison ──
    ("COMPARE", RESUME_P1, "Compare Gokul and Philip Simon Derock",
     {"must_have": ["Gokul", "Philip"],
      "must_not": [],
      "category": "comparison"}),

    # ── Ranking ──
    ("RANK", RESUME_P1, "Who is the most experienced candidate?",
     {"must_have": ["experience", "year"],
      "must_not": ["Here's what you can do", "DocWain"],
      "category": "ranking"}),

    # ── Cross-document ──
    ("CROSS_DOC", RESUME_P1, "What skills are common across all candidates?",
     {"must_have": ["skill"],
      "must_not": ["DocWain", "Here's what"],
      "category": "cross_document"}),

    # ── Specific field extraction ──
    ("FIELD_PHONE", RESUME_P1, "What is Gokul's phone number?",
     {"must_have": ["8838361443"],
      "must_not": ["Philip", "Rahul"],
      "category": "field_extraction"}),

    # ── Response quality (should be clean, not raw data dump) ──
    ("QUALITY_SUMMARY", RESUME_P1, "Give me a brief summary of Gokul's profile",
     {"must_have": ["Gokul"],
      "must_not": ["chunk_type", "section_id", "id:", "embedding_text"],
      "quality_check": "no_raw_dump",
      "category": "quality"}),

    # ── Conversational (should NOT trigger document retrieval) ──
    ("CONV_HELLO", RESUME_P1, "Hello!",
     {"must_have": ["hello", "hi", "help", "welcome", "how can"],
      "must_not": ["experience", "resume", "candidate", "skills"],
      "category": "conversational"}),

    ("CONV_THANKS", RESUME_P1, "Thank you!",
     {"must_have": ["welcome", "glad", "help", "happy"],
      "must_not": ["experience", "resume", "candidate"],
      "category": "conversational"}),

    # ── No-info boundary (query about something NOT in documents) ──
    ("NO_DATA", RESUME_P1, "What is the weather today?",
     {"must_have": ["couldn't find", "not available", "no information", "unable", "don't have"],
      "must_not": [],
      "category": "boundary"}),

    # ── Medical profile ──
    ("MEDICAL", MEDICAL_P, "What are the patient's vital signs?",
     {"must_have": ["vital", "blood", "pressure", "heart", "temperature", "pulse", "patient"],
      "must_not": ["DocWain", "Here's what"],
      "category": "medical"}),

    # ── Follow-up simulation ──
    ("FOLLOWUP", RESUME_P1, "What about his certifications?",
     {"must_have": ["certif"],
      "must_not": [],
      "category": "followup"}),
]

print("=" * 70)
print("DocWain Comprehensive Quality Audit")
print("=" * 70)

results = []
categories = {}
for name, profile, query, checks in TESTS:
    cat = checks.get("category", "other")
    print(f"\n[{name}] {query[:60]}...")
    sys.stdout.flush()

    data, lat = ask(profile, query)
    ans = data.get("answer", data)
    text = ans.get("response", "") if isinstance(ans, dict) else str(ans)
    grounded = ans.get("grounded", False) if isinstance(ans, dict) else False

    issues = []
    # Check must_not patterns
    for bad in checks.get("must_not", []):
        if bad.lower() in text.lower():
            issues.append(f"BAD: '{bad}' found")
    # Check must_have patterns (at least one)
    must_have = checks.get("must_have", [])
    good_found = any(g.lower() in text.lower() for g in must_have)
    if not good_found and text and must_have:
        issues.append(f"MISSING: none of {must_have[:3]}")

    # Quality checks
    if checks.get("quality_check") == "no_raw_dump":
        # Check for raw data dump indicators
        raw_indicators = ["chunk_type", "section_id", "embedding_text", "canonical_text",
                         "id: ", "person:", "organization:", "subscription_id"]
        raw_found = [r for r in raw_indicators if r.lower() in text.lower()]
        if raw_found:
            issues.append(f"RAW_DUMP: {raw_found}")

    # Latency check
    if lat > 60000:
        issues.append(f"SLOW: {lat/1000:.1f}s")

    status = "PASS" if not issues else "FAIL"
    if cat not in categories:
        categories[cat] = {"pass": 0, "fail": 0}
    categories[cat]["pass" if status == "PASS" else "fail"] += 1

    result = {
        "name": name, "query": query, "status": status,
        "latency_ms": lat, "grounded": grounded,
        "response_len": len(text), "issues": issues,
        "response_preview": text[:200],
    }
    results.append(result)

    print(f"  [{status}] {lat/1000:.1f}s | grounded={grounded} | {len(text)} chars")
    if issues:
        for iss in issues:
            print(f"  ✗ {iss}")
    preview = text[:250].replace("\n", "\\n") if text else "(empty)"
    print(f"  → {preview}")

print(f"\n{'='*70}")
print("Category Results:")
for cat, counts in sorted(categories.items()):
    total = counts["pass"] + counts["fail"]
    print(f"  {cat}: {counts['pass']}/{total}")
total_pass = sum(c["pass"] for c in categories.values())
total_tests = len(TESTS)
print(f"\nOverall: {total_pass}/{total_tests} passed")
print("=" * 70)

# Save results
with open("tests/quality_audit_results.json", "w") as f:
    json.dump({"results": results, "categories": categories, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
print(f"\nResults saved to tests/quality_audit_results.json")
