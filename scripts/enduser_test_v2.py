#!/usr/bin/env python3
"""End-user testing v2 — sequential with long timeouts via /api/ask."""

import json
import time
import requests
import sys

BASE = "http://localhost:8000"
SUB_ID = "67fde0754e36c00b14cea7f5"

RESUME_P1 = "69a6e13623d47adc8e7eeadd"
RESUME_P2 = "69a6dddc23d47adc8e7ee7e4"
RESUME_P3 = "69a553ab23d47adc8e7ee37c"
INVOICE_P = "69a6de9e23d47adc8e7ee90a"
MEDICAL_P = "69a552ac23d47adc8e7ee27a"


def ask(profile_id: str, query: str) -> dict:
    payload = {
        "query": query,
        "profile_id": profile_id,
        "subscription_id": SUB_ID,
        "new_session": True,
        "debug": True,
    }
    start = time.time()
    try:
        resp = requests.post(f"{BASE}/api/ask", json=payload, timeout=300)
        lat = (time.time() - start) * 1000
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json(), "ms": lat}
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}", "ms": lat}
    except Exception as e:
        return {"ok": False, "error": str(e), "ms": (time.time() - start) * 1000}


def extract(result):
    if not result["ok"]:
        return "", [], False, False
    data = result["data"]
    ans = data.get("answer", data)
    if isinstance(ans, dict):
        return ans.get("response", ""), ans.get("sources", []), ans.get("grounded", False), ans.get("context_found", False)
    return str(ans), [], False, False


TESTS = [
    # 1. Factual
    ("factual", RESUME_P1, "What is Gokul's educational qualification?"),
    # 2. Skills
    ("factual_skills", RESUME_P1, "What are Philip Simon Derock's skills?"),
    # 3. Contact
    ("contact", RESUME_P1, "What is Rahul Deshbhratar's email and phone number?"),
    # 4. Comparison
    ("comparison", RESUME_P1, "Compare Gokul and Rajeshkumar's skills and experience"),
    # 5. Ranking
    ("ranking", RESUME_P2, "Who is the most qualified candidate for a data science role?"),
    # 6. Summary
    ("summary", RESUME_P1, "Give me a summary of all candidates"),
    # 7. Cross-doc
    ("cross_doc", RESUME_P1, "Which candidates have Python experience?"),
    # 8. Invoice factual
    ("invoice", INVOICE_P, "What is the total amount on the invoices?"),
    # 9. Medical
    ("medical", MEDICAL_P, "Summarize the patient report"),
    # 10. No-info (should not hallucinate)
    ("no_info", RESUME_P1, "What is the company's annual revenue?"),
    # 11. Generate
    ("generate", RESUME_P1, "Write a professional summary for Gokul based on his resume"),
    # 12. Analytics
    ("analytics", RESUME_P2, "How many candidates have more than 3 years of experience?"),
    # 13. Typo tolerance
    ("typo", RESUME_P1, "Wht is Gokuls educaton?"),
    # 14. Capability
    ("conversational", RESUME_P1, "What can you do?"),
]


def main():
    results = []
    print("=" * 70)
    print(f"DocWain End-User Test — {len(TESTS)} queries")
    print("=" * 70)

    for i, (cat, profile, query) in enumerate(TESTS):
        print(f"\n[{i+1}/{len(TESTS)}] {cat}: {query[:55]}...")
        sys.stdout.flush()

        r = ask(profile, query)
        text, sources, grounded, ctx = extract(r)

        issues = []
        # Check quality
        if not r["ok"]:
            issues.append(f"ERROR: {r['error'][:200]}")
        elif len(text.strip()) < 15:
            issues.append("EMPTY: response too short")
        else:
            # Preamble leak
            for p in ["based on my analysis", "after reviewing", "upon examination", "i have reviewed"]:
                if p in text.lower()[:120]:
                    issues.append(f"PREAMBLE: '{p}'")
                    break
            # Metadata leak
            for m in ["chunk_type", "section_id", "embed_pipeline", "canonical_json", "chunk_kind"]:
                if m in text.lower():
                    issues.append(f"METADATA_LEAK: '{m}'")
                    break
            # Repetition
            sents = [s.strip() for s in text.split(". ") if len(s.strip()) > 30]
            seen = set()
            for s in sents:
                k = s.lower()[:80]
                if k in seen:
                    issues.append("REPETITION")
                    break
                seen.add(k)
            # Category-specific
            if "comparison" in cat and "|" not in text and "vs" not in text.lower() and "**" not in text:
                issues.append("NO_STRUCTURE: comparison lacks table/formatting")
            if "ranking" in cat and not any(c.isdigit() for c in text[:200]):
                issues.append("NO_RANKING: no numbered ordering")
            if "no_info" in cat:
                hallu_signs = ["revenue is", "annual revenue", "the company earns", "$"]
                if any(h in text.lower() for h in hallu_signs) and "not" not in text.lower()[:100]:
                    issues.append("HALLUCINATION: fabricated answer for unknown info")
                elif "not" in text.lower()[:200] or "couldn't" in text.lower()[:200] or "no information" in text.lower()[:200]:
                    pass  # Good — acknowledged missing info
            if not grounded and cat not in ("conversational", "no_info", "generate"):
                issues.append("UNGROUNDED")

        status = "PASS" if not issues else "WARN"
        lat_s = r["ms"] / 1000
        print(f"  [{status}] {lat_s:.1f}s | grounded={grounded} | {len(text)} chars")
        if issues:
            for iss in issues:
                print(f"  ⚠ {iss}")
        # Preview
        preview = text[:250].replace("\n", "\\n") if text else "(empty)"
        print(f"  → {preview}")

        results.append({
            "cat": cat, "query": query, "response": text[:1000],
            "grounded": grounded, "context_found": ctx,
            "latency_s": round(lat_s, 1), "issues": issues,
            "sources": [str(s)[:100] for s in sources[:3]],
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_issues = sum(len(r["issues"]) for r in results)
    passed = sum(1 for r in results if not r["issues"])
    avg_lat = sum(r["latency_s"] for r in results) / len(results) if results else 0
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Total issues: {total_issues}")
    print(f"  Avg latency: {avg_lat:.1f}s")
    print()
    for r in results:
        if r["issues"]:
            print(f"  {r['cat']:20s} → {', '.join(r['issues'])}")

    with open("tests/enduser_test_results.json", "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)
    print(f"\nSaved to tests/enduser_test_results.json")


if __name__ == "__main__":
    main()
