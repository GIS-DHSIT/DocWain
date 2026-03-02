#!/usr/bin/env python3
"""Deep end-to-end quality analysis for DocWain.

Tests:
  1. Context retrieval accuracy — does the system retrieve the RIGHT documents?
  2. Response intelligence — analytical depth, patterns, statistics
  3. Cross-document reasoning — synthesis across multiple documents
  4. Agent mode vs standard mode — quality/latency comparison
  5. Internet-enabled queries — accuracy and source attribution
  6. Tool-specific queries — domain-specific extraction quality
  7. Edge cases — ambiguous queries, follow-ups, meta questions
  8. Invoice metadata (new A5 fix) — invoice numbers and dates
"""
import json
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://localhost:8000"
SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
HR_PROFILE_ID = "6994620d6034385742e45abe"
MEDICAL_PROFILE_ID = "699467806034385742e45b9f"
INVOICE_PROFILE_ID = "699c8f9c1aed615c51b0a4dc"
INSURANCE_PROFILE_ID = "6995a20b2f85f720e96fa486"
TIMEOUT = 300.0


def ask(
    query: str,
    tools: Optional[List[str]] = None,
    enable_internet: bool = False,
    agent_mode: Optional[bool] = None,
    profile_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "profile_id": profile_id or HR_PROFILE_ID,
        "subscription_id": subscription_id or SUBSCRIPTION_ID,
        "user_id": "deep_test@docwain.ai",
        "enable_internet": enable_internet,
    }
    if tools:
        payload["tools"] = tools
        payload["use_tools"] = True
    if agent_mode is not None:
        payload["agent_mode"] = agent_mode
    if session_id:
        payload["session_id"] = session_id

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/api/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


def extract_response(result: Dict) -> str:
    return str(result.get("answer", {}).get("response", ""))


def extract_metadata(result: Dict) -> Dict:
    return result.get("answer", {}).get("metadata", {})


def extract_sources(result: Dict) -> List:
    return result.get("answer", {}).get("sources", [])


# ──────────────────────────────────────────────────────────────────────────────
# Quality check helpers
# ──────────────────────────────────────────────────────────────────────────────

BANNED_PHRASES = [
    "not explicitly mentioned", "i don't have access", "i cannot",
    "i'm unable", "as an ai assistant", "as an ai language model",
    "MISSING_REASON", "Not enough information", "unfortunately, i",
    "tool:resumes", "tool:medical", "tool:insights", "tool:lawhere",
]

INTELLIGENCE_SIGNALS = [
    "across", "average", "range", "total", "compared", "pattern",
    "common", "unique", "distribution", "highest", "lowest",
    "shared", "distinct", "overview", "analyzed", "candidates analyzed",
]


def check_quality(response_text: str, expected_signals: List[str], test_name: str) -> Dict:
    lower = response_text.lower()
    length = len(response_text)

    signals_found = [s for s in expected_signals if s.lower() in lower]
    signals_missing = [s for s in expected_signals if s.lower() not in lower]
    banned_found = [p for p in BANNED_PHRASES if p.lower() in lower]
    intelligence_found = [s for s in INTELLIGENCE_SIGNALS if s in lower]

    # Score: 0-100
    signal_ratio = len(signals_found) / max(len(expected_signals), 1)
    score = 0
    if signal_ratio >= 0.5: score += 30
    if signal_ratio >= 1.0: score += 20
    if length > 100: score += 15
    if length > 300: score += 5
    if not banned_found: score += 15
    if intelligence_found: score += 15

    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"

    return {
        "test": test_name,
        "grade": grade,
        "score": score,
        "length": length,
        "signals_found": signals_found,
        "signals_missing": signals_missing,
        "banned_found": banned_found,
        "intelligence_signals": intelligence_found,
        "preview": response_text[:400],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test Categories
# ──────────────────────────────────────────────────────────────────────────────

def test_context_retrieval():
    """Test 1: Context retrieval accuracy — right documents, right content."""
    print("\n" + "=" * 80)
    print("  TEST 1: Context Retrieval Accuracy")
    print("=" * 80)

    results = []

    # 1a. Specific person query — should find ONLY that person's resume
    print("  [1a] Specific person lookup...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Tell me about the candidate with the most Python experience", tools=["resumes"])
        text = extract_response(r)
        sources = extract_sources(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["python", "experience", "years"], "specific_person")
        q["elapsed"] = round(elapsed, 1)
        q["sources_count"] = len(sources)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars | {len(sources)} sources")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "specific_person", "grade": "F", "error": str(e)[:200]})

    # 1b. Multi-document retrieval — should use ALL resumes
    print("  [1b] Multi-doc retrieval...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("List every candidate's name and their primary skill", tools=["resumes"])
        text = extract_response(r)
        sources = extract_sources(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["candidate", "skill"], "multi_doc_retrieval")
        q["elapsed"] = round(elapsed, 1)
        q["sources_count"] = len(sources)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars | {len(sources)} sources")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "multi_doc_retrieval", "grade": "F", "error": str(e)[:200]})

    # 1c. Cross-profile isolation — medical profile should NOT return HR data
    print("  [1c] Profile isolation...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("List all candidate resumes", tools=["resumes"], profile_id=MEDICAL_PROFILE_ID)
        text = extract_response(r)
        elapsed = time.time() - t0
        # Should fail gracefully or mention no resumes found
        has_resume_data = any(w in text.lower() for w in ["technical skills", "candidate:", "python", "java"])
        q = {
            "test": "profile_isolation",
            "grade": "A" if not has_resume_data else "F",
            "score": 100 if not has_resume_data else 0,
            "length": len(text),
            "elapsed": round(elapsed, 1),
            "preview": text[:300],
            "note": "Should NOT contain HR resume data from wrong profile",
        }
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | isolation={'PASS' if not has_resume_data else 'FAIL'}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "profile_isolation", "grade": "F", "error": str(e)[:200]})

    # 1d. Invoice profile — should return invoice data
    print("  [1d] Invoice profile accuracy...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("What vendors appear in the invoices?", profile_id=INVOICE_PROFILE_ID)
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["vendor", "invoice"], "invoice_profile")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "invoice_profile", "grade": "F", "error": str(e)[:200]})

    return results


def test_response_intelligence():
    """Test 2: Analytical depth — patterns, statistics, synthesis."""
    print("\n" + "=" * 80)
    print("  TEST 2: Response Intelligence & Analytical Depth")
    print("=" * 80)

    results = []

    # 2a. Statistical summary — should include counts, averages, ranges
    print("  [2a] Statistical summary...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Analyze the experience levels across all candidates", tools=["resumes"])
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["experience", "years", "candidate"], "statistical_summary")
        q["elapsed"] = round(elapsed, 1)
        # Extra check: does it have actual numbers/statistics?
        import re
        has_numbers = bool(re.search(r'\d+\s*(?:years?|candidates?)', text.lower()))
        q["has_statistics"] = has_numbers
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | stats={'YES' if has_numbers else 'NO'} | intel_signals={len(q['intelligence_signals'])}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "statistical_summary", "grade": "F", "error": str(e)[:200]})

    # 2b. Pattern detection — skill commonality
    print("  [2b] Pattern detection...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("What skills are shared across multiple candidates, and which skills are unique?", tools=["resumes"])
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["skill", "common", "candidate"], "pattern_detection")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars | intel_signals={len(q['intelligence_signals'])}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "pattern_detection", "grade": "F", "error": str(e)[:200]})

    # 2c. Ranking with justification
    print("  [2c] Ranking with justification...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Rank the top 3 candidates for a senior data engineer role and explain why", tools=["resumes"])
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["1.", "data", "engineer", "experience"], "ranking_justified")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "ranking_justified", "grade": "F", "error": str(e)[:200]})

    # 2d. Cross-document comparison
    print("  [2d] Cross-document comparison...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Compare the car and bike insurance policies. What are the key differences in coverage and premium?",
                profile_id=INSURANCE_PROFILE_ID)
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["car", "bike", "coverage", "premium"], "cross_doc_comparison")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "cross_doc_comparison", "grade": "F", "error": str(e)[:200]})

    return results


def test_agent_vs_standard():
    """Test 3: Agent mode vs standard mode comparison."""
    print("\n" + "=" * 80)
    print("  TEST 3: Agent Mode vs Standard Mode")
    print("=" * 80)

    queries = [
        ("What are the most in-demand skills across all candidates?", ["resumes"], HR_PROFILE_ID),
        ("Summarize the key coverage points of the car insurance policy", None, INSURANCE_PROFILE_ID),
        ("What medications are prescribed and at what dosages?", ["medical"], MEDICAL_PROFILE_ID),
    ]

    results = []
    for query, tools, profile_id in queries:
        short_q = query[:50]

        # Standard mode
        print(f"  [STD] {short_q}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r_std = ask(query, tools=tools, profile_id=profile_id, agent_mode=False)
            text_std = extract_response(r_std)
            elapsed_std = time.time() - t0
            q_std = check_quality(text_std, [], f"std_{short_q[:20]}")
            q_std["elapsed"] = round(elapsed_std, 1)
            q_std["mode"] = "standard"
            print(f"[{q_std['grade']}] {elapsed_std:.1f}s | {q_std['length']} chars")
        except Exception as e:
            print(f"[ERROR] {e}")
            q_std = {"test": f"std_{short_q[:20]}", "grade": "F", "mode": "standard", "error": str(e)[:200]}

        # Agent mode
        print(f"  [AGT] {short_q}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r_agt = ask(query, tools=tools, profile_id=profile_id, agent_mode=True)
            text_agt = extract_response(r_agt)
            elapsed_agt = time.time() - t0
            q_agt = check_quality(text_agt, [], f"agt_{short_q[:20]}")
            q_agt["elapsed"] = round(elapsed_agt, 1)
            q_agt["mode"] = "agent"
            print(f"[{q_agt['grade']}] {elapsed_agt:.1f}s | {q_agt['length']} chars")
        except Exception as e:
            print(f"[ERROR] {e}")
            q_agt = {"test": f"agt_{short_q[:20]}", "grade": "F", "mode": "agent", "error": str(e)[:200]}

        results.append({"query": query, "standard": q_std, "agent": q_agt})
        print()

    return results


def test_internet_enabled():
    """Test 4: Internet-enabled queries."""
    print("\n" + "=" * 80)
    print("  TEST 4: Internet-Enabled Queries")
    print("=" * 80)

    queries = [
        {
            "query": "What is the latest stable version of Python?",
            "expect": ["python", "3."],
            "name": "python_version",
        },
        {
            "query": "What are the current trends in AI and machine learning in 2026?",
            "expect": ["ai", "machine learning"],
            "name": "ai_trends",
        },
        {
            "query": "Search the web for the best practices in resume screening using AI",
            "expect": ["resume", "screen"],
            "name": "web_search_explicit",
        },
    ]

    results = []
    for qdef in queries:
        print(f"  [{qdef['name']}] {qdef['query'][:55]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = ask(qdef["query"], enable_internet=True)
            text = extract_response(r)
            meta = extract_metadata(r)
            sources = extract_sources(r)
            elapsed = time.time() - t0
            q = check_quality(text, qdef["expect"], qdef["name"])
            q["elapsed"] = round(elapsed, 1)
            q["web_search"] = meta.get("web_search", False)
            q["source_type"] = meta.get("source_type", "unknown")
            q["sources_count"] = len(sources)
            # Check for web sources
            web_sources = [s for s in sources if s.get("type") == "web" or "http" in str(s.get("file_name", ""))]
            q["web_sources"] = len(web_sources)
            results.append(q)
            print(f"[{q['grade']}] {elapsed:.1f}s | web={q['web_search']} | {q['length']} chars")
        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({"test": qdef["name"], "grade": "F", "error": str(e)[:200]})

    return results


def test_tool_specific():
    """Test 5: Tool-specific extraction quality."""
    print("\n" + "=" * 80)
    print("  TEST 5: Tool-Specific Extraction Quality")
    print("=" * 80)

    results = []

    # 5a. Invoice metadata (A5 fix)
    print("  [5a] Invoice metadata extraction...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("List all invoice numbers and dates", profile_id=INVOICE_PROFILE_ID)
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["invoice"], "invoice_metadata")
        q["elapsed"] = round(elapsed, 1)
        # Check for actual invoice numbers/dates in response
        import re
        has_inv_numbers = bool(re.search(r'(?:INV|inv|QUT|PO)[-\s]?\d+', text))
        q["has_invoice_numbers"] = has_inv_numbers
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | inv_numbers={'YES' if has_inv_numbers else 'NO'}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "invoice_metadata", "grade": "F", "error": str(e)[:200]})

    # 5b. Medical extraction accuracy
    print("  [5b] Medical extraction...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("What are all the patient diagnoses and prescribed medications?",
                tools=["medical"], profile_id=MEDICAL_PROFILE_ID)
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["diagnos", "medication"], "medical_extraction")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "medical_extraction", "grade": "F", "error": str(e)[:200]})

    # 5c. Email drafting
    print("  [5c] Email drafting...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Draft a professional rejection email for candidates who didn't make the shortlist",
                tools=["email_drafting"])
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["dear", "thank", "position"], "email_drafting")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "email_drafting", "grade": "F", "error": str(e)[:200]})

    # 5d. Action items extraction
    print("  [5d] Action items...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Extract all action items and deadlines from the documents",
                tools=["action_items"])
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["action"], "action_items")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "action_items", "grade": "F", "error": str(e)[:200]})

    # 5e. Insights tool
    print("  [5e] Insights extraction...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("What insights and patterns can you identify across all resumes?",
                tools=["insights"])
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["insight", "skill"], "insights_tool")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "insights_tool", "grade": "F", "error": str(e)[:200]})

    # 5f. Legal analysis (using lawhere tool on insurance docs)
    print("  [5f] Legal/policy analysis...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("What are the exclusions and limitations in the insurance policies?",
                tools=["lawhere"], profile_id=INSURANCE_PROFILE_ID)
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["exclusion"], "legal_policy")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "legal_policy", "grade": "F", "error": str(e)[:200]})

    return results


def test_edge_cases():
    """Test 6: Edge cases — ambiguous queries, meta questions, tool name leak."""
    print("\n" + "=" * 80)
    print("  TEST 6: Edge Cases & Quality Gates")
    print("=" * 80)

    results = []

    # 6a. Tool name leak (A1 fix)
    print("  [6a] Tool name leak check...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Summarize the top candidate", tools=["resumes"])
        text = extract_response(r)
        elapsed = time.time() - t0
        has_leak = "tool:resumes" in text or "tool:resume" in text
        q = {
            "test": "tool_name_leak",
            "grade": "A" if not has_leak else "F",
            "score": 100 if not has_leak else 0,
            "length": len(text),
            "elapsed": round(elapsed, 1),
            "leak_found": has_leak,
            "preview": text[:300],
        }
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | leak={'FOUND' if has_leak else 'NONE'}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "tool_name_leak", "grade": "F", "error": str(e)[:200]})

    # 6b. Query echo check (A2 fix)
    print("  [6b] Query echo/stutter check...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Extract all action items from the documents", tools=["action_items"])
        text = extract_response(r)
        elapsed = time.time() - t0
        # Check for echo: response starts with something like "I'll extract Extract all..."
        has_echo = text.lower().startswith("i'll extract extract") or text.lower().startswith("i'll list list")
        q = {
            "test": "query_echo",
            "grade": "A" if not has_echo else "F",
            "score": 100 if not has_echo else 0,
            "length": len(text),
            "elapsed": round(elapsed, 1),
            "echo_found": has_echo,
            "preview": text[:300],
        }
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | echo={'FOUND' if has_echo else 'NONE'}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "query_echo", "grade": "F", "error": str(e)[:200]})

    # 6c. Meta question handling
    print("  [6c] Meta question (What can DocWain do?)...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("What can DocWain do?")
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["document", "help"], "meta_question")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "meta_question", "grade": "F", "error": str(e)[:200]})

    # 6d. Greeting handling
    print("  [6d] Greeting (Hi there)...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("Hi there")
        text = extract_response(r)
        elapsed = time.time() - t0
        has_greeting = any(w in text.lower() for w in ["hello", "hi", "welcome", "docwain", "help"])
        q = {
            "test": "greeting",
            "grade": "A" if has_greeting else "C",
            "score": 100 if has_greeting else 50,
            "length": len(text),
            "elapsed": round(elapsed, 1),
            "preview": text[:300],
        }
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | greeting={'HANDLED' if has_greeting else 'MISSED'}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "greeting", "grade": "F", "error": str(e)[:200]})

    # 6e. Confidence score check (A3 fix)
    print("  [6e] Confidence score validity...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("List all candidates and their skills", tools=["resumes"])
        meta = extract_metadata(r)
        elapsed = time.time() - t0
        confidence = meta.get("confidence", {})
        evidence_cov = confidence.get("dimensions", {}).get("evidence_coverage", -1)
        overall = confidence.get("score", -1)
        is_valid = evidence_cov > 0.0 and overall > 0.0
        q = {
            "test": "confidence_score",
            "grade": "A" if is_valid else "F",
            "score": 100 if is_valid else 0,
            "elapsed": round(elapsed, 1),
            "confidence_overall": overall,
            "evidence_coverage": evidence_cov,
            "note": "evidence_coverage should be > 0 (A3 fix)",
        }
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | confidence={overall:.3f} | evidence_cov={evidence_cov:.3f}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "confidence_score", "grade": "F", "error": str(e)[:200]})

    # 6f. Empty/no-results handling
    print("  [6f] No results handling...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("What is the quantum entanglement coefficient of the 5th document?")
        text = extract_response(r)
        elapsed = time.time() - t0
        # Should gracefully say not found, not crash
        is_graceful = len(text) > 20 and "error" not in text.lower()[:50]
        q = {
            "test": "no_results",
            "grade": "A" if is_graceful else "F",
            "score": 100 if is_graceful else 0,
            "length": len(text),
            "elapsed": round(elapsed, 1),
            "preview": text[:300],
        }
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | graceful={'YES' if is_graceful else 'NO'}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "no_results", "grade": "F", "error": str(e)[:200]})

    return results


def test_complex_queries():
    """Test 7: Complex multi-step reasoning queries."""
    print("\n" + "=" * 80)
    print("  TEST 7: Complex Multi-Step Reasoning")
    print("=" * 80)

    results = []

    # 7a. Multi-criteria analysis
    print("  [7a] Multi-criteria analysis...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask(
            "Which candidates have both cloud certifications (AWS, Azure, GCP) AND "
            "more than 5 years of experience? List them with their specific certifications.",
            tools=["resumes"]
        )
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["certif", "experience", "candidate"], "multi_criteria")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "multi_criteria", "grade": "F", "error": str(e)[:200]})

    # 7b. Aggregation query
    print("  [7b] Aggregation query...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask("How many candidates have Python skills? What percentage of total candidates is that?",
                tools=["resumes"])
        text = extract_response(r)
        elapsed = time.time() - t0
        import re
        has_count = bool(re.search(r'\d+\s*(?:of|out of|candidates?|%|percent)', text.lower()))
        q = check_quality(text, ["python", "candidate"], "aggregation")
        q["elapsed"] = round(elapsed, 1)
        q["has_count"] = has_count
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | count={'YES' if has_count else 'NO'}")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "aggregation", "grade": "F", "error": str(e)[:200]})

    # 7c. Conditional reasoning
    print("  [7c] Conditional reasoning...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = ask(
            "If I need someone for a full-stack role who can work with both frontend "
            "and backend technologies, who would be the best fit and why?",
            tools=["resumes"]
        )
        text = extract_response(r)
        elapsed = time.time() - t0
        q = check_quality(text, ["candidate", "frontend", "backend"], "conditional_reasoning")
        q["elapsed"] = round(elapsed, 1)
        results.append(q)
        print(f"[{q['grade']}] {elapsed:.1f}s | {q['length']} chars")
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({"test": "conditional_reasoning", "grade": "F", "error": str(e)[:200]})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    print(f"\n{'#' * 80}")
    print(f"  DOCWAIN DEEP END-TO-END QUALITY ANALYSIS")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")

    all_results = {}

    all_results["context_retrieval"] = test_context_retrieval()
    all_results["response_intelligence"] = test_response_intelligence()
    all_results["agent_vs_standard"] = test_agent_vs_standard()
    all_results["internet_enabled"] = test_internet_enabled()
    all_results["tool_specific"] = test_tool_specific()
    all_results["edge_cases"] = test_edge_cases()
    all_results["complex_queries"] = test_complex_queries()

    # ── Overall Summary ─────────────────────────────────────────────
    print(f"\n{'#' * 80}")
    print("  OVERALL SUMMARY")
    print(f"{'#' * 80}")

    total_tests = 0
    total_pass = 0
    total_fail = 0
    issues = []

    for category, tests in all_results.items():
        if category == "agent_vs_standard":
            # Special handling for comparison tests
            for t in tests:
                for mode in ["standard", "agent"]:
                    total_tests += 1
                    result = t.get(mode, {})
                    grade = result.get("grade", "F")
                    if grade in ("A", "B"):
                        total_pass += 1
                    elif grade in ("D", "F"):
                        total_fail += 1
                        issues.append(f"[{category}] {result.get('test', '?')} ({mode}): {grade}")
        else:
            for t in tests:
                total_tests += 1
                grade = t.get("grade", "F")
                if grade in ("A", "B"):
                    total_pass += 1
                elif grade in ("D", "F"):
                    total_fail += 1
                    issues.append(f"[{category}] {t.get('test', '?')}: {grade} — {t.get('preview', t.get('error', ''))[:100]}")

    c_count = total_tests - total_pass - total_fail

    print(f"\n  Total tests: {total_tests}")
    print(f"  Pass (A/B):  {total_pass}")
    print(f"  Warning (C): {c_count}")
    print(f"  Fail (D/F):  {total_fail}")
    print(f"  Pass rate:   {total_pass/max(total_tests,1)*100:.0f}%")

    if issues:
        print(f"\n  ISSUES REQUIRING ATTENTION:")
        for issue in issues:
            print(f"    - {issue}")

    # Save full results
    output_path = "/tmp/deep_test_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_all_tests()
