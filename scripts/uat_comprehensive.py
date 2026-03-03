#!/usr/bin/env python3
"""Comprehensive UAT for DocWain — tests all system capabilities.

Tests: Agents (all 11), Screening, Embedding, Retrieval Accuracy,
Response Intelligence, Agentic API endpoints, /ask pipeline.
"""
import json
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://localhost:8000"
SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
HR_PROFILE = "6994620d6034385742e45abe"
MEDICAL_PROFILE = "699467806034385742e45b9f"
INVOICE_SUBSCRIPTION = "67fde0754e36c00b14cea7f5"
INVOICE_PROFILE = "699c8f9c1aed615c51b0a4dc"
INSURANCE_PROFILE = "6995a20b2f85f720e96fa486"
TIMEOUT = 300.0

BANNED_PHRASES = [
    "not explicitly mentioned",
    "not enough information",
    "i don't have",
    "i cannot",
    "no relevant",
    "unable to find",
    "MISSING_REASON",
    "as an ai",
    "i'm an artificial",
]

client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


def _ask(query: str, profile_id=None, subscription_id=None,
         enable_internet=False, agent_name=None, agent_task=None,
         session_id="uat-session") -> Dict:
    payload = {
        "query": query,
        "subscription_id": subscription_id or SUBSCRIPTION_ID,
        "profile_id": profile_id or HR_PROFILE,
        "session_id": session_id,
    }
    if enable_internet:
        payload["enable_internet"] = True
    if agent_name:
        payload["agent_name"] = agent_name
    if agent_task:
        payload["agent_task"] = agent_task
    resp = client.post("/api/ask", json=payload)
    return resp.json()


def _agent_execute(agent_name: str, task_type: str, query: str = "",
                   text: str = "", profile_id=None, subscription_id=None) -> Dict:
    """Call POST /api/agents/{agent_name}/execute."""
    payload = {
        "task_type": task_type,
        "input": {"query": query} if query else {},
        "context": {"query": query},
        "options": {},
    }
    if text:
        payload["input"]["text"] = text
        payload["context"]["text"] = text
    if subscription_id:
        payload["subscription_id"] = subscription_id
    if profile_id:
        payload["profile_id"] = profile_id
    resp = client.post(f"/api/agents/{agent_name}/execute", json=payload)
    return resp.json()


def _grade(test_id: str, response: Dict, expect: List[str],
           check_grounded: bool = True) -> Dict:
    answer = response.get("answer", {})
    text = answer.get("response", "") if isinstance(answer, dict) else str(answer)
    text_lower = text.lower()

    for bp in BANNED_PHRASES:
        if bp.lower() in text_lower:
            return {"id": test_id, "grade": "F", "pass": False,
                    "reason": f"Banned phrase: '{bp}'", "response_len": len(text), "score": 0}

    matched = [kw for kw in expect if kw.lower() in text_lower]
    match_ratio = len(matched) / len(expect) if expect else 1.0

    grounded = answer.get("grounded", False) if isinstance(answer, dict) else False
    context_found = answer.get("context_found", False) if isinstance(answer, dict) else False
    sources = answer.get("sources", []) if isinstance(answer, dict) else []

    score = 0
    if match_ratio >= 0.5:
        score += 40
    if match_ratio >= 1.0:
        score += 20
    if len(text) > 100:
        score += 15
    if len(text) > 300:
        score += 10
    if check_grounded and grounded:
        score += 10
    if sources:
        score += 5

    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 50 else "D" if score >= 30 else "F"
    passed = match_ratio >= 0.5 and len(text) > 20

    return {
        "id": test_id, "grade": grade, "pass": passed,
        "match_ratio": match_ratio, "matched": matched,
        "response_len": len(text), "grounded": grounded,
        "context_found": context_found, "sources": len(sources),
        "score": score,
    }


def _print_result(result: Dict, elapsed: float, text_preview: str = ""):
    status = "\033[92mPASS\033[0m" if result["pass"] else "\033[91mFAIL\033[0m"
    preview = text_preview[:120].replace("\n", " ") if text_preview else ""
    print(f"  [{status}] {result['id']}: grade={result['grade']} "
          f"score={result.get('score', '?')} time={elapsed:.1f}s "
          f"len={result['response_len']} "
          f"matched={result.get('matched', [])}")
    if preview:
        print(f"         Preview: {preview}...")
    if not result["pass"]:
        print(f"         REASON: {result.get('reason', 'Low keyword match')}")
    sys.stdout.flush()


# =========================================================================
# TEST BATCHES
# =========================================================================

def test_health_and_endpoints():
    """Batch 0: Health checks and endpoint discovery."""
    print("\n" + "=" * 70)
    print("BATCH 0: Health & Endpoint Discovery")
    print("=" * 70)
    sys.stdout.flush()
    results = []

    t0 = time.time()
    resp = client.get("/api/health")
    elapsed = time.time() - t0
    ok = resp.status_code == 200 and resp.json().get("status") == "healthy"
    results.append({"id": "HEALTH-01", "pass": ok, "grade": "A" if ok else "F", "score": 100 if ok else 0, "response_len": 0})
    print(f"  [{'PASS' if ok else 'FAIL'}] HEALTH-01: status={resp.json().get('status')} time={elapsed:.1f}s")

    t0 = time.time()
    resp = client.get("/api/agents")
    elapsed = time.time() - t0
    data = resp.json()
    agents = data.get("agents", [])
    ok = resp.status_code == 200 and len(agents) >= 11
    results.append({"id": "AGENTS-LIST", "pass": ok, "grade": "A" if ok else "F", "score": 100 if ok else 0, "response_len": 0})
    print(f"  [{'PASS' if ok else 'FAIL'}] AGENTS-LIST: count={len(agents)} time={elapsed:.1f}s")
    for a in agents:
        print(f"         {a['name']}: {a.get('capabilities', [])}")

    for agent_name in ["hr", "medical", "legal", "invoice", "content", "translation", "education", "image", "web", "analytics", "screening"]:
        resp = client.get(f"/api/agents/{agent_name}")
        ok = resp.status_code == 200
        results.append({"id": f"AGENT-INFO-{agent_name}", "pass": ok, "grade": "A" if ok else "F", "score": 100 if ok else 0, "response_len": 0})
        if not ok:
            print(f"  [FAIL] AGENT-INFO-{agent_name}: status={resp.status_code}")

    resp = client.get("/api/admin/domain-agents/status")
    ok = resp.status_code == 200
    results.append({"id": "AGENTS-STATUS", "pass": ok, "grade": "A" if ok else "F", "score": 100 if ok else 0, "response_len": 0})
    print(f"  [{'PASS' if ok else 'FAIL'}] AGENTS-STATUS: time={time.time()-t0:.1f}s")
    sys.stdout.flush()
    return results


def test_retrieval_accuracy():
    """Batch 1: Response retrieval accuracy via /api/ask."""
    print("\n" + "=" * 70)
    print("BATCH 1: Retrieval Accuracy (RAG Pipeline)")
    print("=" * 70)
    sys.stdout.flush()
    results = []
    queries = [
        ("RA-01", "How many documents are in this profile?", HR_PROFILE, None, ["document"], False),
        ("RA-02", "What types of documents do I have?", HR_PROFILE, None, ["resume"], False),
        ("RA-03", "List all candidates and their skills", HR_PROFILE, None, ["skill"], True),
        ("RA-04", "What programming languages are mentioned across resumes?", HR_PROFILE, None, ["python"], True),
        ("RA-05", "What are the patient diagnoses?", MEDICAL_PROFILE, None, ["patient"], True),
        ("RA-06", "What are the total amounts on all invoices?", INVOICE_PROFILE, INVOICE_SUBSCRIPTION, ["total", "amount"], True),
        ("RA-07", "List all invoice numbers and dates", INVOICE_PROFILE, INVOICE_SUBSCRIPTION, ["invoice"], True),
        ("RA-08", "What is covered under the car insurance policy?", INSURANCE_PROFILE, None, ["cover"], True),
    ]

    for test_id, query, profile, sub_id, expect, grounded in queries:
        t0 = time.time()
        try:
            resp = _ask(query, profile_id=profile, subscription_id=sub_id)
            elapsed = time.time() - t0
            answer = resp.get("answer", {})
            text = answer.get("response", "") if isinstance(answer, dict) else str(answer)
            result = _grade(test_id, resp, expect, check_grounded=grounded)
            _print_result(result, elapsed, text)
        except Exception as e:
            elapsed = time.time() - t0
            result = {"id": test_id, "pass": False, "grade": "F", "score": 0,
                      "response_len": 0, "reason": str(e)}
            _print_result(result, elapsed)
        results.append(result)
    return results


def test_response_intelligence():
    """Batch 2: Response intelligence features."""
    print("\n" + "=" * 70)
    print("BATCH 2: Response Intelligence")
    print("=" * 70)
    sys.stdout.flush()
    results = []
    queries = [
        ("RI-01", "Compare all candidates for a Python developer role", HR_PROFILE, None, ["python", "candidate"]),
        ("RI-02", "Who has the most experience?", HR_PROFILE, None, ["experience"]),
        ("RI-03", "What are the contact details for all candidates?", HR_PROFILE, None, ["email"]),
        ("RI-04", "Compare the car and bike insurance policies", INSURANCE_PROFILE, None, ["car", "bike"]),
    ]

    for test_id, query, profile, sub_id, expect in queries:
        t0 = time.time()
        try:
            resp = _ask(query, profile_id=profile, subscription_id=sub_id)
            elapsed = time.time() - t0
            answer = resp.get("answer", {})
            text = answer.get("response", "") if isinstance(answer, dict) else str(answer)
            meta = answer.get("metadata", {}) if isinstance(answer, dict) else {}
            result = _grade(test_id, resp, expect)

            intel_info = []
            if "confidence" in meta:
                conf = meta["confidence"]
                score_val = conf.get("score")
                if isinstance(score_val, (int, float)):
                    intel_info.append(f"confidence={score_val:.2f}")
                else:
                    intel_info.append(f"confidence={conf.get('level', '?')}")
            if "suggested_followups" in meta:
                intel_info.append(f"followups={len(meta['suggested_followups'])}")
            intel_str = " | ".join(intel_info) if intel_info else "no-intel-metadata"

            _print_result(result, elapsed, text)
            print(f"         Intelligence: {intel_str}")
        except Exception as e:
            elapsed = time.time() - t0
            result = {"id": test_id, "pass": False, "grade": "F", "score": 0,
                      "response_len": 0, "reason": str(e)}
            _print_result(result, elapsed)
        results.append(result)
        sys.stdout.flush()
    return results


def test_agents_via_ask():
    """Batch 3: Agent invocation via /ask endpoint with agent_name parameter."""
    print("\n" + "=" * 70)
    print("BATCH 3: Agent Invocation via /ask (agent_name)")
    print("=" * 70)
    sys.stdout.flush()
    results = []
    queries = [
        # (test_id, agent_name, agent_task, query, profile, sub_id, expect)
        ("ASK-AG-01", "hr", "generate_interview_questions", "Prepare interview questions for Python developer candidates", HR_PROFILE, None, ["interview", "question"]),
        ("ASK-AG-02", "hr", "skill_gap_analysis", "Analyze skill gaps across all candidates", HR_PROFILE, None, ["skill"]),
        ("ASK-AG-03", "medical", "drug_interaction_check", "Check for any drug interactions in patient records", MEDICAL_PROFILE, None, ["drug"]),
        ("ASK-AG-04", "invoice", "payment_anomaly_detection", "Find any anomalies across all invoices", INVOICE_PROFILE, INVOICE_SUBSCRIPTION, ["invoice"]),
        ("ASK-AG-05", "content", "draft_email", "Draft an email summarizing the top candidate", HR_PROFILE, None, ["email"]),
        ("ASK-AG-06", "education", "explain_concept", "Explain how insurance deductibles work", INSURANCE_PROFILE, None, ["deductible"]),
        ("ASK-AG-07", "analytics", "find_patterns", "Find patterns across all the resumes", HR_PROFILE, None, ["pattern"]),
    ]

    for test_id, agent, task, query, profile, sub_id, expect in queries:
        t0 = time.time()
        try:
            resp = _ask(query, profile_id=profile, subscription_id=sub_id,
                        agent_name=agent, agent_task=task)
            elapsed = time.time() - t0
            answer = resp.get("answer", {})
            text = answer.get("response", "") if isinstance(answer, dict) else str(answer)
            result = _grade(test_id, resp, expect, check_grounded=False)
            _print_result(result, elapsed, text)
        except Exception as e:
            elapsed = time.time() - t0
            result = {"id": test_id, "pass": False, "grade": "F", "score": 0,
                      "response_len": 0, "reason": str(e)}
            _print_result(result, elapsed)
        results.append(result)
        sys.stdout.flush()
    return results


def test_agents_via_endpoints():
    """Batch 4: Agent invocation via /api/agents/{name}/execute endpoint."""
    print("\n" + "=" * 70)
    print("BATCH 4: Agentic API Endpoints (/api/agents/{name}/execute)")
    print("=" * 70)
    sys.stdout.flush()
    results = []
    tests = [
        # (test_id, agent_name, task_type, query, text, profile, sub_id)
        ("EP-AG-01", "hr", "candidate_summary", "Summarize the top candidates", "", HR_PROFILE, SUBSCRIPTION_ID),
        ("EP-AG-02", "medical", "clinical_summary", "Summarize the clinical records", "", MEDICAL_PROFILE, SUBSCRIPTION_ID),
        ("EP-AG-03", "legal", "clause_risk_assessment", "Analyze risky clauses", "", None, None),
        ("EP-AG-04", "content", "generate_content", "Create a professional summary",
         "John Smith is a Python developer with 5 years experience at Google.", None, None),
        ("EP-AG-05", "translation", "detect_language", "",
         "Bonjour le monde, comment allez-vous aujourd'hui?", None, None),
        ("EP-AG-06", "analytics", "generate_report", "Generate analytics report", "", HR_PROFILE, SUBSCRIPTION_ID),
        ("EP-AG-07", "screening", "assess_readability", "",
         "The cat sat on the mat. It was a sunny day.", None, None),
        ("EP-AG-08", "screening", "screen_pii", "",
         "John Smith, SSN 123-45-6789, email john@example.com, DOB 1990-01-15", None, None),
        ("EP-AG-09", "education", "explain_concept", "Explain machine learning in simple terms", "", None, None),
        ("EP-AG-10", "invoice", "financial_summary", "Summarize all invoice totals", "", INVOICE_PROFILE, INVOICE_SUBSCRIPTION),
        ("EP-AG-11", "screening", "detect_ai_content", "",
         "The unprecedented confluence of disruptive innovations necessitates a paradigm shift in transformative synergistic methodologies.", None, None),
    ]

    for test_id, agent, task_type, query, text, profile, sub_id in tests:
        t0 = time.time()
        try:
            resp = _agent_execute(agent, task_type, query=query, text=text,
                                  profile_id=profile, subscription_id=sub_id)
            elapsed = time.time() - t0
            ok = resp.get("status") == "success"
            output = resp.get("output", "") or ""
            structured = resp.get("structured_data") or {}
            text_preview = output if output else json.dumps(structured)[:300] if structured else json.dumps(resp)[:300]
            score = 80 if ok and len(text_preview) > 20 else 40 if ok else 0
            grade = "A" if score >= 80 else "B" if score >= 60 else "F"
            result = {"id": test_id, "pass": ok, "grade": grade, "score": score,
                      "response_len": len(text_preview)}
            print(f"  [{'PASS' if ok else 'FAIL'}] {test_id} ({agent}/{task_type}): "
                  f"grade={grade} time={elapsed:.1f}s len={len(text_preview)}")
            if text_preview:
                print(f"         Preview: {text_preview[:150].replace(chr(10), ' ')}...")
            if not ok:
                error = resp.get("error", resp.get("detail", "unknown"))
                print(f"         ERROR: {error}")
        except Exception as e:
            elapsed = time.time() - t0
            result = {"id": test_id, "pass": False, "grade": "F", "score": 0, "response_len": 0}
            print(f"  [FAIL] {test_id} ({agent}/{task_type}): ERROR={e} time={elapsed:.1f}s")
        results.append(result)
        sys.stdout.flush()
    return results


def test_screening():
    """Batch 5: Screening tools via /api/tools/run (backward compat)."""
    print("\n" + "=" * 70)
    print("BATCH 5: Screening Tools (via /api/tools/run)")
    print("=" * 70)
    sys.stdout.flush()
    results = []

    screening_tests = [
        ("SCR-01", "screen_pii",
         {"text": "John Smith lives at 123 Main St, his SSN is 123-45-6789 and email is john@example.com"},
         ["pii"]),
        ("SCR-02", "screen_ai_authorship",
         {"text": "The multifaceted nature of this paradigm-shifting innovation represents a quantum leap in transformative synergistic methodologies that leverage cutting-edge frameworks."},
         ["ai"]),
        ("SCR-03", "screen_readability",
         {"text": "The patient presented with acute myocardial infarction complicated by cardiogenic shock requiring emergent percutaneous coronary intervention with stent placement."},
         ["readab"]),
        ("SCR-04", "screen_resume",
         {"text": "RESUME\nJohn Smith\nSoftware Engineer - 5 years Python\nSkills: Python, Django, Flask, AWS\nGoogle (2019-2024) Senior Developer\nBS Computer Science, MIT"},
         ["score"]),
    ]

    for test_id, tool_name, input_data, expect in screening_tests:
        t0 = time.time()
        try:
            resp = client.post("/api/tools/run", json={
                "tool_name": tool_name,
                "input": input_data,
                "context": {},
                "options": {},
                "subscription_id": SUBSCRIPTION_ID,
                "profile_id": HR_PROFILE,
            })
            elapsed = time.time() - t0
            data = resp.json()
            ok = resp.status_code == 200
            text = json.dumps(data)[:500]
            text_lower = text.lower()
            matched = [kw for kw in expect if kw.lower() in text_lower]
            match_ratio = len(matched) / len(expect) if expect else 1.0
            score = 80 if ok and match_ratio >= 0.5 else 40 if ok else 0
            grade = "A" if score >= 80 else "B" if score >= 60 else "F"
            result = {"id": test_id, "pass": ok and match_ratio >= 0.5, "grade": grade,
                      "score": score, "response_len": len(text), "matched": matched}
            print(f"  [{'PASS' if result['pass'] else 'FAIL'}] {test_id} ({tool_name}): "
                  f"grade={grade} time={elapsed:.1f}s matched={matched}")
            print(f"         Preview: {text[:200].replace(chr(10), ' ')}...")
        except Exception as e:
            elapsed = time.time() - t0
            result = {"id": test_id, "pass": False, "grade": "F", "score": 0, "response_len": 0}
            print(f"  [FAIL] {test_id} ({tool_name}): ERROR={e} time={elapsed:.1f}s")
        results.append(result)
        sys.stdout.flush()
    return results


def test_content_generation():
    """Batch 6: Content generation and email drafting via /api/ask."""
    print("\n" + "=" * 70)
    print("BATCH 6: Content Generation & Email Drafting")
    print("=" * 70)
    sys.stdout.flush()
    results = []
    queries = [
        ("CG-01", "Draft an interview invitation email for the top candidate", HR_PROFILE, None, ["interview"]),
        ("CG-02", "Generate a cover letter for a Python developer based on the best candidate", HR_PROFILE, None, ["python"]),
        ("CG-03", "Create a summary of key findings across all documents", HR_PROFILE, None, ["summary"]),
    ]

    for test_id, query, profile, sub_id, expect in queries:
        t0 = time.time()
        try:
            resp = _ask(query, profile_id=profile, subscription_id=sub_id)
            elapsed = time.time() - t0
            answer = resp.get("answer", {})
            text = answer.get("response", "") if isinstance(answer, dict) else str(answer)
            result = _grade(test_id, resp, expect, check_grounded=False)
            _print_result(result, elapsed, text)
        except Exception as e:
            elapsed = time.time() - t0
            result = {"id": test_id, "pass": False, "grade": "F", "score": 0, "response_len": 0}
            _print_result(result, elapsed)
        results.append(result)
        sys.stdout.flush()
    return results


def test_web_search():
    """Batch 7: Web search."""
    print("\n" + "=" * 70)
    print("BATCH 7: Web Search (enable_internet)")
    print("=" * 70)
    sys.stdout.flush()
    results = []

    t0 = time.time()
    try:
        resp = _ask("What is the latest version of Python programming language?",
                     enable_internet=True)
        elapsed = time.time() - t0
        answer = resp.get("answer", {})
        text = answer.get("response", "") if isinstance(answer, dict) else str(answer)
        result = _grade("WEB-01", resp, ["python"], check_grounded=False)
        _print_result(result, elapsed, text)
    except Exception as e:
        elapsed = time.time() - t0
        result = {"id": "WEB-01", "pass": False, "grade": "F", "score": 0, "response_len": 0, "reason": str(e)}
        print(f"  [FAIL] WEB-01: ERROR={e} time={elapsed:.1f}s")
    results.append(result)
    sys.stdout.flush()
    return results


def test_backward_compat():
    """Batch 8: Backward compatibility."""
    print("\n" + "=" * 70)
    print("BATCH 8: Backward Compatibility (/api/tools/run)")
    print("=" * 70)
    sys.stdout.flush()
    results = []

    t0 = time.time()
    try:
        resp = client.post("/api/tools/run", json={
            "tool_name": "content_generate",
            "input": {"prompt": "Write a brief professional greeting"},
            "context": {"text": "Company overview: We build AI document intelligence products."},
            "options": {"format": "text"},
            "subscription_id": SUBSCRIPTION_ID,
            "profile_id": HR_PROFILE,
        })
        elapsed = time.time() - t0
        ok = resp.status_code == 200
        text = json.dumps(resp.json())[:300]
        result = {"id": "BC-01", "pass": ok, "grade": "A" if ok else "F",
                  "score": 80 if ok else 0, "response_len": len(text)}
        print(f"  [{'PASS' if ok else 'FAIL'}] BC-01 (tools/run content_generate): "
              f"status={resp.status_code} time={elapsed:.1f}s")
        if text:
            print(f"         Preview: {text[:150]}...")
    except Exception as e:
        elapsed = time.time() - t0
        result = {"id": "BC-01", "pass": False, "grade": "F", "score": 0, "response_len": 0}
        print(f"  [FAIL] BC-01: ERROR={e} time={elapsed:.1f}s")
    results.append(result)

    t0 = time.time()
    try:
        resp = client.post("/api/tools/run", json={
            "tool_name": "translator",
            "input": {"text": "Hello, how are you?", "target_language": "es"},
            "context": {},
            "options": {},
        })
        elapsed = time.time() - t0
        ok = resp.status_code == 200
        result = {"id": "BC-02", "pass": ok, "grade": "A" if ok else "F",
                  "score": 80 if ok else 0, "response_len": 0}
        print(f"  [{'PASS' if ok else 'FAIL'}] BC-02 (tools/run translator): "
              f"status={resp.status_code} time={elapsed:.1f}s")
        if ok:
            print(f"         Result: {json.dumps(resp.json())[:200]}")
    except Exception as e:
        elapsed = time.time() - t0
        result = {"id": "BC-02", "pass": False, "grade": "F", "score": 0, "response_len": 0}
        print(f"  [FAIL] BC-02: ERROR={e} time={elapsed:.1f}s")
    results.append(result)
    sys.stdout.flush()
    return results


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("DocWain Comprehensive UAT")
    print(f"Server: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    sys.stdout.flush()

    all_results = []
    total_start = time.time()

    # Run batches in order (light → heavy)
    all_results.extend(test_health_and_endpoints())
    all_results.extend(test_retrieval_accuracy())
    all_results.extend(test_response_intelligence())
    all_results.extend(test_screening())
    all_results.extend(test_agents_via_ask())
    all_results.extend(test_agents_via_endpoints())
    all_results.extend(test_content_generation())
    all_results.extend(test_web_search())
    all_results.extend(test_backward_compat())

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("UAT SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in all_results if r["pass"])
    failed = sum(1 for r in all_results if not r["pass"])
    total = len(all_results)
    grades = {}
    for r in all_results:
        g = r.get("grade", "?")
        grades[g] = grades.get(g, 0) + 1

    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Pass rate: {passed/total*100:.0f}%")
    print(f"Grades: {' '.join(f'{g}={c}' for g, c in sorted(grades.items()))}")
    print(f"Total time: {total_elapsed:.0f}s")

    if failed:
        print(f"\nFailed tests:")
        for r in all_results:
            if not r["pass"]:
                print(f"  - {r['id']}: grade={r.get('grade', '?')} reason={r.get('reason', 'low match')}")

    with open("tests/uat_results.json", "w") as f:
        json.dump({"timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                    "total": total, "passed": passed, "failed": failed,
                    "grades": grades, "results": all_results}, f, indent=2)
    print(f"\nResults saved to tests/uat_results.json")
    sys.stdout.flush()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
