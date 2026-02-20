"""Comprehensive pre-flight analysis for DocWain.

Exercises every major subsystem against live data:
  Phase 1: RAG pipeline (20 queries across domains/intents/scopes)
  Phase 2: Screening engine (8 categories against real documents)
  Phase 3: Gateway integration (multi-category HTTP requests)
  Phase 4: Agentic mode (tool selection + LLM health + content detection)
  Phase 5: Report generation (aggregate JSON to tests/preflight_report.json)

Gate: LIVE_TEST=1 environment variable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("preflight")

LIVE = os.environ.get("LIVE_TEST", "0") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="LIVE_TEST=1 not set")

# ── Constants ────────────────────────────────────────────────────────────────

SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
PROFILE_ID = "698c46e6bcae2c45eca1d8d9"
DOC_IDS = {
    "gaurav": "698d5e8fbcae2c45eca1dcd3",
    "gokul": "698d70fbbcae2c45eca1de38",
    "dev": "698d5e8fbcae2c45eca1dcbe",
    "dhayal": "698d5e8fbcae2c45eca1dcce",
    "bharath": "698d5e8fbcae2c45eca1dcbb",
}

_BANNED_PHRASES = [
    "not explicitly mentioned",
    "not available in retrieved context",
    "not enough information",
]

REPORT_PATH = Path(__file__).parent / "preflight_report.json"

# ── Module-level accumulators ────────────────────────────────────────────────

_rag_results: List[Dict[str, Any]] = []
_screening_results: List[Dict[str, Any]] = []
_gateway_results: List[Dict[str, Any]] = []
_agentic_results: List[Dict[str, Any]] = []

# ── Shared fixtures ──────────────────────────────────────────────────────────

_cached_deps = None


def _get_rag_deps():
    """Build real RAG dependencies, with caching and CUDA fallback."""
    global _cached_deps
    if _cached_deps is not None:
        return _cached_deps
    from src.api.config import Config
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=30)
    try:
        embedder = SentenceTransformer(Config.Model.EMBEDDING_MODEL)
    except (RuntimeError, Exception) as exc:
        if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
            embedder = SentenceTransformer(Config.Model.EMBEDDING_MODEL, device="cpu")
        else:
            raise
    _cached_deps = (qdrant_client, embedder)
    return _cached_deps


def _get_llm_client():
    """Try to get the LLM gateway; return None if unavailable."""
    try:
        from src.llm.gateway import get_llm_gateway
        return get_llm_gateway()
    except Exception:
        return None


def _run_query(query: str, document_id: str | None = None, use_llm: bool = False) -> dict:
    """Run a RAG v3 query and return the result dict."""
    from src.rag_v3.pipeline import run
    qdrant_client, embedder = _get_rag_deps()
    llm_client = _get_llm_client() if use_llm else None
    result = run(
        query=query,
        subscription_id=SUBSCRIPTION_ID,
        profile_id=PROFILE_ID,
        document_id=document_id,
        llm_client=llm_client,
        qdrant_client=qdrant_client,
        redis_client=None,
        embedder=embedder,
        cross_encoder=None,
    )
    return result


def _check_banned_phrases(text: str) -> List[str]:
    """Return list of banned phrases found in text."""
    lowered = text.lower()
    return [p for p in _BANNED_PHRASES if p in lowered]


def _evaluate_rag_result(
    result: dict,
    query: str,
    *,
    expected_scope: str | None = None,
    expected_domain: str | None = None,
    expect_entity: str | None = None,
    expect_multi_doc: bool = False,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    """Evaluate a RAG result and return a structured assessment."""
    response = result.get("response", "")
    metadata = result.get("metadata", {})
    sources = result.get("sources", [])

    checks: Dict[str, bool] = {}
    issues: List[str] = []

    # 1. Non-empty response
    checks["non_empty"] = bool(response and len(response.strip()) > 10)
    if not checks["non_empty"]:
        issues.append("Empty or trivially short response")

    # 2. No banned phrases
    banned = _check_banned_phrases(response)
    checks["no_banned_phrases"] = len(banned) == 0
    if banned:
        issues.append(f"Banned phrases: {banned}")

    # 3. Not a fallback
    checks["not_fallback"] = "I apologize" not in response and "cannot provide" not in response
    if not checks["not_fallback"]:
        issues.append("Got fallback response")

    # 4. Scope detection — pipeline may return dict instead of string
    actual_scope = metadata.get("scope", "unknown")
    if expected_scope:
        if isinstance(actual_scope, dict):
            # Pipeline returns {'profile_id': '...'} for targeted, {'document_id': '...'} for specific_document
            if expected_scope == "targeted":
                checks["scope_correct"] = "profile_id" in actual_scope
            elif expected_scope == "specific_document":
                checks["scope_correct"] = "document_id" in actual_scope
            else:
                checks["scope_correct"] = False
        else:
            checks["scope_correct"] = actual_scope == expected_scope
        if not checks["scope_correct"]:
            issues.append(f"Scope: expected={expected_scope}, got={actual_scope}")

    # 5. Sources present
    checks["has_sources"] = len(sources) > 0 or allow_mismatch
    if not checks["has_sources"]:
        issues.append("No sources returned")

    # 6. Entity isolation
    if expect_entity:
        checks["entity_mentioned"] = expect_entity.lower() in response.lower()
        if not checks["entity_mentioned"]:
            issues.append(f"Entity '{expect_entity}' not mentioned in response")

    # 7. Multi-doc coverage
    if expect_multi_doc:
        doc_count = metadata.get("document_count", 0)
        checks["multi_doc"] = doc_count >= 2
        if not checks["multi_doc"]:
            issues.append(f"Expected multi-doc, got {doc_count} docs")

    passed = all(checks.values())
    return {
        "query": query,
        "passed": passed,
        "checks": checks,
        "issues": issues,
        "response_length": len(response),
        "sources_count": len(sources),
        "scope_detected": actual_scope,
        "domain_detected": metadata.get("domain", "unknown"),
        "quality": metadata.get("quality", "unknown"),
        "latency_ms": metadata.get("latency_ms", 0),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: RAG Pipeline Analysis
# ═════════════════════════════════════════════════════════════════════════════


class TestPhase1RAG:
    """Live RAG pipeline tests across domains, intents, and scopes."""

    # ── HR Factual ────────────────────────────────────────────────────────

    def test_hr_factual_experience(self):
        result = _run_query("What is Dhayal's total years of experience?")
        ev = _evaluate_rag_result(result, "hr_factual_experience",
                                  expected_scope="targeted", expect_entity="Dhayal")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_factual_experience: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_summary_profile(self):
        result = _run_query("Summarize Gaurav's resume and profile")
        ev = _evaluate_rag_result(result, "hr_summary_profile",
                                  expected_scope="targeted", expect_entity="Gaurav")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_summary_profile: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_comparison_all(self):
        result = _run_query("Compare all candidates")
        ev = _evaluate_rag_result(result, "hr_comparison_all",
                                  expected_scope="all_profile", expect_multi_doc=True)
        _rag_results.append(ev)
        print(f"\n[RAG] hr_comparison_all: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_ranking(self):
        result = _run_query("Rank all candidates by skills and experience")
        ev = _evaluate_rag_result(result, "hr_ranking",
                                  expected_scope="all_profile", expect_multi_doc=True)
        _rag_results.append(ev)
        print(f"\n[RAG] hr_ranking: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_multi_field(self):
        result = _run_query("What are Bharath's skills, education, and certifications?")
        ev = _evaluate_rag_result(result, "hr_multi_field",
                                  expected_scope="targeted", expect_entity="Bharath")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_multi_field: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_contact(self):
        result = _run_query("What is Gokul's contact information?")
        ev = _evaluate_rag_result(result, "hr_contact",
                                  expected_scope="targeted", expect_entity="Gokul")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_contact: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_cross_document(self):
        result = _run_query("Which candidates have SAP experience across all resumes?")
        ev = _evaluate_rag_result(result, "hr_cross_document",
                                  expected_scope="all_profile", expect_multi_doc=True)
        _rag_results.append(ev)
        print(f"\n[RAG] hr_cross_document: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_analytics(self):
        result = _run_query("How many candidates are in this profile?")
        ev = _evaluate_rag_result(result, "hr_analytics", expected_scope="all_profile")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_analytics: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_timeline(self):
        result = _run_query("Show Dev's career progression and timeline")
        ev = _evaluate_rag_result(result, "hr_timeline",
                                  expected_scope="targeted", expect_entity="Dev")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_timeline: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_reasoning(self):
        result = _run_query("Is Dhayal qualified for a sales manager role?")
        ev = _evaluate_rag_result(result, "hr_reasoning",
                                  expected_scope="targeted", expect_entity="Dhayal")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_reasoning: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_specific_document(self):
        result = _run_query("What are the skills listed in this document?",
                            document_id=DOC_IDS["dhayal"])
        ev = _evaluate_rag_result(result, "hr_specific_document",
                                  expected_scope="specific_document")
        _rag_results.append(ev)
        print(f"\n[RAG] hr_specific_document: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_hr_list_all(self):
        result = _run_query("List all candidates with their roles")
        ev = _evaluate_rag_result(result, "hr_list_all",
                                  expected_scope="all_profile", expect_multi_doc=True)
        _rag_results.append(ev)
        print(f"\n[RAG] hr_list_all: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    # ── Domain Mismatch Probes ────────────────────────────────────────────

    def test_mismatch_invoice(self):
        result = _run_query("What are the invoice totals and line items?")
        ev = _evaluate_rag_result(result, "mismatch_invoice", allow_mismatch=True)
        _rag_results.append(ev)
        print(f"\n[RAG] mismatch_invoice: {'PASS' if ev['passed'] else 'FAIL'} | response={ev['response_length']} chars")

    def test_mismatch_legal(self):
        result = _run_query("What are the contract terms and conditions?")
        ev = _evaluate_rag_result(result, "mismatch_legal", allow_mismatch=True)
        _rag_results.append(ev)
        print(f"\n[RAG] mismatch_legal: {'PASS' if ev['passed'] else 'FAIL'} | response={ev['response_length']} chars")

    def test_mismatch_medical(self):
        result = _run_query("What medications are listed in the patient record?")
        ev = _evaluate_rag_result(result, "mismatch_medical", allow_mismatch=True)
        _rag_results.append(ev)
        print(f"\n[RAG] mismatch_medical: {'PASS' if ev['passed'] else 'FAIL'} | response={ev['response_length']} chars")

    def test_generic_query(self):
        result = _run_query("What information is available in these documents?")
        ev = _evaluate_rag_result(result, "generic_query")
        _rag_results.append(ev)
        print(f"\n[RAG] generic_query: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    # ── LLM Path Queries ─────────────────────────────────────────────────

    def test_llm_factual(self):
        result = _run_query("What is Dhayal's total experience?", use_llm=True)
        ev = _evaluate_rag_result(result, "llm_factual",
                                  expect_entity="Dhayal")
        ev["llm_path"] = True
        _rag_results.append(ev)
        print(f"\n[RAG/LLM] llm_factual: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_llm_summary(self):
        result = _run_query("Summarize Gaurav's profile", use_llm=True)
        ev = _evaluate_rag_result(result, "llm_summary",
                                  expect_entity="Gaurav")
        ev["llm_path"] = True
        _rag_results.append(ev)
        print(f"\n[RAG/LLM] llm_summary: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_llm_comparison(self):
        result = _run_query("Compare all candidates by skills", use_llm=True)
        ev = _evaluate_rag_result(result, "llm_comparison",
                                  expected_scope="all_profile", expect_multi_doc=True)
        ev["llm_path"] = True
        _rag_results.append(ev)
        print(f"\n[RAG/LLM] llm_comparison: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]

    def test_llm_reasoning(self):
        result = _run_query("Assess whether Bharath is suitable for an SAP SD consultant role", use_llm=True)
        ev = _evaluate_rag_result(result, "llm_reasoning",
                                  expect_entity="Bharath")
        ev["llm_path"] = True
        _rag_results.append(ev)
        print(f"\n[RAG/LLM] llm_reasoning: {'PASS' if ev['passed'] else 'FAIL'} | {ev['response_length']} chars")
        assert ev["checks"].get("non_empty", False), ev["issues"]


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: Screening Pipeline
# ═════════════════════════════════════════════════════════════════════════════


def _run_screening_category(category: str, doc_id: str) -> Dict[str, Any]:
    """Run a screening category and return structured result."""
    start = time.time()
    try:
        from src.screening.engine import ScreeningEngine
        engine = ScreeningEngine()
        results = engine.run_category(category, doc_id, doc_type="RESUME")
        elapsed = (time.time() - start) * 1000
        return {
            "category": category,
            "status": "success",
            "tool_count": len(results),
            "results": [
                {
                    "tool": r.tool_name,
                    "score": round(r.score_0_1, 4),
                    "risk_level": r.risk_level,
                    "reasons_count": len(r.reasons),
                    "reasons": r.reasons[:3],  # top 3 only
                }
                for r in results
            ],
            "latency_ms": round(elapsed, 1),
        }
    except Exception as exc:
        elapsed = (time.time() - start) * 1000
        return {
            "category": category,
            "status": "error",
            "error": str(exc),
            "latency_ms": round(elapsed, 1),
        }


def _run_screening_all(doc_id: str) -> Dict[str, Any]:
    """Run full screening (run_all) and return report summary."""
    start = time.time()
    try:
        from src.screening.engine import ScreeningEngine
        engine = ScreeningEngine()
        report = engine.run_all(doc_id, doc_type="RESUME")
        elapsed = (time.time() - start) * 1000
        d = report.to_dict()
        return {
            "status": "success",
            "overall_score": d.get("overall_score_0_100"),
            "risk_level": d.get("risk_level"),
            "tools_run": len(d.get("results", [])),
            "top_findings": d.get("top_findings", [])[:5],
            "latency_ms": round(elapsed, 1),
        }
    except Exception as exc:
        elapsed = (time.time() - start) * 1000
        return {"status": "error", "error": str(exc), "latency_ms": round(elapsed, 1)}


class TestPhase2Screening:
    """Screening engine tests per category against real documents."""

    def test_screening_integrity(self):
        r = _run_screening_category("integrity", DOC_IDS["gaurav"])
        _screening_results.append(r)
        print(f"\n[SCREEN] integrity: {r['status']} | tools={r.get('tool_count', 0)}")
        assert r["status"] == "success", r.get("error")
        assert r["tool_count"] >= 1

    def test_screening_compliance(self):
        r = _run_screening_category("compliance", DOC_IDS["gaurav"])
        _screening_results.append(r)
        print(f"\n[SCREEN] compliance: {r['status']} | tools={r.get('tool_count', 0)}")
        assert r["status"] == "success", r.get("error")

    def test_screening_quality(self):
        r = _run_screening_category("quality", DOC_IDS["gaurav"])
        _screening_results.append(r)
        print(f"\n[SCREEN] quality: {r['status']} | tools={r.get('tool_count', 0)}")
        assert r["status"] == "success", r.get("error")
        assert r["tool_count"] >= 1

    def test_screening_language(self):
        r = _run_screening_category("language", DOC_IDS["gaurav"])
        _screening_results.append(r)
        print(f"\n[SCREEN] language: {r['status']} | tools={r.get('tool_count', 0)}")
        assert r["status"] == "success", r.get("error")

    def test_screening_security(self):
        r = _run_screening_category("security", DOC_IDS["gaurav"])
        _screening_results.append(r)
        print(f"\n[SCREEN] security: {r['status']} | tools={r.get('tool_count', 0)}")
        assert r["status"] == "success", r.get("error")
        # Resume should have PII (email, phone)
        if r["results"]:
            print(f"  PII score: {r['results'][0]['score']}")

    def test_screening_ai_authorship(self):
        r = _run_screening_category("ai-authorship", DOC_IDS["gaurav"])
        _screening_results.append(r)
        print(f"\n[SCREEN] ai_authorship: {r['status']} | tools={r.get('tool_count', 0)}")
        assert r["status"] == "success", r.get("error")

    def test_screening_resume(self):
        r = _run_screening_category("resume", DOC_IDS["gaurav"])
        _screening_results.append(r)
        tools_run = [t["tool"] for t in r.get("results", [])]
        print(f"\n[SCREEN] resume: {r['status']} | tools={tools_run}")
        assert r["status"] == "success", r.get("error")
        assert r["tool_count"] >= 2, f"Expected multiple resume tools, got {r['tool_count']}"

    def test_screening_legality(self):
        # Legality may not apply to resumes
        r = _run_screening_category("legality", DOC_IDS["gaurav"])
        _screening_results.append(r)
        print(f"\n[SCREEN] legality: {r['status']} | tools={r.get('tool_count', 0)}")
        # Don't assert success — legality may legitimately skip resumes

    def test_screening_full_report(self):
        """Run all screening tools and get aggregated report."""
        r = _run_screening_all(DOC_IDS["gaurav"])
        _screening_results.append({"category": "FULL_REPORT", **r})
        print(f"\n[SCREEN] FULL: {r['status']} | score={r.get('overall_score')} | risk={r.get('risk_level')} | tools={r.get('tools_run')}")
        if r["status"] == "success":
            assert 0 <= r["overall_score"] <= 100
            assert r["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_screening_security_deep(self):
        """Security screening on inline text with known PII."""
        from src.screening.security_service import SecurityScreeningService
        text = (
            "Gaurav Fegade\n"
            "Email: gaurav.fegade@company.com\n"
            "Phone: +91-98765-43210\n"
            "SAP EWM Consultant with 3 years experience."
        )
        response = SecurityScreeningService().screen_text(text)
        _screening_results.append({
            "category": "security_deep",
            "status": "success",
            "classification": response.get("classification"),
            "findings_count": len(response.get("security_findings", [])),
            "overall_risk": response.get("overall_risk_score"),
        })
        findings = response.get("security_findings", [])
        print(f"\n[SCREEN] security_deep: findings={len(findings)} | risk={response.get('overall_risk_score')}")
        # Should detect email and phone as PII
        finding_types = {f.get("type") for f in findings}
        assert "PII" in finding_types or len(findings) > 0, "Should detect PII in resume text"


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: Gateway Integration
# ═════════════════════════════════════════════════════════════════════════════


class TestPhase3Gateway:
    """HTTP endpoint tests for POST /api/gateway/screen."""

    @pytest.fixture(autouse=True)
    def _client(self):
        from fastapi.testclient import TestClient
        from src.main import app
        self.client = TestClient(app)

    def _post_screen(self, category: list, doc_ids: list | None = None) -> dict:
        body: dict = {"category": category}
        if doc_ids is not None:
            body["doc_ids"] = doc_ids
        resp = self.client.post("/api/gateway/screen", json=body)
        return {"http_status": resp.status_code, **resp.json()}

    def test_gateway_single_category(self):
        r = self._post_screen(["integrity"], [DOC_IDS["gaurav"]])
        _gateway_results.append({"test": "single_category", **r})
        print(f"\n[GW] single_category: status={r.get('status')} | http={r['http_status']}")
        assert r["http_status"] == 200
        assert r["status"] in ("success", "partial")
        assert len(r.get("documents", [])) == 1

    def test_gateway_multi_category(self):
        r = self._post_screen(["security", "integrity"], [DOC_IDS["gaurav"]])
        _gateway_results.append({"test": "multi_category", **r})
        print(f"\n[GW] multi_category: status={r.get('status')} | http={r['http_status']}")
        assert r["http_status"] == 200
        assert r["status"] in ("success", "partial")
        doc = r.get("documents", [{}])[0]
        assert "categories" in doc, "Multi-category should return grouped results"
        assert "security" in doc["categories"]
        assert "integrity" in doc["categories"]

    def test_gateway_resume_category(self):
        r = self._post_screen(["resume"], [DOC_IDS["gaurav"]])
        _gateway_results.append({"test": "resume_category", **r})
        print(f"\n[GW] resume_category: status={r.get('status')} | http={r['http_status']}")
        assert r["http_status"] == 200
        assert r["status"] in ("success", "partial")

    def test_gateway_all_category(self):
        r = self._post_screen(["all"], [DOC_IDS["gaurav"]])
        _gateway_results.append({"test": "all_category", **r})
        print(f"\n[GW] all_category: status={r.get('status')} | http={r['http_status']}")
        assert r["http_status"] == 200

    def test_gateway_multi_doc(self):
        r = self._post_screen(["integrity"], [DOC_IDS["gaurav"], DOC_IDS["dhayal"]])
        _gateway_results.append({"test": "multi_doc", **r})
        print(f"\n[GW] multi_doc: status={r.get('status')} | docs={len(r.get('documents', []))}")
        assert r["http_status"] == 200
        assert len(r.get("documents", [])) == 2


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: Agentic Mode
# ═════════════════════════════════════════════════════════════════════════════


class TestPhase4Agentic:
    """Tool selection accuracy, LLM health, and content detection."""

    # ── Tool Selection ────────────────────────────────────────────────────

    _TOOL_SELECTION_CASES = [
        ("Write a cover letter for Dhayal", {"content_generate"}),
        ("Translate this resume to French", {"translator"}),
        ("Draft an email about the candidate", {"email_drafting"}),
        ("Screen this document for PII", {"screen_pii"}),
        ("Explain how SAP EWM works in detail", {"tutor"}),
        ("Compare all candidates by skills", {"content_generate"}),
        ("Rank candidates by experience", {"resumes"}),
        ("Get Dhayal's contact info", {"resumes"}),
        ("Summarize the documents", {"content_generate"}),
        ("Extract all skills from the resume", {"resumes"}),
        ("Analyze this legal clause for risks", {"lawhere"}),
        ("Summarize the medical report findings", {"medical"}),
        ("Generate API documentation for this module", {"code_docs"}),
        ("Check the JIRA ticket status", {"jira_confluence"}),
        ("Extract text from this image using OCR", {"image_analysis"}),
        ("What is the weather today?", set()),  # no tools expected
    ]

    def test_tool_selection_accuracy(self):
        """Verify ToolSelector picks the right tools for diverse queries."""
        from src.agentic.tool_selector import ToolSelector

        # Provide a broad registered set so filtering doesn't block
        known = frozenset({
            "content_generate", "content_types", "resumes", "lawhere", "medical",
            "translator", "tutor", "creator", "email_drafting", "screen_pii",
            "code_docs", "web_extract", "jira_confluence", "image_analysis", "stt", "tts", "db_connector",
        })
        selector = ToolSelector(registered_tools=known)

        correct = 0
        total = len(self._TOOL_SELECTION_CASES)
        details = []

        for query, expected in self._TOOL_SELECTION_CASES:
            selected = set(selector.select_tools(query))
            # Pass if selected tools contain at least one expected tool (or both empty)
            if expected:
                match = bool(selected & expected)
            else:
                match = len(selected) == 0
            if match:
                correct += 1
            details.append({
                "query": query[:60],
                "expected": sorted(expected),
                "selected": sorted(selected),
                "match": match,
            })

        accuracy = correct / total if total else 0
        _agentic_results.append({
            "test": "tool_selection",
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": total,
            "details": details,
        })

        print(f"\n[AGENT] Tool selection: {correct}/{total} = {accuracy:.0%}")
        for d in details:
            status = "OK" if d["match"] else "MISS"
            print(f"  [{status}] {d['query'][:50]:50s} expected={d['expected']} got={d['selected']}")

        assert accuracy >= 0.6, f"Tool selection accuracy {accuracy:.0%} below 60%"

    # ── LLM Gateway Health ────────────────────────────────────────────────

    def test_llm_gateway_health(self):
        """Check if any LLM backend is available and responsive."""
        result: Dict[str, Any] = {"test": "llm_health"}
        try:
            from src.llm.gateway import get_llm_gateway
            gw = get_llm_gateway()
            health = gw.health_check()
            result.update(health)
            print(f"\n[AGENT] LLM health: {health.get('status')} | backend={health.get('backend')} | "
                  f"model={health.get('model')} | latency={health.get('latency_ms')}ms")
        except Exception as exc:
            result["status"] = "unavailable"
            result["error"] = str(exc)
            print(f"\n[AGENT] LLM health: UNAVAILABLE | {exc}")

        _agentic_results.append(result)
        # Don't assert — LLM may legitimately be offline

    # ── Content Type Detection ────────────────────────────────────────────

    def test_content_type_detection(self):
        """Verify content type detection for common queries."""
        from src.content_generation.registry import detect_content_type

        cases = [
            ("Write a cover letter for this candidate", "cover_letter"),
            ("Create a professional summary", "professional_summary"),
            ("Build a skills matrix", "skills_matrix"),
            ("Generate interview prep questions", "interview_prep"),
            ("Summarize the key points", "key_points"),
        ]

        correct = 0
        details = []
        for query, expected_type in cases:
            detected = detect_content_type(query)
            match = detected == expected_type
            if match:
                correct += 1
            details.append({
                "query": query,
                "expected": expected_type,
                "detected": detected,
                "match": match,
            })

        accuracy = correct / len(cases) if cases else 0
        _agentic_results.append({
            "test": "content_detection",
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": len(cases),
            "details": details,
        })

        print(f"\n[AGENT] Content detection: {correct}/{len(cases)} = {accuracy:.0%}")
        for d in details:
            status = "OK" if d["match"] else "MISS"
            print(f"  [{status}] {d['query'][:40]:40s} expected={d['expected']} got={d['detected']}")

        assert accuracy >= 0.6, f"Content detection accuracy {accuracy:.0%} below 60%"


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: Report Generation
# ═════════════════════════════════════════════════════════════════════════════


class TestPhase5Report:
    """Aggregate all results into a structured report."""

    def test_generate_preflight_report(self):
        """Generate and write the full pre-flight report."""
        # ── RAG summary ──
        rag_pass = sum(1 for r in _rag_results if r.get("passed", False))
        rag_total = len(_rag_results)

        domain_stats: Dict[str, Dict[str, int]] = {}
        for r in _rag_results:
            domain = r.get("domain_detected", "unknown")
            domain_stats.setdefault(domain, {"pass": 0, "total": 0})
            domain_stats[domain]["total"] += 1
            if r.get("passed"):
                domain_stats[domain]["pass"] += 1

        # ── Screening summary ──
        screen_pass = sum(1 for r in _screening_results if r.get("status") == "success")
        screen_total = len(_screening_results)

        # ── Gateway summary ──
        gw_pass = sum(1 for r in _gateway_results if r.get("status") in ("success", "partial"))
        gw_total = len(_gateway_results)

        # ── Agentic summary ──
        tool_accuracy = 0.0
        llm_status = "unknown"
        for r in _agentic_results:
            if r.get("test") == "tool_selection":
                tool_accuracy = r.get("accuracy", 0)
            if r.get("test") == "llm_health":
                llm_status = r.get("status", "unknown")

        # ── Improvement targets ──
        improvements = []
        failing_rag = [r for r in _rag_results if not r.get("passed")]
        if failing_rag:
            improvements.append({
                "area": "RAG Pipeline",
                "issue": f"{len(failing_rag)} queries failed",
                "queries": [r["query"] for r in failing_rag],
            })
        failing_screen = [r for r in _screening_results if r.get("status") == "error"]
        if failing_screen:
            improvements.append({
                "area": "Screening",
                "issue": f"{len(failing_screen)} categories errored",
                "categories": [r["category"] for r in failing_screen],
            })
        if tool_accuracy < 0.8:
            improvements.append({
                "area": "Tool Selection",
                "issue": f"Accuracy {tool_accuracy:.0%} below 80% target",
            })
        if llm_status not in ("healthy", "degraded"):
            improvements.append({
                "area": "LLM Gateway",
                "issue": f"LLM status: {llm_status}",
            })

        total = rag_total + screen_total + gw_total
        passed = rag_pass + screen_pass + gw_pass
        pass_rate = passed / total if total else 0

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_id": PROFILE_ID,
            "subscription_id": SUBSCRIPTION_ID,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round(pass_rate, 2),
            },
            "rag": {
                "total": rag_total,
                "passed": rag_pass,
                "pass_rate": round(rag_pass / rag_total, 2) if rag_total else 0,
                "domain_stats": {
                    d: round(s["pass"] / s["total"], 2) if s["total"] else 0
                    for d, s in domain_stats.items()
                },
                "per_query": _rag_results,
            },
            "screening": {
                "total": screen_total,
                "passed": screen_pass,
                "per_category": _screening_results,
            },
            "gateway": {
                "total": gw_total,
                "passed": gw_pass,
                "per_test": _gateway_results,
            },
            "agentic": {
                "tool_selection_accuracy": tool_accuracy,
                "llm_status": llm_status,
                "details": _agentic_results,
            },
            "improvements_needed": improvements,
        }

        # Write report
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 70)
        print("DOCWAIN PRE-FLIGHT REPORT")
        print("=" * 70)
        print(f"  Total tests:   {total}")
        print(f"  Passed:        {passed}")
        print(f"  Failed:        {total - passed}")
        print(f"  Pass rate:     {pass_rate:.0%}")
        print(f"  RAG:           {rag_pass}/{rag_total}")
        print(f"  Screening:     {screen_pass}/{screen_total}")
        print(f"  Gateway:       {gw_pass}/{gw_total}")
        print(f"  Tool accuracy: {tool_accuracy:.0%}")
        print(f"  LLM status:    {llm_status}")
        if improvements:
            print(f"\n  IMPROVEMENTS NEEDED ({len(improvements)}):")
            for imp in improvements:
                print(f"    - [{imp['area']}] {imp['issue']}")
        print(f"\n  Report written to: {REPORT_PATH}")
        print("=" * 70)

        # Soft assertion — report should have reasonable pass rate
        assert pass_rate >= 0.4, f"Overall pass rate {pass_rate:.0%} critically low"
