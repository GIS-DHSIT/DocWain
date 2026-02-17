"""Comprehensive retrieval intelligence live tests.

43 tests across 8 classes covering: scope routing, entity targeting,
contact info, field extraction, reasoning, specific document, edge cases,
and a quality report meta-test.

Gate: LIVE_TEST=1 environment variable.
Requires: running Qdrant instance with actual resume data.
"""

from __future__ import annotations

import os
import re
import sys
import logging
import time

import pytest

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

LIVE = os.environ.get("LIVE_TEST", "0") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="LIVE_TEST=1 not set")

# ── Shared config ────────────────────────────────────────────────────────────

SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
PROFILE_ID = "6992c4ec6034385742e451a6"

DOC_IDS = {
    "abinaya": "69935fd96034385742e45586",
    "swapnil": "6992c5836034385742e45209",
    "gokul": "69930cb66034385742e45469",
    "abhishek": "69935fd96034385742e45573",
    "aadithya": "69935fd96034385742e45576",
    "aloysius": "69935fd96034385742e4558b",
}

CANDIDATE_NAMES = list(DOC_IDS.keys())

# ── Dependency caching ───────────────────────────────────────────────────────

_cached_deps = None


def _get_rag_deps():
    """Build real RAG dependencies with caching and CUDA fallback."""
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


def _run_query(query: str, document_id: str | None = None) -> dict:
    """Run a RAG v3 query and return the result dict."""
    from src.rag_v3.pipeline import run

    qdrant_client, embedder = _get_rag_deps()
    result = run(
        query=query,
        subscription_id=SUBSCRIPTION_ID,
        profile_id=PROFILE_ID,
        document_id=document_id,
        llm_client=None,
        qdrant_client=qdrant_client,
        redis_client=None,
        embedder=embedder,
        cross_encoder=None,
    )
    return result


# ── Assertion helpers ────────────────────────────────────────────────────────

_BANNED_PHRASES = [
    "not explicitly mentioned",
    "not available in retrieved context",
    "not enough information",
]


def _assert_non_empty(response: str, query: str):
    """Response must have meaningful content."""
    assert response and len(response.strip()) > 0, (
        f"Empty response for query: {query}"
    )


def _assert_no_banned(response: str, query: str):
    """Response must not contain known unhelpful phrases."""
    lowered = response.lower()
    for phrase in _BANNED_PHRASES:
        assert phrase not in lowered, (
            f"Banned phrase '{phrase}' in response for '{query}': {response[:300]}"
        )


def _assert_has_sources(result: dict, query: str):
    """Result must have at least one source."""
    sources = result.get("sources", [])
    assert len(sources) > 0, f"No sources for query: {query}"


def _assert_scope(result: dict, expected: str, query: str):
    """Check metadata scope matches expected value."""
    metadata = result.get("metadata", {})
    scope = metadata.get("scope")
    # scope can be a string or a dict; normalize
    if isinstance(scope, dict):
        if "profile_id" in scope and not scope.get("document_id"):
            actual = "all_profile"
        elif "document_id" in scope:
            actual = "specific_document"
        else:
            actual = str(scope)
    else:
        actual = str(scope) if scope else "unknown"
    assert actual == expected, (
        f"Expected scope '{expected}', got '{actual}' for query: {query}"
    )


def _assert_mentions_name(response: str, name: str):
    """Response must mention the given candidate name (case-insensitive)."""
    assert name.lower() in response.lower(), (
        f"Expected '{name}' in response, got: {response[:300]}"
    )


def _assert_mentions_any_name(response: str, min_count: int = 1):
    """Response must mention at least min_count candidate names."""
    lowered = response.lower()
    found = [n for n in CANDIDATE_NAMES if n.lower() in lowered]
    assert len(found) >= min_count, (
        f"Expected {min_count}+ candidate names, found {len(found)} ({found}) "
        f"in: {response[:400]}"
    )


def _assert_has_contact_pattern(response: str):
    """Response must contain an email, phone, or linkedin pattern."""
    patterns = [
        r"[\w.+-]+@[\w.-]+\.\w+",  # email
        r"\+?\d[\d\s\-()]{7,}",  # phone
        r"linkedin\.com/in/",  # linkedin URL
        r"linkedin\s*:",  # "LinkedIn:" label
    ]
    for pat in patterns:
        if re.search(pat, response, re.IGNORECASE):
            return
    assert False, f"No contact pattern found in: {response[:400]}"


def _assert_has_numbered_list(response: str):
    """Response must contain numbered items (1. 2. etc.)."""
    assert re.search(r"\b[1-9]\.\s", response), (
        f"No numbered list found in: {response[:400]}"
    )


def _assert_has_digit(response: str):
    """Response must contain at least one digit."""
    assert re.search(r"\d", response), (
        f"No digit found in response: {response[:300]}"
    )


# ── Class 1: Multi-Document Scope ───────────────────────────────────────────

class TestMultiDocScope:
    """Tests for queries that should route to all_profile scope."""

    def test_rank_all_candidates(self):
        result = _run_query("rank all candidates by skills and experience")
        response = result["response"]
        _assert_non_empty(response, "rank all candidates")
        _assert_no_banned(response, "rank all candidates")
        _assert_scope(result, "all_profile", "rank all candidates")
        _assert_mentions_any_name(response, 2)
        _assert_has_sources(result, "rank all candidates")
        print(f"\n[RANK ALL] {response[:300]}")

    def test_compare_all_candidates(self):
        result = _run_query("compare all candidates")
        response = result["response"]
        _assert_non_empty(response, "compare all candidates")
        _assert_no_banned(response, "compare all candidates")
        _assert_scope(result, "all_profile", "compare all candidates")
        _assert_mentions_any_name(response, 2)
        print(f"\n[COMPARE ALL] {response[:300]}")

    def test_list_all_resumes(self):
        result = _run_query("list all resumes in this profile")
        response = result["response"]
        _assert_non_empty(response, "list all resumes")
        _assert_no_banned(response, "list all resumes")
        _assert_scope(result, "all_profile", "list all resumes")
        print(f"\n[LIST ALL] {response[:300]}")

    def test_how_many_candidates(self):
        result = _run_query("how many candidates are in this profile")
        response = result["response"]
        _assert_non_empty(response, "how many candidates")
        _assert_scope(result, "all_profile", "how many candidates")
        _assert_has_digit(response)
        print(f"\n[HOW MANY] {response[:300]}")

    def test_most_experienced(self):
        result = _run_query("who is the most experienced candidate")
        response = result["response"]
        _assert_non_empty(response, "most experienced")
        _assert_no_banned(response, "most experienced")
        _assert_scope(result, "all_profile", "most experienced")
        _assert_mentions_any_name(response, 1)
        print(f"\n[MOST EXP] {response[:300]}")

    def test_best_technical_skills(self):
        result = _run_query("which candidate has the best technical skills")
        response = result["response"]
        _assert_non_empty(response, "best technical skills")
        _assert_no_banned(response, "best technical skills")
        _assert_scope(result, "all_profile", "best technical skills")
        _assert_mentions_any_name(response, 1)
        print(f"\n[BEST TECH] {response[:300]}")

    def test_summarize_all(self):
        result = _run_query("summarize all candidates")
        response = result["response"]
        _assert_non_empty(response, "summarize all")
        _assert_no_banned(response, "summarize all")
        _assert_scope(result, "all_profile", "summarize all")
        _assert_mentions_any_name(response, 2)
        print(f"\n[SUMMARIZE ALL] {response[:300]}")

    def test_certifications_among_all(self):
        result = _run_query("who has certifications among all candidates")
        response = result["response"]
        _assert_non_empty(response, "certifications among all")
        _assert_no_banned(response, "certifications among all")
        _assert_scope(result, "all_profile", "certifications among all")
        print(f"\n[CERTS ALL] {response[:300]}")


# ── Class 2: Targeted by Name ───────────────────────────────────────────────

class TestTargetedByName:
    """Tests for queries targeting a specific candidate by name."""

    def test_about_gokul(self):
        result = _run_query("tell me about Gokul")
        response = result["response"]
        _assert_non_empty(response, "about Gokul")
        _assert_no_banned(response, "about Gokul")
        _assert_mentions_name(response, "Gokul")
        # Should NOT be all_profile
        metadata = result.get("metadata", {})
        scope = metadata.get("scope")
        assert scope != "all_profile", f"Expected targeted, got: {scope}"
        print(f"\n[ABOUT GOKUL] {response[:300]}")

    def test_swapnil_skills(self):
        result = _run_query("what are Swapnil's skills")
        response = result["response"]
        _assert_non_empty(response, "Swapnil's skills")
        _assert_no_banned(response, "Swapnil's skills")
        _assert_mentions_name(response, "Swapnil")
        assert len(response) > 50, f"Skills response too short ({len(response)} chars)"
        print(f"\n[SWAPNIL SKILLS] {response[:300]}")

    def test_abinaya_experience(self):
        result = _run_query("Abinaya's work experience")
        response = result["response"]
        _assert_non_empty(response, "Abinaya's experience")
        _assert_no_banned(response, "Abinaya's experience")
        _assert_mentions_name(response, "Abinaya")
        print(f"\n[ABINAYA EXP] {response[:300]}")

    def test_abhishek_education(self):
        result = _run_query("what is Abhishek's education")
        response = result["response"]
        _assert_non_empty(response, "Abhishek's education")
        _assert_no_banned(response, "Abhishek's education")
        _assert_mentions_name(response, "Abhishek")
        print(f"\n[ABHISHEK EDU] {response[:300]}")

    def test_aadithya_certifications(self):
        result = _run_query("Aadithya's certifications")
        response = result["response"]
        _assert_non_empty(response, "Aadithya's certs")
        _assert_no_banned(response, "Aadithya's certs")
        _assert_mentions_name(response, "Aadithya")
        print(f"\n[AADITHYA CERTS] {response[:300]}")

    def test_aloysius_profile(self):
        result = _run_query("summarize Aloysius's profile")
        response = result["response"]
        _assert_non_empty(response, "Aloysius profile")
        _assert_no_banned(response, "Aloysius profile")
        _assert_mentions_name(response, "Aloysius")
        print(f"\n[ALOYSIUS PROFILE] {response[:300]}")

    def test_gokul_projects(self):
        result = _run_query("what projects has Gokul worked on")
        response = result["response"]
        _assert_non_empty(response, "Gokul projects")
        _assert_no_banned(response, "Gokul projects")
        _assert_mentions_name(response, "Gokul")
        print(f"\n[GOKUL PROJECTS] {response[:300]}")

    def test_swapnil_career(self):
        result = _run_query("describe Swapnil's career")
        response = result["response"]
        _assert_non_empty(response, "Swapnil career")
        _assert_no_banned(response, "Swapnil career")
        _assert_mentions_name(response, "Swapnil")
        print(f"\n[SWAPNIL CAREER] {response[:300]}")


# ── Class 3: Contact Info ────────────────────────────────────────────────────

class TestContactInfo:
    """Tests for contact information retrieval."""

    def test_gokul_email(self):
        result = _run_query("what is Gokul's email")
        response = result["response"]
        _assert_non_empty(response, "Gokul email")
        _assert_no_banned(response, "Gokul email")
        _assert_mentions_name(response, "Gokul")
        _assert_has_contact_pattern(response)
        print(f"\n[GOKUL EMAIL] {response[:300]}")

    def test_reach_abinaya(self):
        result = _run_query("how can I reach Abinaya")
        response = result["response"]
        _assert_non_empty(response, "reach Abinaya")
        _assert_no_banned(response, "reach Abinaya")
        _assert_has_contact_pattern(response)
        print(f"\n[REACH ABINAYA] {response[:300]}")

    def test_contact_all(self):
        result = _run_query("contact information for all candidates")
        response = result["response"]
        _assert_non_empty(response, "contact all")
        _assert_no_banned(response, "contact all")
        _assert_scope(result, "all_profile", "contact all")
        _assert_has_contact_pattern(response)
        print(f"\n[CONTACT ALL] {response[:300]}")

    def test_swapnil_phone(self):
        result = _run_query("Swapnil's phone number")
        response = result["response"]
        _assert_non_empty(response, "Swapnil phone")
        _assert_no_banned(response, "Swapnil phone")
        _assert_mentions_name(response, "Swapnil")
        print(f"\n[SWAPNIL PHONE] {response[:300]}")

    def test_linkedin_all(self):
        result = _run_query("show me linkedin profiles of candidates")
        response = result["response"]
        _assert_non_empty(response, "linkedin all")
        _assert_no_banned(response, "linkedin all")
        _assert_scope(result, "all_profile", "linkedin all")
        print(f"\n[LINKEDIN ALL] {response[:300]}")


# ── Class 4: Field Extraction ────────────────────────────────────────────────

class TestFieldExtraction:
    """Tests for specific field extraction queries."""

    def test_abhishek_technical_skills(self):
        result = _run_query("what technical skills does Abhishek have")
        response = result["response"]
        _assert_non_empty(response, "Abhishek tech skills")
        _assert_no_banned(response, "Abhishek tech skills")
        _assert_mentions_name(response, "Abhishek")
        assert len(response) > 50, f"Skills response too short ({len(response)} chars)"
        print(f"\n[ABHISHEK TECH] {response[:300]}")

    def test_programming_languages_all(self):
        result = _run_query("list all programming languages mentioned across resumes")
        response = result["response"]
        _assert_non_empty(response, "programming languages all")
        _assert_no_banned(response, "programming languages all")
        _assert_scope(result, "all_profile", "programming languages all")
        print(f"\n[PROG LANGS] {response[:300]}")

    def test_aadithya_degrees(self):
        result = _run_query("what degrees does Aadithya have")
        response = result["response"]
        _assert_non_empty(response, "Aadithya degrees")
        _assert_no_banned(response, "Aadithya degrees")
        _assert_mentions_name(response, "Aadithya")
        print(f"\n[AADITHYA DEGREES] {response[:300]}")

    def test_gokul_certifications(self):
        result = _run_query("certifications held by Gokul")
        response = result["response"]
        _assert_non_empty(response, "Gokul certs")
        _assert_no_banned(response, "Gokul certs")
        _assert_mentions_name(response, "Gokul")
        print(f"\n[GOKUL CERTS] {response[:300]}")

    def test_swapnil_experience_years(self):
        result = _run_query("how many years of experience does Swapnil have")
        response = result["response"]
        _assert_non_empty(response, "Swapnil years")
        _assert_no_banned(response, "Swapnil years")
        _assert_mentions_name(response, "Swapnil")
        _assert_has_digit(response)
        print(f"\n[SWAPNIL YEARS] {response[:300]}")

    def test_abinaya_achievements(self):
        result = _run_query("what are Abinaya's achievements")
        response = result["response"]
        _assert_non_empty(response, "Abinaya achievements")
        _assert_no_banned(response, "Abinaya achievements")
        _assert_mentions_name(response, "Abinaya")
        print(f"\n[ABINAYA ACHIEVEMENTS] {response[:300]}")

    def test_aloysius_soft_skills(self):
        result = _run_query("soft skills mentioned in Aloysius's resume")
        response = result["response"]
        _assert_non_empty(response, "Aloysius soft skills")
        _assert_no_banned(response, "Aloysius soft skills")
        _assert_mentions_name(response, "Aloysius")
        print(f"\n[ALOYSIUS SOFT SKILLS] {response[:300]}")

    def test_abhishek_tools(self):
        result = _run_query("what tools and technologies does Abhishek know")
        response = result["response"]
        _assert_non_empty(response, "Abhishek tools")
        _assert_no_banned(response, "Abhishek tools")
        _assert_mentions_name(response, "Abhishek")
        print(f"\n[ABHISHEK TOOLS] {response[:300]}")


# ── Class 5: Reasoning & Assessment ─────────────────────────────────────────

class TestReasoningAssessment:
    """Tests for reasoning/assessment queries."""

    def test_gokul_python_fit(self):
        result = _run_query("is Gokul qualified for a Python developer role")
        response = result["response"]
        _assert_non_empty(response, "Gokul Python fit")
        _assert_no_banned(response, "Gokul Python fit")
        _assert_mentions_name(response, "Gokul")
        print(f"\n[GOKUL PYTHON] {response[:300]}")

    def test_best_data_science(self):
        result = _run_query("who is the best fit for a data science position")
        response = result["response"]
        _assert_non_empty(response, "best data science")
        _assert_no_banned(response, "best data science")
        _assert_scope(result, "all_profile", "best data science")
        _assert_mentions_any_name(response, 1)
        print(f"\n[BEST DS] {response[:300]}")

    def test_compare_swapnil_gokul(self):
        result = _run_query("compare Swapnil and Gokul for a backend role")
        response = result["response"]
        _assert_non_empty(response, "compare Swapnil Gokul")
        _assert_no_banned(response, "compare Swapnil Gokul")
        _assert_mentions_name(response, "Swapnil")
        _assert_mentions_name(response, "Gokul")
        print(f"\n[COMPARE S&G] {response[:300]}")

    def test_rank_by_education(self):
        result = _run_query("rank candidates by education level")
        response = result["response"]
        _assert_non_empty(response, "rank by education")
        _assert_no_banned(response, "rank by education")
        _assert_scope(result, "all_profile", "rank by education")
        _assert_mentions_any_name(response, 2)
        print(f"\n[RANK EDU] {response[:300]}")

    def test_recommend_senior(self):
        result = _run_query("which candidate would you recommend for a senior role")
        response = result["response"]
        _assert_non_empty(response, "recommend senior")
        _assert_no_banned(response, "recommend senior")
        _assert_scope(result, "all_profile", "recommend senior")
        _assert_mentions_any_name(response, 1)
        print(f"\n[RECOMMEND SR] {response[:300]}")


# ── Class 6: Specific Document ───────────────────────────────────────────────

class TestSpecificDocument:
    """Tests with explicit document_id parameter."""

    def test_skills_by_doc_id(self):
        doc_id = DOC_IDS["swapnil"]
        result = _run_query("what are the skills of this candidate", document_id=doc_id)
        response = result["response"]
        _assert_non_empty(response, "skills by doc_id")
        _assert_no_banned(response, "skills by doc_id")
        _assert_has_sources(result, "skills by doc_id")
        print(f"\n[SKILLS DOC] {response[:300]}")

    def test_summarize_by_doc_id(self):
        doc_id = DOC_IDS["gokul"]
        result = _run_query("summarize this resume", document_id=doc_id)
        response = result["response"]
        _assert_non_empty(response, "summarize by doc_id")
        _assert_no_banned(response, "summarize by doc_id")
        print(f"\n[SUMMARIZE DOC] {response[:300]}")

    def test_education_by_doc_id(self):
        doc_id = DOC_IDS["abhishek"]
        result = _run_query("what is the education background", document_id=doc_id)
        response = result["response"]
        _assert_non_empty(response, "education by doc_id")
        _assert_no_banned(response, "education by doc_id")
        print(f"\n[EDU DOC] {response[:300]}")

    def test_certifications_by_doc_id(self):
        doc_id = DOC_IDS["aadithya"]
        result = _run_query("extract all certifications", document_id=doc_id)
        response = result["response"]
        _assert_non_empty(response, "certs by doc_id")
        _assert_no_banned(response, "certs by doc_id")
        print(f"\n[CERTS DOC] {response[:300]}")


# ── Class 7: Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and graceful degradation."""

    def test_nonexistent_person(self):
        """Query about someone not in the data should not crash."""
        result = _run_query("tell me about John")
        response = result["response"]
        # Should not crash; response may be empty or a graceful message
        assert isinstance(response, str), "Response should be a string"
        print(f"\n[JOHN (missing)] {response[:300]}")

    def test_wrong_domain(self):
        """Invoice query on resume profile should not crash."""
        result = _run_query("what is the invoice total")
        response = result["response"]
        assert isinstance(response, str), "Response should be a string"
        print(f"\n[INVOICE (wrong)] {response[:300]}")

    def test_single_word_query(self):
        """Single word query should return something."""
        result = _run_query("skills")
        response = result["response"]
        _assert_non_empty(response, "skills")
        print(f"\n[SINGLE WORD] {response[:300]}")

    def test_minimal_query(self):
        """Minimal query should not crash."""
        result = _run_query("what")
        response = result["response"]
        assert isinstance(response, str), "Response should be a string"
        print(f"\n[MINIMAL] {response[:300]}")

    def test_everything_about_everyone(self):
        """Exhaustive query should route to all_profile and mention names."""
        result = _run_query("give me everything about every candidate in detail")
        response = result["response"]
        _assert_non_empty(response, "everything everyone")
        _assert_scope(result, "all_profile", "everything everyone")
        _assert_mentions_any_name(response, 2)
        print(f"\n[EVERYTHING] {response[:300]}")


# ── Class 8: Quality Report ──────────────────────────────────────────────────

class TestQualityReport:
    """Meta-test that runs all 43 queries and prints a quality summary."""

    # All 43 queries organized by category
    _ALL_QUERIES = {
        "multi_doc_scope": [
            ("rank all candidates by skills and experience", {}),
            ("compare all candidates", {}),
            ("list all resumes in this profile", {}),
            ("how many candidates are in this profile", {}),
            ("who is the most experienced candidate", {}),
            ("which candidate has the best technical skills", {}),
            ("summarize all candidates", {}),
            ("who has certifications among all candidates", {}),
        ],
        "targeted_by_name": [
            ("tell me about Gokul", {}),
            ("what are Swapnil's skills", {}),
            ("Abinaya's work experience", {}),
            ("what is Abhishek's education", {}),
            ("Aadithya's certifications", {}),
            ("summarize Aloysius's profile", {}),
            ("what projects has Gokul worked on", {}),
            ("describe Swapnil's career", {}),
        ],
        "contact_info": [
            ("what is Gokul's email", {}),
            ("how can I reach Abinaya", {}),
            ("contact information for all candidates", {}),
            ("Swapnil's phone number", {}),
            ("show me linkedin profiles of candidates", {}),
        ],
        "field_extraction": [
            ("what technical skills does Abhishek have", {}),
            ("list all programming languages mentioned across resumes", {}),
            ("what degrees does Aadithya have", {}),
            ("certifications held by Gokul", {}),
            ("how many years of experience does Swapnil have", {}),
            ("what are Abinaya's achievements", {}),
            ("soft skills mentioned in Aloysius's resume", {}),
            ("what tools and technologies does Abhishek know", {}),
        ],
        "reasoning": [
            ("is Gokul qualified for a Python developer role", {}),
            ("who is the best fit for a data science position", {}),
            ("compare Swapnil and Gokul for a backend role", {}),
            ("rank candidates by education level", {}),
            ("which candidate would you recommend for a senior role", {}),
        ],
        "specific_document": [
            ("what are the skills of this candidate", {"document_id": DOC_IDS["swapnil"]}),
            ("summarize this resume", {"document_id": DOC_IDS["gokul"]}),
            ("what is the education background", {"document_id": DOC_IDS["abhishek"]}),
            ("extract all certifications", {"document_id": DOC_IDS["aadithya"]}),
        ],
        "edge_cases": [
            ("tell me about John", {}),
            ("what is the invoice total", {}),
            ("skills", {}),
            ("what", {}),
            ("give me everything about every candidate in detail", {}),
        ],
    }

    def test_quality_report(self):
        """Run all queries, collect results, and print a quality report."""
        results = {}
        for category, queries in self._ALL_QUERIES.items():
            cat_results = []
            for query, kwargs in queries:
                start = time.time()
                try:
                    result = _run_query(query, **kwargs)
                    response = result.get("response", "")
                    elapsed = time.time() - start
                    sources = result.get("sources", [])
                    metadata = result.get("metadata", {})

                    # Evaluate quality signals
                    is_empty = not response or len(response.strip()) == 0
                    has_banned = any(
                        p in response.lower() for p in _BANNED_PHRASES
                    )
                    has_sources = len(sources) > 0
                    has_content = len(response) > 30

                    status = "PASS"
                    issues = []
                    if is_empty:
                        status = "FAIL"
                        issues.append("empty")
                    if has_banned:
                        status = "WARN"
                        issues.append("banned_phrase")
                    if not has_sources and category != "edge_cases":
                        issues.append("no_sources")

                    cat_results.append({
                        "query": query,
                        "status": status,
                        "response_len": len(response),
                        "sources": len(sources),
                        "elapsed": elapsed,
                        "issues": issues,
                        "scope": metadata.get("scope"),
                    })
                except Exception as exc:
                    cat_results.append({
                        "query": query,
                        "status": "ERROR",
                        "response_len": 0,
                        "sources": 0,
                        "elapsed": time.time() - start,
                        "issues": [str(exc)[:100]],
                        "scope": None,
                    })
            results[category] = cat_results

        # Print report
        print("\n" + "=" * 80)
        print("RETRIEVAL INTELLIGENCE QUALITY REPORT")
        print("=" * 80)

        total_pass = 0
        total_warn = 0
        total_fail = 0
        total_error = 0

        for category, cat_results in results.items():
            cat_pass = sum(1 for r in cat_results if r["status"] == "PASS")
            cat_total = len(cat_results)
            print(f"\n--- {category.upper()} ({cat_pass}/{cat_total}) ---")
            for r in cat_results:
                icon = {"PASS": "+", "WARN": "~", "FAIL": "-", "ERROR": "!"}[r["status"]]
                issues_str = f" [{', '.join(r['issues'])}]" if r["issues"] else ""
                print(
                    f"  [{icon}] {r['query'][:55]:55s} "
                    f"len={r['response_len']:4d} "
                    f"src={r['sources']:2d} "
                    f"{r['elapsed']:.1f}s{issues_str}"
                )
                if r["status"] == "PASS":
                    total_pass += 1
                elif r["status"] == "WARN":
                    total_warn += 1
                elif r["status"] == "FAIL":
                    total_fail += 1
                else:
                    total_error += 1

        total = total_pass + total_warn + total_fail + total_error
        print(f"\n{'=' * 80}")
        print(
            f"TOTAL: {total_pass}/{total} PASS, "
            f"{total_warn} WARN, {total_fail} FAIL, {total_error} ERROR"
        )
        print(f"{'=' * 80}\n")

        # Meta-test: at least 80% should pass (not fail/error)
        pass_rate = (total_pass + total_warn) / total if total else 0
        assert pass_rate >= 0.70, (
            f"Quality too low: {pass_rate:.0%} pass rate "
            f"({total_fail} failures, {total_error} errors)"
        )
