"""Tests for neural schema — profile intelligence and intent understanding.

Covers:
1. QueryIntent parsing (action, top_n, criteria, entities)
2. CandidateDigest construction
3. ProfileIntelligence building
4. Intelligent ranking by criteria
5. Response formatting (direct answers, no generic preamble)
6. Enterprise rendering integration (ranking vs comparison routing)
"""

from types import SimpleNamespace
from typing import List

import pytest

from src.rag_v3.neural_schema import (
    QueryIntent,
    CandidateDigest,
    ProfileIntelligence,
    parse_query_intent,
    build_candidate_digest,
    build_profile_intelligence,
    rank_by_criteria,
    format_ranking_response,
    format_comparison_response,
    _parse_years_float,
    _infer_education_level,
    _format_criteria,
)


# ---------------------------------------------------------------------------
# Helpers — fake Candidate objects matching HRSchema.candidates.items
# ---------------------------------------------------------------------------

def _make_candidate(
    name: str = "Candidate",
    technical_skills: List[str] = None,
    functional_skills: List[str] = None,
    certifications: List[str] = None,
    education: List[str] = None,
    total_years_experience: str = "",
    experience_summary: str = "",
    role: str = "",
    designation: str = "",
    achievements: List[str] = None,
    evidence_spans: List[str] = None,
    emails: List[str] = None,
    phones: List[str] = None,
    linkedins: List[str] = None,
):
    return SimpleNamespace(
        name=name,
        technical_skills=technical_skills or [],
        functional_skills=functional_skills or [],
        certifications=certifications or [],
        education=education or [],
        total_years_experience=total_years_experience,
        experience_summary=experience_summary,
        role=role,
        designation=designation,
        achievements=achievements or [],
        evidence_spans=evidence_spans or [],
        emails=emails or [],
        phones=phones or [],
        linkedins=linkedins or [],
    )


# ===========================================================================
# Test Class 1: QueryIntent Parsing
# ===========================================================================

class TestParseQueryIntent:
    """Tests for intelligent intent parsing without asking the user."""

    def test_rank_top_2_resumes(self):
        intent = parse_query_intent("rank the top 2 resumes")
        assert intent.action == "rank"
        assert intent.top_n == 2

    def test_who_are_the_top_2(self):
        intent = parse_query_intent("who are the top 2?")
        assert intent.action == "rank"
        assert intent.top_n == 2

    def test_best_3_candidates(self):
        intent = parse_query_intent("show me the best 3 candidates")
        assert intent.action == "rank"
        assert intent.top_n == 3

    def test_top_five_word(self):
        intent = parse_query_intent("top five resumes")
        assert intent.action == "rank"
        assert intent.top_n == 5

    def test_rank_without_n(self):
        intent = parse_query_intent("rank all candidates")
        assert intent.action == "rank"
        assert intent.top_n is None

    def test_compare_intent(self):
        intent = parse_query_intent("compare Alice and Bob")
        assert intent.action == "compare"

    def test_filter_intent(self):
        intent = parse_query_intent("who has SAP experience?")
        assert intent.action == "filter"
        assert "sap" in intent.criteria

    def test_list_intent(self):
        intent = parse_query_intent("list all candidates")
        assert intent.action == "list"

    def test_summarize_intent(self):
        intent = parse_query_intent("give me a summary of all resumes")
        assert intent.action == "summarize"

    def test_criteria_extraction_sap(self):
        intent = parse_query_intent("rank the top 2 SAP consultants")
        assert "sap" in intent.criteria

    def test_criteria_extraction_experience(self):
        intent = parse_query_intent("who has the most experience?")
        assert "experience" in intent.criteria

    def test_criteria_extraction_supply_chain(self):
        intent = parse_query_intent("rank candidates by supply chain expertise")
        assert "supply_chain" in intent.criteria

    def test_criteria_extraction_data_science(self):
        intent = parse_query_intent("top 3 data science candidates")
        assert "data_science" in intent.criteria

    def test_entity_extraction(self):
        intent = parse_query_intent(
            "compare Alice and Bob",
            candidate_names=["Alice", "Bob", "Carol"],
        )
        assert "Alice" in intent.entities
        assert "Bob" in intent.entities
        assert "Carol" not in intent.entities

    def test_question_detection(self):
        intent = parse_query_intent("who is the best candidate?")
        assert intent.is_question is True

    def test_non_question(self):
        intent = parse_query_intent("rank the top 2 resumes")
        assert intent.is_question is False

    def test_empty_query(self):
        intent = parse_query_intent("")
        assert intent.action == "detail"
        assert intent.top_n is None

    def test_top_n_overrides_filter_to_rank(self):
        """'who are the top 2?' has 'who' (filter) but top_n should force rank."""
        intent = parse_query_intent("who are the top 2 candidates?")
        assert intent.action == "rank"
        assert intent.top_n == 2


# ===========================================================================
# Test Class 2: CandidateDigest
# ===========================================================================

class TestCandidateDigest:
    """Tests for compact candidate representation."""

    def test_build_from_candidate(self):
        cand = _make_candidate(
            name="Aloysius",
            technical_skills=["SAP", "Scala", "Supply Chain", "Excel", "Logistics"],
            certifications=["SAP MM"],
            education=["MBA from Xaviers"],
            total_years_experience="16 years",
            role="Supply Chain Manager",
        )
        digest = build_candidate_digest(cand)
        assert digest.name == "Aloysius"
        assert digest.years_experience == 16.0
        assert digest.skill_count == 5
        assert digest.cert_count == 1
        assert digest.education_level == "Masters"  # MBA = Masters
        assert digest.role == "Supply Chain Manager"
        assert "SAP" in digest.key_skills

    def test_completeness_score(self):
        full = _make_candidate(
            name="Alice",
            technical_skills=["Python", "Java"],
            certifications=["AWS"],
            education=["B.Tech CS"],
            total_years_experience="5 years",
            role="Software Engineer",
            experience_summary="5 years at Google...",
        )
        empty = _make_candidate()

        full_digest = build_candidate_digest(full)
        empty_digest = build_candidate_digest(empty)

        assert full_digest.completeness_score > 0.8
        assert empty_digest.completeness_score < 0.2

    def test_education_level_inference(self):
        assert _infer_education_level(["PhD in Computer Science"]) == "PhD"
        assert _infer_education_level(["M.Tech in Data Science"]) == "Masters"
        assert _infer_education_level(["MBA Marketing"]) == "Masters"
        assert _infer_education_level(["B.Tech Computer Science"]) == "Bachelors"
        assert _infer_education_level(["Diploma in IT"]) == "Diploma"
        assert _infer_education_level([]) == "N/A"

    def test_parse_years(self):
        assert _parse_years_float("16 years") == 16.0
        assert _parse_years_float("3.5 years") == 3.5
        assert _parse_years_float("") is None
        assert _parse_years_float(None) is None

    def test_domain_keywords_extracted(self):
        cand = _make_candidate(
            technical_skills=["SAP MM", "Procurement", "Supply Chain", "Inventory"],
        )
        digest = build_candidate_digest(cand)
        assert "procurement" in digest.domain_keywords
        assert "supply" in digest.domain_keywords


# ===========================================================================
# Test Class 3: ProfileIntelligence
# ===========================================================================

class TestProfileIntelligence:
    """Tests for profile-level context understanding."""

    def test_build_from_candidates(self):
        candidates = [
            _make_candidate(
                name="Alice",
                technical_skills=["Python", "Java", "SAP"],
                total_years_experience="5 years",
            ),
            _make_candidate(
                name="Bob",
                technical_skills=["Python", "React", "Docker"],
                total_years_experience="3 years",
            ),
        ]
        profile = build_profile_intelligence(candidates)
        assert profile.candidate_count == 2
        assert len(profile.candidates) == 2
        assert "python" in profile.common_skills  # appears in both
        assert profile.avg_experience_years == 4.0
        assert "Alice" in profile.candidate_names
        assert "Bob" in profile.candidate_names

    def test_empty_candidates(self):
        profile = build_profile_intelligence([])
        assert profile.candidate_count == 0
        assert profile.candidates == []

    def test_common_skills_threshold(self):
        """Only skills appearing in 2+ candidates are 'common'."""
        candidates = [
            _make_candidate(name="A", technical_skills=["Python", "Java"]),
            _make_candidate(name="B", technical_skills=["Python", "React"]),
            _make_candidate(name="C", technical_skills=["Go", "Rust"]),
        ]
        profile = build_profile_intelligence(candidates)
        assert "python" in profile.common_skills
        assert "java" not in profile.common_skills


# ===========================================================================
# Test Class 4: Intelligent Ranking
# ===========================================================================

class TestRankByCriteria:
    """Tests for multi-criteria ranking with profile awareness."""

    def _make_profile(self):
        """Profile with 3 candidates of varying strength."""
        candidates = [
            _make_candidate(
                name="Senior Pro",
                technical_skills=["SAP", "Procurement", "Supply Chain", "Logistics", "Excel",
                                  "Inventory", "GSCM", "S/4 HANA"],
                certifications=["SAP MM Certified", "PMP"],
                education=["MBA from Xaviers"],
                total_years_experience="16 years",
                role="Supply Chain Manager",
                experience_summary="16 years in supply chain and procurement management.",
            ),
            _make_candidate(
                name="Mid Level",
                technical_skills=["SAP", "Procurement", "Python"],
                certifications=["SAP MM"],
                education=["B.Tech CS"],
                total_years_experience="3 years",
                role="SAP Consultant",
            ),
            _make_candidate(
                name="Junior",
                technical_skills=["Python", "React"],
                education=["B.Tech CS"],
                total_years_experience="1 year",
                role="Developer",
            ),
        ]
        return build_profile_intelligence(candidates)

    def test_overall_ranking(self):
        """Without criteria, rank by overall profile strength."""
        profile = self._make_profile()
        intent = QueryIntent(action="rank")
        ranked = rank_by_criteria(profile, intent)
        assert len(ranked) == 3
        assert ranked[0][0].name == "Senior Pro"

    def test_top_n_selection(self):
        """'top 2' returns exactly 2 candidates."""
        profile = self._make_profile()
        intent = QueryIntent(action="rank", top_n=2)
        ranked = rank_by_criteria(profile, intent)
        assert len(ranked) == 2
        assert ranked[0][0].name == "Senior Pro"

    def test_top_1_selection(self):
        """'top 1' / 'best' returns exactly 1."""
        profile = self._make_profile()
        intent = QueryIntent(action="rank", top_n=1)
        ranked = rank_by_criteria(profile, intent)
        assert len(ranked) == 1

    def test_sap_criteria_boosts_sap_candidate(self):
        """SAP criteria should boost SAP-focused candidates."""
        profile = self._make_profile()
        intent = QueryIntent(action="rank", criteria=["sap"])
        ranked = rank_by_criteria(profile, intent)
        # Senior Pro has 8 skills including SAP + 2 SAP certs → should rank first
        assert ranked[0][0].name == "Senior Pro"
        # Junior has no SAP → should rank last
        assert ranked[-1][0].name == "Junior"

    def test_reason_provided(self):
        """Each ranking includes a reason string."""
        profile = self._make_profile()
        intent = QueryIntent(action="rank", top_n=2)
        ranked = rank_by_criteria(profile, intent)
        for digest, score, reason in ranked:
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_ranking_score_ordering(self):
        """Scores should be in descending order."""
        profile = self._make_profile()
        intent = QueryIntent(action="rank")
        ranked = rank_by_criteria(profile, intent)
        scores = [score for _, score, _ in ranked]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Test Class 5: Response Formatting
# ===========================================================================

class TestFormatRankingResponse:
    """Tests for direct, query-answering format."""

    def _ranked_results(self):
        d1 = CandidateDigest(
            name="Alice Chen", role="Senior Engineer", years_experience=8,
            skill_count=12, cert_count=2, education_level="Masters",
            key_skills=["Python", "Java", "React", "Docker", "Kubernetes"],
            key_certs=["AWS Solutions Architect", "PMP"],
        )
        d2 = CandidateDigest(
            name="Bob Smith", role="Junior Developer", years_experience=2,
            skill_count=5, cert_count=0, education_level="Bachelors",
            key_skills=["Python", "JavaScript"],
            key_certs=[],
        )
        return [
            (d1, 45.0, "8 years experience; 12 skills; Masters"),
            (d2, 20.0, "2 years experience; 5 skills"),
        ]

    def test_format_top_n(self):
        ranked = self._ranked_results()
        intent = QueryIntent(action="rank", top_n=2)
        profile = ProfileIntelligence(candidate_count=5, candidates=[])
        result = format_ranking_response(ranked, intent, profile)
        assert "**Top 2 of 5 candidates**" in result
        assert "**Alice Chen**" in result
        assert "**Bob Smith**" in result
        assert "1." in result
        assert "2." in result

    def test_no_generic_preamble(self):
        ranked = self._ranked_results()
        intent = QueryIntent(action="rank", top_n=2)
        profile = ProfileIntelligence(candidate_count=5, candidates=[])
        result = format_ranking_response(ranked, intent, profile)
        assert "I understand your question" not in result
        assert "Here's what I found" not in result

    def test_includes_reasoning(self):
        ranked = self._ranked_results()
        intent = QueryIntent(action="rank", top_n=2)
        profile = ProfileIntelligence(candidate_count=5, candidates=[])
        result = format_ranking_response(ranked, intent, profile)
        assert "*Why:" in result

    def test_remaining_count(self):
        ranked = self._ranked_results()
        intent = QueryIntent(action="rank", top_n=2)
        profile = ProfileIntelligence(candidate_count=5, candidates=[])
        result = format_ranking_response(ranked, intent, profile)
        assert "3 other candidates not shown" in result

    def test_criteria_in_header(self):
        ranked = self._ranked_results()
        intent = QueryIntent(action="rank", top_n=2, criteria=["sap", "experience"])
        profile = ProfileIntelligence(candidate_count=5, candidates=[])
        result = format_ranking_response(ranked, intent, profile)
        assert "sap" in result.lower()
        assert "experience" in result.lower()

    def test_format_all_ranking(self):
        ranked = self._ranked_results()
        intent = QueryIntent(action="rank")
        profile = ProfileIntelligence(candidate_count=2, candidates=[])
        result = format_ranking_response(ranked, intent, profile)
        assert "**Ranking of 2 candidates**" in result

    def test_empty_results(self):
        intent = QueryIntent(action="rank", top_n=2)
        profile = ProfileIntelligence(candidate_count=0, candidates=[])
        result = format_ranking_response([], intent, profile)
        assert "No candidates found" in result


# ===========================================================================
# Test Class 6: Format Comparison
# ===========================================================================

class TestFormatComparison:

    def test_two_way_comparison(self):
        d1 = CandidateDigest(
            name="Alice", years_experience=5, skill_count=10, cert_count=2,
            education_level="Masters", role="Engineer",
            key_skills=["Python", "Java", "React"],
        )
        d2 = CandidateDigest(
            name="Bob", years_experience=3, skill_count=6, cert_count=1,
            education_level="Bachelors", role="Developer",
            key_skills=["Python", "Go", "Docker"],
        )
        intent = QueryIntent(action="compare")
        profile = ProfileIntelligence(candidate_count=2, candidates=[d1, d2])
        result = format_comparison_response([d1, d2], intent, profile)
        assert "Alice vs Bob" in result
        assert "Experience" in result
        assert "Skills" in result


# ===========================================================================
# Test Class 7: Enterprise Integration
# ===========================================================================

class TestEnterpriseRendering:
    """Tests that _render_hr correctly routes ranking vs comparison."""

    def _make_hr_schema(self, candidates):
        """Build a minimal HRSchema-like object."""
        return SimpleNamespace(
            candidates=SimpleNamespace(
                items=candidates,
                missing_reason=None,
            ),
        )

    def test_rank_query_produces_ranking_not_comparison(self):
        """'rank the top 2' should produce a ranking, not a comparison table."""
        from src.rag_v3.enterprise import _render_hr

        candidates = [
            _make_candidate(
                name="Senior Pro",
                technical_skills=["SAP", "Procurement", "Supply Chain", "Logistics",
                                  "Excel", "Inventory", "GSCM", "S/4 HANA"],
                certifications=["SAP MM Certified", "PMP"],
                education=["MBA from Xaviers"],
                total_years_experience="16 years",
                role="Supply Chain Manager",
                experience_summary="16 years in supply chain management.",
            ),
            _make_candidate(
                name="Mid Level",
                technical_skills=["SAP", "Procurement", "Python"],
                certifications=["SAP MM"],
                total_years_experience="3 years",
                role="SAP Consultant",
            ),
            _make_candidate(
                name="Junior",
                technical_skills=["Python", "React"],
                total_years_experience="1 year",
                role="Developer",
            ),
        ]
        schema = self._make_hr_schema(candidates)
        result = _render_hr(schema, "rank", query="rank the top 2 resumes")

        # Should be a ranking, not a comparison table
        assert "**Top 2 of 3 candidates**" in result
        assert "**Senior Pro**" in result
        # Should NOT contain all-fields comparison format
        assert "Comparison of" not in result
        # Should only show 2 candidates
        assert "1." in result
        assert "2." in result

    def test_rank_without_top_n_shows_all(self):
        """'rank all candidates' should rank all, not compare."""
        from src.rag_v3.enterprise import _render_hr

        candidates = [
            _make_candidate(name="A", technical_skills=["Python"] * 5, total_years_experience="10 years"),
            _make_candidate(name="B", technical_skills=["Java"] * 3, total_years_experience="5 years"),
        ]
        schema = self._make_hr_schema(candidates)
        result = _render_hr(schema, "rank", query="rank all candidates")

        assert "Ranking of 2 candidates" in result
        assert "1." in result
        assert "2." in result

    def test_compare_query_produces_comparison(self):
        """'compare' intent should produce comparison, not ranking."""
        from src.rag_v3.enterprise import _render_hr

        candidates = [
            _make_candidate(name="Alice", technical_skills=["Python", "Java"], total_years_experience="5 years"),
            _make_candidate(name="Bob", technical_skills=["Python", "Go"], total_years_experience="3 years"),
        ]
        schema = self._make_hr_schema(candidates)
        result = _render_hr(schema, "compare", query="compare Alice and Bob")

        # Should be a comparison
        assert "Alice" in result
        assert "Bob" in result

    def test_multi_candidate_with_top_n_query_routes_to_ranking(self):
        """Multi-candidate view with a 'top 2' query should auto-detect ranking."""
        from src.rag_v3.enterprise import _render_hr

        candidates = [
            _make_candidate(name="A", technical_skills=["SAP"] * 8, total_years_experience="16 years"),
            _make_candidate(name="B", technical_skills=["SAP"] * 4, total_years_experience="5 years"),
            _make_candidate(name="C", technical_skills=["Python"] * 2, total_years_experience="1 year"),
        ]
        schema = self._make_hr_schema(candidates)
        # Intent is "detail" but query says "who are the top 2?"
        result = _render_hr(schema, "detail", query="who are the top 2?")

        # Should detect ranking intent and show top 2
        assert "**Top 2 of 3 candidates**" in result
        assert "1." in result
        assert "2." in result

    def test_no_generic_preamble_in_rank_response(self):
        """Ranking response should not start with 'I understand your question'."""
        from src.rag_v3.enterprise import _render_hr

        candidates = [
            _make_candidate(name="A", technical_skills=["SAP"] * 5, total_years_experience="10 years"),
            _make_candidate(name="B", technical_skills=["SAP"] * 3, total_years_experience="5 years"),
        ]
        schema = self._make_hr_schema(candidates)
        result = _render_hr(schema, "rank", query="rank the top 2 resumes")

        assert not result.startswith("I understand")
        assert "Here's what I found" not in result


# ===========================================================================
# Test Class 8: Criteria Formatting
# ===========================================================================

class TestFormatCriteria:

    def test_single_criterion(self):
        assert _format_criteria(["sap"]) == "sap"

    def test_two_criteria(self):
        result = _format_criteria(["sap", "experience"])
        assert "sap" in result
        assert "experience" in result

    def test_empty_criteria(self):
        assert _format_criteria([]) == "overall profile strength"

    def test_underscore_replaced(self):
        result = _format_criteria(["supply_chain"])
        assert "supply chain" in result
