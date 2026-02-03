from __future__ import annotations

from datetime import date

from src.rag.experience_extractor import extract_experience


def test_explicit_years_high_confidence():
    text = "Senior Engineer with 14+ years of experience in software development."
    result = extract_experience(text)
    assert result.total_years_experience == 14
    assert result.experience_confidence == "high"
    assert result.experience_basis == "explicit_years"


def test_date_ranges_compute_years_and_ignore_dob():
    text = """
    Experience
    Software Engineer — Jan 2019 - Present
    DOB: 1980
    """
    result = extract_experience(text, as_of=date(2024, 1, 1))
    assert result.total_years_experience == 5
    assert result.experience_basis == "computed_date_ranges"


def test_conflicting_explicit_statements():
    text = "10 years of experience in IT. Also noted: 12 years of experience in consulting."
    result = extract_experience(text)
    assert result.total_years_experience is None
    assert result.experience_basis == "conflicting"
