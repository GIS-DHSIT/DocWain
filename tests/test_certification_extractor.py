from __future__ import annotations

from src.rag.certification_extractor import extract_certifications


def test_heading_noise_returns_empty():
    text = "Certifications, Contact"
    result = extract_certifications(text)
    assert result == []


def test_extracts_real_certifications():
    text = """
    Certifications
    - AWS Certified Solutions Architect
    - Certified Scrum Master (CSM)
    """
    result = extract_certifications(text)
    assert "AWS Certified Solutions Architect" in result
    assert "Certified Scrum Master (CSM)" in result
