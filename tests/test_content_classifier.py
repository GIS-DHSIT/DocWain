"""Tests for the consolidated content classifier."""
from src.embedding.pipeline.content_classifier import classify_doc_domain, classify_section_kind


# ── classify_section_kind — title-based ──────────────────────────────────

def test_classify_skills_technical_from_title():
    assert classify_section_kind("Python, Java", "Technical Skills") == "skills_technical"


def test_classify_skills_functional_from_title():
    assert classify_section_kind("Leadership, Communication", "Soft Skills") == "skills_functional"


def test_classify_skills_default_technical():
    """Bare 'Skills' title defaults to skills_technical."""
    assert classify_section_kind("Python, Docker", "Skills") == "skills_technical"


def test_classify_education_from_title():
    assert classify_section_kind("B.Tech CS, MIT", "Education") == "education"


def test_classify_certifications_from_title():
    assert classify_section_kind("AWS Solutions Architect", "Certifications") == "certifications"


def test_classify_experience_from_title():
    assert classify_section_kind("Led team of 5", "Work Experience") == "experience"


def test_classify_summary_from_title():
    assert classify_section_kind("Experienced developer", "Professional Summary") == "summary_objective"


def test_classify_contact_from_title():
    assert classify_section_kind("john@example.com", "Contact Information") == "identity_contact"


def test_classify_achievements_from_title():
    assert classify_section_kind("Best Employee 2024", "Awards & Achievements") == "achievements"


def test_classify_projects_from_title():
    """PROJECTS title maps to experience."""
    assert classify_section_kind("Built microservices platform", "PROJECTS") == "experience"


# ── classify_section_kind — content-based ────────────────────────────────

def test_classify_experience_from_content():
    text = "Led team of 5 developers, managed project delivery, 8 years of experience"
    assert classify_section_kind(text, "") == "experience"


def test_classify_education_from_content():
    text = "B.Tech in Computer Science from MIT, GPA 3.8, graduation 2020, bachelor degree, university coursework"
    assert classify_section_kind(text, "") == "education"


def test_classify_generic_content_returns_section_text():
    """Content with fewer than 2 keyword matches falls back to section_text."""
    assert classify_section_kind("The quick brown fox", "") == "section_text"


def test_classify_empty_text():
    assert classify_section_kind("", "") == "section_text"


def test_title_takes_priority_over_content():
    """A 'Professional Summary' title should win even if content has tech keywords."""
    text = "10+ years of experience in Python, Java, AWS, Docker, Kubernetes development"
    assert classify_section_kind(text, "Professional Summary") == "summary_objective"


# ── classify_doc_domain ──────────────────────────────────────────────────

def test_classify_domain_resume_from_content():
    text = "Skills: Python. Education: MIT. Work Experience: 5 years. Resume summary objective"
    assert classify_doc_domain(text) == "resume"


def test_classify_domain_invoice_from_content():
    text = "Invoice #123. Total: $500. Amount due: $500. Bill to: Acme Corp. Due date: Jan 2026"
    assert classify_doc_domain(text) == "invoice"


def test_classify_domain_from_filename():
    assert classify_doc_domain("some text", "John_Resume.pdf") == "resume"


def test_classify_domain_from_filename_cv():
    assert classify_doc_domain("some text", "candidate_cv.docx") == "resume"


def test_classify_domain_from_doc_type():
    assert classify_doc_domain("some text", "", "resume") == "resume"


def test_classify_domain_generic_fallback():
    result = classify_doc_domain("The quick brown fox jumps over the lazy dog")
    assert result == "generic"


# ── Integration: payload builder uses classifier ─────────────────────────

def test_payload_builder_classifies_section_kind():
    from src.embedding.pipeline.payload_normalizer import build_qdrant_payload

    raw = {
        "subscription_id": "sub1",
        "profile_id": "prof1",
        "document_id": "doc1",
        "text": "Python, Java, AWS, Docker, Kubernetes, React, Angular",
        "source": {"name": "test.pdf"},
        "section": {"id": "s1", "title": "Technical Skills"},
    }
    payload = build_qdrant_payload(raw)
    assert payload["section_kind"] == "skills_technical"


def test_payload_builder_enriches_embedding_text():
    from src.embedding.pipeline.payload_normalizer import build_qdrant_payload

    raw = {
        "subscription_id": "sub1",
        "profile_id": "prof1",
        "document_id": "doc1",
        "text": "Python, Java, AWS",
        "source": {"name": "test.pdf"},
        "section": {"id": "s1", "title": "Technical Skills"},
    }
    payload = build_qdrant_payload(raw)
    assert payload["embedding_text"].startswith("[Skills Technical]")
    assert "Python" in payload["canonical_text"]
    assert not payload["canonical_text"].startswith("[")  # canonical stays clean


def test_payload_builder_classifies_doc_domain():
    from src.embedding.pipeline.payload_normalizer import build_qdrant_payload

    raw = {
        "subscription_id": "sub1",
        "profile_id": "prof1",
        "document_id": "doc1",
        "text": "Skills: Python. Education: MIT. Experience summary resume objective",
        "source": {"name": "candidate_resume.pdf"},
    }
    payload = build_qdrant_payload(raw)
    assert payload["doc_domain"] == "resume"
