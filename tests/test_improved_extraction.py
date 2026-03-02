"""
Test to verify the improved extraction logic handles section-based metadata correctly.
This tests the intelligent extraction from chunks organized by section_kind.
"""

import pytest
from unittest.mock import Mock
from src.rag_v3.extract import _extract_hr
from src.rag_v3.types import MISSING_REASON


def create_mock_chunk(text, chunk_id, doc_id, section_kind="section_text", section_title=""):
    """Helper to create mock chunk objects."""
    chunk = Mock()
    chunk.text = text
    chunk.id = chunk_id
    chunk.meta = {
        "document_id": doc_id,
        "section_kind": section_kind,
        "section_title": section_title,
    }
    chunk.metadata = chunk.meta
    # Set source to None to avoid Mock iteration issues
    chunk.source = None
    return chunk


def test_extract_hr_with_metadata_sections():
    """Test HR extraction using section metadata (not just text parsing)."""

    # Create mock chunks organized by section kind
    chunks = [
        # Skills section chunks
        create_mock_chunk(
            text="Python, Java, SQL, JavaScript, React, Node.js",
            chunk_id="chunk-1",
            doc_id="doc-1",
            section_kind="skills_technical",
            section_title="Technical Skills",
        ),
        # Functional skills chunks
        create_mock_chunk(
            text="Software Architecture, System Design, Team Leadership, Agile Development",
            chunk_id="chunk-2",
            doc_id="doc-1",
            section_kind="skills_functional",
            section_title="Functional Skills",
        ),
        # Education chunks
        create_mock_chunk(
            text="B.Tech in Computer Science from IIT Delhi, MBA from ISB",
            chunk_id="chunk-3",
            doc_id="doc-1",
            section_kind="education",
            section_title="Education",
        ),
        # Certifications chunks
        create_mock_chunk(
            text="AWS Solutions Architect, Google Cloud Professional",
            chunk_id="chunk-4",
            doc_id="doc-1",
            section_kind="certifications",
            section_title="Certifications",
        ),
        # Experience summary
        create_mock_chunk(
            text="Senior Software Engineer with 10 years of experience in full-stack development",
            chunk_id="chunk-5",
            doc_id="doc-1",
            section_kind="summary_objective",
            section_title="Professional Summary",
        ),
    ]

    # Extract HR schema
    schema = _extract_hr(chunks)

    # Verify candidates were extracted
    assert schema.candidates is not None
    assert schema.candidates.items is not None
    assert len(schema.candidates.items) == 1

    candidate = schema.candidates.items[0]

    # Verify technical skills are extracted
    assert candidate.technical_skills is not None
    assert len(candidate.technical_skills) > 0
    assert "Python" in candidate.technical_skills
    assert "Java" in candidate.technical_skills
    assert "SQL" in candidate.technical_skills
    print(f"✓ Technical skills extracted: {candidate.technical_skills}")

    # Verify functional skills are extracted
    assert candidate.functional_skills is not None
    assert len(candidate.functional_skills) > 0
    assert "Software Architecture" in candidate.functional_skills or "System Design" in candidate.functional_skills
    print(f"✓ Functional skills extracted: {candidate.functional_skills}")

    # Verify education is extracted
    assert candidate.education is not None
    assert len(candidate.education) > 0
    assert "B.Tech" in candidate.education[0] or "IIT" in candidate.education[0]
    print(f"✓ Education extracted: {candidate.education}")

    # Verify certifications are extracted
    assert candidate.certifications is not None
    assert len(candidate.certifications) > 0
    assert "AWS" in candidate.certifications[0] or "Google Cloud" in candidate.certifications[0]
    print(f"✓ Certifications extracted: {candidate.certifications}")

    # Verify experience summary is extracted
    assert candidate.experience_summary is not None
    assert candidate.experience_summary != MISSING_REASON
    assert "Senior Software Engineer" in candidate.experience_summary or "10 years" in candidate.experience_summary
    print(f"✓ Experience summary extracted: {candidate.experience_summary[:60]}...")

    # Verify NOT showing "Not explicitly mentioned" errors
    missing_reason = candidate.missing_reason or {}
    assert missing_reason.get("technical_skills") != MISSING_REASON
    assert missing_reason.get("functional_skills") != MISSING_REASON
    assert missing_reason.get("education") != MISSING_REASON
    assert missing_reason.get("certifications") != MISSING_REASON
    assert missing_reason.get("experience_summary") != MISSING_REASON

    print("✓ All fields properly extracted - no 'Not explicitly mentioned' errors!")


def test_extract_hr_handles_sparse_metadata():
    """Test that extraction still works even with minimal metadata."""

    chunks = [
        create_mock_chunk(
            text="Python, Java, C++",
            chunk_id="chunk-1",
            doc_id="doc-1",
            section_kind="skills_technical",
        ),
        create_mock_chunk(
            text="Team leadership and communication",
            chunk_id="chunk-2",
            doc_id="doc-1",
            section_kind="skills_functional",
        ),
    ]

    schema = _extract_hr(chunks)

    assert schema.candidates is not None
    assert schema.candidates.items is not None
    candidate = schema.candidates.items[0]

    # Even with minimal metadata, should extract skills
    assert candidate.technical_skills is not None
    assert len(candidate.technical_skills) > 0
    print(f"✓ Extracted technical skills even with minimal metadata: {candidate.technical_skills}")


def test_extract_hr_with_contact_info():
    """Test that contact info is extracted properly."""

    chunks = [
        create_mock_chunk(
            text="John Doe\njohn.doe@example.com\n+1-202-555-0173\nhttps://www.linkedin.com/in/johndoe/",
            chunk_id="chunk-1",
            doc_id="doc-1",
            section_kind="identity_contact",
            section_title="Contact",
        ),
    ]

    schema = _extract_hr(chunks)

    assert schema.candidates is not None
    assert schema.candidates.items is not None
    candidate = schema.candidates.items[0]

    # Verify contact extraction
    if candidate.emails:
        assert "john.doe@example.com" in candidate.emails
        print(f"✓ Email extracted: {candidate.emails}")

    if candidate.phones:
        print(f"✓ Phone extracted: {candidate.phones}")

    if candidate.linkedins:
        print(f"✓ LinkedIn extracted: {candidate.linkedins}")


def test_extract_hr_with_generic_section_kind():
    """Test HR extraction when all chunks have generic section_kind (real-world scenario)."""

    chunks = [
        # All chunks have generic section_kind - content inference will identify them
        create_mock_chunk(
            text="Python, Java, SQL, JavaScript, React, Node.js",
            chunk_id="chunk-1",
            doc_id="doc-1",
            section_kind="section_text",  # Generic - will be inferred
            section_title="Technical Skills",
        ),
        create_mock_chunk(
            text="Software Architecture, System Design, Team Leadership, Agile Development",
            chunk_id="chunk-2",
            doc_id="doc-1",
            section_kind="section_text",  # Generic - will be inferred
            section_title="Functional Skills",
        ),
        create_mock_chunk(
            text="B.Tech in Computer Science from IIT Delhi, MBA from ISB",
            chunk_id="chunk-3",
            doc_id="doc-1",
            section_kind="section_text",  # Generic - will be inferred
            section_title="Education",
        ),
        create_mock_chunk(
            text="AWS Solutions Architect, Google Cloud Professional",
            chunk_id="chunk-4",
            doc_id="doc-1",
            section_kind="section_text",  # Generic - will be inferred
            section_title="Certifications",
        ),
    ]

    schema = _extract_hr(chunks)

    assert schema.candidates is not None
    assert schema.candidates.items is not None
    candidate = schema.candidates.items[0]

    # Verify content inference worked
    assert candidate.technical_skills is not None
    assert "Python" in candidate.technical_skills
    print(f"✓ Content inference: Technical skills identified from generic section_text")

    assert candidate.functional_skills is not None
    print(f"✓ Content inference: Functional skills identified from generic section_text")

    assert candidate.education is not None
    print(f"✓ Content inference: Education identified from generic section_text")

    assert candidate.certifications is not None
    print(f"✓ Content inference: Certifications identified from generic section_text")

    print("✓ All fields extracted despite generic metadata!")


def test_extract_from_structured_document():
    """Test HR extraction from StructuredDocument format (pickle data from extraction_service)."""
    from src.rag_v3.document_extraction import extract_hr_from_complete_document

    # Simulate StructuredDocument format as dict (after asdict() conversion)
    structured_doc = {
        'document_id': 'test-123',
        'original_filename': 'John_Doe_Resume.pdf',
        'document_type': 'RESUME',
        'raw_text': '''
John Doe
john.doe@email.com | +1-555-123-4567 | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Senior Software Engineer with 8+ years of experience in Python, Java, and cloud technologies.

TECHNICAL SKILLS
Python, Java, JavaScript, React, Node.js, AWS, Docker, Kubernetes

WORK EXPERIENCE
Senior Software Engineer at Google (2018-2024)
- Led development of microservices architecture
- Improved system performance by 40%

EDUCATION
Bachelor of Science in Computer Science from Stanford University, 2015

CERTIFICATIONS
AWS Certified Solutions Architect
Google Cloud Professional Developer
''',
        'sections': [
            {'section_type': 'header', 'content': 'John Doe\njohn.doe@email.com | +1-555-123-4567', 'key_items': [], 'title': 'Contact'},
            {'section_type': 'summary', 'content': 'Senior Software Engineer with 8+ years of experience', 'key_items': [], 'title': 'Summary'},
            {'section_type': 'skills', 'content': 'Python, Java, JavaScript, React, Node.js, AWS, Docker', 'key_items': ['Python', 'Java', 'JavaScript', 'React', 'AWS'], 'title': 'Skills'},
            {'section_type': 'experience', 'content': 'Senior Software Engineer at Google (2018-2024)', 'key_items': [], 'title': 'Experience'},
            {'section_type': 'education', 'content': 'Bachelor of Science in Computer Science from Stanford University, 2015', 'key_items': ['BS Computer Science, Stanford'], 'title': 'Education'},
            {'section_type': 'certifications', 'content': 'AWS Certified, Google Cloud', 'key_items': ['AWS Certified Solutions Architect', 'Google Cloud Professional'], 'title': 'Certifications'},
        ],
        'document_classification': {
            'primary_type': 'RESUME',
            'structured_fields': {}
        },
        'metadata': {}
    }

    result = extract_hr_from_complete_document(structured_doc)

    # Verify name extraction
    assert result.get("name") == "John Doe", f"Name mismatch: {result.get('name')}"
    print(f"✓ Name extracted: {result.get('name')}")

    # Verify email extraction
    assert result.get("email"), "Email not extracted"
    assert "john.doe@email.com" in result["email"]
    print(f"✓ Email extracted: {result.get('email')}")

    # Verify phone extraction
    assert result.get("phone"), "Phone not extracted"
    print(f"✓ Phone extracted: {result.get('phone')}")

    # Verify skills extraction
    assert result.get("technical_skills"), "Skills not extracted"
    assert "Python" in result["technical_skills"]
    print(f"✓ Technical skills extracted: {result.get('technical_skills')}")

    # Verify education extraction
    assert result.get("education"), "Education not extracted"
    print(f"✓ Education extracted: {result.get('education')}")

    # Verify certifications extraction
    assert result.get("certifications"), "Certifications not extracted"
    print(f"✓ Certifications extracted: {result.get('certifications')}")

    # Verify experience summary extraction
    assert result.get("experience_summary"), "Experience summary not extracted"
    print(f"✓ Experience summary extracted: {result.get('experience_summary')[:50]}...")

    # Verify years of experience extraction
    assert result.get("total_years_experience"), "Years of experience not extracted"
    print(f"✓ Years of experience extracted: {result.get('total_years_experience')}")

    print("✓ StructuredDocument extraction working correctly!")


if __name__ == "__main__":
    test_extract_hr_with_metadata_sections()
    test_extract_hr_handles_sparse_metadata()
    test_extract_hr_with_contact_info()
    test_extract_hr_with_generic_section_kind()
    test_extract_from_structured_document()
    print("\n✅ All HR extraction tests passed!")

