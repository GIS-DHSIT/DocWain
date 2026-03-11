#!/usr/bin/env python3
"""Test script to verify document extraction works with ResumeProfile objects."""

import sys
sys.path.insert(0, '/')

from src.screening.resume.models import ResumeProfile, ExperienceItem, EducationItem, CertificationItem
from src.rag_v3.document_extraction import extract_hr_from_complete_document

# Create a test ResumeProfile object
profile = ResumeProfile(
    name="John Doe",
    headline="Senior Software Engineer - Full Stack Developer",
    summary="Experienced full-stack developer with 8 years in Python, Java, and modern web frameworks.",
    skills=["Python", "Java", "JavaScript", "React", "SQL", "Docker", "AWS"],
    experience=[
        ExperienceItem(
            title="Senior Software Engineer",
            company="Tech Corp",
            start_date="2019-01-01",
            end_date="2026-02-07",
            location="San Francisco, CA",
            description="Led development of microservices architecture using Python and Kubernetes"
        ),
        ExperienceItem(
            title="Software Engineer",
            company="StartUp Inc",
            start_date="2017-06-01",
            end_date="2018-12-31",
            description="Full-stack development using React and Python"
        )
    ],
    education=[
        EducationItem(
            institution="Stanford University",
            degree="B.S.",
            field="Computer Science",
            start_year="2013",
            end_year="2017"
        )
    ],
    certifications=[
        CertificationItem(
            name="AWS Solutions Architect",
            issuer="Amazon",
        ),
        CertificationItem(
            name="Google Cloud Professional",
            issuer="Google",
        )
    ],
    links=[
        "john.doe@company.com",
        "https://www.linkedin.com/in/johndoe",
        "+1-202-555-0173"
    ]
)

print("=" * 60)
print("Testing Document Extraction with ResumeProfile")
print("=" * 60)

try:
    result = extract_hr_from_complete_document(profile)

    print(f"\n✅ Extraction successful!")
    print(f"\nExtracted Data:")
    print(f"  Name: {result.get('name')}")
    print(f"  Technical Skills: {result.get('technical_skills', [])}")
    print(f"  Education: {result.get('education', [])}")
    print(f"  Certifications: {result.get('certifications', [])}")
    print(f"  Experience Summary: {result.get('experience_summary')}")
    print(f"  Total Years Experience: {result.get('total_years_experience')}")
    print(f"  Email: {result.get('email', [])}")
    print(f"  LinkedIn: {result.get('linkedin', [])}")
    print(f"  Phone: {result.get('phone', [])}")
    print(f"  Source Type: {result.get('source_type')}")

    # Check for "Not explicitly mentioned" errors
    not_mentioned_count = 0
    for key, value in result.items():
        if value == "Not explicitly mentioned in documents.":
            not_mentioned_count += 1
            print(f"  ❌ {key}: Not explicitly mentioned")

    if not_mentioned_count == 0:
        print(f"\n✅ SUCCESS: All fields extracted (no 'Not mentioned' errors)")
    else:
        print(f"\n⚠️ WARNING: {not_mentioned_count} fields show 'Not mentioned'")

except Exception as e:
    print(f"\n❌ Extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

