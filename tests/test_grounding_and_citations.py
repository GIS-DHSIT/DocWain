from src.rag.citations import build_citations
from src.rag.grounding import enforce_grounding, filter_chunks_by_query_entity


def test_cross_document_contamination_blocked():
    chunks = [
        {
            "text": "Certifications include AWS and Azure.",
            "metadata": {
                "profile_name": "Nishanthan",
                "source_file": "nishanthan_resume.pdf",
                "section": "CERTIFICATIONS",
                "page": 1,
            },
            "source_id": 1,
        },
        {
            "text": "Certifications include GCP.",
            "metadata": {
                "profile_name": "Nischay",
                "source_file": "nischay_resume.pdf",
                "section": "CERTIFICATIONS",
                "page": 2,
            },
            "source_id": 2,
        },
    ]

    filtered = filter_chunks_by_query_entity("certifications of Nishanthan", chunks)
    citations = build_citations(filtered)
    assert "nishanthan_resume.pdf" in citations
    assert "nischay_resume.pdf" not in citations


def test_unsupported_claim_replaced_with_not_mentioned():
    answer = "DOCUMENT / INFORMATION\n- Certifications: AWS\nCitations: Sample"
    evidence = [{"text": "Skills include Python and SQL."}]
    grounded, _ = enforce_grounding(answer, evidence)
    assert "DOCUMENT / INFORMATION" in grounded
    assert "Not explicitly mentioned in the provided documents." in grounded
