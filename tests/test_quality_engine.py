"""Tests for the output quality engine."""

from __future__ import annotations

import pytest

from src.docwain_intel.quality_engine import (
    QualityResult,
    validate_output,
    _extract_claims,
    _verify_claim,
    _strip_meta_commentary,
    _restructure_to_spec,
)
from src.docwain_intel.rendering_spec import RenderingSpec
from src.docwain_intel.evidence_organizer import (
    OrganizedEvidence,
    EvidenceGroup,
    EvidenceGap,
)
from src.docwain_intel.models import ExtractionResult, EntitySpan, FactTriple


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_evidence(
    entity_text: str = "John",
    facts: list | None = None,
    chunks: list | None = None,
    gaps: list | None = None,
) -> OrganizedEvidence:
    """Helper to build an OrganizedEvidence with one entity group."""
    grp = EvidenceGroup(
        entity_id="e1",
        entity_text=entity_text,
        entity_label="PERSON",
        facts=facts or [],
        chunks=chunks or [],
        relevance_score=0.9,
    )
    return OrganizedEvidence(
        entity_groups=[grp],
        ungrouped_chunks=[],
        gaps=gaps or [],
        total_facts=len(facts or []),
        total_chunks=len(chunks or []),
    )


def _make_extraction(
    entities: list | None = None,
    facts: list | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        document_id="doc1",
        entities=entities or [],
        facts=facts or [],
    )


# ---------------------------------------------------------------------------
# Test 1: Table spec with prose output → restructured to table
# ---------------------------------------------------------------------------

def test_table_spec_prose_restructured():
    """Prose output should be restructured to a markdown table when spec requires table."""
    spec = RenderingSpec(layout_mode="table", use_table=True)
    evidence = _make_evidence(
        facts=[{"predicate": "name", "object_value": "John"}],
        chunks=[{"text": "Name: John, Role: Engineer"}],
    )
    output = "Name: John\nRole: Engineer\nDepartment: Sales"
    result = validate_output(output, spec, evidence)

    assert "|" in result.cleaned_text
    assert "---" in result.cleaned_text
    assert "John" in result.cleaned_text
    assert result.structural_conformance > 0.0


# ---------------------------------------------------------------------------
# Test 2: Single value spec with preamble → stripped to value
# ---------------------------------------------------------------------------

def test_single_value_preamble_stripped():
    """Preamble should be stripped for single_value layout, leaving just the answer."""
    spec = RenderingSpec(layout_mode="single_value")
    evidence = _make_evidence(
        chunks=[{"text": "The salary is $95,000"}],
    )
    output = "Based on the provided documents, the answer is:\n$95,000\nThis is the annual salary."
    result = validate_output(output, spec, evidence)

    lines = [l for l in result.cleaned_text.split("\n") if l.strip()]
    assert len(lines) <= 2
    assert "$95,000" in result.cleaned_text


# ---------------------------------------------------------------------------
# Test 3: Claim verified against matching FactTriple
# ---------------------------------------------------------------------------

def test_claim_verified_with_fact_triple():
    """Claim 'John works at Google' should verify against a matching FactTriple."""
    evidence = _make_evidence(
        entity_text="John",
        facts=[{"predicate": "works_at", "object_value": "Google"}],
    )
    extraction = _make_extraction(
        entities=[
            EntitySpan(
                entity_id="e1", text="John", normalized="john",
                label="PERSON", unit_id="u1",
            ),
        ],
        facts=[
            FactTriple(
                fact_id="f1", subject_id="e1", predicate="works_at",
                object_value="Google", unit_id="u1", raw_text="John works at Google",
            ),
        ],
    )

    assert _verify_claim("John works at Google", evidence, extraction) is True


# ---------------------------------------------------------------------------
# Test 4: Claim unverified with no backing
# ---------------------------------------------------------------------------

def test_claim_unverified_no_backing():
    """Claim 'John works at Meta' should NOT verify when evidence only mentions Google."""
    evidence = _make_evidence(
        entity_text="John",
        facts=[{"predicate": "works_at", "object_value": "Google"}],
    )
    extraction = _make_extraction(
        entities=[
            EntitySpan(
                entity_id="e1", text="John", normalized="john",
                label="PERSON", unit_id="u1",
            ),
        ],
        facts=[
            FactTriple(
                fact_id="f1", subject_id="e1", predicate="works_at",
                object_value="Google", unit_id="u1", raw_text="John works at Google",
            ),
        ],
    )

    # "Meta" doesn't appear anywhere in evidence.
    assert _verify_claim("Sarah works at Meta", evidence, extraction) is False


# ---------------------------------------------------------------------------
# Test 5: "Based on the provided documents..." → stripped
# ---------------------------------------------------------------------------

def test_strip_based_on():
    text = "Based on the provided documents, the revenue is $1M."
    result = _strip_meta_commentary(text)
    assert not result.lower().startswith("based on")
    assert "$1M" in result


# ---------------------------------------------------------------------------
# Test 6: "Here are the findings:" → stripped
# ---------------------------------------------------------------------------

def test_strip_here_are():
    text = "Here are the findings:\n- Item 1\n- Item 2"
    result = _strip_meta_commentary(text)
    assert "Here are" not in result
    assert "Item 1" in result
    assert "Item 2" in result


# ---------------------------------------------------------------------------
# Test 7: Clean output matching spec → returned unchanged
# ---------------------------------------------------------------------------

def test_clean_output_unchanged():
    """Already-conformant output should not be modified."""
    spec = RenderingSpec(layout_mode="narrative")
    evidence = _make_evidence(
        entity_text="John",
        chunks=[{"text": "John is a senior engineer at Google with 10 years of experience."}],
    )
    output = "John is a senior engineer at Google with 10 years of experience."
    result = validate_output(output, spec, evidence)

    assert result.was_modified is False
    assert result.cleaned_text == output


# ---------------------------------------------------------------------------
# Test 8: Output completeness — missing fields noted in issues
# ---------------------------------------------------------------------------

def test_missing_fields_noted():
    """When spec lists fields that are missing from output, issues should note them."""
    spec = RenderingSpec(
        layout_mode="card",
        field_ordering=["name", "email", "phone"],
        use_bold_values=True,
    )
    evidence = _make_evidence(
        facts=[{"predicate": "name", "object_value": "John"}],
        chunks=[{"text": "John, email: john@example.com"}],
    )
    output = "**Name**: John\n**Email**: john@example.com"
    result = validate_output(output, spec, evidence)

    # "phone" is missing from the output.
    assert any("phone" in issue.lower() for issue in result.issues_found)


# ---------------------------------------------------------------------------
# Test 9: Empty output → handled gracefully
# ---------------------------------------------------------------------------

def test_empty_output():
    spec = RenderingSpec(layout_mode="narrative")
    evidence = _make_evidence()
    result = validate_output("", spec, evidence)

    assert result.cleaned_text == ""
    assert result.structural_conformance == 0.0
    assert result.content_integrity == 0.0
    assert "empty" in result.issues_found[0].lower()


# ---------------------------------------------------------------------------
# Test 10: Multiple LLM artifacts → all stripped
# ---------------------------------------------------------------------------

def test_multiple_artifacts_stripped():
    text = (
        "Based on my analysis of the documents:\n"
        "After reviewing the evidence, the key points are:\n"
        "- Revenue: $5M\n"
        "- Profit: $1M\n"
        "Let me summarize the findings for you."
    )
    result = _strip_meta_commentary(text)

    assert "Based on" not in result
    assert "After reviewing" not in result
    assert "Let me" not in result
    assert "$5M" in result
    assert "$1M" in result


# ---------------------------------------------------------------------------
# Test 11: Structural conformance score calculated correctly
# ---------------------------------------------------------------------------

def test_structural_conformance_score():
    """Table spec with actual table output should score 1.0 conformance."""
    spec = RenderingSpec(layout_mode="table", use_table=True, use_bold_values=True)
    evidence = _make_evidence(
        chunks=[{"text": "Name: John, Role: Engineer"}],
    )
    output = "| **Name** | **Role** |\n| --- | --- |\n| John | Engineer |"
    result = validate_output(output, spec, evidence)

    assert result.structural_conformance == 1.0


# ---------------------------------------------------------------------------
# Test 12: Content integrity score calculated correctly
# ---------------------------------------------------------------------------

def test_content_integrity_score():
    """Integrity score should reflect ratio of verified to total claims."""
    spec = RenderingSpec(layout_mode="narrative")
    evidence = _make_evidence(
        entity_text="John",
        facts=[{"predicate": "role", "object_value": "Engineer"}],
        chunks=[{"text": "John is an Engineer at Google."}],
    )
    extraction = _make_extraction(
        entities=[
            EntitySpan(
                entity_id="e1", text="John", normalized="john",
                label="PERSON", unit_id="u1",
            ),
            EntitySpan(
                entity_id="e2", text="Google", normalized="google",
                label="ORG", unit_id="u1",
            ),
        ],
        facts=[
            FactTriple(
                fact_id="f1", subject_id="e1", predicate="role",
                object_value="Engineer", unit_id="u1",
                raw_text="John is an Engineer",
            ),
        ],
    )

    # First claim grounded, second not.
    output = "John is an Engineer at Google.\nSarah works at Meta with 5 years experience."
    result = validate_output(output, spec, evidence, extraction)

    assert result.claims_verified >= 1
    assert result.claims_unverified >= 1
    assert 0.0 < result.content_integrity < 1.0


# ---------------------------------------------------------------------------
# Test 13: Card spec with plain text → converted to bold labels
# ---------------------------------------------------------------------------

def test_card_spec_conversion():
    """Plain key-value text should be converted to bold-label card format."""
    spec = RenderingSpec(layout_mode="card", use_bold_values=True)
    evidence = _make_evidence(
        chunks=[{"text": "Name: Alice, Age: 30"}],
    )
    output = "Name: Alice\nAge: 30"
    result = validate_output(output, spec, evidence)

    assert "**Name**" in result.cleaned_text
    assert "**Age**" in result.cleaned_text


# ---------------------------------------------------------------------------
# Test 14: List spec with prose → converted to bullets
# ---------------------------------------------------------------------------

def test_list_spec_conversion():
    """Prose output should be converted to bullet list for list layout."""
    spec = RenderingSpec(layout_mode="list")
    evidence = _make_evidence(
        chunks=[{"text": "Python is popular. Java is enterprise."}],
    )
    output = "Python is a popular language. Java is used in enterprise. Go is fast."
    result = validate_output(output, spec, evidence)

    assert result.cleaned_text.count("- ") >= 2


# ---------------------------------------------------------------------------
# Test 15: Whitespace-only output → treated as empty
# ---------------------------------------------------------------------------

def test_whitespace_only_output():
    spec = RenderingSpec(layout_mode="narrative")
    evidence = _make_evidence()
    result = validate_output("   \n  \n  ", spec, evidence)

    assert result.cleaned_text == ""
    assert result.structural_conformance == 0.0
    assert "empty" in result.issues_found[0].lower()


# ---------------------------------------------------------------------------
# Test 16: extract_claims returns correct claims
# ---------------------------------------------------------------------------

def test_extract_claims_filters():
    """Only sentences with entities or numbers should be extracted as claims."""
    text = "the sky is blue.\nJohn works at Google.\nRevenue is $5M."
    claims = _extract_claims(text)

    # "the sky is blue" has no proper nouns or numbers — excluded.
    assert any("John" in c for c in claims)
    assert any("$5M" in c for c in claims)
