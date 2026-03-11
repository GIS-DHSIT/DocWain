"""Tests for constrained_prompter — prompt construction from rendering spec + evidence."""

import pytest

from src.docwain_intel.constrained_prompter import (
    ConstrainedPrompt,
    RenderingSpec,
    build_prompt,
    _format_evidence_section,
    MAX_PROMPT_CHARS,
)
from src.docwain_intel.evidence_organizer import (
    EvidenceGap,
    EvidenceGroup,
    OrganizedEvidence,
    ProvenanceRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(
    *,
    entity_groups: list | None = None,
    ungrouped: list | None = None,
    gaps: list | None = None,
    provenance: list | None = None,
) -> OrganizedEvidence:
    return OrganizedEvidence(
        entity_groups=entity_groups or [],
        ungrouped_chunks=ungrouped or [],
        gaps=gaps or [],
        provenance=provenance or [],
        total_facts=0,
        total_chunks=0,
    )


def _simple_evidence() -> OrganizedEvidence:
    """Evidence with one entity group and one ungrouped chunk."""
    return _make_evidence(
        entity_groups=[
            EvidenceGroup(
                entity_id="e1",
                entity_text="Alice",
                facts=[{"predicate": "role", "value": "Engineer"}],
                chunks=[{"text": "Alice is a software engineer.", "source_document": "resume.pdf"}],
            )
        ],
        ungrouped=[
            {"text": "Company was founded in 2010.", "source_document": "about.pdf"}
        ],
    )


# ---------------------------------------------------------------------------
# 1. Table spec -> prompt contains column instructions from field_ordering
# ---------------------------------------------------------------------------

def test_table_spec_columns():
    spec = RenderingSpec(layout_mode="table", field_ordering=["Name", "Role", "Experience"])
    prompt = build_prompt(spec, _simple_evidence(), "List all candidates")

    assert "table" in prompt.user_prompt.lower()
    assert "Name | Role | Experience" in prompt.user_prompt


# ---------------------------------------------------------------------------
# 2. Card spec -> prompt contains field labels
# ---------------------------------------------------------------------------

def test_card_spec_field_labels():
    spec = RenderingSpec(layout_mode="card", field_ordering=["Name", "Email", "Phone"])
    prompt = build_prompt(spec, _simple_evidence(), "Show contact info")

    assert "labeled fields" in prompt.user_prompt.lower()
    assert "Name, Email, Phone" in prompt.user_prompt


# ---------------------------------------------------------------------------
# 3. Single value spec -> prompt says "one line only"
# ---------------------------------------------------------------------------

def test_single_value_one_line():
    spec = RenderingSpec(layout_mode="single_value")
    prompt = build_prompt(spec, _simple_evidence(), "What is Alice's role?")

    assert "one line only" in prompt.user_prompt.lower()


# ---------------------------------------------------------------------------
# 4. Narrative spec -> prompt mentions paragraphs + detail level
# ---------------------------------------------------------------------------

def test_narrative_detail_level():
    spec = RenderingSpec(layout_mode="narrative", detail_level="comprehensive")
    prompt = build_prompt(spec, _simple_evidence(), "Summarize Alice's background")

    assert "paragraphs" in prompt.user_prompt.lower()
    assert "comprehensive" in prompt.user_prompt.lower()


# ---------------------------------------------------------------------------
# 5. Gap markers injected when gaps present
# ---------------------------------------------------------------------------

def test_gaps_injected():
    evidence = _make_evidence(
        gaps=[
            EvidenceGap(field_name="salary", description="No salary information found"),
            EvidenceGap(field_name="education", description="No education records found"),
        ]
    )
    spec = RenderingSpec(include_gaps=True)
    prompt = build_prompt(spec, evidence, "Tell me about the candidate")

    assert "No evidence found for" in prompt.user_prompt
    assert "salary" in prompt.user_prompt
    assert "education" in prompt.user_prompt


# ---------------------------------------------------------------------------
# 6. No gaps -> no gap section
# ---------------------------------------------------------------------------

def test_no_gaps_no_section():
    evidence = _make_evidence(gaps=[])
    spec = RenderingSpec(include_gaps=True)
    prompt = build_prompt(spec, evidence, "question")

    assert "No evidence found for" not in prompt.user_prompt


# ---------------------------------------------------------------------------
# 7. Evidence grouped by entity in prompt
# ---------------------------------------------------------------------------

def test_evidence_grouped_by_entity():
    evidence = _make_evidence(
        entity_groups=[
            EvidenceGroup(entity_text="Alice", facts=[{"predicate": "role", "value": "Dev"}], chunks=[]),
            EvidenceGroup(entity_text="Bob", facts=[{"predicate": "role", "value": "PM"}], chunks=[]),
        ],
        ungrouped=[{"text": "Extra info", "source_document": "misc.pdf"}],
    )
    spec = RenderingSpec()
    prompt = build_prompt(spec, evidence, "Compare")

    assert "## Alice" in prompt.user_prompt
    assert "## Bob" in prompt.user_prompt
    assert "## Additional Evidence" in prompt.user_prompt


# ---------------------------------------------------------------------------
# 8. Provenance instruction included when spec.include_provenance=True
# ---------------------------------------------------------------------------

def test_provenance_instruction():
    spec = RenderingSpec(include_provenance=True)
    prompt = build_prompt(spec, _simple_evidence(), "question")

    assert "source document" in prompt.user_prompt.lower()
    assert "page number" in prompt.user_prompt.lower()


def test_no_provenance_instruction():
    spec = RenderingSpec(include_provenance=False)
    prompt = build_prompt(spec, _simple_evidence(), "question")

    assert "PROVENANCE" not in prompt.user_prompt


# ---------------------------------------------------------------------------
# 9. System prompt always contains anti-hallucination rules
# ---------------------------------------------------------------------------

def test_system_prompt_anti_hallucination():
    spec = RenderingSpec()
    prompt = build_prompt(spec, _simple_evidence(), "anything")

    assert "ONLY using the provided evidence" in prompt.system_prompt
    assert "Do NOT invent" in prompt.system_prompt
    assert "Do NOT add preambles" in prompt.system_prompt
    assert "missing" in prompt.system_prompt.lower()


# ---------------------------------------------------------------------------
# 10. max_tokens varies by detail_level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detail_level,expected_tokens", [
    ("minimal", 256),
    ("concise", 512),
    ("standard", 1024),
    ("comprehensive", 2048),
])
def test_max_tokens_by_detail(detail_level, expected_tokens):
    spec = RenderingSpec(detail_level=detail_level)
    prompt = build_prompt(spec, _simple_evidence(), "q")
    assert prompt.max_tokens == expected_tokens


# ---------------------------------------------------------------------------
# 11. Temperature varies by layout_mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout_mode,expected_temp", [
    ("single_value", 0.1),
    ("card", 0.1),
    ("table", 0.1),
    ("comparison", 0.1),
    ("narrative", 0.3),
    ("summary", 0.3),
    ("list", 0.2),
    ("timeline", 0.2),
])
def test_temperature_by_layout(layout_mode, expected_temp):
    spec = RenderingSpec(layout_mode=layout_mode)
    prompt = build_prompt(spec, _simple_evidence(), "q")
    assert prompt.temperature == pytest.approx(expected_temp)


# ---------------------------------------------------------------------------
# 12. Prompt truncation when evidence exceeds budget
# ---------------------------------------------------------------------------

def test_prompt_truncation_on_budget():
    # Create evidence with many chunks to exceed budget
    chunks = [
        {"text": f"Detail chunk number {i} with content about topic {i}. " * 20, "source_document": f"doc_{i}.pdf"}
        for i in range(100)
    ]
    evidence = _make_evidence(
        entity_groups=[
            EvidenceGroup(
                entity_text="BigEntity",
                chunks=chunks,
                facts=[],
            )
        ]
    )
    spec = RenderingSpec()
    prompt = build_prompt(spec, evidence, "question", max_prompt_chars=2000)

    total_len = len(prompt.system_prompt) + len(prompt.user_prompt)
    # Should be within budget (with some margin for framing text)
    assert total_len <= 3000
    assert "truncated" in prompt.user_prompt.lower()


# ---------------------------------------------------------------------------
# 13. List layout with max_items
# ---------------------------------------------------------------------------

def test_list_layout_max_items():
    spec = RenderingSpec(layout_mode="list", max_items=5)
    prompt = build_prompt(spec, _simple_evidence(), "list things")

    assert "bulleted list" in prompt.user_prompt.lower()
    assert "at most 5" in prompt.user_prompt


# ---------------------------------------------------------------------------
# 14. Comparison layout with field ordering
# ---------------------------------------------------------------------------

def test_comparison_with_fields():
    spec = RenderingSpec(layout_mode="comparison", field_ordering=["Name", "Skills", "Experience"])
    prompt = build_prompt(spec, _simple_evidence(), "Compare candidates")

    assert "comparison" in prompt.user_prompt.lower()
    assert "Name | Skills | Experience" in prompt.user_prompt


# ---------------------------------------------------------------------------
# 15. _format_evidence_section truncates long chunks
# ---------------------------------------------------------------------------

def test_chunk_text_truncation():
    long_text = "x" * 1000
    evidence = _make_evidence(
        entity_groups=[
            EvidenceGroup(
                entity_text="Test",
                chunks=[{"text": long_text, "source_document": "doc.pdf"}],
                facts=[],
            )
        ]
    )
    formatted = _format_evidence_section(evidence)

    # Each chunk excerpt capped at 500 chars + "..."
    # The line also has source prefix, so total is longer, but raw text portion <= 503
    assert "..." in formatted
    assert len(long_text) > 500  # sanity
    # The formatted text should be significantly shorter than the raw input
    assert len(formatted) < len(long_text)
