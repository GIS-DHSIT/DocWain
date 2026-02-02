from __future__ import annotations

import re

from src.api import dw_newron
from src.api.dw_newron import (
    AnswerPlan,
    DraftPoint,
    RetrievedChunk,
    assemble_response_text,
    build_evidence_set,
    validate_answer_plan,
)
from src.api.pipeline_state import pipeline_stage
from src.api.dataHandler import train_on_document
from src.embedding.chunking.section_chunker import SectionChunker


def test_entity_scope_filters_mismatch():
    chunks = [
        RetrievedChunk(
            id="1",
            text="Resume for Nishanthan.",
            score=0.9,
            metadata={"profile_name": "Nishanthan"},
        ),
        RetrievedChunk(
            id="2",
            text="Resume for Nischay.",
            score=0.8,
            metadata={"profile_name": "Nischay"},
        ),
    ]
    filtered = dw_newron.EnterpriseRAGSystem._apply_entity_scope(chunks, "Nishanthan")
    assert len(filtered) == 1
    assert filtered[0].metadata.get("profile_name") == "Nishanthan"


def test_grounding_replaces_unsupported():
    chunks = [
        RetrievedChunk(
            id="1",
            text="Email: user@example.com",
            score=0.9,
            metadata={"chunk_id": "c1"},
        )
    ]
    evidence_set = build_evidence_set(chunks)
    plan = AnswerPlan(
        answer_type="FACT_LOOKUP",
        requested_fields=["Email"],
        required_sections=[],
        candidate_entities=[],
        must_cite=True,
        draft_points=[DraftPoint(field="Email", value="other@example.com", chunk_ids=["c1"])],
    )
    validated = validate_answer_plan(plan, evidence_set)
    assert validated[0].supported is False
    assert "Not explicitly mentioned" in validated[0].value


def test_citations_only_from_supported_points():
    chunks = [
        RetrievedChunk(
            id="1",
            text="Email: user@example.com",
            score=0.9,
            metadata={"chunk_id": "c1", "file_name": "resume.pdf", "section_title": "Contact", "page": 1},
            source="resume.pdf",
        ),
        RetrievedChunk(
            id="2",
            text="Phone: 555-1212",
            score=0.8,
            metadata={"chunk_id": "c2", "file_name": "resume.pdf", "section_title": "Contact", "page": 1},
            source="resume.pdf",
        ),
    ]
    evidence_set = build_evidence_set(chunks)
    plan = AnswerPlan(
        answer_type="FACT_LOOKUP",
        requested_fields=["Email", "Phone"],
        required_sections=[],
        candidate_entities=[],
        must_cite=True,
        draft_points=[
            DraftPoint(field="Email", value="user@example.com", chunk_ids=["c1"]),
            DraftPoint(field="Phone", value="555-9999", chunk_ids=["c2"]),
        ],
    )
    validated = validate_answer_plan(plan, evidence_set)
    text, sources = assemble_response_text(
        header="DOCUMENT / INFORMATION",
        preface_lines=["Here's what the document states."],
        plan=plan,
        validated_points=validated,
        evidence_set=evidence_set,
    )
    assert "SOURCE-1" in text
    assert "SOURCE-2" not in text
    assert len(sources) == 1


def test_chunking_no_broken_tokens():
    text = "This is an Unstruc-\ntured section with hyphenation."
    chunker = SectionChunker()
    chunks = chunker.chunk_document(text, doc_internal_id="doc", source_filename="doc.txt")
    assert chunks
    for chunk in chunks:
        stripped = chunk.text.strip()
        assert not re.search(r"[A-Za-z]-\Z", stripped)
        assert "Unstruc-" not in stripped


def test_screening_blocks_embedding():
    with pipeline_stage("SCREENING"):
        result = train_on_document("test text", "sub", "profile", "doc", "doc.txt")
    assert result.get("status") == "blocked"


def test_output_structure_preserved():
    chunks = [
        RetrievedChunk(
            id="1",
            text="Email: user@example.com",
            score=0.9,
            metadata={"chunk_id": "c1", "file_name": "resume.pdf", "section_title": "Contact", "page": 1},
            source="resume.pdf",
        ),
    ]
    evidence_set = build_evidence_set(chunks)
    plan = AnswerPlan(
        answer_type="FACT_LOOKUP",
        requested_fields=["Email"],
        required_sections=[],
        candidate_entities=[],
        must_cite=True,
        draft_points=[DraftPoint(field="Email", value="user@example.com", chunk_ids=["c1"])],
    )
    validated = validate_answer_plan(plan, evidence_set)
    text, _sources = assemble_response_text(
        header="DOCUMENT / INFORMATION",
        preface_lines=["Here's what the document states."],
        plan=plan,
        validated_points=validated,
        evidence_set=evidence_set,
    )
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0] == "DOCUMENT / INFORMATION"
    assert lines[-1].startswith("Citations:")
