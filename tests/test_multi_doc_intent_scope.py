from __future__ import annotations

from dataclasses import dataclass

from src.rag.doc_inventory import DocInventoryItem
from src.rag.intent_scope_orchestrator import decide_intent_scope, retrieve_per_doc
from src.rag.context_reasoning import KeyFact, WorkingContext
from src.rag.evidence_selector import select_evidence_lines
from src.rag.response_formatter import format_candidate_profile_response


@dataclass
class DummyChunk:
    text: str
    score: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def test_scope_detect_all_documents():
    docs = [
        DocInventoryItem(doc_id="1", source_file="ajay resume.pdf", document_name="ajay resume.pdf", doc_type="resume"),
        DocInventoryItem(doc_id="2", source_file="sara resume.pdf", document_name="sara resume.pdf", doc_type="resume"),
    ]
    decision = decide_intent_scope(
        query_text="perform the same for all the documents",
        doc_inventory=docs,
        session_id=None,
        subscription_id=None,
        profile_id=None,
    )
    assert decision.scope == "multi_doc"


def test_scope_detect_single_doc_only_when_named():
    docs = [
        DocInventoryItem(doc_id="1", source_file="ajay resume.pdf", document_name="ajay resume.pdf", doc_type="resume"),
        DocInventoryItem(doc_id="2", source_file="sara resume.pdf", document_name="sara resume.pdf", doc_type="resume"),
    ]
    decision = decide_intent_scope(
        query_text="from ajay resume.pdf extract name",
        doc_inventory=docs,
        session_id=None,
        subscription_id=None,
        profile_id=None,
    )
    assert decision.scope == "single_doc"
    assert any(doc.source_file == "ajay resume.pdf" for doc in decision.target_docs)


def test_multi_doc_does_not_collapse():
    docs = [
        DocInventoryItem(doc_id="1", source_file="a.pdf", document_name="a.pdf", doc_type="resume"),
        DocInventoryItem(doc_id="2", source_file="b.pdf", document_name="b.pdf", doc_type="resume"),
        DocInventoryItem(doc_id="3", source_file="c.pdf", document_name="c.pdf", doc_type="resume"),
    ]

    def retrieve_fn(doc: DocInventoryItem, top_k: int):
        return [DummyChunk(text=f"{doc.source_file} chunk", score=0.9, metadata={"source_file": doc.source_file})]

    all_chunks, chunks_by_doc, docs_used = retrieve_per_doc(
        doc_inventory=docs,
        retrieve_fn=retrieve_fn,
        per_doc_top_k=3,
        min_chunks_per_doc=1,
        max_docs=3,
    )

    assert len(all_chunks) >= 2
    assert len(chunks_by_doc) >= 2
    assert len(set(docs_used)) >= 2


def test_evidence_trimming():
    context = WorkingContext(
        resolved_scope=["a.pdf", "b.pdf"],
        key_facts=[
            KeyFact(label="Name", value="Ajay", doc_name="a.pdf", section="Header", chunk_id="1"),
            KeyFact(label="Skills", value="Python, SQL", doc_name="a.pdf", section="Skills", chunk_id="2"),
            KeyFact(label="Education", value="B.Sc", doc_name="a.pdf", section="Education", chunk_id="3"),
            KeyFact(label="Name", value="Sara", doc_name="b.pdf", section="Header", chunk_id="4"),
            KeyFact(label="Skills", value="Java, AWS", doc_name="b.pdf", section="Skills", chunk_id="5"),
            KeyFact(label="Education", value="M.Sc", doc_name="b.pdf", section="Education", chunk_id="6"),
        ],
        tables=[],
        numeric_claims=[],
        contradictions=[],
        missing_fields=[],
    )
    lines = select_evidence_lines(query="name skills education", context=context, max_lines_per_doc=3)
    assert len(lines) <= 6
    assert all("Name" in line or "Skills" in line or "Education" in line for line in lines)


def test_response_length_and_structure():
    profile = {
        "candidate_name": "Ajay Kumar",
        "total_years_experience": 5.0,
        "experience_summary": "Roles referenced include Software Engineer.",
        "technical_skills": ["Python", "SQL"],
        "functional_skills": ["Agile"],
        "certifications": ["AWS Certified"],
        "education": ["B.Sc Computer Science"],
        "achievements_awards": ["Employee of the Month"],
        "source_type": "resume",
        "source_document": "ajay resume.pdf",
    }
    response = format_candidate_profile_response(profile=profile)
    sentence_count = len([s for s in response.replace("\n", " ").split(".") if s.strip()])
    assert sentence_count >= 6
    assert "Takeaways:" in response
    assert "\n-" in response
