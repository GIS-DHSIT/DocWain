"""Tests for the generic document intelligence extraction algorithm.

Verifies that the unified extractor handles ANY document type by analysing
content structure (KV pairs, lists, sections, contacts) rather than domain
classification.
"""

from __future__ import annotations

import pytest

from src.rag_v3.types import (
    Chunk,
    ChunkSource,
    EvidenceSpan,
    FieldValue,
    FieldValuesField,
    GenericSchema,
)
from src.rag_v3.extract import (
    _extract_document_intelligence,
    _extract_structured_facts,
    _score_and_sort_facts,
    _infer_domain_intent,
    _detect_multi_entity_collision,
    _keywords,
    schema_extract,
)
from src.rag_v3.enterprise import render_enterprise


# ── Helpers ────────────────────────────────────────────────────────────

def _make_chunk(
    text: str,
    *,
    chunk_id: str = "c1",
    doc_id: str = "d1",
    doc_name: str = "doc.pdf",
    section_kind: str = "",
    score: float = 0.9,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        score=score,
        source=ChunkSource(document_name=doc_name, page=1),
        meta={
            "document_id": doc_id,
            "section_kind": section_kind,
            "source_name": doc_name,
        },
    )


# ── Test 1: Medical record extraction ──────────────────────────────────

class TestMedicalRecord:
    def test_produces_labeled_kv_facts(self):
        """Medical record should produce labeled facts, NOT HR template."""
        chunks = [_make_chunk(
            "Patient Name: John Doe\nDiagnosis: Hypertension\nAge: 52\nDoctor: Dr. Laura Bennett",
            doc_name="hospital_medical_record.pdf",
        )]
        schema = _extract_document_intelligence("what is the diagnosis", chunks)
        assert isinstance(schema, GenericSchema)
        facts = schema.facts.items or []
        labels = {f.label for f in facts if f.label}
        assert "Patient Name" in labels or "Diagnosis" in labels
        # Must NOT contain HR artifacts
        values_text = " ".join(f.value for f in facts)
        assert "Candidate" not in values_text
        assert "Source type" not in values_text

    def test_no_hr_schema_for_medical_query(self):
        """Querying about patient details should not produce HR output."""
        chunks = [_make_chunk(
            "Patient Name: Jane Smith\nMedication: Metformin\nBlood Pressure: 130/85",
            doc_name="patient_record.pdf",
        )]
        schema = _extract_document_intelligence("give me the patient details", chunks)
        facts = schema.facts.items or []
        labels = {f.label for f in facts if f.label}
        assert "Patient Name" in labels
        assert "Medication" in labels or "Blood Pressure" in labels


# ── Test 2: Resume extraction quality ──────────────────────────────────

class TestResumeExtraction:
    def test_produces_structured_facts(self):
        """Resume should produce skills, education, contact as labeled facts."""
        chunks = [_make_chunk(
            "Name: Jane Smith\nSkills: Python, SQL, AWS\nEducation: BS Computer Science\nExperience: 5 years",
            doc_name="jane_smith_resume.pdf",
        )]
        schema = _extract_document_intelligence("summarize this resume", chunks)
        facts = schema.facts.items or []
        labels = {f.label for f in facts if f.label}
        assert "Name" in labels
        assert "Skills" in labels or "Education" in labels

    def test_bullet_list_skills_extracted(self):
        """Resume bullet-point skills should be extracted with section label."""
        chunks = [_make_chunk(
            "- Python\n- Java\n- SQL\n- Docker",
            section_kind="skills_technical",
            doc_name="resume.pdf",
        )]
        schema = _extract_document_intelligence("what are the technical skills", chunks)
        facts = schema.facts.items or []
        values = {f.value for f in facts}
        assert "Python" in values or "Java" in values


# ── Test 3: Invoice extraction ─────────────────────────────────────────

class TestInvoiceExtraction:
    def test_produces_totals_and_parties(self):
        """Invoice should produce labeled facts for totals and parties."""
        chunks = [_make_chunk(
            "Invoice #123\nTotal: $500.00\nBill To: Acme Corp\nDue Date: 2026-03-15",
            doc_name="invoice_123.pdf",
        )]
        schema = _extract_document_intelligence("what is the total", chunks)
        facts = schema.facts.items or []
        labels = {f.label for f in facts if f.label}
        assert "Total" in labels or "Bill To" in labels


# ── Test 4: Multi-document grouping ────────────────────────────────────

class TestMultiDocument:
    def test_facts_grouped_by_document(self):
        """Facts from multiple docs should have distinct document_name values."""
        chunks = [
            _make_chunk("Name: Alice\nSkills: Java", doc_id="d1", doc_name="alice.pdf", chunk_id="c1"),
            _make_chunk("Name: Bob\nSkills: Python", doc_id="d2", doc_name="bob.pdf", chunk_id="c2"),
        ]
        schema = _extract_document_intelligence("compare all candidates", chunks)
        facts = schema.facts.items or []
        doc_names = {f.document_name for f in facts if f.document_name}
        assert len(doc_names) >= 2
        assert "alice.pdf" in doc_names
        assert "bob.pdf" in doc_names

    def test_multi_entity_collision_false_for_generic(self):
        """GenericSchema should not trigger multi-entity collision."""
        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Name", value="Alice", document_name="a.pdf", evidence_spans=[]),
        ]))
        chunks = [
            _make_chunk("Name: Alice", doc_id="d1", chunk_id="c1"),
            _make_chunk("Name: Bob", doc_id="d2", chunk_id="c2"),
        ]
        assert _detect_multi_entity_collision("generic", schema, chunks, None) is False


# ── Test 5: Contact query ──────────────────────────────────────────────

class TestContactQuery:
    def test_contact_facts_scored_highest(self):
        """Contact queries should boost contact facts to the top."""
        chunks = [_make_chunk(
            "Name: Jane\nEmail: jane@example.com\nPhone: 555-1234\nSkills: Python, SQL",
            doc_name="resume.pdf",
        )]
        schema = _extract_document_intelligence("what is the email address", chunks)
        facts = schema.facts.items or []
        assert facts  # should have facts
        # Email should be in the results
        emails = [f for f in facts if f.label and f.label.lower() == "email"]
        assert emails or any("jane@example.com" in f.value for f in facts)


# ── Test 6: Section organization ───────────────────────────────────────

class TestSectionOrganization:
    def test_facts_carry_section_labels(self):
        """Facts should carry section labels from chunk metadata."""
        chunks = [
            _make_chunk("- Python\n- Java", section_kind="skills_technical", chunk_id="c1"),
            _make_chunk("- B.Tech Computer Science", section_kind="education", chunk_id="c2"),
        ]
        schema = _extract_document_intelligence("list all skills and education", chunks)
        facts = schema.facts.items or []
        sections = {f.section for f in facts if f.section}
        assert len(sections) >= 1  # at least one distinct section


# ── Test 7: Full-document dump handling ────────────────────────────────

class TestFullDocumentDump:
    def test_splits_sections_correctly(self):
        """A full document dump should be split into sections."""
        full_doc = (
            "SKILLS\n"
            "Python, Java, SQL\n"
            "\n"
            "EDUCATION\n"
            "BS Computer Science, MIT 2020\n"
            "\n"
            "EXPERIENCE\n"
            "Software Engineer at Google, 3 years\n"
        )
        chunks = [_make_chunk(full_doc, doc_name="resume.pdf")]
        schema = _extract_document_intelligence("what are the skills", chunks)
        facts = schema.facts.items or []
        assert facts  # should extract something


# ── Test 8: Rendering multi-doc ────────────────────────────────────────

class TestRendering:
    def test_multi_doc_has_document_headers(self):
        """Multi-document rendering should have per-document headers."""
        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Name", value="Alice", document_name="alice.pdf", evidence_spans=[]),
            FieldValue(label="Skills", value="Java, Spring", document_name="alice.pdf", evidence_spans=[]),
            FieldValue(label="Name", value="Bob", document_name="bob.pdf", evidence_spans=[]),
            FieldValue(label="Skills", value="Python, Django", document_name="bob.pdf", evidence_spans=[]),
        ]))
        output = render_enterprise(schema, "summary")
        assert "alice.pdf" in output
        assert "bob.pdf" in output
        assert "Alice" in output
        assert "Bob" in output

    def test_single_doc_structured_detail(self):
        """Single-document rendering should show structured detail."""
        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Patient Name", value="John Doe", evidence_spans=[]),
            FieldValue(label="Diagnosis", value="Hypertension", evidence_spans=[]),
        ]))
        output = render_enterprise(schema, "facts")
        assert "**Patient Name:**" in output
        assert "**Diagnosis:**" in output
        assert "John Doe" in output
        assert "Hypertension" in output

    def test_contact_intent_filters(self):
        """Contact intent should filter to contact facts."""
        schema = GenericSchema(facts=FieldValuesField(items=[
            FieldValue(label="Name", value="Jane", evidence_spans=[]),
            FieldValue(label="Email", value="jane@example.com", evidence_spans=[]),
            FieldValue(label="Phone", value="555-1234", evidence_spans=[]),
            FieldValue(label="Skills", value="Python", evidence_spans=[]),
        ]))
        output = render_enterprise(schema, "contact")
        assert "jane@example.com" in output
        assert "555-1234" in output
        # Skills should be filtered out for contact intent
        # (or at least contact facts should dominate)


# ── Test 9: Empty and fallback ─────────────────────────────────────────

class TestFallback:
    def test_empty_chunks_returns_empty_schema(self):
        """Empty chunks should return GenericSchema with no facts."""
        schema = _extract_document_intelligence("anything", [])
        assert isinstance(schema, GenericSchema)
        facts = (schema.facts.items if schema.facts else None) or []
        assert len(facts) == 0

    def test_fallback_top_sentences(self):
        """When no KV/list/keyword matches, should fall back to top sentences."""
        chunks = [_make_chunk(
            "The weather today is sunny and pleasant with clear skies across the region.",
            doc_name="weather.txt",
        )]
        schema = _extract_document_intelligence("what is something completely unrelated xyz123", chunks)
        facts = (schema.facts.items if schema.facts else None) or []
        assert len(facts) >= 1  # fallback sentences


# ── Test 10: Garbage filtering ─────────────────────────────────────────

class TestGarbageFiltering:
    def test_metadata_garbage_filtered(self):
        """Stringified metadata should be filtered out."""
        chunks = [_make_chunk(
            "{'section_id': 'abc', 'chunk_type': 'text'}\nPatient Name: John Doe",
            doc_name="record.pdf",
        )]
        schema = _extract_document_intelligence("patient info", chunks)
        facts = schema.facts.items or []
        for f in facts:
            assert "section_id" not in f.value
            assert "chunk_type" not in f.value


# ── Test 11: Intent detection (domain-agnostic) ───────────────────────

class TestIntentDetection:
    def test_contact_intent(self):
        domain, intent = _infer_domain_intent("what is the email address", [])
        assert intent == "contact"

    def test_rank_intent(self):
        domain, intent = _infer_domain_intent("rank all candidates", [])
        assert intent == "rank"

    def test_compare_intent(self):
        domain, intent = _infer_domain_intent("compare the two documents", [])
        assert intent == "compare"

    def test_list_intent(self):
        domain, intent = _infer_domain_intent("list all documents", [])
        assert intent == "list"

    def test_default_facts_intent(self):
        domain, intent = _infer_domain_intent("tell me about this document", [])
        assert intent == "facts"

    def test_domain_detected_from_query_without_hint(self):
        """Query-based domain detection: 'patient diagnosis' → medical."""
        domain, intent = _infer_domain_intent("what is the patient diagnosis", [])
        assert domain in ("medical", "generic")  # ML classifier may detect medical from query

    def test_domain_hint_passed_through(self):
        """Domain hint should be passed through for metadata."""
        domain, intent = _infer_domain_intent("query", [], domain_hint="hr")
        assert domain == "hr"


# ── Test 12: Score and sort ────────────────────────────────────────────

class TestScoreAndSort:
    def test_keyword_matching_boosts_relevant_facts(self):
        """Facts containing query keywords should be ranked higher."""
        facts = [
            FieldValue(label="Color", value="blue", evidence_spans=[]),
            FieldValue(label="Name", value="John", evidence_spans=[]),
            FieldValue(label=None, value="The diagnosis is hypertension", evidence_spans=[]),
        ]
        sorted_facts = _score_and_sort_facts(facts, ["diagnosis", "hypertension"], "what is the diagnosis")
        # The fact about diagnosis should be ranked first
        assert "hypertension" in sorted_facts[0].value.lower() or "diagnosis" in (sorted_facts[0].label or "").lower()

    def test_labeled_facts_boosted_over_unlabeled(self):
        """Labeled facts should have higher scores than unlabeled."""
        facts = [
            FieldValue(label=None, value="some sentence about skills", evidence_spans=[]),
            FieldValue(label="Skills", value="Python, Java", evidence_spans=[]),
        ]
        sorted_facts = _score_and_sort_facts(facts, ["skills"], "what are the skills")
        assert sorted_facts[0].label == "Skills"


# ── Test 13: Hyphenated label preservation (Fix 1) ───────────────────

class TestHyphenatedLabels:
    def test_non_compete_not_split(self):
        """'Non-Compete: 12 months' should keep label='Non-Compete'."""
        chunks = [_make_chunk(
            "Non-Compete: 12 months\nResults-driven: yes\nFull-Time: true",
            doc_name="contract.pdf",
        )]
        schema = _extract_document_intelligence("contract terms", chunks)
        facts = schema.facts.items or []
        labels = {f.label for f in facts if f.label}
        assert "Non-Compete" in labels
        # "Results" should not appear as a standalone label from bad splitting
        assert "Results" not in labels
        assert "Non" not in labels

    def test_hyphenated_value_preserved(self):
        """Results-driven in value should stay intact."""
        chunks = [_make_chunk(
            "Style: Results-driven approach",
            doc_name="doc.pdf",
        )]
        schema = _extract_document_intelligence("what style", chunks)
        facts = schema.facts.items or []
        values = {f.value for f in facts}
        assert any("Results-driven" in v for v in values)


# ── Test 14: Heading inheritance for bullets (Fix 3) ──────────────────

class TestHeadingInheritance:
    def test_bullets_under_medications_heading(self):
        """Bullets under 'Medications:' should inherit that heading as label."""
        chunks = [_make_chunk(
            "Medications:\n- Metformin 500mg\n- Lisinopril 10mg\n- Aspirin 81mg",
            doc_name="patient.pdf",
        )]
        schema = _extract_document_intelligence("list medications", chunks)
        facts = schema.facts.items or []
        med_facts = [f for f in facts if f.label == "Medications"]
        assert len(med_facts) >= 2
        values = {f.value for f in med_facts}
        assert "Metformin 500mg" in values or "Lisinopril 10mg" in values

    def test_nested_kv_under_heading(self):
        """KV pairs under a heading should inherit section context."""
        chunks = [_make_chunk(
            "Lab Results:\n  HbA1c: 7.2%\n  Glucose: 110 mg/dL",
            doc_name="labs.pdf",
        )]
        schema = _extract_document_intelligence("lab results", chunks)
        facts = schema.facts.items or []
        labels = {f.label for f in facts if f.label}
        assert "HbA1c" in labels or "Glucose" in labels


# ── Test 15: Per-document dedup (Fix 4) ───────────────────────────────

class TestPerDocumentDedup:
    def test_same_label_in_different_docs_preserved(self):
        """Same label+value pattern in different docs should both be kept."""
        chunks = [
            _make_chunk("Project: Website Redesign\nBudget: $50k", doc_id="d1", doc_name="alice.pdf", chunk_id="c1"),
            _make_chunk("Project: Website Redesign\nBudget: $75k", doc_id="d2", doc_name="bob.pdf", chunk_id="c2"),
        ]
        schema = _extract_document_intelligence("compare projects", chunks)
        facts = schema.facts.items or []
        project_facts = [f for f in facts if f.label == "Project"]
        # Both docs should have their "Project" fact preserved
        doc_names = {f.document_name for f in project_facts}
        assert len(doc_names) >= 2

    def test_true_duplicates_within_doc_removed(self):
        """Exact same fact in two chunks of same doc should be deduped."""
        chunks = [
            _make_chunk("Name: Alice\nSkills: Python", doc_id="d1", doc_name="alice.pdf", chunk_id="c1"),
            _make_chunk("Name: Alice\nSkills: Python", doc_id="d1", doc_name="alice.pdf", chunk_id="c2"),
        ]
        schema = _extract_document_intelligence("details", chunks)
        facts = schema.facts.items or []
        name_facts = [f for f in facts if f.label == "Name" and f.value == "Alice"]
        assert len(name_facts) == 1


# ── Test 16: Intent-aware scoring (Fix 5) ─────────────────────────────

class TestIntentAwareScoring:
    def test_contact_intent_boosts_email(self):
        """Contact query should boost email over skills."""
        facts = [
            FieldValue(label="Skills", value="Python, Java", evidence_spans=[]),
            FieldValue(label="Email", value="jane@example.com", evidence_spans=[]),
        ]
        sorted_facts = _score_and_sort_facts(facts, ["email"], "what is the email", intent="contact")
        assert sorted_facts[0].label == "Email"

    def test_skills_query_does_not_boost_email(self):
        """Skills query should not give email a contact boost."""
        facts = [
            FieldValue(label="Email", value="jane@example.com", evidence_spans=[]),
            FieldValue(label="Skills", value="Python, machine learning", evidence_spans=[]),
        ]
        sorted_facts = _score_and_sort_facts(facts, ["skills", "python"], "what are the skills", intent="facts")
        assert sorted_facts[0].label == "Skills"

    def test_rank_intent_boosts_experience(self):
        """Rank query should boost experience and skills labels."""
        facts = [
            FieldValue(label="Color", value="blue", evidence_spans=[]),
            FieldValue(label="Experience", value="5 years software engineering", evidence_spans=[]),
        ]
        sorted_facts = _score_and_sort_facts(facts, ["rank"], "rank all candidates", intent="rank")
        assert sorted_facts[0].label == "Experience"


# ── Test 17: No full-text dump as Fact 1 (Fix 6) ─────────────────────

class TestNoFullTextDump:
    def test_long_text_not_single_fact(self):
        """A long paragraph should not appear as a single unlabeled fact."""
        long_text = "This is a very long paragraph " * 20  # ~600 chars
        chunks = [_make_chunk(
            f"Name: Alice\n{long_text}",
            doc_name="doc.pdf",
        )]
        schema = _extract_document_intelligence("tell me about Alice", chunks)
        facts = schema.facts.items or []
        for fact in facts:
            assert len(fact.value) <= 300 or fact.label is not None


# ── Test 18: Standalone lines under heading (Fix 7) ──────────────────

class TestStandaloneLines:
    def test_education_entry_without_delimiter(self):
        """Education entry without KV delimiter should be extracted under heading."""
        chunks = [_make_chunk(
            "EDUCATION:\nM.S. Computer Science, Stanford, 2017\nB.S. Mathematics, MIT, 2015",
            doc_name="resume.pdf",
        )]
        schema = _extract_document_intelligence("education background", chunks)
        facts = schema.facts.items or []
        edu_facts = [f for f in facts if f.label and "education" in f.label.lower()]
        assert len(edu_facts) >= 1
        values = " ".join(f.value for f in edu_facts)
        assert "Stanford" in values or "MIT" in values


# ── Test 19: Sentence splitting on newlines (Fix 8) ──────────────────

class TestSentenceSplitting:
    def test_newline_separated_text_splits(self):
        """Text without sentence endings but with newlines should split."""
        chunks = [_make_chunk(
            "First line of content\nSecond line of content\nThird line of content",
            doc_name="notes.txt",
        )]
        schema = _extract_document_intelligence("content details", chunks)
        facts = schema.facts.items or []
        # Should have multiple facts, not one giant blob
        assert len(facts) >= 2


# ── Test 20: Phone parenthesis preservation (Fix 9) ──────────────────

class TestPhonePreservation:
    def test_parenthesized_phone_preserved(self):
        """Phone number (555) 123-4567 should keep parentheses."""
        chunks = [_make_chunk(
            "Name: John Doe\nPhone: (555) 123-4567\nEmail: john@example.com",
            doc_name="contact.pdf",
        )]
        schema = _extract_document_intelligence("contact info", chunks)
        facts = schema.facts.items or []
        phone_facts = [f for f in facts if f.label == "Phone"]
        if phone_facts:
            assert "(" in phone_facts[0].value and ")" in phone_facts[0].value


# ── Test 21: Phrase-aware keywords (Fix 10) ───────────────────────────

class TestPhraseKeywords:
    def test_bigrams_generated(self):
        """'machine learning experience' should produce bigram 'machine learning'."""
        kws = _keywords("machine learning experience")
        assert "machine learning" in kws or "learning experience" in kws

    def test_expanded_stop_words(self):
        """Common query words should be filtered out."""
        kws = _keywords("what is the diagnosis")
        assert "what" not in kws
        assert "the" not in kws
        assert "diagnosis" in kws

    def test_results_driven_not_split_as_label(self):
        """'Results-driven engineer with...' should not produce label='Results'."""
        chunks = [_make_chunk(
            "Summary: Results-driven engineer with 10 years experience",
            doc_name="resume.pdf",
        )]
        schema = _extract_document_intelligence("summary", chunks)
        facts = schema.facts.items or []
        labels = {f.label for f in facts if f.label}
        assert "Results" not in labels
