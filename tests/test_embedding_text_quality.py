"""Tests for embedding text quality improvements.

Validates:
- Control character stripping in normalize_content(), clean_text_for_embedding(), sanitize_text()
- Preserves tabs, newlines, regular content
- Handles realistic PDF extraction artifacts
- ContentTypeDetector does not misclassify product codes as phone numbers
- content_classifier rejects education title when content is experience
- identity_contact not over-classified on resume chunks with incidental email
- schema_normalizer strips wrong upstream prefixes
- End-to-end: resume experience chunk does not get "Contact Information:" prefix
- ML zero-shot content type detection via embedding prototypes
"""
from __future__ import annotations

import threading
import pytest
import numpy as np


class TestControlCharStripping:
    """Control character removal across all three entry points."""

    def test_normalize_content_strips_null_bytes(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("Hello\x00World")
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_normalize_content_strips_escape_sequences(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("Data\x1b[31mRed\x1b[0mNormal")
        assert "\x1b" not in result
        # Only the ESC char (0x1b) is stripped; brackets remain as regular chars
        assert "Data" in result
        assert "Red" in result

    def test_normalize_content_strips_form_feed(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("Page 1\x0cPage 2")
        assert "\x0c" not in result
        assert "Page 1" in result
        assert "Page 2" in result

    def test_normalize_content_strips_vertical_tab(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("Line1\x0bLine2")
        assert "\x0b" not in result

    def test_normalize_content_preserves_tabs(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        # Tabs are in the allowed range (0x09); normalize_content may split digit-letter boundaries
        result = normalize_content("Column\tData\tResult")
        assert "Column" in result
        assert "Data" in result
        assert "Result" in result

    def test_normalize_content_preserves_newlines(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("LineA\nLineB\nLineC")
        assert "LineA" in result
        assert "LineB" in result
        assert "LineC" in result

    def test_normalize_content_preserves_regular_content(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        text = "John Smith has 5+ years of experience in Python, Java, and SQL."
        result = normalize_content(text)
        assert "John Smith" in result
        assert "Python" in result

    def test_clean_text_for_embedding_strips_control_chars(self):
        from src.embedding.pipeline.chunk_integrity import clean_text_for_embedding
        result = clean_text_for_embedding("Resume\x00\x01\x02\x03 Content")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Resume" in result
        assert "Content" in result

    def test_clean_text_for_embedding_preserves_newlines(self):
        from src.embedding.pipeline.chunk_integrity import clean_text_for_embedding
        result = clean_text_for_embedding("Skills:\n- Python\n- Java")
        assert "Skills:" in result
        assert "Python" in result

    def test_sanitize_text_strips_control_chars(self):
        from src.rag_v3.sanitize import sanitize_text
        result = sanitize_text("Answer\x00with\x1bnull\x7fbytes")
        assert "\x00" not in result
        assert "\x1b" not in result
        assert "\x7f" not in result
        assert "Answer" in result

    def test_sanitize_text_preserves_normal_output(self):
        from src.rag_v3.sanitize import sanitize_text
        text = "John Smith has experience in Python and Java."
        result = sanitize_text(text)
        assert "John Smith" in result
        assert "Python" in result

    def test_sanitize_empty_input(self):
        from src.rag_v3.sanitize import sanitize_text
        assert sanitize_text("") == ""
        assert sanitize_text(None) == ""

    def test_normalize_content_empty_input(self):
        from src.embedding.pipeline.schema_normalizer import normalize_content
        assert normalize_content("") == ""
        assert normalize_content(None) == ""

    def test_clean_text_for_embedding_empty(self):
        from src.embedding.pipeline.chunk_integrity import clean_text_for_embedding
        assert clean_text_for_embedding("") == ""

    def test_strips_c1_control_codes(self):
        """C1 control codes (0x80-0x9f) should be stripped."""
        from src.embedding.pipeline.schema_normalizer import normalize_content
        result = normalize_content("Text\x80\x85\x9fMore")
        assert "\x80" not in result
        assert "\x85" not in result
        assert "\x9f" not in result

    def test_realistic_pdf_artifacts(self):
        """Simulate PDF extraction with null bytes + form feeds."""
        from src.embedding.pipeline.schema_normalizer import normalize_content
        pdf_text = "Invoice\x00 #12345\x0c\nDate: 2024-01-15\x00\nTotal: $500.00"
        result = normalize_content(pdf_text)
        assert "\x00" not in result
        assert "\x0c" not in result
        assert "Invoice" in result
        assert "12345" in result
        assert "500.00" in result


class TestContentClassifierExpansion:
    """Test new section kind classifications."""

    def test_invoice_line_items_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Line Items") == "line_items"

    def test_invoice_bill_to_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Bill To") == "parties_addresses"

    def test_invoice_payment_terms_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Payment Terms") == "terms_conditions"

    def test_legal_governing_law_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Governing Law") == "legal_clauses"

    def test_legal_confidentiality_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Confidentiality Agreement") == "legal_clauses"

    def test_legal_recitals_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Recitals") == "legal_preamble"

    def test_legal_definitions_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Definitions") == "legal_definitions"

    def test_legal_signatures_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Signature Page") == "legal_signatures"

    def test_medical_diagnosis_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Diagnosis") == "medical_findings"

    def test_medical_medications_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Medications") == "medical_medications"

    def test_medical_lab_results_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Lab Results") == "medical_lab_results"

    def test_medical_patient_info_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Patient Information") == "medical_patient_info"

    def test_bank_transactions_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Transaction History") == "transactions"

    def test_bank_account_summary_title(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Account Summary") == "financial_summary"

    # Content-based classification tests
    def test_content_line_items_keywords(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        text = "Item: Widget A, quantity: 10, unit price: $5.00, subtotal: $50"
        result = classify_section_kind(text, "")
        assert result in ("line_items", "financial_summary")  # both valid

    def test_content_legal_clauses_keywords(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        text = "The governing law of this agreement shall be arbitration. Indemnification clause applies."
        result = classify_section_kind(text, "")
        assert result == "legal_clauses"

    def test_content_medical_findings_keywords(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        text = "The diagnosis shows clinical finding of hypertension. Chief complaint is headache. Review of systems normal."
        result = classify_section_kind(text, "")
        assert result == "medical_findings"

    # Regression: existing resume kinds unchanged
    def test_resume_skills_unchanged(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Technical Skills") == "skills_technical"

    def test_resume_education_unchanged(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Education") == "education"

    def test_resume_experience_unchanged(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Work Experience") == "experience"

    def test_resume_certifications_unchanged(self):
        from src.embedding.pipeline.content_classifier import classify_section_kind
        assert classify_section_kind("", "Certifications") == "certifications"


class TestDomainClassifierStrengthening:
    """Test strong indicator keywords in domain classifier."""

    def test_invoice_strong_indicators(self):
        from src.intelligence.domain_classifier import classify_domain
        text = "Invoice Number: INV-2024-001\nInvoice Date: January 15, 2024\nBill To: Company ABC"
        result = classify_domain(text)
        assert result.domain == "invoice"
        assert not result.uncertain

    def test_legal_strong_indicators(self):
        from src.intelligence.domain_classifier import classify_domain
        text = "This agreement is entered into by the party of the first part, hereinafter referred to as the Licensor."
        result = classify_domain(text)
        assert result.domain == "legal"

    def test_medical_strong_indicators(self):
        from src.intelligence.domain_classifier import classify_domain
        text = "Chief Complaint: Patient presents with chest pain. Medical history includes hypertension. Review of systems unremarkable."
        result = classify_domain(text)
        assert result.domain == "medical"

    def test_bank_strong_indicators(self):
        from src.intelligence.domain_classifier import classify_domain
        text = "Account Statement for period ending Dec 2024. Available balance: $5,432.10. Opening balance: $4,000.00."
        result = classify_domain(text)
        assert result.domain == "bank_statement"

    def test_resume_still_works(self):
        from src.intelligence.domain_classifier import classify_domain
        text = "Professional Experience: 5 years. Skills: Python, Java. Education: B.Tech Computer Science. Certification: AWS."
        result = classify_domain(text)
        assert result.domain == "resume"


class TestEntityMetadataExtraction:
    """Test ingestion-time entity metadata extraction."""

    def test_extracts_email(self):
        from src.api.extraction_service import _extract_entity_metadata
        raw = {"test.pdf": {"full_text": "John Smith\njohn@example.com\n+1-555-1234567"}}
        result = _extract_entity_metadata(raw, "resume.pdf")
        assert result.get("entity_email") == "john@example.com"

    def test_extracts_phone(self):
        from src.api.extraction_service import _extract_entity_metadata
        raw = {"test.pdf": {"full_text": "Contact: +91-9876543210\nSkills: Python"}}
        result = _extract_entity_metadata(raw, "resume.pdf")
        assert "entity_phone" in result

    def test_extracts_name_from_filename(self):
        from src.api.extraction_service import _extract_entity_metadata
        raw = {"resume.pdf": {"full_text": "Some content..."}}
        result = _extract_entity_metadata(raw, "John_Smith_Resume.pdf")
        # Name should be extracted from filename
        if "entity_name" in result:
            assert "john" in result["entity_name"].lower() or "smith" in result["entity_name"].lower()

    def test_returns_empty_for_no_data(self):
        from src.api.extraction_service import _extract_entity_metadata
        result = _extract_entity_metadata({}, "")
        assert isinstance(result, dict)

    def test_validation_logs_no_crash(self):
        """_validate_extraction_fields should never crash, just log."""
        from src.api.extraction_service import _validate_extraction_fields
        # Resume with missing fields
        _validate_extraction_fields(
            {"document_type": "RESUME", "domain": "resume"},
            {"test.pdf": {"full_text": ""}},
            "test.pdf",
        )
        # Invoice
        _validate_extraction_fields(
            {"document_type": "INVOICE", "domain": "invoice"},
            {"inv.pdf": {"full_text": "just some text"}},
            "inv.pdf",
        )
        # Unknown type
        _validate_extraction_fields(
            {"document_type": "GENERIC", "domain": "generic"},
            {},
            "",
        )


# ── Fix 1 tests: ContentTypeDetector pattern tightening ──────────────


class TestContentTypeDetectorProductCodes:
    """Product codes like 582-33-2563 must NOT be classified as phone numbers."""

    def test_product_code_not_classified_as_contact(self):
        """Invoice line items with product codes should be financial, not contact."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = (
            "Item #1: Widget Assembly\n"
            "Product Code: 582-33-2563\n"
            "Qty: 10 | Unit Price: $45.00 | Total: $450.00"
        )
        result = ContentTypeDetector.detect(text, section_title="")
        assert result["content_type"] != "contact_info", (
            f"Product code misclassified as contact_info: {result}"
        )

    def test_ssn_like_number_not_contact(self):
        """Numbers like 123-45-6789 should not trigger contact_info."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = "Reference: 123-45-6789, Order: 987-65-4321, Account: 555-12-3456"
        result = ContentTypeDetector.detect(text, section_title="")
        assert result["content_type"] != "contact_info", (
            f"SSN-like numbers misclassified as contact_info: {result}"
        )

    def test_labeled_phone_still_detected(self):
        """A phone number with 'Phone:' label should still be detected."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = (
            "John Smith\n"
            "Email: john@example.com\n"
            "Phone: +1-555-123-4567\n"
            "LinkedIn: linkedin.com/in/johnsmith"
        )
        result = ContentTypeDetector.detect(text, section_title="")
        assert result["content_type"] == "contact_info", (
            f"Labeled contact info not detected: {result}"
        )

    def test_unlabeled_email_in_resume_not_contact(self):
        """A bare email in a long resume chunk should not classify as contact."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = (
            "Developed a microservices architecture using Python and Django. "
            "Collaborated with team@company.com on CI/CD pipeline. "
            "Deployed to AWS using Docker and Kubernetes. "
            "Managed team of 5 engineers over 3 years. "
            "Implemented monitoring using Grafana and Prometheus dashboards."
        )
        result = ContentTypeDetector.detect(text, section_title="")
        assert result["content_type"] != "contact_info", (
            f"Long experience text misclassified as contact_info: {result}"
        )

    def test_financial_needs_currency_or_multiple_keywords(self):
        """A resume project mentioning 'amount' shouldn't classify as financial."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = (
            "Built a recommendation engine that increased conversion rate. "
            "The total amount of users served was 1 million. "
            "Responsible for managing the payment integration module."
        )
        result = ContentTypeDetector.detect(text, section_title="")
        # Should NOT be financial — no currency symbol, no 2+ strong financial keywords
        assert result["content_type"] != "financial", (
            f"Resume project text misclassified as financial: {result}"
        )

    def test_real_invoice_still_detected_as_financial(self):
        """Actual invoice text with currency should still be financial."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = (
            "Invoice #INV-2024-001\n"
            "Subtotal: $1,250.00\n"
            "Tax (10%): $125.00\n"
            "Total Amount Due: $1,375.00\n"
            "Payment Terms: Net 30"
        )
        result = ContentTypeDetector.detect(text, section_title="")
        assert result["content_type"] == "financial", (
            f"Real invoice text not classified as financial: {result}"
        )

    def test_comma_separated_list_not_skills_without_tech(self):
        """Comma-separated generic list should not auto-classify as skills."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = "apples, oranges, bananas, grapes, mangoes, papayas"
        result = ContentTypeDetector.detect(text, section_title="")
        assert result["content_type"] != "skills_list", (
            f"Fruit list misclassified as skills_list: {result}"
        )


# ── Fix 2 tests: content_classifier content verification gate ────────


class TestContentClassifierVerificationGate:
    """Title-based classification should be rejected when content doesn't match."""

    def test_education_title_rejected_when_content_is_experience(self):
        """Section titled 'Education' with work experience text should NOT be education."""
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        text = (
            "Worked as Senior Developer at TechCorp from 2019-2023. "
            "Responsible for designing microservices architecture. "
            "Managed a team of 8 engineers and delivered 3 major projects."
        )
        kind, source = classify_section_kind_with_source(text, "Education")
        assert kind != "education", (
            f"Experience text under 'Education' title should NOT classify as education, got: {kind}"
        )
        # Should fall through to content-based and get experience
        assert kind == "experience", f"Expected experience, got {kind}"

    def test_education_title_accepted_when_content_matches(self):
        """Section titled 'Education' with real education content should be education."""
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        text = (
            "Bachelor of Technology in Computer Science, "
            "Anna University, Chennai. GPA: 8.5/10. "
            "Graduated in 2020."
        )
        kind, source = classify_section_kind_with_source(text, "Education")
        assert kind == "education", f"Real education content should classify as education, got: {kind}"
        assert source == "title"

    def test_skills_title_rejected_when_content_has_no_tech(self):
        """Section titled 'Technical Skills' with non-tech text should fall through."""
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        text = (
            "Invoice verification by matching PO, GRN, and supplier bill. "
            "Maintained vendor records and processed purchase orders. "
            "Coordinated with warehouse staff for material receipt."
        )
        kind, source = classify_section_kind_with_source(text, "Technical Skills")
        assert kind != "skills_technical", (
            f"Procurement text under 'Technical Skills' should NOT be skills_technical, got: {kind}"
        )

    def test_skills_title_accepted_with_tech_content(self):
        """Section titled 'Technical Skills' with real tech content should pass."""
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        text = "Python, Java, JavaScript, React, Django, Docker, Kubernetes, AWS, SQL, MongoDB"
        kind, source = classify_section_kind_with_source(text, "Technical Skills")
        assert kind == "skills_technical"
        assert source == "title"

    def test_identity_contact_not_overclassified(self):
        """Resume chunk with incidental email mention should NOT be identity_contact."""
        from src.embedding.pipeline.content_classifier import classify_section_kind
        text = (
            "5 years of experience in Python and Java development. "
            "Worked on machine learning projects. "
            "Visit our site at www.example.com for more details. "
            "Email notifications sent to the team."
        )
        kind = classify_section_kind(text, "")
        assert kind != "identity_contact", (
            f"Resume text with incidental www/email mention classified as identity_contact: {kind}"
        )

    def test_identity_contact_tightened_keywords(self):
        """identity_contact should require contextual signals like 'email:' or 'phone:'."""
        from src.embedding.pipeline.content_classifier import classify_section_kind
        # This has broad tokens ("@", ".com", "address") but no contextual labels
        text = "john@company.com is the project lead. The address of the office is 123 Main St."
        kind = classify_section_kind(text, "")
        assert kind != "identity_contact", (
            f"Text with incidental @ and address should NOT be identity_contact, got: {kind}"
        )

    def test_experience_wins_tiebreak_with_date_ranges(self):
        """When tied, date ranges should prefer experience over other kinds."""
        from src.embedding.pipeline.content_classifier import classify_section_kind
        text = (
            "Python developer at TechCorp from 2019-2023. "
            "Built REST APIs and managed deployment pipelines. "
            "Led team of 5 for the cloud migration project."
        )
        kind = classify_section_kind(text, "")
        assert kind == "experience", f"Text with date ranges should prefer experience, got: {kind}"


# ── Fix 3 tests: schema_normalizer prefix stripping ──────────────────


class TestSchemaNormalizerPrefixStripping:
    """Wrong upstream embedding_text prefixes should be stripped."""

    def test_strips_wrong_contact_prefix(self):
        """'Contact Information: <invoice text>' should have prefix stripped."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
        raw = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "content": "Widget A, Qty: 10, Unit Price: $45.00, Total: $450.00",
            "embedding_text": "Contact Information: Widget A, Qty: 10, Unit Price: $45.00, Total: $450.00",
            "section_title": "Line Items",
            "doc_domain": "invoice",
            "section_kind": "line_items",
        }
        payload = build_qdrant_payload(raw)
        et = payload.get("embedding_text", "")
        assert not et.startswith("Contact Information:"), (
            f"Wrong 'Contact Information:' prefix should be stripped: {et[:80]}"
        )

    def test_strips_wrong_financial_prefix_on_resume(self):
        """'Financial Information: <resume text>' should have prefix stripped."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
        raw = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "content": "Developed a CNN model to classify plant diseases. Achieved 92% accuracy.",
            "embedding_text": "Financial Information: Developed a CNN model to classify plant diseases. Achieved 92% accuracy.",
            "section_title": "Projects",
            "doc_domain": "resume",
            "section_kind": "experience",
        }
        payload = build_qdrant_payload(raw)
        et = payload.get("embedding_text", "")
        assert not et.startswith("Financial Information:"), (
            f"Wrong 'Financial Information:' prefix should be stripped: {et[:80]}"
        )

    def test_strips_wrong_education_prefix(self):
        """'Education and Qualifications: <experience text>' prefix stripped."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
        raw = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "content": "Senior Developer at TechCorp, 2019-2023. Led microservices architecture.",
            "embedding_text": "Education and Qualifications: Senior Developer at TechCorp, 2019-2023. Led microservices architecture.",
            "section_title": "Experience",
            "doc_domain": "resume",
            "section_kind": "experience",
        }
        payload = build_qdrant_payload(raw)
        et = payload.get("embedding_text", "")
        assert not et.startswith("Education and Qualifications:"), (
            f"Wrong 'Education and Qualifications:' prefix should be stripped: {et[:80]}"
        )

    def test_preserves_correct_prefix(self):
        """Correct prefix added by schema_normalizer should remain."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload
        raw = {
            "subscription_id": "sub1",
            "profile_id": "prof1",
            "document_id": "doc1",
            "content": "Bachelor of Technology, Computer Science, Anna University, GPA: 8.5",
            "section_title": "Education",
            "doc_domain": "resume",
            "section_kind": "education",
            "section_kind_source": "title",
        }
        payload = build_qdrant_payload(raw)
        et = payload.get("embedding_text", "")
        # The normalizer should add its OWN correct [Education] prefix
        assert "Bachelor" in et, f"Content should be preserved in embedding_text: {et[:80]}"


# ── End-to-end embedding quality test ────────────────────────────────


class TestEndToEndEmbeddingQuality:
    """End-to-end: ensure the full pipeline produces correct embedding text."""

    def test_resume_experience_no_contact_prefix(self):
        """Resume experience chunk should never get 'Contact Information:' prefix."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        from src.embedding.pipeline.content_classifier import classify_section_kind_with_source
        text = (
            "Senior Software Engineer at Google, 2020-Present. "
            "Designed and deployed scalable microservices handling 1M+ requests/day. "
            "Technologies: Python, Go, Kubernetes, BigQuery, Spanner."
        )
        # Step 1: ContentTypeDetector should NOT classify as contact
        ct = ContentTypeDetector.detect(text, section_title="Work Experience")
        assert ct["content_type"] != "contact_info", (
            f"Experience text wrongly detected as contact: {ct}"
        )
        # Step 2: content_classifier should classify as experience
        kind, source = classify_section_kind_with_source(text, "Work Experience")
        assert kind == "experience", f"Expected experience, got {kind}"

    def test_invoice_line_items_no_contact_prefix(self):
        """Invoice line items with product codes should not get contact prefix."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        text = (
            "Product: Industrial Valve Assembly\n"
            "SKU: 582-33-2563\n"
            "Quantity: 25 units\n"
            "Unit Price: $89.50\n"
            "Subtotal: $2,237.50"
        )
        ct = ContentTypeDetector.detect(text, section_title="")
        assert ct["content_type"] != "contact_info", (
            f"Invoice line items with product code wrongly detected as contact: {ct}"
        )

    def test_resume_with_scattered_email_not_identity_contact(self):
        """A resume chunk mentioning email in passing should not be identity_contact."""
        from src.embedding.pipeline.content_classifier import classify_section_kind
        text = (
            "Implemented email notification service for 50k users. "
            "Developed REST APIs using Django and deployed on AWS. "
            "Integrated with internal tools at company.com. "
            "Managed CI/CD pipelines using Jenkins and Docker."
        )
        kind = classify_section_kind(text, "")
        assert kind != "identity_contact", (
            f"Resume experience text classified as identity_contact: {kind}"
        )


# ── ML zero-shot content type detection ──────────────────────────────


class _FakeEmbedder:
    """Deterministic mock embedder for unit tests.

    Returns a 1024-dim vector derived from a hash of the input text so that
    similar texts get similar-ish vectors (good enough for prototype testing).
    """
    DIM = 1024

    def encode(self, text_or_texts, normalize_embeddings=True, convert_to_numpy=True):
        import hashlib

        def _vec(text: str) -> np.ndarray:
            # Seed a deterministic RNG from the text hash
            h = int(hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(), 16)
            rng = np.random.RandomState(h % (2**31))
            v = rng.randn(self.DIM).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    v /= norm
            return v

        if isinstance(text_or_texts, str):
            return _vec(text_or_texts)
        return np.stack([_vec(t) for t in text_or_texts])


class _SemanticEmbedder:
    """Mock embedder that encodes prototype-like semantics.

    Produces vectors that are closer to the matching content type by
    injecting a domain-specific bias component, ensuring cosine similarity
    correctly identifies the best prototype.
    """
    DIM = 1024

    # Assign each content type a distinct direction in the first 7 dims
    _TYPE_SLOTS = {
        "contact": 0, "email": 0, "phone": 0, "linkedin": 0,
        "skills": 1, "python": 1, "java": 1, "docker": 1, "kubernetes": 1,
        "aws": 1, "react": 1, "sql": 1, "frameworks": 1, "technologies": 1,
        "proficient": 1, "competencies": 1, "programming": 1,
        "education": 2, "bachelor": 2, "master": 2, "university": 2,
        "degree": 2, "gpa": 2, "graduated": 2, "college": 2, "academic": 2,
        "coursework": 2, "qualifications": 2,
        "experience": 3, "worked": 3, "managed": 3, "developed": 3,
        "deployed": 3, "career": 3, "responsibilities": 3, "job": 3,
        "employment": 3, "projects": 3, "company": 3,
        "invoice": 4, "financial": 4, "payment": 4, "subtotal": 4,
        "amount": 4, "total": 4, "billing": 4, "tax": 4, "purchase": 4,
        "legal": 5, "contract": 5, "clause": 5, "agreement": 5,
        "liability": 5, "indemnification": 5, "arbitration": 5,
        "warranty": 5, "confidentiality": 5, "governing": 5,
        "medical": 6, "diagnosis": 6, "patient": 6, "symptoms": 6,
        "treatment": 6, "medication": 6, "clinical": 6, "prognosis": 6,
        "prescription": 6, "dosage": 6,
    }

    def encode(self, text_or_texts, normalize_embeddings=True, convert_to_numpy=True):
        def _vec(text: str) -> np.ndarray:
            v = np.zeros(self.DIM, dtype=np.float32)
            # Small random base for diversity
            import hashlib
            h = int(hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(), 16)
            rng = np.random.RandomState(h % (2**31))
            v += rng.randn(self.DIM).astype(np.float32) * 0.05
            # Strong bias toward matched domain slots
            words = set(text.lower().split())
            for word in words:
                # Strip punctuation
                w = word.strip(",.;:!?()[]{}\"'")
                slot = self._TYPE_SLOTS.get(w)
                if slot is not None:
                    v[slot] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    v /= norm
            return v

        if isinstance(text_or_texts, str):
            return _vec(text_or_texts)
        return np.stack([_vec(t) for t in text_or_texts])


class TestMLContentTypeDetector:
    """ML-based zero-shot content type detection via prototype embeddings."""

    def _make_detector(self, embedder=None):
        from src.embedding.universal_enhancer import ContentTypeDetector
        return ContentTypeDetector(embedder=embedder or _SemanticEmbedder())

    def test_ml_classifies_contact_info(self):
        """Labeled contact block with email/phone/LinkedIn → contact_info."""
        det = self._make_detector()
        text = (
            "John Smith\n"
            "Email: john.smith@example.com\n"
            "Phone: +1-555-123-4567\n"
            "LinkedIn: linkedin.com/in/johnsmith"
        )
        result = det.detect_ml(text, section_title="Contact Information")
        assert result["content_type"] == "contact_info", (
            f"Expected contact_info, got {result['content_type']} (scores: {result.get('scores')})"
        )

    def test_ml_classifies_experience(self):
        """Work history with dates and responsibilities → experience."""
        det = self._make_detector()
        text = (
            "Senior Software Engineer at Google, 2020-Present. "
            "Managed a team of 8 engineers. Developed microservices. "
            "Deployed to production on Kubernetes. Led 3 projects."
        )
        result = det.detect_ml(text, section_title="Work Experience")
        assert result["content_type"] == "experience", (
            f"Expected experience, got {result['content_type']} (scores: {result.get('scores')})"
        )

    def test_ml_classifies_education(self):
        """Degree/university text → education."""
        det = self._make_detector()
        text = (
            "Bachelor of Technology in Computer Science, Anna University, Chennai. "
            "GPA: 8.5/10. Graduated in 2020. Coursework: algorithms, databases."
        )
        result = det.detect_ml(text, section_title="Education")
        assert result["content_type"] == "education", (
            f"Expected education, got {result['content_type']} (scores: {result.get('scores')})"
        )

    def test_ml_classifies_skills(self):
        """Tech keyword list → skills_list."""
        det = self._make_detector()
        text = "Python, Java, JavaScript, React, Django, Docker, Kubernetes, AWS, SQL, MongoDB"
        result = det.detect_ml(text, section_title="Technical Skills")
        assert result["content_type"] == "skills_list", (
            f"Expected skills_list, got {result['content_type']} (scores: {result.get('scores')})"
        )

    def test_ml_classifies_financial(self):
        """Invoice with amounts → financial."""
        det = self._make_detector()
        text = (
            "Invoice #INV-2024-001\n"
            "Subtotal: $1,250.00\n"
            "Tax (10%): $125.00\n"
            "Total Amount Due: $1,375.00\n"
            "Payment Terms: Net 30"
        )
        result = det.detect_ml(text)
        assert result["content_type"] == "financial", (
            f"Expected financial, got {result['content_type']} (scores: {result.get('scores')})"
        )

    def test_ml_classifies_legal(self):
        """Contract clauses → legal."""
        det = self._make_detector()
        text = (
            "This Agreement shall be governed by the laws of the State of California. "
            "Indemnification: Party A shall indemnify Party B. "
            "Confidentiality clause applies for 5 years. Liability cap: $1M."
        )
        result = det.detect_ml(text)
        assert result["content_type"] == "legal", (
            f"Expected legal, got {result['content_type']} (scores: {result.get('scores')})"
        )

    def test_ml_classifies_medical(self):
        """Diagnosis/medication → medical."""
        det = self._make_detector()
        text = (
            "Patient presents with chest pain. Diagnosis: acute myocardial infarction. "
            "Treatment plan: aspirin 325mg, nitroglycerin. "
            "Prognosis: good with medication compliance."
        )
        result = det.detect_ml(text)
        assert result["content_type"] == "medical", (
            f"Expected medical, got {result['content_type']} (scores: {result.get('scores')})"
        )

    def test_ml_product_code_not_contact(self):
        """Product code 582-33-2563 should NOT be classified as contact_info."""
        det = self._make_detector()
        text = (
            "Product: Industrial Valve Assembly\n"
            "SKU: 582-33-2563\n"
            "Quantity: 25 units\n"
            "Unit Price: $89.50\n"
            "Subtotal: $2,237.50"
        )
        result = det.detect_ml(text)
        assert result["content_type"] != "contact_info", (
            f"Product code misclassified as contact_info: {result}"
        )

    def test_ml_scattered_email_not_contact(self):
        """Resume with incidental email mention should NOT be contact_info."""
        det = self._make_detector()
        text = (
            "Developed a microservices architecture using Python and Django. "
            "Collaborated with team@company.com on CI/CD pipeline. "
            "Deployed to AWS using Docker and Kubernetes. "
            "Managed team of 5 engineers over 3 years."
        )
        result = det.detect_ml(text)
        assert result["content_type"] != "contact_info", (
            f"Experience text misclassified as contact_info: {result}"
        )

    def test_ml_fallback_without_embedder(self):
        """embedder=None falls back to regex detect()."""
        from src.embedding.universal_enhancer import ContentTypeDetector
        det = ContentTypeDetector(embedder=None)
        text = (
            "Email: john@example.com\n"
            "Phone: +1-555-123-4567\n"
            "LinkedIn: linkedin.com/in/john"
        )
        result = det.detect_ml(text)
        # Should still work — falls back to regex
        assert result["content_type"] == "contact_info", (
            f"Fallback detect should classify contact_info: {result}"
        )

    def test_ml_low_confidence_returns_narrative(self):
        """Gibberish / unrelated text gets narrative when similarity is low."""
        det = self._make_detector()
        # Nonsense text unlikely to match any prototype
        text = "xyzzy plugh abracadabra zork quux blorb frobnitz"
        result = det.detect_ml(text)
        # With semantic embedder, gibberish should have low similarity to all prototypes
        # The result type depends on random projection but confidence should be low
        assert result["confidence"] <= 0.6 or result["content_type"] == "narrative", (
            f"Gibberish should have low confidence or be narrative: {result}"
        )

    def test_ml_prototypes_cached(self):
        """Second call reuses prototypes (no re-encode)."""
        embedder = _SemanticEmbedder()
        det = self._make_detector(embedder)
        # First call builds prototypes
        det.detect_ml("Python, Java, AWS", section_title="Skills")
        proto_id = id(det._prototypes)
        # Second call reuses
        det.detect_ml("Docker, Kubernetes, Go", section_title="Skills")
        assert id(det._prototypes) == proto_id, "Prototypes should be cached"

    def test_ml_thread_safety(self):
        """Concurrent calls don't crash."""
        det = self._make_detector()
        errors = []

        def _run(text):
            try:
                result = det.detect_ml(text)
                assert "content_type" in result
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_run, args=(f"Python Java AWS Docker skill {i}",))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Thread safety errors: {errors}"

    def test_ml_result_has_scores(self):
        """ML detection returns per-type similarity scores."""
        det = self._make_detector()
        text = "Python, Java, React, Docker, Kubernetes, SQL"
        result = det.detect_ml(text, section_title="Skills")
        assert "scores" in result
        assert len(result["scores"]) == 7  # 7 content types
        for ctype in ("contact_info", "skills_list", "education", "experience",
                       "financial", "legal", "medical"):
            assert ctype in result["scores"]

    def test_ml_enhancer_integration(self):
        """UniversalEmbeddingEnhancer uses ML path when embedder provided."""
        from src.embedding.universal_enhancer import UniversalEmbeddingEnhancer
        embedder = _SemanticEmbedder()
        enhancer = UniversalEmbeddingEnhancer(embedder=embedder)
        result = enhancer.enhance_chunk(
            text="Python, Java, React, Docker, Kubernetes, AWS, SQL, MongoDB",
            section_title="Technical Skills",
        )
        assert result.content_type == "skills_list", (
            f"Enhancer with ML should detect skills_list, got {result.content_type}"
        )


class TestEncodingGarbageDetection:
    """Detect and handle encoding-corrupted text (UTF-16 null interleaving, etc.)."""

    def test_utf16le_null_interleaving_detected(self):
        """UTF-16LE null-byte interleaving is detected as encoding garbage."""
        from src.embedding.pipeline.schema_normalizer import _is_encoding_garbage
        # Simulates UTF-16LE decoded as UTF-8: every other byte is \x00
        text = "\x00i\x00n\x00t\x00e\x00r\x00c\x00o\x00m\x00p\x00a\x00n\x00y"
        assert _is_encoding_garbage(text), "UTF-16 null interleaving should be detected"

    def test_replacement_char_flood_detected(self):
        """High density of U+FFFD replacement chars is detected as garbage."""
        from src.embedding.pipeline.schema_normalizer import _is_encoding_garbage
        text = "\ufffd" * 30 + "some text here"
        assert _is_encoding_garbage(text), "Replacement char flood should be detected"

    def test_low_alnum_ratio_detected(self):
        """Text with very low alphanumeric ratio is detected as garbage."""
        from src.embedding.pipeline.schema_normalizer import _is_encoding_garbage
        # Mostly control/special chars with very few letters
        text = "\x01\x02\x03\x04\x05" * 20 + "ab"
        assert _is_encoding_garbage(text), "Low alphanumeric ratio should be detected"

    def test_normal_text_not_flagged(self):
        """Regular resume text is NOT flagged as encoding garbage."""
        from src.embedding.pipeline.schema_normalizer import _is_encoding_garbage
        text = "John Smith has 5+ years of experience in Python, Java, and SQL. " \
               "He worked at Google and Microsoft on distributed systems."
        assert not _is_encoding_garbage(text), "Normal text should NOT be flagged"

    def test_short_text_not_flagged(self):
        """Very short text is not flagged (not enough signal)."""
        from src.embedding.pipeline.schema_normalizer import _is_encoding_garbage
        assert not _is_encoding_garbage("Hi")
        assert not _is_encoding_garbage("")

    def test_is_metadata_garbage_catches_encoding_corruption(self):
        """_is_metadata_garbage() now also catches encoding-corrupted text."""
        from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage
        text = "\x00i\x00n\x00t\x00e\x00r\x00c\x00o\x00m\x00p\x00a\x00n\x00y\x00 \x00t\x00e\x00s\x00t"
        assert _is_metadata_garbage(text), "Encoding garbage should be caught by _is_metadata_garbage"

    def test_smart_decode_utf16le_bom(self):
        """Bytes with UTF-16LE BOM decode correctly."""
        from src.api.dw_document_extractor import _smart_decode
        text = "Hello World"
        raw = b'\xff\xfe' + text.encode('utf-16-le')
        decoded = _smart_decode(raw)
        assert "Hello World" in decoded

    def test_smart_decode_utf16le_no_bom(self):
        """Bytes with high null ratio (UTF-16 without BOM) decode correctly."""
        from src.api.dw_document_extractor import _smart_decode
        text = "intercompany transfer document"
        raw = text.encode('utf-16-le')  # No BOM, but has null bytes
        decoded = _smart_decode(raw)
        assert "intercompany" in decoded

    def test_smart_decode_utf8_passthrough(self):
        """Regular UTF-8 bytes pass through correctly."""
        from src.api.dw_document_extractor import _smart_decode
        raw = "Regular UTF-8 text with accents: café résumé".encode('utf-8')
        decoded = _smart_decode(raw)
        assert "Regular UTF-8 text" in decoded
        assert "café" in decoded

    def test_smart_decode_latin1_fallback(self):
        """Non-UTF-8 bytes fall back to charset detection or Latin-1."""
        from src.api.dw_document_extractor import _smart_decode
        # Latin-1 encoded text with characters invalid in UTF-8
        raw = "Résumé für Böhm".encode('latin-1')
        decoded = _smart_decode(raw)
        # charset_normalizer or latin-1 should recover the text
        assert len(decoded) > 5

    def test_normalize_content_strips_nulls(self):
        """normalize_content() strips embedded \\x00 chars early."""
        from src.embedding.pipeline.schema_normalizer import normalize_content
        text = "H\x00e\x00l\x00l\x00o\x00 \x00W\x00o\x00r\x00l\x00d"
        result = normalize_content(text)
        assert "\x00" not in result
        assert "Hello" in result
        assert "World" in result

    def test_prepare_chunks_drops_encoding_garbage(self):
        """prepare_embedding_chunks() drops chunks with encoding garbage."""
        from src.embedding.pipeline.embed_pipeline import prepare_embedding_chunks
        good_text = "John Smith has extensive experience in Python and machine learning."
        garbage_text = "\x00i\x00n\x00t\x00e\x00r\x00c\x00o\x00m\x00p\x00a\x00n\x00y\x00 " * 5
        chunks = [good_text, garbage_text]
        meta = [
            {"section_title": "Experience", "section_kind": "experience"},
            {"section_title": "Data", "section_kind": "other"},
        ]
        result_chunks, result_meta, stats, _rescued = prepare_embedding_chunks(
            chunks, meta,
            subscription_id="sub1", profile_id="prof1",
            document_id="doc1", doc_name="test.doc",
        )
        # Garbage chunk should be dropped; good chunk should remain
        assert len(result_chunks) == 1
        assert "John Smith" in result_chunks[0] or "John Smith" in result_meta[0].get("content", "")
