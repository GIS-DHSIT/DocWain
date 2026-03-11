"""Real-world validation tests for the DocWain Intelligence Engine.

These tests use realistic document content (not toy examples) to validate
that the pipeline produces accurate, complete, traceable results.
"""
import pytest
from src.docwain_intel.models import (
    Block, Page, Section, Table, ExtractedDocumentJSON,
    UnitType, SemanticUnit, StructuredDocument,
)
from src.docwain_intel.structure_parser import parse_structure
from src.docwain_intel.entity_engine import extract_entities_and_facts
from src.docwain_intel.entity_resolver import resolve_entities
from src.docwain_intel.document_fingerprint import compute_fingerprint
from src.docwain_intel.verification import verify_extraction
from src.docwain_intel.intel_pipeline import process_document, PipelineStage
from src.docwain_intel.query_router import route_query, QueryRoute
from src.docwain_intel.response_assembler import assemble_response


# ── Realistic Document Fixtures ───────────────────────────────────

def _resume_doc():
    """Realistic software engineer resume."""
    blocks = [
        Block(block_id="b0", type="heading", text="Gokul Ramanathan", page_number=1, reading_order=0),
        Block(block_id="b1", type="key_value", text="Email: gokul.ramanathan@gmail.com", key="Email", value="gokul.ramanathan@gmail.com", page_number=1, reading_order=1),
        Block(block_id="b2", type="key_value", text="Phone: +1-408-555-0142", key="Phone", value="+1-408-555-0142", page_number=1, reading_order=2),
        Block(block_id="b3", type="key_value", text="Location: San Jose, CA", key="Location", value="San Jose, CA", page_number=1, reading_order=3),
        Block(block_id="b4", type="heading", text="Professional Summary", page_number=1, reading_order=4),
        Block(block_id="b5", type="paragraph", text="Senior Software Engineer with 8 years of experience designing and building scalable distributed systems. Expert in Python, Go, and Kubernetes. Led a team of 12 engineers at Google Cloud, delivering a real-time data pipeline processing 500 million events per day.", page_number=1, reading_order=5),
        Block(block_id="b6", type="heading", text="Work Experience", page_number=1, reading_order=6),
        Block(block_id="b7", type="paragraph", text="Senior Software Engineer at Google Cloud, Mountain View, CA. January 2020 to Present. Architected and deployed a distributed event processing system handling 500M events/day with 99.99% uptime. Reduced infrastructure costs by $2.3M annually through optimization of Kubernetes cluster autoscaling. Mentored 5 junior engineers and conducted 200+ code reviews.", page_number=1, reading_order=7),
        Block(block_id="b8", type="paragraph", text="Software Engineer at Microsoft Azure, Redmond, WA. June 2016 to December 2019. Built core components of Azure Functions serverless platform serving 10,000+ enterprise customers. Implemented automated deployment pipelines reducing release cycles from 2 weeks to 4 hours. Contributed to open-source .NET runtime with 3 accepted PRs.", page_number=2, reading_order=8),
        Block(block_id="b9", type="heading", text="Skills", page_number=2, reading_order=9),
        Block(block_id="b10", type="list_item", text="- Python, Go, Java, C#, TypeScript", page_number=2, reading_order=10),
        Block(block_id="b11", type="list_item", text="- Kubernetes, Docker, Terraform, AWS, GCP, Azure", page_number=2, reading_order=11),
        Block(block_id="b12", type="list_item", text="- PostgreSQL, Redis, Apache Kafka, BigQuery", page_number=2, reading_order=12),
        Block(block_id="b13", type="list_item", text="- System Design, Distributed Systems, CI/CD, Agile", page_number=2, reading_order=13),
        Block(block_id="b14", type="heading", text="Education", page_number=2, reading_order=14),
        Block(block_id="b15", type="paragraph", text="Master of Science in Computer Science, Stanford University, 2016. Bachelor of Technology in Computer Science, IIT Bombay, 2014.", page_number=2, reading_order=15),
        Block(block_id="b16", type="heading", text="Certifications", page_number=2, reading_order=16),
        Block(block_id="b17", type="list_item", text="- Google Cloud Professional Cloud Architect (2022)", page_number=2, reading_order=17),
        Block(block_id="b18", type="list_item", text="- Certified Kubernetes Administrator (CKA, 2021)", page_number=2, reading_order=18),
    ]
    sections = [
        Section(section_path=["Gokul Ramanathan"], content_refs=["b0", "b1", "b2", "b3"], page_range=[1, 1]),
        Section(section_path=["Professional Summary"], content_refs=["b4", "b5"], page_range=[1, 1]),
        Section(section_path=["Work Experience"], content_refs=["b6", "b7", "b8"], page_range=[1, 2]),
        Section(section_path=["Skills"], content_refs=["b9", "b10", "b11", "b12", "b13"], page_range=[2, 2]),
        Section(section_path=["Education"], content_refs=["b14", "b15"], page_range=[2, 2]),
        Section(section_path=["Certifications"], content_refs=["b16", "b17", "b18"], page_range=[2, 2]),
    ]
    pages = [
        Page(page_number=1, blocks=[b for b in blocks if b.page_number == 1]),
        Page(page_number=2, blocks=[b for b in blocks if b.page_number == 2]),
    ]
    return ExtractedDocumentJSON(document_id="resume_gokul", pages=pages, sections=sections)


def _invoice_doc():
    """Realistic business invoice."""
    blocks = [
        Block(block_id="b0", type="heading", text="INVOICE", page_number=1, reading_order=0),
        Block(block_id="b1", type="key_value", text="Invoice Number: INV-2024-0847", key="Invoice Number", value="INV-2024-0847", page_number=1, reading_order=1),
        Block(block_id="b2", type="key_value", text="Date: March 15, 2024", key="Date", value="March 15, 2024", page_number=1, reading_order=2),
        Block(block_id="b3", type="key_value", text="Due Date: April 14, 2024", key="Due Date", value="April 14, 2024", page_number=1, reading_order=3),
        Block(block_id="b4", type="key_value", text="Bill To: Acme Corporation", key="Bill To", value="Acme Corporation", page_number=1, reading_order=4),
        Block(block_id="b5", type="key_value", text="Bill From: TechSolutions Ltd.", key="Bill From", value="TechSolutions Ltd.", page_number=1, reading_order=5),
        Block(block_id="b6", type="key_value", text="Payment Terms: Net 30", key="Payment Terms", value="Net 30", page_number=1, reading_order=6),
    ]
    tables = [
        Table(
            table_id="t1",
            headers=["Description", "Quantity", "Unit Price", "Amount"],
            rows=[
                ["Cloud Infrastructure Setup", "1", "$15,000", "$15,000"],
                ["API Development (hours)", "120", "$150", "$18,000"],
                ["Security Audit", "1", "$5,000", "$5,000"],
                ["Training & Documentation", "40", "$100", "$4,000"],
            ],
            page_number=1,
        ),
    ]
    total_blocks = [
        Block(block_id="b7", type="key_value", text="Subtotal: $42,000", key="Subtotal", value="$42,000", page_number=1, reading_order=7),
        Block(block_id="b8", type="key_value", text="Tax (8.5%): $3,570", key="Tax (8.5%)", value="$3,570", page_number=1, reading_order=8),
        Block(block_id="b9", type="key_value", text="Total Due: $45,570", key="Total Due", value="$45,570", page_number=1, reading_order=9),
    ]
    blocks.extend(total_blocks)
    pages = [Page(page_number=1, blocks=blocks)]
    return ExtractedDocumentJSON(document_id="invoice_0847", pages=pages, tables=tables)


def _legal_contract_doc():
    """Realistic legal contract excerpt."""
    blocks = [
        Block(block_id="b0", type="heading", text="CONSULTING SERVICES AGREEMENT", page_number=1, reading_order=0),
        Block(block_id="b1", type="paragraph", text="This Consulting Services Agreement (the \"Agreement\") is entered into as of January 1, 2024 (the \"Effective Date\"), by and between Meridian Holdings Inc., a Delaware corporation with its principal office at 500 Technology Drive, Suite 400, San Francisco, CA 94105 (\"Client\"), and Dr. Priya Sharma, an individual residing at 2847 Oak Valley Road, Palo Alto, CA 94303 (\"Consultant\").", page_number=1, reading_order=1),
        Block(block_id="b2", type="heading", text="1. Services", page_number=1, reading_order=2),
        Block(block_id="b3", type="paragraph", text="Consultant shall provide strategic technology advisory services including but not limited to: architecture review, cloud migration planning, and technical due diligence for potential acquisitions. The Consultant shall devote a minimum of 20 hours per week to the Services.", page_number=1, reading_order=3),
        Block(block_id="b4", type="heading", text="2. Compensation", page_number=1, reading_order=4),
        Block(block_id="b5", type="paragraph", text="Client shall pay Consultant a monthly retainer of $25,000, payable on the first business day of each calendar month. In addition, Consultant shall be eligible for a performance bonus of up to $50,000 per quarter based on achievement of mutually agreed milestones.", page_number=1, reading_order=5),
        Block(block_id="b6", type="heading", text="3. Term and Termination", page_number=2, reading_order=6),
        Block(block_id="b7", type="paragraph", text="This Agreement shall commence on January 1, 2024 and continue for a period of twelve (12) months, unless earlier terminated. Either party may terminate this Agreement upon thirty (30) days' written notice. In the event of termination without cause, Client shall pay Consultant for all Services rendered through the date of termination plus a termination fee equal to two months' retainer ($50,000).", page_number=2, reading_order=7),
        Block(block_id="b8", type="heading", text="4. Confidentiality", page_number=2, reading_order=8),
        Block(block_id="b9", type="paragraph", text="Consultant agrees to maintain the confidentiality of all proprietary information disclosed by Client during the term of this Agreement and for a period of three (3) years following termination.", page_number=2, reading_order=9),
    ]
    sections = [
        Section(section_path=["CONSULTING SERVICES AGREEMENT"], content_refs=["b0", "b1"], page_range=[1, 1]),
        Section(section_path=["1. Services"], content_refs=["b2", "b3"], page_range=[1, 1]),
        Section(section_path=["2. Compensation"], content_refs=["b4", "b5"], page_range=[1, 1]),
        Section(section_path=["3. Term and Termination"], content_refs=["b6", "b7"], page_range=[2, 2]),
        Section(section_path=["4. Confidentiality"], content_refs=["b8", "b9"], page_range=[2, 2]),
    ]
    pages = [
        Page(page_number=1, blocks=[b for b in blocks if b.page_number == 1]),
        Page(page_number=2, blocks=[b for b in blocks if b.page_number == 2]),
    ]
    return ExtractedDocumentJSON(document_id="contract_meridian", pages=pages, sections=sections)


# ── STRUCTURE PARSER VALIDATION ───────────────────────────────────

class TestResumeStructureParsing:
    def test_resume_section_structure(self):
        doc = _resume_doc()
        result = parse_structure(doc, document_id="resume_gokul")
        unit_types = [u.unit_type for u in result.units]
        assert UnitType.KV_GROUP in unit_types, "Contact info KV group not detected"
        assert UnitType.PARAGRAPH in unit_types, "Work experience paragraph not detected"
        assert UnitType.LIST in unit_types, "Skills list not detected"

    def test_resume_heading_path_propagation(self):
        doc = _resume_doc()
        result = parse_structure(doc, document_id="resume_gokul")
        work_units = [u for u in result.units if any("Work Experience" in h for h in u.heading_path)]
        assert len(work_units) >= 1, "Work Experience heading path not propagated"

    def test_resume_no_text_dropped(self):
        doc = _resume_doc()
        result = parse_structure(doc, document_id="resume_gokul")
        all_block_text = " ".join(b.text or "" for p in doc.pages for b in p.blocks if b.type != "heading")
        all_unit_text = " ".join(u.text for u in result.units)
        # Key facts should survive
        assert "gokul.ramanathan@gmail.com" in all_unit_text
        assert "500 million events" in all_unit_text.lower() or "500M" in all_unit_text or "500m" in all_unit_text.lower()
        assert "Stanford University" in all_unit_text


class TestInvoiceStructureParsing:
    def test_invoice_kv_extraction(self):
        doc = _invoice_doc()
        result = parse_structure(doc, document_id="invoice_0847")
        kv_units = [u for u in result.units if u.unit_type == UnitType.KV_GROUP]
        assert len(kv_units) >= 1
        all_kv = {}
        for kv_unit in kv_units:
            if kv_unit.kv_pairs:
                all_kv.update(kv_unit.kv_pairs)
        assert all_kv.get("Invoice Number") == "INV-2024-0847"
        assert all_kv.get("Total Due") == "$45,570"

    def test_invoice_table_preservation(self):
        doc = _invoice_doc()
        result = parse_structure(doc, document_id="invoice_0847")
        table_units = [u for u in result.units if u.unit_type == UnitType.TABLE]
        assert len(table_units) == 1
        assert table_units[0].table_headers == ["Description", "Quantity", "Unit Price", "Amount"]
        assert len(table_units[0].table_rows) == 4
        assert table_units[0].table_rows[1]["Description"] == "API Development (hours)"


# ── ENTITY EXTRACTION VALIDATION ─────────────────────────────────

class TestResumeEntityExtraction:
    def test_extracts_person_name(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        result = extract_entities_and_facts(structured)
        person_ents = [e for e in result.entities if e.label == "PERSON"]
        person_names = [e.normalized for e in person_ents]
        # Must find the candidate's name
        assert any("gokul" in n for n in person_names), f"Gokul not found in persons: {person_names}"

    def test_extracts_organizations(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        result = extract_entities_and_facts(structured)
        org_ents = [e for e in result.entities if e.label == "ORG"]
        org_names = [e.normalized for e in org_ents]
        assert any("google" in n for n in org_names), f"Google not found in orgs: {org_names}"

    def test_extracts_email(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        result = extract_entities_and_facts(structured)
        email_ents = [e for e in result.entities if e.label == "EMAIL"]
        assert len(email_ents) >= 1, "Email not extracted"
        assert any("gokul" in e.normalized for e in email_ents)

    def test_extracts_kv_facts(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        result = extract_entities_and_facts(structured)
        assert len(result.kv_pairs) >= 3, f"Expected 3+ KV pairs, got {len(result.kv_pairs)}"
        kv_keys = [kv.get("key", "").lower() for kv in result.kv_pairs]
        assert any("email" in k for k in kv_keys), f"Email not in KV pairs: {kv_keys}"

    def test_no_hallucinated_entities(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        result = extract_entities_and_facts(structured)
        # Include heading paths in searchable text (entities can come from headings)
        all_text = " ".join(u.text.lower() for u in structured.units)
        heading_text = " ".join(h.lower() for u in structured.units for h in (u.heading_path or []))
        searchable = f"{all_text} {heading_text}"
        for ent in result.entities:
            assert ent.text.lower() in searchable, f"HALLUCINATED entity: '{ent.text}' not in source text or headings"

    def test_all_facts_have_provenance(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        result = extract_entities_and_facts(structured)
        for fact in result.facts:
            assert fact.raw_text, f"Fact {fact.fact_id} ({fact.predicate}) has no provenance"
            assert len(fact.raw_text) >= 5, f"Provenance too short: '{fact.raw_text}'"


class TestInvoiceEntityExtraction:
    def test_extracts_monetary_values(self):
        doc = _invoice_doc()
        structured = parse_structure(doc, document_id="invoice_0847")
        result = extract_entities_and_facts(structured)
        money_ents = [e for e in result.entities if e.label == "MONEY"]
        money_texts = [e.text for e in money_ents]
        # Should find at least some dollar amounts
        all_money_text = " ".join(money_texts)
        has_money = len(money_ents) > 0 or any("$" in kv.get("value", "") for kv in result.kv_pairs)
        assert has_money, f"No monetary values found. Entities: {[e.label + ':' + e.text for e in result.entities]}"

    def test_table_facts_extracted(self):
        doc = _invoice_doc()
        structured = parse_structure(doc, document_id="invoice_0847")
        result = extract_entities_and_facts(structured)
        assert len(result.tables_structured) >= 1, "Invoice table not structured"
        assert len(result.facts) >= 4, f"Expected 4+ table row facts, got {len(result.facts)}"


class TestContractEntityExtraction:
    def test_extracts_parties(self):
        doc = _legal_contract_doc()
        structured = parse_structure(doc, document_id="contract_meridian")
        result = extract_entities_and_facts(structured)
        all_ents = [e.normalized for e in result.entities]
        has_meridian = any("meridian" in n for n in all_ents)
        has_priya = any("priya" in n or "sharma" in n for n in all_ents)
        assert has_meridian or has_priya, f"Contract parties not found. Entities: {all_ents[:20]}"

    def test_extracts_dates(self):
        doc = _legal_contract_doc()
        structured = parse_structure(doc, document_id="contract_meridian")
        result = extract_entities_and_facts(structured)
        date_ents = [e for e in result.entities if e.label == "DATE"]
        assert len(date_ents) >= 1, "Contract dates not extracted"

    def test_extracts_monetary_terms(self):
        doc = _legal_contract_doc()
        structured = parse_structure(doc, document_id="contract_meridian")
        result = extract_entities_and_facts(structured)
        money_ents = [e for e in result.entities if e.label == "MONEY"]
        assert len(money_ents) >= 1, "Compensation terms not extracted"


# ── ENTITY RESOLUTION VALIDATION ─────────────────────────────────

class TestCrossDocumentResolution:
    def test_same_person_different_docs(self):
        """Gokul Ramanathan in resume should merge with G. Ramanathan in reference letter."""
        from src.docwain_intel.models import EntitySpan
        entities = [
            EntitySpan(entity_id="e1", text="Gokul Ramanathan", normalized="gokul ramanathan", label="PERSON", unit_id="u1", confidence=0.9, source="spacy"),
            EntitySpan(entity_id="e2", text="G. Ramanathan", normalized="g. ramanathan", label="PERSON", unit_id="u2", confidence=0.85, source="spacy"),
            EntitySpan(entity_id="e3", text="Dr. Priya Sharma", normalized="dr. priya sharma", label="PERSON", unit_id="u3", confidence=0.9, source="spacy"),
        ]
        resolved = resolve_entities(entities)
        ramanathan_ents = [e for e in resolved if "ramanathan" in e.normalized]
        assert len(ramanathan_ents) == 1, f"Expected 1 merged Ramanathan, got {len(ramanathan_ents)}: {[e.text for e in ramanathan_ents]}"


# ── DOCUMENT FINGERPRINT VALIDATION ──────────────────────────────

class TestFingerprintAccuracy:
    def test_resume_fingerprint(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        extraction = extract_entities_and_facts(structured)
        fp = compute_fingerprint(structured, extraction)
        assert fp.entity_density > 0, "Resume should have high entity density"
        assert "PERSON" in fp.entity_distribution, "Resume should have PERSON entities"
        assert fp.structure_profile.get("list", 0) > 0, "Resume should have list sections"

    def test_invoice_fingerprint(self):
        doc = _invoice_doc()
        structured = parse_structure(doc, document_id="invoice_0847")
        extraction = extract_entities_and_facts(structured)
        fp = compute_fingerprint(structured, extraction)
        assert fp.numeric_density > 0.05, f"Invoice should have high numeric density, got {fp.numeric_density}"
        assert fp.structure_profile.get("kv_group", 0) > 0, "Invoice should have KV groups"

    def test_contract_formality(self):
        doc = _legal_contract_doc()
        structured = parse_structure(doc, document_id="contract_meridian")
        extraction = extract_entities_and_facts(structured)
        fp = compute_fingerprint(structured, extraction)
        assert fp.formality_score >= 0.5, f"Legal contract should be formal, got {fp.formality_score}"


# ── VERIFICATION VALIDATION ──────────────────────────────────────

class TestVerificationOnRealDocs:
    def test_resume_passes_verification(self):
        doc = _resume_doc()
        structured = parse_structure(doc, document_id="resume_gokul")
        extraction = extract_entities_and_facts(structured)
        verification = verify_extraction(extraction, structured)
        assert verification.quality_score >= 0.5, f"Resume quality too low: {verification.quality_score}"

    def test_invoice_passes_verification(self):
        doc = _invoice_doc()
        structured = parse_structure(doc, document_id="invoice_0847")
        extraction = extract_entities_and_facts(structured)
        verification = verify_extraction(extraction, structured)
        assert verification.quality_score >= 0.5, f"Invoice quality too low: {verification.quality_score}"


# ── FULL PIPELINE END-TO-END ─────────────────────────────────────

class TestEndToEndPipeline:
    def test_resume_pipeline(self):
        result = process_document(
            extracted_doc=_resume_doc(),
            document_id="resume_gokul",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.stage_reached == PipelineStage.VERIFIED.value, f"Pipeline stopped at {result.stage_reached}: {result.error}"
        assert len(result.extraction.entities) >= 5, f"Too few entities: {len(result.extraction.entities)}"
        assert len(result.extraction.facts) >= 3, f"Too few facts: {len(result.extraction.facts)}"
        assert result.fingerprint.entity_density > 0

    def test_invoice_pipeline(self):
        result = process_document(
            extracted_doc=_invoice_doc(),
            document_id="invoice_0847",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.stage_reached == PipelineStage.VERIFIED.value, f"Pipeline stopped at {result.stage_reached}: {result.error}"
        assert len(result.extraction.kv_pairs) >= 5

    def test_contract_pipeline(self):
        result = process_document(
            extracted_doc=_legal_contract_doc(),
            document_id="contract_meridian",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        assert result.stage_reached == PipelineStage.VERIFIED.value, f"Pipeline stopped at {result.stage_reached}: {result.error}"
        assert len(result.extraction.entities) >= 3

    def test_pipeline_timing(self):
        """Pipeline should complete in reasonable time (no LLM calls)."""
        import time
        start = time.monotonic()
        result = process_document(
            extracted_doc=_resume_doc(),
            document_id="resume_gokul",
            subscription_id="sub_001",
            profile_id="prof_001",
        )
        elapsed = time.monotonic() - start
        assert elapsed < 30, f"Pipeline took {elapsed:.1f}s — too slow for ingestion"
        total_stage_time = sum(result.stage_timings.values())
        assert total_stage_time < 30, f"Cumulative stage time {total_stage_time:.1f}s too high"


# ── QUERY ROUTING VALIDATION ─────────────────────────────────────

class TestQueryRoutingAccuracy:
    def test_entity_lookup_queries(self):
        queries = [
            "What is Gokul Ramanathan's email?",
            "Show me the invoice total",
            "What is the contract value?",
        ]
        for q in queries:
            analysis = route_query(q)
            assert analysis.route in (QueryRoute.GRAPH_DIRECT, QueryRoute.HYBRID_SEARCH), \
                f"'{q}' should route to graph/hybrid, got {analysis.route}"

    def test_comparison_queries(self):
        queries = [
            "Compare Gokul Ramanathan and Priya Sharma",
            "What are the differences between the two candidates?",
        ]
        for q in queries:
            analysis = route_query(q)
            assert analysis.route in (QueryRoute.HYBRID_SEARCH, QueryRoute.FULL_SEARCH), \
                f"'{q}' should route to hybrid/full, got {analysis.route}"

    def test_reasoning_queries(self):
        queries = [
            "Based on the resumes, who would be the best candidate for a senior role and why?",
            "Analyze the risk factors in this contract and recommend changes",
        ]
        for q in queries:
            analysis = route_query(q)
            assert analysis.route == QueryRoute.LLM_GENERATION, \
                f"'{q}' should route to LLM, got {analysis.route}"


# ── RESPONSE ASSEMBLY VALIDATION ─────────────────────────────────

class TestResponseAssemblyQuality:
    def test_entity_response_completeness(self):
        facts = [
            {"subject": "Gokul Ramanathan", "predicate": "EMAIL", "value": "gokul.ramanathan@gmail.com", "source": "resume.pdf", "page": 1},
            {"subject": "Gokul Ramanathan", "predicate": "PHONE", "value": "+1-408-555-0142", "source": "resume.pdf", "page": 1},
            {"subject": "Gokul Ramanathan", "predicate": "LOCATION", "value": "San Jose, CA", "source": "resume.pdf", "page": 1},
        ]
        response = assemble_response(
            query="What is Gokul's contact information?",
            route=QueryRoute.GRAPH_DIRECT,
            facts=facts,
        )
        assert "gokul.ramanathan@gmail.com" in response.text
        assert "+1-408-555-0142" in response.text
        assert "San Jose" in response.text
        assert len(response.sources) >= 1
        assert response.confidence > 0

    def test_comparison_table_structure(self):
        facts = [
            {"subject": "Gokul Ramanathan", "predicate": "EXPERIENCE", "value": "8 years", "source": "gokul_resume.pdf", "page": 1},
            {"subject": "Gokul Ramanathan", "predicate": "ROLE", "value": "Senior Engineer", "source": "gokul_resume.pdf", "page": 1},
            {"subject": "Priya Sharma", "predicate": "EXPERIENCE", "value": "10 years", "source": "priya_resume.pdf", "page": 1},
            {"subject": "Priya Sharma", "predicate": "ROLE", "value": "Consultant", "source": "priya_resume.pdf", "page": 1},
        ]
        response = assemble_response(
            query="Compare Gokul and Priya",
            route=QueryRoute.HYBRID_SEARCH,
            facts=facts,
            is_comparison=True,
        )
        assert "|" in response.text, "Comparison should produce markdown table"
        assert "Gokul" in response.text
        assert "Priya" in response.text
        assert "8 years" in response.text
        assert "10 years" in response.text
