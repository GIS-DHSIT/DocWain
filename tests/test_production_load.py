"""Production load tests for the DocWain intelligence pipeline.

Tests large documents, concurrent processing, memory stability,
and throughput for the 6-stage intel pipeline.
"""
from __future__ import annotations

import gc
import random
import string
import time
import tracemalloc
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pytest

from src.docwain_intel.models import (
    Block,
    ExtractedDocumentJSON,
    Page,
    Section,
    Table,
)
from src.docwain_intel.structure_parser import parse_structure
from src.docwain_intel.entity_engine import extract_entities_and_facts
from src.docwain_intel.entity_resolver import resolve_entities
from src.docwain_intel.intel_pipeline import process_document
from src.docwain_intel.query_router import route_query


# ---------------------------------------------------------------------------
# Domain-specific text generators
# ---------------------------------------------------------------------------

_RESUME_PARAGRAPHS = [
    "John Smith is a senior software engineer with over 10 years of experience "
    "in distributed systems, cloud architecture, and machine learning. He holds "
    "a Master of Science degree from Stanford University and a Bachelor of Science "
    "from MIT. His email is john.smith@example.com and phone is +1-555-123-4567.",

    "At Google LLC from 2018 to 2023, he led a team of 12 engineers building "
    "real-time data pipelines processing 500 million events per day using Apache "
    "Kafka and Apache Flink. He reduced infrastructure costs by 35% through "
    "auto-scaling optimizations and container orchestration with Kubernetes.",

    "Previously at Amazon Web Services from 2014 to 2018, he designed and "
    "implemented a microservices architecture serving 200,000 requests per second "
    "with 99.99% availability. He earned the AWS Solutions Architect Professional "
    "certification and mentored 8 junior developers.",

    "Technical skills include Python, Java, Go, TypeScript, React, PostgreSQL, "
    "MongoDB, Redis, Docker, Kubernetes, Terraform, AWS, GCP, Azure, TensorFlow, "
    "PyTorch, scikit-learn, and Apache Spark. He is proficient in agile "
    "methodologies and test-driven development.",

    "He published 3 papers on distributed consensus algorithms at IEEE conferences "
    "and holds 2 patents related to real-time anomaly detection in streaming data. "
    "He volunteers as a mentor at Code.org and speaks at PyCon annually.",
]

_INVOICE_PARAGRAPHS = [
    "Invoice Number: INV-2024-0847. Date: March 15, 2024. Due Date: April 14, 2024. "
    "Bill To: Acme Corporation, 123 Business Drive, Suite 400, San Francisco, CA 94102. "
    "Contact: Sarah Johnson, Accounts Payable, sarah.johnson@acme.com.",

    "Item 1: Cloud Infrastructure Services - Annual License. Quantity: 1. "
    "Unit Price: $45,000.00. Total: $45,000.00. Description: Enterprise tier "
    "cloud platform with 99.99% SLA, 24/7 support, and dedicated account manager.",

    "Item 2: Professional Services - Implementation. Quantity: 160 hours. "
    "Rate: $250.00/hour. Total: $40,000.00. Description: Custom integration "
    "with existing SAP ERP system, data migration, and user training sessions.",

    "Item 3: Data Analytics Module. Quantity: 50 user licenses. "
    "Unit Price: $120.00/user/month. Total: $72,000.00. Billed annually. "
    "Includes real-time dashboards, predictive analytics, and custom reports.",

    "Subtotal: $157,000.00. Tax (8.5%): $13,345.00. Grand Total: $170,345.00. "
    "Payment Terms: Net 30. Bank: First National Bank, Account: 9876543210, "
    "Routing: 021000021. Late fee: 1.5% per month on outstanding balance.",
]

_LEGAL_PARAGRAPHS = [
    "This Master Services Agreement (hereinafter referred to as the 'Agreement') "
    "is entered into as of the 1st day of January, 2024, by and between TechCorp "
    "International Inc., a Delaware corporation with its principal offices at "
    "456 Innovation Blvd, Austin, TX 78701 ('Provider'), and GlobalBank Holdings "
    "PLC, a corporation organized under the laws of England ('Client').",

    "WHEREAS, Provider possesses specialized expertise in artificial intelligence "
    "and machine learning solutions; and WHEREAS, Client desires to engage Provider "
    "to develop and deploy certain AI-powered document processing systems pursuant "
    "to the terms and conditions set forth herein; NOW, THEREFORE, in consideration "
    "of the mutual covenants contained herein, the parties agree as follows.",

    "Section 3.1 Confidentiality. Each party shall hold in strict confidence all "
    "Confidential Information disclosed by the other party. 'Confidential Information' "
    "means any non-public technical, business, or financial information, including "
    "but not limited to trade secrets, algorithms, customer lists, pricing structures, "
    "and strategic plans. This obligation shall survive for 5 years after termination.",

    "Section 5.2 Limitation of Liability. IN NO EVENT SHALL EITHER PARTY BE LIABLE "
    "FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, "
    "INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS, DATA, OR BUSINESS OPPORTUNITIES, "
    "ARISING OUT OF OR RELATED TO THIS AGREEMENT, REGARDLESS OF THE THEORY OF "
    "LIABILITY. Provider's total aggregate liability shall not exceed $2,000,000.",

    "Section 8.1 Governing Law. This Agreement shall be governed by and construed "
    "in accordance with the laws of the State of Delaware, without regard to its "
    "conflict of laws principles. Any dispute arising under this Agreement shall "
    "be resolved through binding arbitration administered by the American Arbitration "
    "Association in accordance with its Commercial Arbitration Rules.",
]

_MEDICAL_PARAGRAPHS = [
    "Patient Name: Robert Chen. Date of Birth: 05/12/1975. MRN: MED-2024-88432. "
    "Visit Date: February 28, 2024. Referring Physician: Dr. Emily Watson, MD. "
    "Department: Internal Medicine. Insurance: BlueCross BlueShield PPO, "
    "Policy #BC-449821.",

    "Chief Complaint: Patient presents with persistent chest pain for 3 days, "
    "radiating to the left arm, accompanied by shortness of breath and diaphoresis. "
    "Past Medical History: Hypertension diagnosed 2018, Type 2 Diabetes Mellitus "
    "diagnosed 2020, Hyperlipidemia. Current Medications: Metformin 1000mg BID, "
    "Lisinopril 20mg daily, Atorvastatin 40mg daily.",

    "Physical Examination: Blood Pressure 158/94 mmHg. Heart Rate 92 bpm. "
    "Temperature 98.6F. SpO2 96% on room air. Cardiac auscultation reveals "
    "regular rate and rhythm with no murmurs, gallops, or rubs. Lungs clear "
    "to auscultation bilaterally. No peripheral edema noted.",

    "Laboratory Results: Troponin I 0.08 ng/mL (elevated, normal <0.04). "
    "BNP 245 pg/mL (elevated). CBC within normal limits. BMP shows glucose "
    "187 mg/dL, creatinine 1.1 mg/dL. Lipid panel: Total cholesterol 242, "
    "LDL 165, HDL 38, Triglycerides 195.",

    "Assessment and Plan: 1. Acute coronary syndrome - admit to telemetry, "
    "serial troponins q6h, cardiology consult for stress test vs catheterization. "
    "2. Uncontrolled diabetes - adjust Metformin, consider adding GLP-1 agonist. "
    "3. Hypertension - increase Lisinopril to 40mg. Follow-up in 2 weeks with "
    "Dr. Watson. Patient educated on warning signs requiring emergency care.",
]

_DOMAIN_TEXTS = {
    "resume": _RESUME_PARAGRAPHS,
    "invoice": _INVOICE_PARAGRAPHS,
    "legal": _LEGAL_PARAGRAPHS,
    "medical": _MEDICAL_PARAGRAPHS,
}


def _generate_realistic_text(domain: str) -> str:
    """Return domain-appropriate text by cycling through paragraph templates."""
    paragraphs = _DOMAIN_TEXTS.get(domain, _RESUME_PARAGRAPHS)
    return "\n\n".join(paragraphs)


def _random_text(length: int = 200) -> str:
    """Generate pseudo-random filler text of approximately *length* chars."""
    words = [
        "the", "company", "reported", "quarterly", "revenue", "of",
        "$12.5", "million", "representing", "a", "15%", "increase",
        "year", "over", "from", "previous", "fiscal", "operations",
        "management", "strategic", "initiatives", "delivered", "growth",
        "across", "all", "business", "segments", "including", "technology",
        "services", "and", "consulting", "divisions", "international",
        "market", "expansion", "contributed", "to", "overall", "performance",
        "employee", "headcount", "grew", "by", "200", "reaching", "5000",
        "total", "staff", "members", "globally",
    ]
    result: List[str] = []
    while len(" ".join(result)) < length:
        result.append(random.choice(words))
    return " ".join(result)


# ---------------------------------------------------------------------------
# Document generators
# ---------------------------------------------------------------------------

def _generate_document(
    num_pages: int,
    blocks_per_page: int,
    *,
    block_types: List[str] | None = None,
    domain: str = "resume",
) -> ExtractedDocumentJSON:
    """Create an ExtractedDocumentJSON with the specified page/block counts.

    Each page gets *blocks_per_page* blocks. Block types cycle through
    *block_types* (default: paragraph only). Text content is drawn from
    the domain-specific paragraph pool and padded with random filler
    to ensure uniqueness.
    """
    if block_types is None:
        block_types = ["paragraph"]

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    paragraphs = _DOMAIN_TEXTS.get(domain, _RESUME_PARAGRAPHS)
    pages: List[Page] = []

    for page_num in range(1, num_pages + 1):
        blocks: List[Block] = []
        for blk_idx in range(blocks_per_page):
            blk_type = block_types[blk_idx % len(block_types)]
            blk_id = f"blk_{page_num}_{blk_idx}"
            text = paragraphs[(page_num + blk_idx) % len(paragraphs)]
            # Add unique filler to avoid dedup collisions
            text += f" [Page {page_num}, block {blk_idx}]"

            key = None
            value = None
            if blk_type == "key_value":
                key = f"Field_{page_num}_{blk_idx}"
                value = f"Value for field {page_num}-{blk_idx}: {_random_text(60)}"
                text = f"{key}: {value}"

            blocks.append(Block(
                block_id=blk_id,
                type=blk_type,
                text=text,
                key=key,
                value=value,
                page_number=page_num,
                reading_order=blk_idx,
            ))
        pages.append(Page(page_number=page_num, blocks=blocks))

    return ExtractedDocumentJSON(
        document_id=doc_id,
        pages=pages,
    )


def _generate_kv_document(num_pairs: int) -> ExtractedDocumentJSON:
    """Create a document composed entirely of key-value blocks."""
    doc_id = f"doc_kv_{uuid.uuid4().hex[:8]}"
    blocks_per_page = 20
    pages: List[Page] = []
    pair_idx = 0

    while pair_idx < num_pairs:
        page_num = len(pages) + 1
        blocks: List[Block] = []
        for _ in range(min(blocks_per_page, num_pairs - pair_idx)):
            key = f"Field_{pair_idx:04d}"
            value = f"Value_{pair_idx}: " + _random_text(80)
            blocks.append(Block(
                block_id=f"kv_{pair_idx}",
                type="key_value",
                text=f"{key}: {value}",
                key=key,
                value=value,
                page_number=page_num,
                reading_order=pair_idx % blocks_per_page,
            ))
            pair_idx += 1
        pages.append(Page(page_number=page_num, blocks=blocks))

    return ExtractedDocumentJSON(document_id=doc_id, pages=pages)


def _generate_table_document(
    num_rows: int, num_cols: int
) -> ExtractedDocumentJSON:
    """Create a document with a single large table."""
    doc_id = f"doc_tbl_{uuid.uuid4().hex[:8]}"
    headers = [f"Column_{c}" for c in range(num_cols)]
    rows: List[List[str]] = []
    for r in range(num_rows):
        row = [f"R{r}C{c}_{_random_text(20)}" for c in range(num_cols)]
        rows.append(row)

    table = Table(
        table_id="tbl_0",
        headers=headers,
        rows=rows,
        page_number=1,
    )
    # Add a minimal page so the doc isn't empty
    page = Page(page_number=1, blocks=[
        Block(
            block_id="blk_summary",
            type="paragraph",
            text="This document contains a large data table with financial records.",
            page_number=1,
            reading_order=0,
        ),
    ])
    return ExtractedDocumentJSON(
        document_id=doc_id,
        pages=[page],
        tables=[table],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestProductionLoad:
    """Production load tests for the intel pipeline."""

    def test_large_document_100_pages(self):
        """100-page document with 10 blocks/page through structure_parser.

        Asserts completion in <30s and correct unit count.
        """
        doc = _generate_document(100, 10, domain="resume")
        total_blocks = sum(len(p.blocks) for p in doc.pages)
        assert total_blocks == 1000

        start = time.monotonic()
        structured = parse_structure(doc, document_id=doc.document_id)
        elapsed = time.monotonic() - start

        assert elapsed < 30.0, f"Structure parsing took {elapsed:.1f}s (limit 30s)"
        # Every paragraph block should produce at least one unit (paragraphs
        # are merged by adjacency, so unit count <= block count).
        assert structured.unit_count > 0
        assert structured.total_chars > 0
        # Sanity: at least 50 units from 1000 blocks (merging reduces count)
        assert structured.unit_count >= 50, (
            f"Expected >=50 units from 1000 blocks, got {structured.unit_count}"
        )

    def test_large_document_500_blocks(self):
        """500 blocks with mixed types — verify all blocks are processed."""
        mixed_types = ["paragraph", "list_item", "key_value", "heading"]
        doc = _generate_document(50, 10, block_types=mixed_types, domain="invoice")

        total_blocks = sum(len(p.blocks) for p in doc.pages)
        assert total_blocks == 500

        structured = parse_structure(doc, document_id=doc.document_id)
        assert structured.unit_count > 0
        # Check that different unit types are present
        unit_types = {u.unit_type.value for u in structured.units}
        # Headings are consumed without creating units; the others should appear
        assert len(unit_types) >= 2, (
            f"Expected multiple unit types, got: {unit_types}"
        )
        assert structured.total_chars > 0

    @pytest.mark.slow
    def test_concurrent_pipeline_processing(self):
        """Process 5 documents concurrently — no race conditions."""
        domains = ["resume", "invoice", "legal", "medical", "resume"]
        docs = [
            _generate_document(10, 5, domain=d)
            for d in domains
        ]

        results = []

        def _run(doc_data):
            return process_document(
                extracted_doc=doc_data,
                document_id=doc_data.document_id,
                subscription_id="sub_load_test",
                profile_id="prof_load_test",
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_run, d): i for i, d in enumerate(docs)}
            for future in as_completed(futures):
                idx = futures[future]
                res = future.result()
                results.append((idx, res))

        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        for idx, res in results:
            assert res.error is None, (
                f"Document {idx} failed: {res.error}"
            )
            assert res.stage_reached != "FAILED", (
                f"Document {idx} reached FAILED stage"
            )
            assert res.structured_doc is not None
            assert res.extraction is not None
            assert res.fingerprint is not None

    @pytest.mark.slow
    def test_entity_extraction_large_text(self):
        """Single unit with 10K+ chars — entity extraction completes without OOM."""
        # Build a long realistic text (>10K chars)
        long_text = ""
        for domain in ("resume", "legal", "medical", "invoice"):
            long_text += _generate_realistic_text(domain) + "\n\n"
        # Pad to ensure >10K
        while len(long_text) < 10_000:
            long_text += _random_text(500) + "\n"
        assert len(long_text) >= 10_000

        doc = ExtractedDocumentJSON(
            document_id="doc_large_text",
            pages=[Page(
                page_number=1,
                blocks=[Block(
                    block_id="blk_0",
                    type="paragraph",
                    text=long_text,
                    page_number=1,
                    reading_order=0,
                )],
            )],
        )

        structured = parse_structure(doc, document_id=doc.document_id)
        assert structured.unit_count >= 1

        start = time.monotonic()
        extraction = extract_entities_and_facts(structured)
        elapsed = time.monotonic() - start

        # Should complete and find entities from the rich text
        assert len(extraction.entities) > 0, "Expected entities from 10K+ chars of text"
        assert elapsed < 60.0, f"Entity extraction took {elapsed:.1f}s (limit 60s)"

    def test_many_kv_pairs(self):
        """Document with 200+ key-value pairs — all extracted as facts."""
        num_pairs = 220
        doc = _generate_kv_document(num_pairs)

        structured = parse_structure(doc, document_id=doc.document_id)
        # KV blocks get grouped; check that kv_pairs are populated
        kv_units = [u for u in structured.units if u.kv_pairs]
        total_kv = sum(len(u.kv_pairs) for u in kv_units)
        assert total_kv == num_pairs, (
            f"Expected {num_pairs} KV pairs in units, got {total_kv}"
        )

        extraction = extract_entities_and_facts(structured)
        # Each KV pair should produce at least one fact
        kv_facts = [
            f for f in extraction.facts
            if f.extraction_method == "kv_structured"
        ]
        assert len(kv_facts) >= num_pairs, (
            f"Expected >= {num_pairs} kv_structured facts, got {len(kv_facts)}"
        )

    def test_large_table_extraction(self):
        """Table with 50+ rows and 10 columns — all rows processed."""
        num_rows = 55
        num_cols = 10
        doc = _generate_table_document(num_rows, num_cols)

        structured = parse_structure(doc, document_id=doc.document_id)
        table_units = [u for u in structured.units if u.table_rows is not None]
        assert len(table_units) >= 1, "Expected at least one table unit"

        total_rows = sum(len(u.table_rows) for u in table_units)
        assert total_rows == num_rows, (
            f"Expected {num_rows} table rows, got {total_rows}"
        )

        extraction = extract_entities_and_facts(structured)
        table_facts = [
            f for f in extraction.facts
            if f.extraction_method == "table_structured"
        ]
        # Each row has num_cols cells; some may be empty but most should produce facts
        assert len(table_facts) >= num_rows * (num_cols - 1), (
            f"Expected >= {num_rows * (num_cols - 1)} table facts, got {len(table_facts)}"
        )

        # Check structured table output
        assert len(extraction.tables_structured) >= 1
        tbl = extraction.tables_structured[0]
        assert tbl["row_count"] == num_rows
        assert len(tbl["headers"]) == num_cols

    @pytest.mark.slow
    def test_pipeline_memory_stability(self):
        """Process 10 documents sequentially — no memory leak (< 2x growth)."""
        tracemalloc.start()
        gc.collect()

        # Warm up: run one document to load spaCy etc.
        warmup_doc = _generate_document(5, 3, domain="resume")
        process_document(
            extracted_doc=warmup_doc,
            document_id=warmup_doc.document_id,
            subscription_id="sub_mem",
            profile_id="prof_mem",
        )
        gc.collect()

        # Take baseline snapshot
        snapshot_before = tracemalloc.take_snapshot()
        baseline_current, _ = tracemalloc.get_traced_memory()

        for i in range(10):
            domain = ["resume", "invoice", "legal", "medical"][i % 4]
            doc = _generate_document(10, 5, domain=domain)
            result = process_document(
                extracted_doc=doc,
                document_id=doc.document_id,
                subscription_id="sub_mem",
                profile_id=f"prof_mem_{i}",
            )
            assert result.error is None, f"Iteration {i} failed: {result.error}"

        gc.collect()
        final_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Allow baseline to be at least 1 MB to avoid division by near-zero
        effective_baseline = max(baseline_current, 1_000_000)
        growth_ratio = final_current / effective_baseline

        assert growth_ratio < 2.0, (
            f"Memory grew {growth_ratio:.2f}x "
            f"(baseline={baseline_current / 1e6:.1f}MB, "
            f"final={final_current / 1e6:.1f}MB, "
            f"peak={peak / 1e6:.1f}MB)"
        )

    @pytest.mark.slow
    def test_mixed_document_types(self):
        """Process resume, invoice, legal, medical — each gets appropriate tags."""
        domains = ["resume", "invoice", "legal", "medical"]
        results: Dict[str, object] = {}

        for domain in domains:
            doc = _generate_document(10, 8, domain=domain)
            result = process_document(
                extracted_doc=doc,
                document_id=doc.document_id,
                subscription_id="sub_mixed",
                profile_id="prof_mixed",
            )
            assert result.error is None, f"{domain} pipeline failed: {result.error}"
            assert result.fingerprint is not None, f"{domain} missing fingerprint"
            results[domain] = result

        # Each document should have a meaningful fingerprint (entity_distribution
        # and structure_profile are always populated; auto_tags may be empty
        # depending on the spaCy model available).
        for domain in domains:
            fp = results[domain].fingerprint
            assert len(fp.entity_distribution) > 0, (
                f"{domain} document has empty entity_distribution"
            )

        # Fingerprints should differ across domains — compare entity distributions
        all_entity_dists = [
            frozenset(results[d].fingerprint.entity_distribution.keys())
            for d in domains
        ]
        unique_dists = len(set(all_entity_dists))
        assert unique_dists >= 2, (
            "Expected at least 2 distinct entity distributions across 4 domains, "
            f"got {unique_dists}"
        )

        # Structure profiles should all have content
        for domain in domains:
            fp = results[domain].fingerprint
            assert sum(fp.structure_profile.values()) > 0, (
                f"{domain} has empty structure_profile"
            )

    @pytest.mark.slow
    def test_entity_resolver_many_entities(self):
        """500+ entities with duplicates — resolver runs in <5s, deduplicates."""
        from src.docwain_intel.models import EntitySpan

        base_names = [
            ("John Smith", "PERSON"),
            ("Dr. John Smith", "PERSON"),
            ("J. Smith", "PERSON"),
            ("Acme Corporation", "ORG"),
            ("Acme Corp.", "ORG"),
            ("ACME Corp", "ORG"),
            ("San Francisco", "GPE"),
            ("San Francisco, CA", "GPE"),
            ("Google LLC", "ORG"),
            ("Google", "ORG"),
            ("Stanford University", "ORG"),
            ("Stanford Univ.", "ORG"),
            ("Robert Chen", "PERSON"),
            ("Dr. Robert Chen", "PERSON"),
            ("R. Chen", "PERSON"),
            ("Emily Watson", "PERSON"),
            ("Dr. Emily Watson", "PERSON"),
            ("TechCorp International Inc.", "ORG"),
            ("TechCorp International", "ORG"),
            ("GlobalBank Holdings PLC", "ORG"),
            ("GlobalBank Holdings", "ORG"),
        ]

        entities: List[EntitySpan] = []
        for i in range(520):
            name, label = base_names[i % len(base_names)]
            entities.append(EntitySpan(
                entity_id=f"ent_{i:04d}",
                text=name,
                normalized=name.lower().strip(),
                label=label,
                unit_id=f"unit_{i % 50}",
                confidence=round(random.uniform(0.6, 0.95), 2),
                source="spacy",
            ))

        assert len(entities) >= 500

        start = time.monotonic()
        resolved = resolve_entities(entities)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Entity resolution took {elapsed:.1f}s (limit 5s)"
        # Should have far fewer entities after dedup
        assert len(resolved) < len(entities), (
            f"Expected dedup to reduce {len(entities)} entities, got {len(resolved)}"
        )
        # At minimum, duplicates of exact same name+label should merge
        assert len(resolved) <= len(base_names), (
            f"Expected at most {len(base_names)} unique clusters, got {len(resolved)}"
        )
        # Check that aliases were populated for merged entities
        entities_with_aliases = [e for e in resolved if len(e.aliases) > 0]
        assert len(entities_with_aliases) > 0, "Expected some entities to have aliases"

    def test_query_router_throughput(self):
        """Route 100 diverse queries — verify throughput and correctness.

        Allows up to 5s total (spaCy NLP runs per query). The primary
        goal is correctness and route diversity, not sub-second latency.
        """
        # Warm up spaCy model loading (not counted in timing)
        route_query("warmup query")

        queries = [
            # Factual / entity lookups
            "What is John Smith's email address?",
            "Show me the invoice total for INV-2024-0847",
            "What medications is the patient taking?",
            "List all certifications",
            # Comparisons
            "Compare John Smith and Jane Doe's experience",
            "What are the differences between Provider and Client obligations?",
            # Aggregations
            "How many employees does the company have?",
            "What is the total revenue for Q3?",
            "Count all invoices over $50,000",
            # Reasoning
            "Who is the best candidate for the senior engineer role?",
            "Should we renew this contract based on the terms?",
            "Explain the liability limitations in this agreement",
            "Why was the patient prescribed Metformin?",
            # Broad scope
            "Summarize all documents",
            "Give me an overview of the medical records",
            "Outline the key terms of the agreement",
            # Conversational
            "Hello",
            "Thanks",
            "OK",
            # Numeric constraints
            "Find candidates with more than 5 years experience",
            "Show invoices under $10,000",
            "List items priced between $100 and $500",
            # Complex / long
            "Based on the employment history, technical skills, and certifications, "
            "evaluate whether the candidate meets the requirements for a principal "
            "engineer position at a Fortune 500 technology company",
            "Analyze the medical test results and medication history to determine "
            "if the current treatment plan is effective for managing the patient's "
            "cardiovascular risk factors",
        ]

        # Pad to 100 queries by cycling
        while len(queries) < 100:
            queries.append(queries[len(queries) % len(queries[:24])])

        assert len(queries) >= 100

        start = time.monotonic()
        results = [route_query(q) for q in queries]
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Routing 100 queries took {elapsed:.2f}s (limit 5s)"
        assert len(results) == len(queries)

        # Verify routes are populated and diverse
        routes_seen = {r.route for r in results}
        assert len(routes_seen) >= 2, (
            f"Expected diverse routes, got only: {routes_seen}"
        )
        # Every result should have a valid route and confidence
        for i, res in enumerate(results):
            assert res.route is not None, f"Query {i} has no route"
            assert 0.0 <= res.confidence <= 1.0, (
                f"Query {i} confidence out of range: {res.confidence}"
            )


class TestStructureParserEdgeCases:
    """Non-slow edge-case tests for structure parsing under moderate load."""

    def test_empty_document(self):
        """Empty document produces empty structured output."""
        doc = ExtractedDocumentJSON(document_id="doc_empty", pages=[])
        structured = parse_structure(doc, document_id="doc_empty")
        assert structured.unit_count == 0
        assert structured.total_chars == 0
        assert len(structured.units) == 0

    def test_single_page_many_blocks(self):
        """Single page with 100 blocks."""
        blocks = []
        for i in range(100):
            blocks.append(Block(
                block_id=f"blk_{i}",
                type="paragraph",
                text=f"This is paragraph block number {i} with enough text to pass the minimum character threshold for semantic units.",
                page_number=1,
                reading_order=i,
            ))
        doc = ExtractedDocumentJSON(
            document_id="doc_single_page",
            pages=[Page(page_number=1, blocks=blocks)],
        )
        structured = parse_structure(doc, document_id="doc_single_page")
        assert structured.unit_count >= 1
        assert structured.total_chars > 0

    def test_heading_only_document(self):
        """Document with only heading blocks produces no units (headings are consumed)."""
        blocks = [
            Block(
                block_id=f"h_{i}",
                type="heading",
                text=f"Section {i}: Important Topic",
                page_number=1,
                reading_order=i,
            )
            for i in range(20)
        ]
        doc = ExtractedDocumentJSON(
            document_id="doc_headings",
            pages=[Page(page_number=1, blocks=blocks)],
        )
        structured = parse_structure(doc, document_id="doc_headings")
        # Headings are consumed; no content blocks means no units
        assert structured.unit_count == 0
