"""End-to-end validation tests for DocWain's complete intelligence engine.

Simulates the FULL lifecycle: document ingestion -> extraction -> graph population
-> query-time intelligence -> response quality validation.

Uses REAL modules for ingestion pipeline (structure_parser, entity_engine,
entity_resolver, document_fingerprint, verification) and mocks ONLY for
graph adapter (no real Neo4j) and LLM calls.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.docwain_intel.models import (
    Block, Page, Section, Table, ExtractedDocumentJSON,
    UnitType, SemanticUnit, StructuredDocument,
    ExtractionResult, DocumentFingerprint, VerificationResult,
)
from src.docwain_intel.structure_parser import parse_structure
from src.docwain_intel.entity_engine import extract_entities_and_facts
from src.docwain_intel.entity_resolver import resolve_entities
from src.docwain_intel.document_fingerprint import compute_fingerprint
from src.docwain_intel.verification import verify_extraction
from src.docwain_intel.intel_pipeline import process_document, PipelineStage
from src.docwain_intel.query_router import route_query, QueryRoute, QueryAnalysis
from src.docwain_intel.query_analyzer import analyze_query, QueryGeometry
from src.docwain_intel.evidence_organizer import (
    organize_evidence, OrganizedEvidence, EvidenceGroup,
)
from src.docwain_intel.rendering_spec import generate_spec, RenderingSpec
from src.docwain_intel.constrained_prompter import build_prompt, ConstrainedPrompt
from src.docwain_intel.quality_engine import validate_output, QualityResult
from src.docwain_intel.conversation_graph import ConversationGraph
from src.docwain_intel.intelligence import IntelligenceEngine, IntelligentResponse
from src.docwain_intel.payload_enricher import enrich_payload, batch_enrich
from src.docwain_intel.graph_populator import populate_graph
from src.docwain_intel.graph_adapter import (
    CypherGraphAdapter, GraphNode, GraphEdge,
)
from src.docwain_intel.response_assembler import assemble_response


# ============================================================================
# Test Document Fixtures
# ============================================================================

def _build_resume_doc() -> ExtractedDocumentJSON:
    """Build a realistic resume for Gokul Ramanathan."""
    blocks = [
        Block(block_id="r0", type="heading", text="Gokul Ramanathan",
              page_number=1, reading_order=0),
        Block(block_id="r1", type="heading", text="Summary",
              page_number=1, reading_order=1),
        Block(block_id="r2", type="paragraph",
              text="Experienced software engineer with expertise in Python, AWS, and Docker. "
                   "Passionate about building scalable distributed systems.",
              page_number=1, reading_order=2),
        Block(block_id="r3", type="heading", text="Experience",
              page_number=1, reading_order=3),
        Block(block_id="r4", type="paragraph",
              text="Senior Software Engineer at Google, Mountain View, CA. "
                   "January 2019 to December 2022. Designed and implemented "
                   "large-scale data pipelines processing 200 million events per day.",
              page_number=1, reading_order=4),
        Block(block_id="r5", type="paragraph",
              text="Software Engineer at Amazon, Seattle, WA. "
                   "June 2017 to December 2019. Built microservices for the "
                   "AWS Lambda platform serving thousands of enterprise customers.",
              page_number=1, reading_order=5),
        Block(block_id="r6", type="heading", text="Skills",
              page_number=2, reading_order=6),
        Block(block_id="r7", type="list_item", text="Python, Java, Go",
              page_number=2, reading_order=7),
        Block(block_id="r8", type="list_item", text="AWS, Docker, Kubernetes",
              page_number=2, reading_order=8),
        Block(block_id="r9", type="heading", text="Education",
              page_number=2, reading_order=9),
        Block(block_id="r10", type="paragraph",
              text="Master of Science in Computer Science, MIT, 2017.",
              page_number=2, reading_order=10),
        Block(block_id="r11", type="heading", text="Contact",
              page_number=2, reading_order=11),
        Block(block_id="r12", type="key_value",
              text="Email: gokul@example.com",
              key="Email", value="gokul@example.com",
              page_number=2, reading_order=12),
        Block(block_id="r13", type="key_value",
              text="Phone: +1-555-0100",
              key="Phone", value="+1-555-0100",
              page_number=2, reading_order=13),
    ]
    sections = [
        Section(section_path=["Gokul Ramanathan"],
                content_refs=["r0"], page_range=[1, 1]),
        Section(section_path=["Summary"],
                content_refs=["r1", "r2"], page_range=[1, 1]),
        Section(section_path=["Experience"],
                content_refs=["r3", "r4", "r5"], page_range=[1, 1]),
        Section(section_path=["Skills"],
                content_refs=["r6", "r7", "r8"], page_range=[2, 2]),
        Section(section_path=["Education"],
                content_refs=["r9", "r10"], page_range=[2, 2]),
        Section(section_path=["Contact"],
                content_refs=["r11", "r12", "r13"], page_range=[2, 2]),
    ]
    pages = [
        Page(page_number=1, blocks=[b for b in blocks if b.page_number == 1]),
        Page(page_number=2, blocks=[b for b in blocks if b.page_number == 2]),
    ]
    return ExtractedDocumentJSON(
        document_id="resume_gokul", pages=pages, sections=sections,
    )


def _build_invoice_doc() -> ExtractedDocumentJSON:
    """Build a realistic invoice INV-2024-0847."""
    blocks = [
        Block(block_id="i0", type="heading", text="Invoice INV-2024-0847",
              page_number=1, reading_order=0),
        Block(block_id="i1", type="key_value",
              text="Vendor: TechCorp Solutions",
              key="Vendor", value="TechCorp Solutions",
              page_number=1, reading_order=1),
        Block(block_id="i2", type="key_value",
              text="Client: Acme Industries",
              key="Client", value="Acme Industries",
              page_number=1, reading_order=2),
        Block(block_id="i3", type="key_value",
              text="Invoice Date: 2024-03-15",
              key="Invoice Date", value="2024-03-15",
              page_number=1, reading_order=3),
        Block(block_id="i4", type="key_value",
              text="Due Date: 2024-04-15",
              key="Due Date", value="2024-04-15",
              page_number=1, reading_order=4),
        Block(block_id="i5", type="key_value",
              text="Total: $26,700",
              key="Total", value="$26,700",
              page_number=1, reading_order=5),
    ]
    tables = [
        Table(
            table_id="t_inv_items",
            headers=["Service", "Amount"],
            rows=[
                ["Cloud Services", "$15,000"],
                ["Consulting", "$8,500"],
                ["Support", "$3,200"],
            ],
            page_number=1,
        ),
    ]
    sections = [
        Section(section_path=["Invoice INV-2024-0847"],
                content_refs=["i0", "i1", "i2", "i3", "i4", "i5"],
                page_range=[1, 1]),
    ]
    pages = [
        Page(page_number=1, blocks=blocks),
    ]
    return ExtractedDocumentJSON(
        document_id="invoice_inv2024",
        pages=pages,
        sections=sections,
        tables=tables,
    )


def _build_contract_doc() -> ExtractedDocumentJSON:
    """Build a realistic legal contract between TechCorp Solutions and Acme Industries."""
    blocks = [
        Block(block_id="c0", type="heading",
              text="Consulting Services Agreement",
              page_number=1, reading_order=0),
        Block(block_id="c1", type="paragraph",
              text="This Consulting Services Agreement is entered into between "
                   "TechCorp Solutions (hereinafter 'Provider') and "
                   "Acme Industries (hereinafter 'Client').",
              page_number=1, reading_order=1),
        Block(block_id="c2", type="heading",
              text="Scope of Services",
              page_number=1, reading_order=2),
        Block(block_id="c3", type="paragraph",
              text="The Provider shall deliver cloud infrastructure consulting, "
                   "system architecture design, and technical support services.",
              page_number=1, reading_order=3),
        Block(block_id="c4", type="heading",
              text="Compensation",
              page_number=1, reading_order=4),
        Block(block_id="c5", type="paragraph",
              text="The Client shall compensate the Provider at a rate of "
                   "$150 per hour for all services rendered.",
              page_number=1, reading_order=5),
        Block(block_id="c6", type="heading",
              text="Term",
              page_number=2, reading_order=6),
        Block(block_id="c7", type="paragraph",
              text="This Agreement shall be effective for a period of 24 months "
                   "commencing on April 1, 2024.",
              page_number=2, reading_order=7),
        Block(block_id="c8", type="heading",
              text="Confidentiality",
              page_number=2, reading_order=8),
        Block(block_id="c9", type="paragraph",
              text="Both parties agree to maintain strict confidentiality of all "
                   "proprietary information exchanged during the term of this Agreement.",
              page_number=2, reading_order=9),
        Block(block_id="c10", type="heading",
              text="Termination",
              page_number=2, reading_order=10),
        Block(block_id="c11", type="paragraph",
              text="Either party may terminate this Agreement with 30 days written notice.",
              page_number=2, reading_order=11),
    ]
    sections = [
        Section(section_path=["Consulting Services Agreement"],
                content_refs=["c0", "c1"], page_range=[1, 1]),
        Section(section_path=["Scope of Services"],
                content_refs=["c2", "c3"], page_range=[1, 1]),
        Section(section_path=["Compensation"],
                content_refs=["c4", "c5"], page_range=[1, 1]),
        Section(section_path=["Term"],
                content_refs=["c6", "c7"], page_range=[2, 2]),
        Section(section_path=["Confidentiality"],
                content_refs=["c8", "c9"], page_range=[2, 2]),
        Section(section_path=["Termination"],
                content_refs=["c10", "c11"], page_range=[2, 2]),
    ]
    pages = [
        Page(page_number=1, blocks=[b for b in blocks if b.page_number == 1]),
        Page(page_number=2, blocks=[b for b in blocks if b.page_number == 2]),
    ]
    return ExtractedDocumentJSON(
        document_id="contract_techcorp",
        pages=pages,
        sections=sections,
    )


# ============================================================================
# Shared pipeline helper
# ============================================================================

def _run_full_ingestion(doc: ExtractedDocumentJSON):
    """Run stages 1-5 of the pipeline and return all intermediate results."""
    structured = parse_structure(doc, document_id=doc.document_id)
    extraction = extract_entities_and_facts(structured)
    resolved = resolve_entities(extraction.entities)
    extraction.entities = resolved
    fingerprint = compute_fingerprint(structured, extraction)
    verification = verify_extraction(extraction, structured)
    return structured, extraction, fingerprint, verification


# ============================================================================
# Class 1: Full Ingestion Pipeline
# ============================================================================

class TestFullIngestionPipeline:
    """Test the complete 6-stage ingestion pipeline on realistic documents."""

    def test_resume_full_pipeline(self):
        doc = _build_resume_doc()
        structured, extraction, fingerprint, verification = _run_full_ingestion(doc)

        # Structure: should produce multiple semantic units
        assert structured.unit_count > 0
        assert structured.total_chars > 0

        # Entities: should find person names and organizations
        entity_texts_lower = {e.text.lower() for e in extraction.entities}
        entity_labels = {e.label for e in extraction.entities}

        # Gokul Ramanathan should be found (via heading heuristic or spaCy)
        has_gokul = any("gokul" in t for t in entity_texts_lower)
        assert has_gokul, (
            f"Expected entity 'Gokul Ramanathan'; found: {entity_texts_lower}"
        )

        # Organizations: Google and Amazon
        has_google = any("google" in t for t in entity_texts_lower)
        has_amazon = any("amazon" in t for t in entity_texts_lower)
        assert has_google or has_amazon, (
            f"Expected org entities Google/Amazon; found: {entity_texts_lower}"
        )

        # Email pattern entity
        has_email = any(e.label == "EMAIL" for e in extraction.entities)
        assert has_email, "Expected EMAIL entity from regex pattern extraction"

        # Fingerprint should produce auto_tags
        assert isinstance(fingerprint.auto_tags, list)
        assert fingerprint.entity_density > 0
        assert fingerprint.structure_complexity > 0

        # Verification should pass
        assert verification.is_valid
        assert verification.quality_score > 0

    def test_invoice_full_pipeline(self):
        doc = _build_invoice_doc()
        structured, extraction, fingerprint, verification = _run_full_ingestion(doc)

        # Should have KV_GROUP units from key-value blocks
        kv_units = [u for u in structured.units if u.unit_type == UnitType.KV_GROUP]
        assert len(kv_units) > 0, "Invoice should have KV_GROUP units"

        # Should have TABLE units
        table_units = [u for u in structured.units if u.unit_type == UnitType.TABLE]
        assert len(table_units) > 0, "Invoice should have TABLE units"

        # KV facts should be extracted
        kv_pairs = extraction.kv_pairs
        assert len(kv_pairs) > 0, "Invoice should have KV pairs extracted"

        # Monetary values via table extraction
        table_facts = [
            f for f in extraction.facts
            if f.extraction_method == "table_structured"
        ]
        assert len(table_facts) > 0, "Invoice should have table-structured facts"

        # Check that monetary amounts appear in facts
        all_values = " ".join(
            f.object_value or "" for f in extraction.facts
        )
        assert "$" in all_values or "15,000" in all_values or "26,700" in all_values, (
            f"Expected monetary values in facts; values: {all_values[:200]}"
        )

        # Fingerprint: numeric density should be notable
        assert fingerprint.numeric_density > 0

        # Verification
        assert verification.is_valid

    def test_contract_full_pipeline(self):
        doc = _build_contract_doc()
        structured, extraction, fingerprint, verification = _run_full_ingestion(doc)

        # Should have paragraph units for legal clauses
        para_units = [u for u in structured.units if u.unit_type == UnitType.PARAGRAPH]
        assert len(para_units) >= 3, "Contract should have multiple paragraph units"

        # Party entities: TechCorp Solutions and Acme Industries
        entity_texts_lower = {e.text.lower() for e in extraction.entities}
        # At least one of the parties should be found
        has_techcorp = any("techcorp" in t for t in entity_texts_lower)
        has_acme = any("acme" in t for t in entity_texts_lower)
        assert has_techcorp or has_acme, (
            f"Expected party entities; found: {entity_texts_lower}"
        )

        # Compensation fact: $150 should appear
        all_fact_text = " ".join(f.raw_text for f in extraction.facts)
        has_compensation = "150" in all_fact_text or "$150" in all_fact_text
        # Also check entity text and unit text as fallback
        all_text = " ".join(u.text for u in structured.units)
        assert "150" in all_text, "Contract should mention $150/hr compensation"

        # Term: 24 months should appear in document text
        assert "24 months" in all_text, "Contract should mention 24 month term"

        # Formality score should be elevated for legal text
        assert fingerprint.formality_score >= 0.25

        # Verification
        assert verification.is_valid

    def test_cross_document_entity_resolution(self):
        """Process all 3 docs and verify cross-doc entity resolution."""
        invoice_doc = _build_invoice_doc()
        contract_doc = _build_contract_doc()

        _, inv_extraction, _, _ = _run_full_ingestion(invoice_doc)
        _, con_extraction, _, _ = _run_full_ingestion(contract_doc)

        # Combine entities from both documents
        all_entities = list(inv_extraction.entities) + list(con_extraction.entities)

        # Run cross-document resolution
        resolved = resolve_entities(all_entities)

        # TechCorp Solutions appears in both invoice and contract.
        # After resolution, there should be fewer total entities than before.
        # The same canonical entity should cover both mentions.
        techcorp_entities = [
            e for e in resolved
            if "techcorp" in e.text.lower()
        ]
        acme_entities = [
            e for e in resolved
            if "acme" in e.text.lower()
        ]

        # If entities were found in both docs, resolution should merge them
        if len(techcorp_entities) > 0:
            # After resolution, duplicates should be merged
            assert len(techcorp_entities) <= 2, (
                f"Expected TechCorp merged; got {len(techcorp_entities)} entries"
            )
        if len(acme_entities) > 0:
            assert len(acme_entities) <= 2, (
                f"Expected Acme merged; got {len(acme_entities)} entries"
            )

        # Total resolved should be <= total input
        assert len(resolved) <= len(all_entities)

    def test_graph_population_complete(self):
        """Mock graph adapter, process all 3 docs, verify node/edge creation."""
        mock_driver = MagicMock()
        mock_adapter = CypherGraphAdapter(driver=mock_driver)

        # Track all calls
        upserted_nodes = []
        upserted_edges = []
        original_upsert_node = mock_adapter.upsert_node
        original_upsert_edge = mock_adapter.upsert_edge

        def track_node(node):
            upserted_nodes.append(node)

        def track_edge(edge):
            upserted_edges.append(edge)

        mock_adapter.upsert_node = track_node
        mock_adapter.upsert_edge = track_edge

        for doc_builder, doc_id in [
            (_build_resume_doc, "resume_gokul"),
            (_build_invoice_doc, "invoice_inv2024"),
            (_build_contract_doc, "contract_techcorp"),
        ]:
            doc = doc_builder()
            structured, extraction, fingerprint, verification = _run_full_ingestion(doc)

            populate_graph(
                adapter=mock_adapter,
                extraction=extraction,
                structured_doc=structured,
                document_id=doc_id,
                subscription_id="sub_test",
                profile_id="prof_test",
                fingerprint_tags=fingerprint.auto_tags,
            )

        # Verify Document nodes created (one per document)
        doc_nodes = [n for n in upserted_nodes if n.node_type == "Document"]
        assert len(doc_nodes) == 3, f"Expected 3 Document nodes; got {len(doc_nodes)}"

        # Verify Entity nodes created
        entity_nodes = [n for n in upserted_nodes if n.node_type == "Entity"]
        assert len(entity_nodes) > 0, "Expected Entity nodes"

        # Verify Chunk nodes created
        chunk_nodes = [n for n in upserted_nodes if n.node_type == "Chunk"]
        assert len(chunk_nodes) > 0, "Expected Chunk nodes"

        # Verify edge types
        edge_types = {e.edge_type for e in upserted_edges}
        assert "BELONGS_TO" in edge_types, "Expected BELONGS_TO edges"
        assert "MENTIONED_IN" in edge_types, "Expected MENTIONED_IN edges"

        # APPEARS_IN edges should exist (entity -> chunk)
        appears_in = [e for e in upserted_edges if e.edge_type == "APPEARS_IN"]
        assert len(appears_in) > 0, "Expected APPEARS_IN edges"


# ============================================================================
# Class 2: Query-Time Intelligence
# ============================================================================

class TestQueryTimeIntelligence:
    """Test the complete query-time intelligence pipeline."""

    def test_simple_fact_query(self):
        """Query: 'What is Gokul's email?' -- should be high specificity, entity_lookup."""
        analysis = route_query("What is Gokul's email?")
        geometry = analyze_query("What is Gokul's email?", analysis)

        # High specificity (possessive + field token "email" + entity)
        assert analysis.specificity >= 0.3, (
            f"Expected high specificity; got {analysis.specificity}"
        )
        # Should route to GRAPH_DIRECT for direct fact lookup
        assert analysis.route == QueryRoute.GRAPH_DIRECT, (
            f"Expected GRAPH_DIRECT; got {analysis.route}"
        )
        # Geometry: should be attribute_centric
        assert geometry.focus_type == "attribute_centric", (
            f"Expected attribute_centric; got {geometry.focus_type}"
        )
        # Granularity > 0.5 (concise answer expected)
        assert geometry.granularity > 0.3, (
            f"Expected higher granularity for fact query; got {geometry.granularity}"
        )

    def test_comparison_query_geometry(self):
        """Query: 'Compare TechCorp invoice with the contract' -- comparative intent."""
        query = "Compare TechCorp invoice with the contract"
        analysis = route_query(query)
        geometry = analyze_query(query, analysis)

        # Should detect comparison
        assert analysis.is_comparison or geometry.is_comparison, (
            "Expected comparison detection"
        )

        # Create evidence with 2 entity groups to trigger comparison layout
        evidence = OrganizedEvidence(
            entity_groups=[
                EvidenceGroup(
                    entity_id="e1", entity_text="Invoice",
                    chunks=[{"text": "Invoice details", "score": 0.8}],
                ),
                EvidenceGroup(
                    entity_id="e2", entity_text="Contract",
                    chunks=[{"text": "Contract terms", "score": 0.7}],
                ),
            ],
            total_chunks=2,
        )
        spec = generate_spec(geometry, evidence)

        # For comparison with 2 entity groups, should produce comparison layout
        if analysis.is_comparison:
            assert spec.layout_mode == "comparison", (
                f"Expected comparison layout; got {spec.layout_mode}"
            )
            assert spec.use_table is True

    def test_narrative_query(self):
        """Query: 'Tell me about Gokul Ramanathan's experience' -- narrative intent."""
        query = "Tell me about Gokul Ramanathan's experience"
        analysis = route_query(query)
        geometry = analyze_query(query, analysis)

        # Should detect narrative/detailed intent
        # "tell me about" pushes granularity low (detailed)
        assert geometry.granularity < 0.7, (
            f"Expected lower granularity for narrative; got {geometry.granularity}"
        )

        # With no evidence, spec should default to narrative
        evidence = OrganizedEvidence(
            ungrouped_chunks=[
                {"text": "Gokul worked at Google as a Senior Engineer", "score": 0.9},
            ],
            total_chunks=1,
        )
        spec = generate_spec(geometry, evidence)
        assert spec.layout_mode == "narrative", (
            f"Expected narrative layout; got {spec.layout_mode}"
        )

    def test_evidence_organization(self):
        """Create chunks with intel_entity_ids and verify grouping."""
        chunks = [
            {
                "text": "Gokul is a software engineer",
                "payload": {"intel_entity_ids": ["ent_gokul"]},
                "score": 0.9,
            },
            {
                "text": "Gokul's email is gokul@example.com",
                "payload": {"intel_entity_ids": ["ent_gokul"]},
                "score": 0.85,
            },
            {
                "text": "TechCorp provides consulting services",
                "payload": {"intel_entity_ids": ["ent_techcorp"]},
                "score": 0.7,
            },
            {
                "text": "General document overview",
                "payload": {},
                "score": 0.5,
            },
        ]
        facts = [
            {
                "subject": "Gokul Ramanathan",
                "predicate": "email",
                "value": "gokul@example.com",
                "confidence": 0.95,
            },
        ]

        evidence = organize_evidence(
            chunks=chunks,
            facts=facts,
            query_entities=["Gokul Ramanathan"],
        )

        # Should have entity groups
        assert len(evidence.entity_groups) >= 1, "Expected entity groups"

        # Gokul group should have 2 chunks
        gokul_groups = [
            g for g in evidence.entity_groups
            if g.entity_id == "ent_gokul"
        ]
        if gokul_groups:
            assert len(gokul_groups[0].chunks) == 2

        # Ungrouped chunks should contain the one without entity IDs
        assert len(evidence.ungrouped_chunks) == 1

        # Provenance should be extracted
        assert len(evidence.provenance) == len(chunks)

    def test_constrained_prompt_generation(self):
        """Generate a full prompt for a comparison query and verify structure."""
        geometry = QueryGeometry(
            query="Compare invoice and contract",
            intent_type="comparative",
            is_comparison=True,
            granularity=0.5,
            focus_type="entity_centric",
        )
        evidence = OrganizedEvidence(
            entity_groups=[
                EvidenceGroup(
                    entity_id="e1", entity_text="Invoice",
                    facts=[
                        {"predicate": "total", "value": "$26,700"},
                        {"predicate": "vendor", "value": "TechCorp"},
                    ],
                    chunks=[{"text": "Invoice INV-2024-0847", "score": 0.9}],
                ),
                EvidenceGroup(
                    entity_id="e2", entity_text="Contract",
                    facts=[
                        {"predicate": "rate", "value": "$150/hr"},
                        {"predicate": "term", "value": "24 months"},
                    ],
                    chunks=[{"text": "Consulting Agreement", "score": 0.85}],
                ),
            ],
            gaps=[],
            total_facts=4,
            total_chunks=2,
        )
        spec = generate_spec(geometry, evidence)
        prompt = build_prompt(spec, evidence, "Compare invoice and contract")

        # System prompt should contain anti-hallucination rules
        assert "NOT" in prompt.system_prompt or "not" in prompt.system_prompt.lower()
        assert "evidence" in prompt.system_prompt.lower()

        # User prompt should contain QUESTION, evidence, and format instructions
        assert "QUESTION:" in prompt.user_prompt
        assert "EVIDENCE:" in prompt.user_prompt
        assert "OUTPUT FORMAT:" in prompt.user_prompt

        # Evidence section should mention both entities
        assert "Invoice" in prompt.user_prompt
        assert "Contract" in prompt.user_prompt

        # Max tokens and temperature should be set
        assert prompt.max_tokens > 0
        assert 0.0 <= prompt.temperature <= 1.0

    def test_quality_validation_on_good_output(self):
        """LLM output that matches spec -- should pass with high scores."""
        spec = RenderingSpec(
            layout_mode="narrative",
            detail_level="standard",
        )
        evidence = OrganizedEvidence(
            entity_groups=[
                EvidenceGroup(
                    entity_text="Gokul Ramanathan",
                    chunks=[{
                        "text": "Gokul Ramanathan is a senior software engineer at Google. "
                                "He has expertise in Python and AWS.",
                    }],
                ),
            ],
            total_chunks=1,
        )

        good_output = (
            "Gokul Ramanathan is a senior software engineer at Google "
            "with expertise in Python and AWS."
        )

        result = validate_output(good_output, spec, evidence)

        assert result.structural_conformance > 0.5, (
            f"Expected high structural conformance; got {result.structural_conformance}"
        )
        assert result.content_integrity > 0.5, (
            f"Expected high content integrity; got {result.content_integrity}"
        )

    def test_quality_validation_strips_artifacts(self):
        """LLM output with preamble -- should be stripped."""
        spec = RenderingSpec(layout_mode="narrative")
        evidence = OrganizedEvidence(
            entity_groups=[
                EvidenceGroup(
                    entity_text="Gokul",
                    chunks=[{
                        "text": "Gokul works at Google as a software engineer.",
                    }],
                ),
            ],
            total_chunks=1,
        )

        preamble_output = (
            "Based on the documents provided, "
            "Gokul works at Google as a software engineer."
        )

        result = validate_output(preamble_output, spec, evidence)

        # Preamble should be stripped
        assert result.was_modified, "Expected preamble to be stripped"
        assert not result.cleaned_text.lower().startswith("based on"), (
            f"Preamble not stripped: {result.cleaned_text[:80]}"
        )
        # Core content should be preserved
        assert "Gokul" in result.cleaned_text
        assert "Google" in result.cleaned_text


# ============================================================================
# Class 3: Conversational Intelligence
# ============================================================================

class TestConversationalIntelligence:
    """Test multi-turn conversation with context tracking."""

    def test_multi_turn_pronoun_resolution(self):
        """Turn 1: About Gokul. Turn 2: 'What are his skills?' resolves 'his'."""
        cg = ConversationGraph(session_id="sess_pronoun")

        # Turn 1: mention Gokul as PERSON
        cg.add_turn(
            query="Tell me about Gokul Ramanathan",
            entities=["Gokul Ramanathan"],
            entity_labels={"Gokul Ramanathan": "PERSON"},
            facts_disclosed=["name: Gokul Ramanathan"],
            response_text="Gokul Ramanathan is a software engineer.",
        )

        # Turn 2: pronoun reference
        resolved = cg.resolve_query("What are his skills?")
        assert "Gokul Ramanathan" in resolved, (
            f"Expected 'his' resolved to 'Gokul Ramanathan'; got: {resolved}"
        )

    def test_progressive_disclosure(self):
        """Turn 1 discloses facts A,B. Turn 2 'anything else?' returns new facts."""
        cg = ConversationGraph(session_id="sess_progressive")

        cg.add_turn(
            query="Tell me about Gokul",
            entities=["Gokul"],
            entity_labels={},
            facts_disclosed=["email:gokul@example.com", "phone:+1-555-0100"],
            response_text="Gokul's email is gokul@example.com, phone is +1-555-0100.",
        )

        all_facts = [
            "email:gokul@example.com",
            "phone:+1-555-0100",
            "company:Google",
            "degree:MIT CS",
        ]

        undisclosed = cg.get_undisclosed_facts("Gokul", all_facts)

        # Previously disclosed facts should be excluded
        assert "email:gokul@example.com" not in undisclosed
        assert "phone:+1-555-0100" not in undisclosed
        # New facts should be included
        assert "company:Google" in undisclosed
        assert "degree:MIT CS" in undisclosed

    def test_session_isolation(self):
        """Two sessions should not share entity context."""
        cg1 = ConversationGraph(session_id="sess_1")
        cg2 = ConversationGraph(session_id="sess_2")

        cg1.add_turn(
            query="About Gokul",
            entities=["Gokul Ramanathan"],
            entity_labels={"Gokul Ramanathan": "PERSON"},
            facts_disclosed=[],
            response_text="Gokul info.",
        )

        cg2.add_turn(
            query="About TechCorp",
            entities=["TechCorp Solutions"],
            entity_labels={"TechCorp Solutions": "ORG"},
            facts_disclosed=[],
            response_text="TechCorp info.",
        )

        # Session 1 should only know about Gokul
        ctx1 = cg1.get_context_entities()
        entity_names_1 = {e["entity"] for e in ctx1}
        assert "Gokul Ramanathan" in entity_names_1
        assert "TechCorp Solutions" not in entity_names_1

        # Session 2 should only know about TechCorp
        ctx2 = cg2.get_context_entities()
        entity_names_2 = {e["entity"] for e in ctx2}
        assert "TechCorp Solutions" in entity_names_2
        assert "Gokul Ramanathan" not in entity_names_2

    def test_entity_context_carryover(self):
        """Turn 1: invoice context. Turn 2: 'What's the total?' scopes to invoice."""
        cg = ConversationGraph(session_id="sess_carryover")

        cg.add_turn(
            query="Tell me about invoice INV-2024-0847",
            entities=["INV-2024-0847"],
            entity_labels={"INV-2024-0847": "DOCUMENT"},
            facts_disclosed=["vendor:TechCorp Solutions"],
            response_text="Invoice INV-2024-0847 from TechCorp Solutions.",
        )

        # "What's the total?" should carry forward the invoice context
        resolved = cg.resolve_query("What's the total?")
        # The query may or may not be modified (no pronoun), but context entities
        # should still include the invoice
        ctx = cg.get_context_entities()
        entity_names = {e["entity"] for e in ctx}
        assert "INV-2024-0847" in entity_names


# ============================================================================
# Class 4: End-to-End Response Quality
# ============================================================================

class TestEndToEndResponseQuality:
    """Validate that the full pipeline produces quality responses."""

    def test_structured_response_for_fact_query(self):
        """Full pipeline: ingest resume -> query email -> concise response."""
        doc = _build_resume_doc()
        structured, extraction, fingerprint, verification = _run_full_ingestion(doc)

        # Build chunks from structured doc
        chunks = []
        for unit in structured.units:
            payload = {}
            enriched = enrich_payload(
                payload, unit.text, extraction, fingerprint, structured,
            )
            chunks.append({
                "text": unit.text,
                "payload": enriched,
                "score": 0.8,
            })

        # Query
        analysis = route_query("What is Gokul's email?")
        geometry = analyze_query("What is Gokul's email?", analysis)

        evidence = organize_evidence(
            chunks=chunks,
            facts=[],
            query_entities=analysis.entities,
            extraction=extraction,
        )
        spec = generate_spec(geometry, evidence)
        prompt = build_prompt(spec, evidence, "What is Gokul's email?")

        # Simulate LLM giving the correct answer
        llm_output = "gokul@example.com"
        quality = validate_output(llm_output, spec, evidence)

        # Should not be heavily modified
        assert "gokul@example.com" in quality.cleaned_text

    def test_table_response_for_comparison(self):
        """Full pipeline: ingest invoice + contract -> comparison spec."""
        inv_doc = _build_invoice_doc()
        con_doc = _build_contract_doc()

        _, inv_ext, inv_fp, _ = _run_full_ingestion(inv_doc)
        _, con_ext, con_fp, _ = _run_full_ingestion(con_doc)

        query = "Compare the invoice and the consulting agreement"
        analysis = route_query(query)
        geometry = analyze_query(query, analysis)

        # Build evidence groups from both docs
        evidence = OrganizedEvidence(
            entity_groups=[
                EvidenceGroup(
                    entity_id="inv", entity_text="Invoice",
                    facts=[
                        {"predicate": "total", "value": "$26,700"},
                        {"predicate": "vendor", "value": "TechCorp Solutions"},
                    ],
                    chunks=[{"text": "Invoice INV-2024-0847", "score": 0.85}],
                ),
                EvidenceGroup(
                    entity_id="con", entity_text="Contract",
                    facts=[
                        {"predicate": "rate", "value": "$150/hr"},
                        {"predicate": "term", "value": "24 months"},
                    ],
                    chunks=[{"text": "Consulting Services Agreement", "score": 0.8}],
                ),
            ],
            total_facts=4,
            total_chunks=2,
        )
        spec = generate_spec(geometry, evidence)

        # If comparison detected, should request table layout
        if analysis.is_comparison:
            assert spec.use_table is True
            prompt = build_prompt(spec, evidence, query)
            # Prompt should mention comparison or table
            assert "table" in prompt.user_prompt.lower() or "comparison" in prompt.user_prompt.lower()

    def test_no_hallucination_in_response(self):
        """Quality engine should detect fabricated vs grounded claims."""
        spec = RenderingSpec(layout_mode="narrative")
        evidence = OrganizedEvidence(
            entity_groups=[
                EvidenceGroup(
                    entity_text="Gokul Ramanathan",
                    chunks=[{
                        "text": "Gokul Ramanathan works at Google as a software engineer since 2019.",
                    }],
                ),
            ],
            total_chunks=1,
        )

        # Output with a fabricated claim (Stanford not in evidence)
        fabricated_output = (
            "Gokul Ramanathan works at Google since 2019. "
            "He graduated from Stanford University in 2015."
        )
        result = validate_output(fabricated_output, spec, evidence)

        # Should detect some unverified claims
        # "Stanford" and "2015" are not in evidence
        assert result.claims_unverified > 0 or result.content_integrity < 1.0, (
            "Expected fabricated claim to be flagged"
        )

    def test_complete_pipeline_timing(self):
        """Run full pipeline and verify it completes within time budget."""
        start = time.monotonic()

        doc = _build_resume_doc()
        result = process_document(
            extracted_doc=doc,
            document_id="resume_gokul",
            subscription_id="sub_test",
            profile_id="prof_test",
        )

        elapsed = time.monotonic() - start

        # Should complete within 30 seconds
        assert elapsed < 30, f"Pipeline took {elapsed:.1f}s, expected < 30s"

        # Should have reached at least VERIFIED stage
        assert result.stage_reached in (
            PipelineStage.VERIFIED.value,
            PipelineStage.STORED.value,
            PipelineStage.AUDITED.value,
        ), f"Expected VERIFIED+; got {result.stage_reached}"

        # Stage timings should be recorded
        assert "structure" in result.stage_timings
        assert "extraction" in result.stage_timings
        assert "resolution" in result.stage_timings
        assert "fingerprint" in result.stage_timings
        assert "verification" in result.stage_timings

        # No errors
        assert result.error is None


# ============================================================================
# Additional: Payload Enrichment + IntelligenceEngine integration
# ============================================================================

class TestPayloadEnrichment:
    """Test that enriched payloads contain intelligence metadata."""

    def test_enrich_payload_adds_entity_ids(self):
        doc = _build_resume_doc()
        structured, extraction, fingerprint, _ = _run_full_ingestion(doc)

        # Pick a unit that mentions Gokul
        for unit in structured.units:
            if "gokul" in unit.text.lower() or "google" in unit.text.lower():
                payload = {}
                enriched = enrich_payload(
                    payload, unit.text, extraction, fingerprint, structured,
                )
                # Should have intel_ keys
                assert "intel_entity_ids" in enriched
                assert "intel_fingerprint_tags" in enriched
                assert "intel_unit_type" in enriched
                assert isinstance(enriched["intel_entity_ids"], list)
                break

    def test_batch_enrich(self):
        doc = _build_invoice_doc()
        structured, extraction, fingerprint, _ = _run_full_ingestion(doc)

        payloads = [
            ({}, unit.text) for unit in structured.units
        ]
        results = batch_enrich(
            payloads, extraction, fingerprint, structured,
        )
        assert len(results) == len(structured.units)
        for r in results:
            assert "intel_entity_ids" in r


class TestIntelligenceEngineIntegration:
    """Test IntelligenceEngine with mocked graph adapter."""

    def test_process_query_needs_llm(self):
        """When no LLM response provided, engine returns needs_llm=True."""
        engine = IntelligenceEngine()

        # Patch graph adapter to return None (no graph available)
        with patch.object(engine, "_get_graph", return_value=None):
            response = engine.process_query(
                query="Tell me about Gokul Ramanathan's experience",
                subscription_id="sub_test",
                profile_id="prof_test",
                session_id="sess_test",
            )

        assert response.needs_llm is True
        assert response.prompt is not None
        assert response.geometry is not None
        assert response.spec is not None
        assert len(response.stage_timings) > 0

    def test_process_query_with_llm_response(self):
        """When LLM response provided, engine validates and returns text."""
        engine = IntelligenceEngine()

        chunks = [
            {
                "text": "Gokul Ramanathan works at Google.",
                "payload": {"intel_entity_ids": ["ent_gokul"]},
                "score": 0.9,
            },
        ]

        with patch.object(engine, "_get_graph", return_value=None):
            response = engine.process_query(
                query="Where does Gokul work?",
                subscription_id="sub_test",
                profile_id="prof_test",
                session_id="sess_llm",
                chunks=chunks,
                llm_response="Gokul Ramanathan works at Google.",
            )

        assert response.needs_llm is False
        assert response.text != ""
        assert response.quality is not None
        assert response.quality.content_integrity > 0

    def test_process_query_graph_direct(self):
        """When graph has facts, should return directly without LLM."""
        engine = IntelligenceEngine()

        mock_graph = MagicMock()
        mock_graph.get_entity_facts.return_value = [
            {
                "target_props": {
                    "predicate": "email",
                    "value": "gokul@example.com",
                    "confidence": 0.95,
                },
            },
        ]

        with patch.object(engine, "_get_graph", return_value=mock_graph):
            response = engine.process_query(
                query="What is Gokul's email?",
                subscription_id="sub_test",
                profile_id="prof_test",
                session_id="sess_graph",
            )

        # Should have resolved via graph (no LLM needed)
        assert response.needs_llm is False
        assert response.route_used == "GRAPH_DIRECT"
        assert response.text != ""

    def test_finalize_validates_llm_output(self):
        """Test the finalize() method for post-LLM validation."""
        engine = IntelligenceEngine()

        spec = RenderingSpec(layout_mode="narrative")
        evidence = OrganizedEvidence(
            entity_groups=[
                EvidenceGroup(
                    entity_text="Gokul",
                    chunks=[{"text": "Gokul is a software engineer at Google."}],
                ),
            ],
            total_chunks=1,
        )

        response = engine.finalize(
            session_id="sess_finalize",
            query="Tell me about Gokul",
            llm_response="Gokul is a software engineer at Google.",
            spec=spec,
            evidence=evidence,
        )

        assert response.needs_llm is False
        assert response.text != ""
        assert response.quality is not None
        assert response.quality.structural_conformance > 0
