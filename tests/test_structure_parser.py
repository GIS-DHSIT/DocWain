"""Tests for structural document parsing."""
import pytest
from src.docwain_intel.models import UnitType, ExtractedDocumentJSON, Page, Block, Section, Table
from src.docwain_intel.structure_parser import parse_structure, StructuredDocument


class TestParseStructure:
    def _make_doc(self, blocks, sections=None, tables=None, page_number=1):
        pages = [Page(page_number=page_number, blocks=blocks)]
        return ExtractedDocumentJSON(
            document_id="doc_001", pages=pages,
            sections=sections or [], tables=tables or [],
        )

    def test_paragraph_blocks_become_paragraph_units(self):
        blocks = [
            Block(block_id="b1", type="paragraph", text="John Smith has 8 years of experience in software engineering.", page_number=1, reading_order=0),
            Block(block_id="b2", type="paragraph", text="He specializes in Python and distributed systems.", page_number=1, reading_order=1),
        ]
        result = parse_structure(self._make_doc(blocks), document_id="doc_001")
        assert isinstance(result, StructuredDocument)
        assert len(result.units) >= 1
        assert all(u.unit_type in (UnitType.PARAGRAPH, UnitType.FRAGMENT) for u in result.units)

    def test_heading_creates_section_context(self):
        blocks = [
            Block(block_id="b0", type="heading", text="Work Experience", page_number=1, reading_order=0),
            Block(block_id="b1", type="paragraph", text="Senior Engineer at Google, 2020-2023.", page_number=1, reading_order=1),
        ]
        sections = [Section(section_path=["Work Experience"], content_refs=["b0", "b1"], page_range=[1, 1])]
        result = parse_structure(self._make_doc(blocks, sections=sections), document_id="doc_001")
        para_units = [u for u in result.units if u.unit_type == UnitType.PARAGRAPH]
        assert len(para_units) >= 1
        assert "Work Experience" in para_units[0].heading_path

    def test_kv_blocks_become_kv_group(self):
        blocks = [
            Block(block_id="b1", type="key_value", text="Name: John Smith", key="Name", value="John Smith", page_number=1, reading_order=0),
            Block(block_id="b2", type="key_value", text="Email: john@test.com", key="Email", value="john@test.com", page_number=1, reading_order=1),
            Block(block_id="b3", type="key_value", text="Phone: 555-0123", key="Phone", value="555-0123", page_number=1, reading_order=2),
        ]
        result = parse_structure(self._make_doc(blocks), document_id="doc_001")
        kv_units = [u for u in result.units if u.unit_type == UnitType.KV_GROUP]
        assert len(kv_units) >= 1
        assert kv_units[0].kv_pairs is not None
        assert kv_units[0].kv_pairs.get("Name") == "John Smith"

    def test_table_becomes_table_unit_with_headers(self):
        blocks = [Block(block_id="b1", type="paragraph", text="See table below.", page_number=1, reading_order=0)]
        tables = [Table(table_id="t1", headers=["Name", "Role", "Years"], rows=[["John", "Engineer", "5"], ["Sarah", "Manager", "3"]], page_number=1)]
        result = parse_structure(self._make_doc(blocks, tables=tables), document_id="doc_001")
        table_units = [u for u in result.units if u.unit_type == UnitType.TABLE]
        assert len(table_units) == 1
        assert table_units[0].table_headers == ["Name", "Role", "Years"]
        assert len(table_units[0].table_rows) == 2
        assert table_units[0].table_rows[0]["Name"] == "John"

    def test_list_items_grouped(self):
        blocks = [
            Block(block_id="b1", type="list_item", text="- Python", page_number=1, reading_order=0),
            Block(block_id="b2", type="list_item", text="- Java", page_number=1, reading_order=1),
            Block(block_id="b3", type="list_item", text="- Go", page_number=1, reading_order=2),
        ]
        result = parse_structure(self._make_doc(blocks), document_id="doc_001")
        list_units = [u for u in result.units if u.unit_type == UnitType.LIST]
        assert len(list_units) == 1
        assert "Python" in list_units[0].text
        assert "Go" in list_units[0].text

    def test_provenance_tracking(self):
        blocks = [Block(block_id="b1", type="paragraph", text="Important fact here.", page_number=3, reading_order=0)]
        result = parse_structure(self._make_doc(blocks, page_number=3), document_id="doc_001")
        assert result.units[0].page_start == 3
        assert result.units[0].page_end == 3

    def test_empty_document_produces_no_units(self):
        result = parse_structure(self._make_doc([]), document_id="doc_001")
        assert len(result.units) == 0

    def test_every_char_mapped(self):
        blocks = [
            Block(block_id="b1", type="paragraph", text="First paragraph.", page_number=1, reading_order=0),
            Block(block_id="b2", type="paragraph", text="Second paragraph.", page_number=1, reading_order=1),
        ]
        result = parse_structure(self._make_doc(blocks), document_id="doc_001")
        total_input_chars = sum(len(b.text or "") for b in blocks)
        total_output_chars = sum(len(u.text or "") for u in result.units)
        assert total_output_chars >= total_input_chars * 0.9
