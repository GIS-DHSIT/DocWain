"""Tests for StructuredTable model and TableParser."""

import pytest


class TestStructuredTableModel:

    def test_parse_simple_table(self):
        """Parse a simple table with header row."""
        from src.intelligence.table_parser import TableParser

        raw_rows = [
            ["Item", "Quantity", "Price"],
            ["Widget A", "10", "$15.00"],
            ["Widget B", "5", "$25.00"],
        ]

        parser = TableParser()
        table = parser.parse(raw_rows, page=1)

        assert table.headers == ["Item", "Quantity", "Price"]
        assert len(table.data_rows) == 2
        assert table.data_rows[0]["Item"] == "Widget A"
        assert table.data_rows[0]["Price"] == "$15.00"
        assert table.row_count == 2
        assert table.col_count == 3
        assert table.page == 1

    def test_parse_table_no_headers(self):
        """Table without clear headers should use column indices."""
        from src.intelligence.table_parser import TableParser

        raw_rows = [
            ["abc", "123", "xyz"],
            ["def", "456", "uvw"],
        ]

        parser = TableParser()
        table = parser.parse(raw_rows, page=0)

        # No clear header -- columns named col_0, col_1, col_2
        assert table.headers == ["col_0", "col_1", "col_2"]
        assert table.data_rows[0]["col_0"] == "abc"

    def test_parse_empty_table(self):
        """Empty table returns empty StructuredTable."""
        from src.intelligence.table_parser import TableParser

        parser = TableParser()
        table = parser.parse([], page=0)
        assert table.row_count == 0
        assert table.headers == []

    def test_detect_header_row(self):
        """Header detection identifies text-only first rows as headers."""
        from src.intelligence.table_parser import TableParser

        parser = TableParser()

        # First row all text, subsequent rows have numbers -> first row is header
        assert parser._is_header_row(
            ["Name", "Age", "Salary"],
            [["John", "30", "$50000"], ["Jane", "25", "$60000"]],
        )

        # First row has numbers -> not a header
        assert not parser._is_header_row(
            ["100", "200", "300"],
            [["400", "500", "600"]],
        )

    def test_flat_text_backward_compat(self):
        """flat_text should produce CSV-like output."""
        from src.intelligence.table_parser import TableParser

        raw_rows = [
            ["Item", "Price"],
            ["Widget", "$10"],
        ]

        parser = TableParser()
        table = parser.parse(raw_rows, page=1)
        flat = table.flat_text
        assert "Item, Price" in flat
        assert "Widget, $10" in flat

    def test_parse_with_none_cells(self):
        """None cells should be converted to empty strings."""
        from src.intelligence.table_parser import TableParser

        raw_rows = [
            ["Header A", "Header B", None],
            [None, "value", "data"],
        ]

        parser = TableParser()
        table = parser.parse(raw_rows, page=0)
        assert table.col_count == 3
        assert table.data_rows[0]["Header A"] == ""


class TestTableTypeClassification:

    def test_classifies_invoice_line_items(self):
        from src.intelligence.table_parser import TableParser

        raw = [
            ["Description", "Qty", "Unit Price", "Amount"],
            ["Web Development", "40", "$150.00", "$6,000.00"],
            ["Consulting", "10", "$200.00", "$2,000.00"],
        ]

        parser = TableParser()
        table = parser.parse(raw, page=1)
        table_type = parser.classify_table_type(table)
        assert table_type == "line_items"

    def test_classifies_totals_table(self):
        from src.intelligence.table_parser import TableParser

        raw = [
            ["Subtotal", "$8,000.00"],
            ["Tax (10%)", "$800.00"],
            ["Total", "$8,800.00"],
        ]

        parser = TableParser()
        table = parser.parse(raw, page=2)
        table_type = parser.classify_table_type(table)
        assert table_type == "totals"

    def test_classifies_skills_matrix(self):
        from src.intelligence.table_parser import TableParser

        raw = [
            ["Skill", "Level", "Years"],
            ["Python", "Expert", "8"],
            ["Java", "Advanced", "5"],
            ["AWS", "Intermediate", "3"],
        ]

        parser = TableParser()
        table = parser.parse(raw, page=1)
        table_type = parser.classify_table_type(table)
        assert table_type == "skills_matrix"

    def test_classifies_generic_table(self):
        from src.intelligence.table_parser import TableParser

        raw = [
            ["Column A", "Column B"],
            ["foo", "bar"],
        ]

        parser = TableParser()
        table = parser.parse(raw, page=0)
        table_type = parser.classify_table_type(table)
        assert table_type == "generic"


class TestTableAwareChunking:

    def test_table_chunk_meta_pipe_delimited(self):
        """_build_table_chunk_meta extracts pipe-delimited headers."""
        from src.embedding.chunking.section_chunker import SectionChunker

        chunker = SectionChunker()
        meta = chunker._build_table_chunk_meta(
            "Item | Quantity | Price\nWidget A | 10 | $15.00\nWidget B | 5 | $25.00"
        )
        assert meta is not None
        assert meta["table_headers"] == ["Item", "Quantity", "Price"]
        assert meta["table_rows"] == 2

    def test_table_chunk_meta_comma_delimited(self):
        """_build_table_chunk_meta extracts comma-delimited headers."""
        from src.embedding.chunking.section_chunker import SectionChunker

        chunker = SectionChunker()
        meta = chunker._build_table_chunk_meta(
            "Name, Age, Department\nJohn, 30, Engineering\nJane, 25, Marketing"
        )
        assert meta is not None
        assert meta["table_headers"] == ["Name", "Age", "Department"]
        assert meta["table_rows"] == 2

    def test_table_chunk_meta_returns_none_for_non_table(self):
        """_build_table_chunk_meta returns None for non-table text."""
        from src.embedding.chunking.section_chunker import SectionChunker

        chunker = SectionChunker()
        meta = chunker._build_table_chunk_meta("Just a regular paragraph of text.")
        assert meta is None

    def test_large_table_produces_multiple_chunks(self):
        """Large tables should be split into multiple chunks by SectionChunker."""
        from src.embedding.chunking.section_chunker import SectionChunker

        from src.api.pipeline_models import ExtractedDocument

        # Build a large table (30 pipe-delimited rows)
        header = "Name | Department | Salary | Start Date"
        rows = [f"Employee_{i} | Dept_{i % 5} | ${50000 + i * 1000} | 2020-0{(i % 9) + 1}-01" for i in range(30)]
        table_text = header + "\n" + "\n".join(rows)

        # Create a chunker with a small target so the large table gets split
        chunker = SectionChunker(target_chunk_chars=300, max_chunk_chars=500)

        doc = ExtractedDocument(
            full_text="Employee Directory\n\n" + table_text,
            sections=[],
            tables=[],
            figures=[],
            chunk_candidates=[],
        )
        chunks = chunker.chunk_document(doc, doc_internal_id="d1", source_filename="employees.pdf")

        # Should produce multiple chunks (table is ~2000 chars, target=300)
        assert len(chunks) >= 2, f"Expected >=2 chunks but got {len(chunks)}"


class TestTableMetadataInPayload:

    def test_table_headers_in_qdrant_payload(self):
        """Table headers should appear in Qdrant payload for retrieval boosting."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        chunk_meta = {
            "subscription_id": "sub1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "Item, Qty, Price\nWidget, 10, $15",
            "chunk_type": "table",
            "table_headers": ["Item", "Qty", "Price"],
            "table_type": "line_items",
            "table_row_count": 1,
        }

        payload = build_qdrant_payload(chunk_meta)

        assert payload.get("table_type") == "line_items"
        assert payload.get("table_headers") == ["Item", "Qty", "Price"]
        assert payload.get("table_row_count") == 1

    def test_non_table_chunks_no_table_fields(self):
        """Non-table chunks should not have table metadata."""
        from src.embedding.pipeline.schema_normalizer import build_qdrant_payload

        chunk_meta = {
            "subscription_id": "sub1",
            "profile_id": "p1",
            "document_id": "d1",
            "canonical_text": "John has 10 years experience",
            "chunk_type": "text",
        }

        payload = build_qdrant_payload(chunk_meta)

        assert "table_type" not in payload or payload.get("table_type") is None
        assert "table_headers" not in payload
