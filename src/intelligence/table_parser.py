"""Structured table parsing for pdfplumber 2D arrays.

Preserves header-cell relationships instead of immediately flattening
tables to CSV text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_NUMERIC_RE = re.compile(r"^[\$\d,.\-\+%()]+$")

_LINE_ITEM_HEADERS = {"description", "item", "product", "service", "qty", "quantity",
                      "unit price", "unit_price", "amount", "total", "rate", "hours"}
_TOTALS_HEADERS = {"subtotal", "tax", "total", "grand total", "balance", "amount due",
                   "discount", "shipping", "net", "gross"}
_SKILLS_HEADERS = {"skill", "technology", "competency", "level", "proficiency",
                   "years", "experience", "rating", "tool"}


@dataclass
class StructuredTable:
    """A parsed table with header and data row structure."""

    headers: List[str] = field(default_factory=list)
    data_rows: List[Dict[str, str]] = field(default_factory=list)
    page: int = 0
    row_count: int = 0
    col_count: int = 0
    table_title: Optional[str] = None
    table_type: Optional[str] = None  # "line_items", "totals", "skills_matrix", etc.

    @property
    def flat_text(self) -> str:
        """CSV-like text representation for backward compatibility."""
        lines = []
        if self.headers:
            lines.append(", ".join(self.headers))
        for row in self.data_rows:
            lines.append(", ".join(str(row.get(h, "")) for h in self.headers))
        return "\n".join(lines)


class TableParser:
    """Parses raw 2D arrays (e.g. from pdfplumber) into StructuredTable."""

    def parse(
        self,
        raw_rows: List[List[Optional[str]]],
        page: int = 0,
        title: Optional[str] = None,
    ) -> StructuredTable:
        """Parse a raw 2D table into a StructuredTable.

        Args:
            raw_rows: List of rows, each row a list of cell values (may contain None).
            page: Page number the table was found on.
            title: Optional table title / caption.

        Returns:
            A StructuredTable with headers and typed data rows.
        """
        if not raw_rows:
            return StructuredTable(page=page, table_title=title)

        # Clean None cells to empty strings
        cleaned: List[List[str]] = []
        for row in raw_rows:
            cleaned.append([cell if cell is not None else "" for cell in row])

        # Normalize column count (pad shorter rows to max width)
        max_cols = max(len(row) for row in cleaned)
        for row in cleaned:
            while len(row) < max_cols:
                row.append("")

        # Detect header row
        first_row = cleaned[0]
        rest_rows = cleaned[1:] if len(cleaned) > 1 else []

        if self._is_header_row(first_row, rest_rows):
            headers = first_row
            data = rest_rows
        else:
            headers = [f"col_{i}" for i in range(max_cols)]
            data = cleaned

        # Build data rows as dicts
        data_rows: List[Dict[str, str]] = []
        for row in data:
            row_dict: Dict[str, str] = {}
            for i, h in enumerate(headers):
                row_dict[h] = row[i] if i < len(row) else ""
            data_rows.append(row_dict)

        return StructuredTable(
            headers=headers,
            data_rows=data_rows,
            page=page,
            row_count=len(data_rows),
            col_count=max_cols,
            table_title=title,
        )

    def _is_header_row(
        self, first_row: List[str], rest_rows: List[List[str]]
    ) -> bool:
        """Determine whether *first_row* looks like a header.

        Heuristic:
        - Count how many non-empty cells in the first row are purely numeric.
        - A high text ratio (>= 0.6) combined with more numeric data rows
          indicates a header.
        - A very high text ratio (>= 0.8) is treated as a header even without
          numeric data rows.
        """
        non_empty = [c for c in first_row if c.strip()]
        if not non_empty:
            return False

        numeric_cells = sum(1 for c in non_empty if _NUMERIC_RE.match(c.strip()))
        non_empty_count = len(non_empty)
        text_ratio = 1.0 - (numeric_cells / non_empty_count)

        if text_ratio >= 0.8:
            return True

        if text_ratio >= 0.6 and rest_rows:
            # Check whether data rows have more numeric content
            data_numeric = 0
            data_total = 0
            for row in rest_rows:
                for c in row:
                    if c.strip():
                        data_total += 1
                        if _NUMERIC_RE.match(c.strip()):
                            data_numeric += 1
            if data_total > 0 and (data_numeric / data_total) > (
                numeric_cells / non_empty_count
            ):
                return True

        return False

    def classify_table_type(self, table: StructuredTable) -> str:
        """Classify table semantic type based on column header patterns."""
        if not table.headers:
            return "generic"

        header_lower = {h.lower().strip() for h in table.headers}

        # Check column header overlap with known patterns
        line_item_overlap = len(header_lower & _LINE_ITEM_HEADERS)
        totals_overlap = len(header_lower & _TOTALS_HEADERS)
        skills_overlap = len(header_lower & _SKILLS_HEADERS)

        # Also check first-column values for totals (tables without proper headers)
        if table.data_rows:
            first_col_vals = {row.get(table.headers[0], "").lower().strip() for row in table.data_rows}
            totals_overlap += len(first_col_vals & _TOTALS_HEADERS)

        if line_item_overlap >= 2:
            return "line_items"
        if totals_overlap >= 2:
            return "totals"
        if skills_overlap >= 2:
            return "skills_matrix"
        return "generic"
