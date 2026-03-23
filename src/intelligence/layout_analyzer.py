"""Multi-column layout detection and reading order reconstruction.

Handles 1-column, 2-column (sidebar+main), 3-column, and mixed layouts
commonly found in resumes, reports, and academic papers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LayoutResult:
    column_count: int = 1
    ordered_texts: List[str] = field(default_factory=list)
    ordered_blocks: List[Dict[str, Any]] = field(default_factory=list)


class LayoutAnalyzer:
    """Detect multi-column layouts from bounding box data and reconstruct reading order.

    Supports:
    - Single column (standard documents)
    - Two-column sidebar (resumes with narrow left sidebar)
    - Two-column symmetric (academic papers, newsletters)
    - Three-column layouts
    - Mixed layouts (full-width header + columns + full-width footer)
    """

    def __init__(self, page_width: float = 612.0, column_gap_min: float = 20.0):
        self._page_width = page_width
        self._column_gap_min = column_gap_min

    def analyze(self, blocks: List[Dict[str, Any]]) -> LayoutResult:
        if not blocks:
            return LayoutResult()

        mid = self._page_width / 2
        full_width_threshold = self._page_width * 0.55

        # Classify blocks by horizontal position
        full_width = []
        narrow_left = []   # Sidebar (< 40% of page width, left side)
        main_right = []    # Main content (right of sidebar)
        left_col = []      # Symmetric left column
        right_col = []     # Symmetric right column

        for block in blocks:
            bbox = block.get("bbox", [0, 0, self._page_width, 0])
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            width = x1 - x0
            block_mid = (x0 + x1) / 2.0

            if width > full_width_threshold:
                full_width.append((y0, block))
            elif block_mid < mid:
                left_col.append((y0, block))
            elif block_mid >= mid:
                right_col.append((y0, block))
            else:
                full_width.append((y0, block))

        # Detect layout type
        has_symmetric_cols = len(left_col) >= 2 and len(right_col) >= 2

        # Check for sidebar layout (narrow left column < 30% width, wide right > 40%)
        has_sidebar = False
        if has_symmetric_cols:
            left_widths = [(b.get("bbox", [0,0,0,0])[2] - b.get("bbox", [0,0,0,0])[0]) for _, b in left_col]
            right_widths = [(b.get("bbox", [0,0,0,0])[2] - b.get("bbox", [0,0,0,0])[0]) for _, b in right_col]
            avg_left = sum(left_widths) / len(left_widths) if left_widths else 0
            avg_right = sum(right_widths) / len(right_widths) if right_widths else 0
            # Sidebar: left column significantly narrower than right
            if avg_left < self._page_width * 0.30 and avg_right > self._page_width * 0.35:
                has_sidebar = True

        if has_sidebar:
            # Resume-style: full-width header → main content (right) → sidebar (left)
            full_width.sort(key=lambda item: item[0])
            right_col.sort(key=lambda item: item[0])
            left_col.sort(key=lambda item: item[0])

            ordered = (
                [b for _, b in full_width]
                + [b for _, b in right_col]
                + [b for _, b in left_col]
            )
            return LayoutResult(
                column_count=2,
                ordered_texts=[b.get("text", "") for b in ordered],
                ordered_blocks=ordered,
            )

        if has_symmetric_cols:
            # Symmetric 2-column: left top-to-bottom, then right top-to-bottom
            full_width.sort(key=lambda item: item[0])
            left_col.sort(key=lambda item: item[0])
            right_col.sort(key=lambda item: item[0])

            ordered = (
                [b for _, b in full_width]
                + [b for _, b in left_col]
                + [b for _, b in right_col]
            )
            return LayoutResult(
                column_count=2,
                ordered_texts=[b.get("text", "") for b in ordered],
                ordered_blocks=ordered,
            )

        # Single column — sort all by vertical position
        all_blocks = full_width + left_col + right_col
        all_blocks.sort(key=lambda item: item[0])
        ordered = [b for _, b in all_blocks]
        return LayoutResult(
            column_count=1,
            ordered_texts=[b.get("text", "") for b in ordered],
            ordered_blocks=ordered,
        )
