"""Multi-column layout detection and reading order reconstruction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LayoutResult:
    column_count: int = 1
    ordered_texts: List[str] = field(default_factory=list)
    ordered_blocks: List[Dict[str, Any]] = field(default_factory=list)


class LayoutAnalyzer:
    """Detect multi-column layouts from bounding box data and reconstruct reading order."""

    def __init__(self, page_width: float = 612.0, column_gap_min: float = 20.0):
        self._page_width = page_width
        self._column_gap_min = column_gap_min

    def analyze(self, blocks: List[Dict[str, Any]]) -> LayoutResult:
        if not blocks:
            return LayoutResult()

        # Separate full-width blocks from potential column blocks
        mid = self._page_width / 2
        full_width = []
        left_blocks = []
        right_blocks = []

        for block in blocks:
            bbox = block.get("bbox", [0, 0, self._page_width, 0])
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            width = x1 - x0

            if width > self._page_width * 0.6:
                full_width.append((y0, block))
            elif x1 <= mid + self._column_gap_min:
                left_blocks.append((y0, block))
            elif x0 >= mid - self._column_gap_min:
                right_blocks.append((y0, block))
            else:
                full_width.append((y0, block))

        # Determine column count
        is_multi = len(left_blocks) >= 2 and len(right_blocks) >= 2
        column_count = 2 if is_multi else 1

        if not is_multi:
            # Single column -- sort all by vertical position
            all_blocks = full_width + left_blocks + right_blocks
            all_blocks.sort(key=lambda item: item[0])
            ordered = [b for _, b in all_blocks]
            return LayoutResult(
                column_count=1,
                ordered_texts=[b.get("text", "") for b in ordered],
                ordered_blocks=ordered,
            )

        # Multi-column: full-width headers first, then left column top-to-bottom, then right
        full_width.sort(key=lambda item: item[0])
        left_blocks.sort(key=lambda item: item[0])
        right_blocks.sort(key=lambda item: item[0])

        ordered = (
            [b for _, b in full_width]
            + [b for _, b in left_blocks]
            + [b for _, b in right_blocks]
        )

        return LayoutResult(
            column_count=2,
            ordered_texts=[b.get("text", "") for b in ordered],
            ordered_blocks=ordered,
        )
