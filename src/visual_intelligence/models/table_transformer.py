"""Table Transformer wrapper for table detection and structure recognition.

Uses Microsoft DETR-based Table Transformer models:
- ``table_det`` for detecting table regions in a page image.
- ``table_str`` for recognizing cell-level structure within a table crop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from PIL import Image

from src.visual_intelligence.datatypes import StructuredTableResult
from src.visual_intelligence.model_pool import get_model_pool

logger = logging.getLogger(__name__)

MIN_TABLE_CONFIDENCE = 0.7
MIN_CELL_CONFIDENCE = 0.5


class TableTransformerDetector:
    """Detect tables and extract cell-level structure using Table Transformer."""

    _DET_KEY = "table_det"
    _STR_KEY = "table_str"

    def __init__(self) -> None:
        self._det_model: Optional[Any] = None
        self._det_processor: Optional[Any] = None
        self._str_model: Optional[Any] = None
        self._str_processor: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image: Image.Image,
        page_number: int,
        table_regions: Optional[List[Dict]] = None,
    ) -> List[StructuredTableResult]:
        """Extract structured tables from *image*.

        Parameters
        ----------
        image:
            Full page image.
        page_number:
            1-based page index for provenance.
        table_regions:
            Optional pre-detected table regions (e.g. from DiT).  Each dict
            must contain ``"bbox"`` and ``"confidence"`` keys.  When provided,
            the detection step is skipped.

        Returns an empty list when models are unavailable or on any error.
        """
        try:
            pool = get_model_pool()

            if not pool.is_available(self._DET_KEY) and table_regions is None:
                logger.debug("Table detection model unavailable; skipping.")
                return []

            if not pool.is_available(self._STR_KEY):
                logger.debug("Table structure model unavailable; skipping.")
                return []

            if not self._ensure_loaded():
                return []

            # Determine table locations
            if table_regions is not None:
                regions = table_regions
            else:
                regions = self._detect_tables(image)

            if not regions:
                return []

            results: List[StructuredTableResult] = []
            for region in regions:
                bbox = region["bbox"]
                crop = image.crop(bbox)
                table = self._recognize_structure(crop, page_number, bbox)
                if table is not None:
                    results.append(table)

            return results

        except Exception:
            logger.warning(
                "Table extraction failed for page %s", page_number,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load detection and structure models from the pool.

        Returns ``True`` if all models and processors are ready.
        """
        if (
            self._det_model is not None
            and self._det_processor is not None
            and self._str_model is not None
            and self._str_processor is not None
        ):
            return True

        try:
            pool = get_model_pool()

            self._det_model = pool.load_model(self._DET_KEY)
            self._det_processor = pool.load_processor(self._DET_KEY)
            self._str_model = pool.load_model(self._STR_KEY)
            self._str_processor = pool.load_processor(self._STR_KEY)

            if any(
                obj is None
                for obj in (
                    self._det_model,
                    self._det_processor,
                    self._str_model,
                    self._str_processor,
                )
            ):
                logger.warning("One or more table models/processors could not be loaded.")
                return False

            return True
        except Exception:
            logger.warning("Failed to load table transformer models.", exc_info=True)
            return False

    def _detect_tables(self, image: Image.Image) -> List[Dict]:
        """Run table detection on *image* and return bounding boxes.

        Each returned dict has ``"bbox"`` (x0, y0, x1, y1) and
        ``"confidence"`` keys.
        """
        pool = get_model_pool()
        device = pool.get_device(self._DET_KEY)

        inputs = self._det_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._det_model(**inputs)

        target_sizes = torch.tensor(
            [[image.height, image.width]], dtype=torch.float32,
        )
        results = self._det_processor.post_process_object_detection(
            outputs, threshold=MIN_TABLE_CONFIDENCE, target_sizes=target_sizes,
        )

        tables: List[Dict] = []
        for result in results:
            for score, box in zip(result["scores"], result["boxes"]):
                conf = float(score.item() if hasattr(score, "item") else score)
                bbox = tuple(
                    b.item() if hasattr(b, "item") else float(b) for b in box
                )
                tables.append({"bbox": bbox, "confidence": conf})

        return tables

    def _recognize_structure(
        self,
        table_image: Image.Image,
        page_number: int,
        bbox: Tuple[float, ...],
    ) -> Optional[StructuredTableResult]:
        """Recognize cell-level structure in a cropped table image.

        Returns a ``StructuredTableResult`` or ``None`` on failure.
        """
        try:
            pool = get_model_pool()
            device = pool.get_device(self._STR_KEY)

            inputs = self._str_processor(images=table_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._str_model(**inputs)

            target_sizes = torch.tensor(
                [[table_image.height, table_image.width]], dtype=torch.float32,
            )
            results = self._str_processor.post_process_object_detection(
                outputs, threshold=MIN_CELL_CONFIDENCE, target_sizes=target_sizes,
            )

            id2label = self._str_model.config.id2label

            cells: List[Dict] = []
            confidences: List[float] = []

            for result in results:
                for score, label_id, box in zip(
                    result["scores"], result["labels"], result["boxes"],
                ):
                    label_name = id2label.get(
                        label_id.item() if hasattr(label_id, "item") else int(label_id),
                        f"class_{label_id}",
                    )

                    # Only keep cell and header detections
                    if "cell" not in label_name.lower() and "header" not in label_name.lower():
                        continue

                    conf = float(score.item() if hasattr(score, "item") else score)
                    cell_bbox = tuple(
                        b.item() if hasattr(b, "item") else float(b) for b in box
                    )
                    confidences.append(conf)
                    cells.append({
                        "bbox": cell_bbox,
                        "label": label_name,
                        "confidence": conf,
                        "text": "",
                    })

            if not cells:
                return None

            # Assign grid positions based on spatial clustering
            cells = self._assign_grid_positions(cells)
            headers, rows = self._cells_to_grid(cells)

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return StructuredTableResult(
                page=page_number,
                bbox=bbox,
                headers=headers,
                rows=rows,
                spans=[],
                confidence=round(avg_conf, 4),
            )

        except Exception:
            logger.warning(
                "Structure recognition failed for table on page %s",
                page_number,
                exc_info=True,
            )
            return None

    def _assign_grid_positions(self, cells: List[Dict]) -> List[Dict]:
        """Cluster cells into rows by y-center proximity and assign row/col indices.

        Cells whose y-centers are within 15 px of each other are placed in the
        same row.  Within each row, cells are sorted left-to-right by x-center.
        """
        ROW_THRESHOLD = 15  # px

        # Compute y-center for each cell
        for cell in cells:
            x0, y0, x1, y1 = cell["bbox"]
            cell["_y_center"] = (y0 + y1) / 2.0
            cell["_x_center"] = (x0 + x1) / 2.0

        # Sort by y-center first
        cells.sort(key=lambda c: c["_y_center"])

        # Cluster into rows
        row_groups: List[List[Dict]] = []
        for cell in cells:
            placed = False
            for group in row_groups:
                group_y = sum(c["_y_center"] for c in group) / len(group)
                if abs(cell["_y_center"] - group_y) <= ROW_THRESHOLD:
                    group.append(cell)
                    placed = True
                    break
            if not placed:
                row_groups.append([cell])

        # Sort rows by average y, then sort cells within each row by x
        row_groups.sort(key=lambda g: sum(c["_y_center"] for c in g) / len(g))

        for row_idx, group in enumerate(row_groups):
            group.sort(key=lambda c: c["_x_center"])
            for col_idx, cell in enumerate(group):
                cell["row"] = row_idx
                cell["col"] = col_idx

        # Flatten back
        flat: List[Dict] = []
        for group in row_groups:
            flat.extend(group)

        return flat

    def _cells_to_grid(self, cells: List[Dict]) -> Tuple[List[str], List[List[str]]]:
        """Convert positioned cells into headers and data rows.

        The first row is treated as headers; remaining rows become data.
        """
        if not cells:
            return [], []

        max_row = max(c["row"] for c in cells)
        max_col = max(c["col"] for c in cells)

        # Build a grid initialized with empty strings
        grid: List[List[str]] = [
            [""] * (max_col + 1) for _ in range(max_row + 1)
        ]

        for cell in cells:
            r, c = cell["row"], cell["col"]
            grid[r][c] = cell.get("text", "")

        headers = grid[0]
        rows = grid[1:] if len(grid) > 1 else []

        return headers, rows
