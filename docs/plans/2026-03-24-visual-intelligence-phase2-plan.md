# Visual Intelligence Layer — Phase 2: Tables + OCR

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Table Transformer (table detection + cell-level structure recognition) and TrOCR (enhanced OCR for low-confidence regions), plus parallel KG enrichment for visual entities.

**Architecture:** Extends Phase 1's visual intelligence layer with two new model wrappers plugged into the existing orchestrator, plus a KG enricher that fires async after merging.

**Tech Stack:** HuggingFace transformers (Table Transformer DETR, TrOCR), Neo4j (KG writes)

**Prerequisite:** Phase 1 complete (all 11 tasks passing)

---

### Task 1: Table Transformer Detection Wrapper

**Files:**
- Create: `src/visual_intelligence/models/table_transformer.py`
- Test: `tests/test_visual_intelligence_table.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_table.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import VisualRegion, StructuredTableResult


def test_table_detector_creation():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    with patch("src.visual_intelligence.model_pool.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        detector = TableTransformerDetector()
        assert detector is not None


def test_table_detect_returns_bboxes():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True
    mock_pool.load_model.return_value = MagicMock()
    mock_pool.load_processor.return_value = MagicMock()
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        with patch.object(TableTransformerDetector, "_detect_tables") as mock_det:
            mock_det.return_value = [
                {"bbox": (100, 200, 500, 600), "confidence": 0.92},
            ]
            with patch.object(TableTransformerDetector, "_recognize_structure") as mock_str:
                mock_str.return_value = StructuredTableResult(
                    page=1,
                    bbox=(100, 200, 500, 600),
                    headers=["Col1", "Col2"],
                    rows=[["A", "1"], ["B", "2"]],
                    spans=[],
                    confidence=0.88,
                )
                from src.visual_intelligence.models.table_transformer import TableTransformerDetector
                detector = TableTransformerDetector()
                img = Image.new("RGB", (800, 1000), "white")
                tables = detector.extract(img, page_number=1)
                assert len(tables) == 1
                assert tables[0].headers == ["Col1", "Col2"]
                assert len(tables[0].rows) == 2


def test_table_unavailable_returns_empty():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        detector = TableTransformerDetector()
        img = Image.new("RGB", (800, 1000), "white")
        tables = detector.extract(img, page_number=1)
        assert tables == []


def test_table_structure_cell_extraction():
    from src.visual_intelligence.models.table_transformer import TableTransformerDetector
    detector = TableTransformerDetector.__new__(TableTransformerDetector)
    # Test the cell grid building logic
    cells = [
        {"bbox": (100, 200, 250, 240), "row": 0, "col": 0, "text": "Name"},
        {"bbox": (260, 200, 400, 240), "row": 0, "col": 1, "text": "Value"},
        {"bbox": (100, 250, 250, 290), "row": 1, "col": 0, "text": "A"},
        {"bbox": (260, 250, 400, 290), "row": 1, "col": 1, "text": "1"},
    ]
    headers, rows = detector._cells_to_grid(cells)
    assert headers == ["Name", "Value"]
    assert rows == [["A", "1"]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_table.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/models/table_transformer.py
"""Table Transformer — table detection and cell-level structure recognition."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.visual_intelligence.datatypes import StructuredTableResult

logger = logging.getLogger(__name__)

MIN_TABLE_CONFIDENCE = 0.7
MIN_CELL_CONFIDENCE = 0.5


class TableTransformerDetector:
    """Detects tables and extracts cell-level structure using DETR-based Table Transformer."""

    def __init__(self) -> None:
        self._det_model = None
        self._det_processor = None
        self._str_model = None
        self._str_processor = None

    def _ensure_loaded(self) -> bool:
        if self._det_model is not None and self._str_model is not None:
            return True
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("table_det") or not pool.is_available("table_str"):
            return False
        self._det_model = pool.load_model("table_det")
        self._det_processor = pool.load_processor("table_det")
        self._str_model = pool.load_model("table_str")
        self._str_processor = pool.load_processor("table_str")
        return all(x is not None for x in [
            self._det_model, self._det_processor, self._str_model, self._str_processor
        ])

    def extract(
        self,
        image: Image.Image,
        page_number: int,
        table_regions: Optional[List[Dict]] = None,
    ) -> List[StructuredTableResult]:
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("table_det"):
            return []
        if not self._ensure_loaded():
            return []

        try:
            # Step 1: Detect table regions (or use DiT-provided regions)
            if table_regions:
                detected = table_regions
            else:
                detected = self._detect_tables(image)

            if not detected:
                return []

            # Step 2: For each detected table, extract cell structure
            results: List[StructuredTableResult] = []
            for table_info in detected:
                bbox = table_info["bbox"]
                cropped = image.crop(bbox)
                structured = self._recognize_structure(cropped, page_number, bbox)
                if structured:
                    results.append(structured)

            return results
        except Exception as exc:
            logger.warning("Table extraction failed on page %d: %s", page_number, exc)
            return []

    def _detect_tables(self, image: Image.Image) -> List[Dict]:
        from src.visual_intelligence.model_pool import get_model_pool
        import torch

        pool = get_model_pool()
        device = pool.get_device("table_det")
        inputs = self._det_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._det_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self._det_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=MIN_TABLE_CONFIDENCE
        )[0]

        detected = []
        for score, box in zip(
            results["scores"].cpu().tolist(),
            results["boxes"].cpu().tolist(),
        ):
            detected.append({"bbox": tuple(box), "confidence": score})

        return detected

    def _recognize_structure(
        self,
        table_image: Image.Image,
        page_number: int,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[StructuredTableResult]:
        from src.visual_intelligence.model_pool import get_model_pool
        import torch

        pool = get_model_pool()
        device = pool.get_device("table_str")
        inputs = self._str_processor(images=table_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._str_model(**inputs)

        target_sizes = torch.tensor([table_image.size[::-1]])
        results = self._str_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=MIN_CELL_CONFIDENCE
        )[0]

        id2label = self._str_model.config.id2label
        cells = []
        for score, label_id, box in zip(
            results["scores"].cpu().tolist(),
            results["labels"].cpu().tolist(),
            results["boxes"].cpu().tolist(),
        ):
            label = id2label.get(label_id, "")
            if "cell" in label.lower() or "header" in label.lower():
                cells.append({
                    "bbox": tuple(box),
                    "label": label,
                    "confidence": score,
                })

        if not cells:
            return None

        # Sort cells into grid by position
        cells_with_pos = self._assign_grid_positions(cells)
        headers, rows = self._cells_to_grid(cells_with_pos)

        return StructuredTableResult(
            page=page_number,
            bbox=bbox,
            headers=headers,
            rows=rows,
            spans=[],  # TODO: detect merged cells
            confidence=min(c.get("confidence", 0) for c in cells) if cells else 0.0,
        )

    def _assign_grid_positions(self, cells: List[Dict]) -> List[Dict]:
        if not cells:
            return []

        # Sort by y-center then x-center
        for c in cells:
            x0, y0, x1, y1 = c["bbox"]
            c["y_center"] = (y0 + y1) / 2
            c["x_center"] = (x0 + x1) / 2

        # Cluster into rows by y-center proximity
        cells.sort(key=lambda c: c["y_center"])
        rows_clusters: List[List[Dict]] = []
        current_row: List[Dict] = [cells[0]]
        for c in cells[1:]:
            if abs(c["y_center"] - current_row[-1]["y_center"]) < 15:
                current_row.append(c)
            else:
                rows_clusters.append(current_row)
                current_row = [c]
        rows_clusters.append(current_row)

        # Assign row/col indices
        result = []
        for row_idx, row_cells in enumerate(rows_clusters):
            row_cells.sort(key=lambda c: c["x_center"])
            for col_idx, cell in enumerate(row_cells):
                cell["row"] = row_idx
                cell["col"] = col_idx
                cell["text"] = cell.get("text", "")
                result.append(cell)

        return result

    def _cells_to_grid(
        self, cells: List[Dict]
    ) -> Tuple[List[str], List[List[str]]]:
        if not cells:
            return [], []

        max_row = max(c["row"] for c in cells)
        max_col = max(c["col"] for c in cells)

        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for c in cells:
            grid[c["row"]][c["col"]] = c.get("text", "")

        headers = grid[0] if grid else []
        rows = grid[1:] if len(grid) > 1 else []
        return headers, rows
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_table.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/models/table_transformer.py tests/test_visual_intelligence_table.py
git commit -m "feat(visual-intelligence): add Table Transformer wrapper for table detection and structure recognition"
```

---

### Task 2: TrOCR Enhancer Wrapper

**Files:**
- Create: `src/visual_intelligence/models/trocr_enhancer.py`
- Test: `tests/test_visual_intelligence_trocr.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_trocr.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import OCRPatch


def test_trocr_enhancer_creation():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer
    with patch("src.visual_intelligence.model_pool.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        enhancer = TrOCREnhancer()
        assert enhancer is not None


def test_trocr_enhance_low_confidence_region():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer

    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_pool.load_model.return_value = mock_model
    mock_pool.load_processor.return_value = mock_processor
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        with patch.object(TrOCREnhancer, "_run_trocr") as mock_run:
            mock_run.return_value = ("Enhanced text here", 0.92)
            enhancer = TrOCREnhancer()
            img = Image.new("RGB", (200, 50), "white")
            patch_result = enhancer.enhance_region(
                image=img,
                page=1,
                bbox=(10, 20, 210, 70),
                original_text="Enhancod toxt hore",
                original_confidence=0.45,
            )
            assert isinstance(patch_result, OCRPatch)
            assert patch_result.enhanced_text == "Enhanced text here"
            assert patch_result.enhanced_confidence == 0.92
            assert patch_result.method == "trocr_printed"


def test_trocr_skips_high_confidence():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer

    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        enhancer = TrOCREnhancer()
        img = Image.new("RGB", (200, 50), "white")
        result = enhancer.enhance_region(
            image=img,
            page=1,
            bbox=(10, 20, 210, 70),
            original_text="Good text",
            original_confidence=0.95,
        )
        assert result is None  # No enhancement needed


def test_trocr_unavailable_returns_none():
    from src.visual_intelligence.models.trocr_enhancer import TrOCREnhancer
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        enhancer = TrOCREnhancer()
        img = Image.new("RGB", (200, 50), "white")
        result = enhancer.enhance_region(
            image=img, page=1, bbox=(0, 0, 200, 50),
            original_text="bad", original_confidence=0.3,
        )
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_trocr.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/models/trocr_enhancer.py
"""TrOCR — enhanced OCR for low-confidence text regions."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from PIL import Image

from src.visual_intelligence.datatypes import OCRPatch

logger = logging.getLogger(__name__)

# Only enhance regions below this confidence
ENHANCEMENT_THRESHOLD = 0.70


class TrOCREnhancer:
    """Enhances OCR output for low-confidence regions using TrOCR."""

    def __init__(self) -> None:
        self._printed_model = None
        self._printed_processor = None
        self._handwritten_model = None
        self._handwritten_processor = None

    def _ensure_printed(self) -> bool:
        if self._printed_model is not None:
            return True
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("trocr_printed"):
            return False
        self._printed_model = pool.load_model("trocr_printed")
        self._printed_processor = pool.load_processor("trocr_printed")
        return self._printed_model is not None

    def _ensure_handwritten(self) -> bool:
        if self._handwritten_model is not None:
            return True
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("trocr_handwritten"):
            return False
        self._handwritten_model = pool.load_model("trocr_handwritten")
        self._handwritten_processor = pool.load_processor("trocr_handwritten")
        return self._handwritten_model is not None

    def enhance_region(
        self,
        image: Image.Image,
        page: int,
        bbox: Tuple[float, float, float, float],
        original_text: str,
        original_confidence: float,
        is_handwritten: bool = False,
    ) -> Optional[OCRPatch]:
        # Skip if confidence is already good
        if original_confidence >= ENHANCEMENT_THRESHOLD:
            return None

        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()

        model_key = "trocr_handwritten" if is_handwritten else "trocr_printed"
        if not pool.is_available(model_key):
            return None

        try:
            enhanced_text, enhanced_conf = self._run_trocr(image, is_handwritten)
            if enhanced_text and enhanced_conf > original_confidence:
                return OCRPatch(
                    page=page,
                    bbox=bbox,
                    original_text=original_text,
                    enhanced_text=enhanced_text,
                    original_confidence=original_confidence,
                    enhanced_confidence=enhanced_conf,
                    method=f"trocr_{'handwritten' if is_handwritten else 'printed'}",
                )
            return None
        except Exception as exc:
            logger.debug("TrOCR enhancement failed: %s", exc)
            return None

    def _run_trocr(
        self, image: Image.Image, is_handwritten: bool = False
    ) -> Tuple[str, float]:
        import torch
        from src.visual_intelligence.model_pool import get_model_pool

        pool = get_model_pool()

        if is_handwritten:
            self._ensure_handwritten()
            model = self._handwritten_model
            processor = self._handwritten_processor
            device = pool.get_device("trocr_handwritten")
        else:
            self._ensure_printed()
            model = self._printed_model
            processor = self._printed_processor
            device = pool.get_device("trocr_printed")

        if model is None or processor is None:
            return "", 0.0

        # TrOCR expects RGB images
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated = model.generate(
                pixel_values,
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]

        # Estimate confidence from generation scores
        if hasattr(generated, "scores") and generated.scores:
            import torch.nn.functional as F
            probs = [F.softmax(s, dim=-1).max().item() for s in generated.scores]
            confidence = sum(probs) / len(probs) if probs else 0.5
        else:
            confidence = 0.75  # Default if scores unavailable

        return text.strip(), confidence
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_trocr.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/models/trocr_enhancer.py tests/test_visual_intelligence_trocr.py
git commit -m "feat(visual-intelligence): add TrOCR enhancer for low-confidence OCR regions"
```

---

### Task 3: KG Enricher

**Files:**
- Create: `src/visual_intelligence/kg_enricher.py`
- Test: `tests/test_visual_intelligence_kg.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_kg.py
import pytest
from unittest.mock import MagicMock, patch

from src.visual_intelligence.datatypes import (
    VisualEnrichmentResult, VisualRegion, StructuredTableResult, KVPair,
)


def test_kg_enricher_builds_payload():
    from src.visual_intelligence.kg_enricher import VisualKGEnricher
    enricher = VisualKGEnricher()

    result = VisualEnrichmentResult(doc_id="doc-kg-1")
    result.regions = [
        VisualRegion(bbox=(10, 20, 300, 400), label="table", confidence=0.9, page=1),
        VisualRegion(bbox=(10, 420, 300, 500), label="figure", confidence=0.85, page=1),
    ]
    result.tables = [
        StructuredTableResult(page=1, bbox=(10, 20, 300, 400),
                             headers=["A", "B"], rows=[["1", "2"]], spans=[], confidence=0.88),
    ]
    result.kv_pairs = [
        KVPair(key="Invoice No", value="12345", confidence=0.91, page=1),
    ]

    payload = enricher.build_payload(
        doc_id="doc-kg-1",
        subscription_id="sub-1",
        profile_id="prof-1",
        result=result,
    )

    assert payload["doc_id"] == "doc-kg-1"
    assert len(payload["nodes"]) >= 4  # 2 regions + 1 table + 1 kv
    assert len(payload["edges"]) >= 3  # doc->region, region->table, kv->region


def test_kg_enricher_enqueue_fires_and_forgets():
    from src.visual_intelligence.kg_enricher import VisualKGEnricher
    enricher = VisualKGEnricher()

    result = VisualEnrichmentResult(doc_id="doc-kg-2")
    result.regions = [
        VisualRegion(bbox=(10, 20, 300, 400), label="text", confidence=0.9, page=1),
    ]

    with patch.object(enricher, "_enqueue") as mock_enqueue:
        enricher.enqueue_enrichment(
            doc_id="doc-kg-2",
            subscription_id="sub-1",
            profile_id="prof-1",
            result=result,
        )
        mock_enqueue.assert_called_once()


def test_kg_enricher_empty_result_skips():
    from src.visual_intelligence.kg_enricher import VisualKGEnricher
    enricher = VisualKGEnricher()

    result = VisualEnrichmentResult(doc_id="doc-empty")
    # No regions, tables, or kv_pairs

    with patch.object(enricher, "_enqueue") as mock_enqueue:
        enricher.enqueue_enrichment(
            doc_id="doc-empty",
            subscription_id="sub-1",
            profile_id="prof-1",
            result=result,
        )
        mock_enqueue.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_kg.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/kg_enricher.py
"""Parallel KG enrichment for visual intelligence entities."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from src.visual_intelligence.datatypes import (
    VisualEnrichmentResult,
    VisualRegion,
    StructuredTableResult,
    KVPair,
)

logger = logging.getLogger(__name__)


class VisualKGEnricher:
    """Writes visual intelligence entities to Knowledge Graph (non-blocking, non-fatal)."""

    def build_payload(
        self,
        doc_id: str,
        subscription_id: str,
        profile_id: str,
        result: VisualEnrichmentResult,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Layout regions → nodes
        for region in result.regions:
            node_id = f"region_{doc_id}_{region.page}_{region.label}_{id(region)}"
            nodes.append({
                "id": node_id,
                "type": "LayoutRegion",
                "properties": {
                    "label": region.label,
                    "bbox": list(region.bbox),
                    "page": region.page,
                    "confidence": region.confidence,
                },
            })
            edges.append({
                "from": doc_id,
                "to": node_id,
                "type": "HAS_REGION",
            })

        # Structured tables → nodes with edges to their region
        for table in result.tables:
            table_id = f"table_{doc_id}_{table.page}_{id(table)}"
            nodes.append({
                "id": table_id,
                "type": "StructuredTable",
                "properties": {
                    "page": table.page,
                    "headers": table.headers,
                    "num_rows": len(table.rows),
                    "num_cols": len(table.headers),
                    "confidence": table.confidence,
                },
            })
            # Find matching region
            matching_region = next(
                (r for r in result.regions
                 if r.page == table.page and r.label == "table"),
                None,
            )
            if matching_region:
                region_id = f"region_{doc_id}_{matching_region.page}_{matching_region.label}_{id(matching_region)}"
                edges.append({
                    "from": region_id,
                    "to": table_id,
                    "type": "CONTAINS",
                })
            else:
                edges.append({
                    "from": doc_id,
                    "to": table_id,
                    "type": "HAS_TABLE",
                })

        # KV pairs → nodes
        for kv in result.kv_pairs:
            kv_id = f"formfield_{doc_id}_{kv.page}_{kv.key}_{id(kv)}"
            nodes.append({
                "id": kv_id,
                "type": "FormField",
                "properties": {
                    "key": kv.key,
                    "value": kv.value,
                    "confidence": kv.confidence,
                    "page": kv.page,
                },
            })
            # Find matching region
            matching_region = next(
                (r for r in result.regions if r.page == kv.page),
                None,
            )
            if matching_region:
                region_id = f"region_{doc_id}_{matching_region.page}_{matching_region.label}_{id(matching_region)}"
                edges.append({
                    "from": kv_id,
                    "to": region_id,
                    "type": "FOUND_IN",
                })

        return {
            "doc_id": doc_id,
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "nodes": nodes,
            "edges": edges,
        }

    def enqueue_enrichment(
        self,
        doc_id: str,
        subscription_id: str,
        profile_id: str,
        result: VisualEnrichmentResult,
    ) -> None:
        if not result.regions and not result.tables and not result.kv_pairs:
            logger.debug("No visual entities to write to KG for doc %s", doc_id)
            return

        payload = self.build_payload(doc_id, subscription_id, profile_id, result)
        self._enqueue(payload)

    def _enqueue(self, payload: Dict[str, Any]) -> None:
        """Fire-and-forget KG write via daemon thread."""
        thread = threading.Thread(
            target=self._write_to_kg,
            args=(payload,),
            daemon=True,
            name=f"visual-kg-{payload['doc_id']}",
        )
        thread.start()

    def _write_to_kg(self, payload: Dict[str, Any]) -> None:
        try:
            from src.kg.ingest import GraphIngestQueue, GraphIngestPayload
            queue = GraphIngestQueue()
            kg_payload = GraphIngestPayload(
                document_id=payload["doc_id"],
                subscription_id=payload["subscription_id"],
                profile_id=payload["profile_id"],
                nodes=payload["nodes"],
                edges=payload["edges"],
                source="visual_intelligence",
            )
            queue.enqueue(kg_payload)
            logger.debug("Enqueued %d visual KG nodes for doc %s",
                        len(payload["nodes"]), payload["doc_id"])
        except ImportError:
            logger.debug("KG module not available, skipping visual KG enrichment")
        except Exception as exc:
            logger.warning("Visual KG enrichment failed for doc %s: %s",
                          payload["doc_id"], exc)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_kg.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/kg_enricher.py tests/test_visual_intelligence_kg.py
git commit -m "feat(visual-intelligence): add KG enricher for parallel visual entity writes"
```

---

### Task 4: Orchestrator Updates — Wire Table Transformer + TrOCR + KG

**Files:**
- Modify: `src/visual_intelligence/orchestrator.py`
- Test: `tests/test_visual_intelligence_orchestrator_p2.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_orchestrator_p2.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, StructuredTableResult, OCRPatch,
)
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_orchestrator_runs_table_transformer_on_table_regions():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.6)]
        extracted.sections = []
        extracted.full_text = "text"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.3, has_tables=True, has_forms=False),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        # DiT finds a table region
        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 200, 500, 600), label="table", confidence=0.9, page=1),
        ])

        # Table Transformer extracts structure
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[
            StructuredTableResult(page=1, bbox=(100, 200, 500, 600),
                                 headers=["A", "B"], rows=[["1", "2"]], spans=[], confidence=0.88),
        ])

        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)

        import fitz
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "t")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-p2", extracted, pdf_bytes)
        orch.table_detector.extract.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_fires_kg_enrichment():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = []
        extracted.sections = []
        extracted.full_text = "text"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.2, has_tables=True, has_forms=False),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(10, 20, 300, 400), label="text", confidence=0.9, page=1),
        ])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)

        mock_kg = MagicMock()
        orch.kg_enricher = mock_kg

        import fitz
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "t")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich(
            "doc-kg", extracted, pdf_bytes,
            subscription_id="sub-1", profile_id="prof-1",
        )
        mock_kg.enqueue_enrichment.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_orchestrator_p2.py -v`
Expected: FAIL (orchestrator doesn't have table_detector, trocr_enhancer, kg_enricher yet)

**Step 3: Update orchestrator**

Update `src/visual_intelligence/orchestrator.py` to add:

1. Lazy properties for `table_detector`, `trocr_enhancer`, `kg_enricher`
2. In `_process_page`: after DiT, run Table Transformer on table regions (Tier 1+) and TrOCR on low-confidence regions (Tier 2)
3. In `enrich`: after merging, fire KG enrichment async
4. Accept optional `subscription_id` and `profile_id` params in `enrich()`

Key additions to `_process_page`:
```python
# After DiT detection:
if tier >= Tier.LIGHT:
    table_regions = [r for r in dit_regions if r.label == "table"]
    if table_regions:
        tables = self.table_detector.extract(
            page_img.image, page_img.page_number,
            table_regions=[{"bbox": r.bbox, "confidence": r.confidence} for r in table_regions]
        )
        # Store tables in result

if tier >= Tier.FULL:
    # TrOCR on low-confidence regions
    for region in dit_regions:
        if region.label == "text":
            patch = self.trocr_enhancer.enhance_region(...)
```

Key addition to `enrich`:
```python
# After merger, fire KG
if subscription_id and profile_id:
    self.kg_enricher.enqueue_enrichment(doc_id, subscription_id, profile_id, visual_result)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_orchestrator_p2.py -v`
Expected: PASS (2 tests)

**Step 5: Run all Phase 1+2 tests**

Run: `pytest tests/test_visual_intelligence_*.py -v`
Expected: All passing

**Step 6: Commit**

```bash
git add src/visual_intelligence/orchestrator.py tests/test_visual_intelligence_orchestrator_p2.py
git commit -m "feat(visual-intelligence): wire Table Transformer, TrOCR, and KG enricher into orchestrator"
```

---

### Task 5: Phase 2 End-to-End Smoke Test

**Files:**
- Create: `tests/test_visual_intelligence_e2e_p2.py`

**Step 1: Write the smoke test**

```python
# tests/test_visual_intelligence_e2e_p2.py
"""End-to-end smoke test for Phase 2 (Tables + OCR + KG)."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, StructuredTableResult, OCRPatch,
)
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_full_pipeline_table_and_ocr():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.sections = []
        extracted.tables = [MagicMock(page=1, csv="a,b\n1,2")]
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.4)]
        extracted.full_text = "Garblod toxt on page 1"
        extracted.metrics = {"total_pages": 1}

        # Tier 2 (complex)
        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.4,
                          image_ratio=0.7, has_tables=True, has_forms=True),
        ])

        mock_img = Image.new("RGB", (2550, 3300), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=2550, height=3300, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 50, 2400, 1500), label="text", confidence=0.93, page=1),
            VisualRegion(bbox=(100, 1520, 2400, 2800), label="table", confidence=0.89, page=1),
        ])

        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[
            StructuredTableResult(page=1, bbox=(100, 1520, 2400, 2800),
                                 headers=["Name", "Amount"], rows=[["Rent", "$1200"]],
                                 spans=[], confidence=0.87),
        ])

        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=OCRPatch(
            page=1, bbox=(100, 50, 2400, 1500),
            original_text="Garblod toxt",
            enhanced_text="Garbled text",
            original_confidence=0.4,
            enhanced_confidence=0.92,
            method="trocr_printed",
        ))

        mock_kg = MagicMock()
        orch.kg_enricher = mock_kg

        import fitz
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "t")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich(
            "doc-e2e-p2", extracted, pdf_bytes,
            subscription_id="sub-1", profile_id="prof-1",
        )

        # Verify all models were invoked
        orch.dit_detector.detect.assert_called_once()
        orch.table_detector.extract.assert_called_once()
        orch.trocr_enhancer.enhance_region.assert_called()
        mock_kg.enqueue_enrichment.assert_called_once()
        assert result.metrics.get("visual_intelligence_applied") is True
```

**Step 2: Run test**

Run: `pytest tests/test_visual_intelligence_e2e_p2.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_visual_intelligence_e2e_p2.py
git commit -m "test(visual-intelligence): add Phase 2 end-to-end smoke test"
```

---

## Phase 2 Complete Checklist

After all 5 tasks:

- [ ] `src/visual_intelligence/models/table_transformer.py`
- [ ] `src/visual_intelligence/models/trocr_enhancer.py`
- [ ] `src/visual_intelligence/kg_enricher.py`
- [ ] `src/visual_intelligence/orchestrator.py` updated (table + OCR + KG wired in)
- [ ] 4 new test files passing

**All tests passing:** `pytest tests/test_visual_intelligence_*.py -v`
