# Visual Intelligence Layer — Phase 3: Deep Understanding

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LayoutLMv3 for semantic document understanding (key information extraction, form field associations, token classification), plus batch inference optimization.

**Architecture:** LayoutLMv3 processes Tier 2 pages with full page image + OCR text + bounding boxes. Outputs semantic labels per token and spatial key-value pairs, fed into Enrichment Merger.

**Tech Stack:** HuggingFace transformers (LayoutLMv3), torch

**Prerequisite:** Phase 1 + Phase 2 complete

---

### Task 1: LayoutLMv3 Wrapper

**Files:**
- Create: `src/visual_intelligence/models/layoutlmv3.py`
- Test: `tests/test_visual_intelligence_layoutlmv3.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_layoutlmv3.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import KVPair, VisualRegion


def test_layoutlmv3_creation():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
    with patch("src.visual_intelligence.model_pool.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        extractor = LayoutLMv3Extractor()
        assert extractor is not None


def test_layoutlmv3_extracts_kv_pairs():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor

    mock_pool = MagicMock()
    mock_pool.is_available.return_value = True
    mock_pool.load_model.return_value = MagicMock()
    mock_pool.load_processor.return_value = MagicMock()
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        with patch.object(LayoutLMv3Extractor, "_run_inference") as mock_inf:
            mock_inf.return_value = {
                "token_labels": [
                    {"text": "Invoice No", "label": "question", "bbox": (10, 20, 100, 40)},
                    {"text": "12345", "label": "answer", "bbox": (110, 20, 200, 40)},
                    {"text": "Date", "label": "question", "bbox": (10, 50, 60, 70)},
                    {"text": "2026-01-15", "label": "answer", "bbox": (70, 50, 180, 70)},
                ],
            }
            extractor = LayoutLMv3Extractor()
            img = Image.new("RGB", (800, 1000), "white")
            words = ["Invoice", "No", "12345", "Date", "2026-01-15"]
            boxes = [(10, 20, 60, 40), (65, 20, 100, 40), (110, 20, 200, 40),
                     (10, 50, 60, 70), (70, 50, 180, 70)]

            kv_pairs = extractor.extract(img, words, boxes, page_number=1)
            assert len(kv_pairs) == 2
            assert kv_pairs[0].key == "Invoice No"
            assert kv_pairs[0].value == "12345"
            assert kv_pairs[1].key == "Date"
            assert kv_pairs[1].value == "2026-01-15"


def test_layoutlmv3_unavailable_returns_empty():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        extractor = LayoutLMv3Extractor()
        img = Image.new("RGB", (800, 1000), "white")
        kv_pairs = extractor.extract(img, [], [], page_number=1)
        assert kv_pairs == []


def test_layoutlmv3_pairs_question_answer_tokens():
    from src.visual_intelligence.models.layoutlmv3 import LayoutLMv3Extractor
    extractor = LayoutLMv3Extractor.__new__(LayoutLMv3Extractor)

    tokens = [
        {"text": "Name", "label": "question", "bbox": (10, 10, 50, 20)},
        {"text": "John Doe", "label": "answer", "bbox": (60, 10, 150, 20)},
        {"text": "Some text", "label": "other", "bbox": (10, 30, 100, 40)},
        {"text": "Total", "label": "question", "bbox": (10, 50, 50, 60)},
        {"text": "$500", "label": "answer", "bbox": (60, 50, 100, 60)},
    ]

    pairs = extractor._pair_question_answer(tokens, page=1)
    assert len(pairs) == 2
    assert pairs[0].key == "Name"
    assert pairs[0].value == "John Doe"
    assert pairs[1].key == "Total"
    assert pairs[1].value == "$500"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_layoutlmv3.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/models/layoutlmv3.py
"""LayoutLMv3 — multi-modal document understanding for KIE and form extraction."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.visual_intelligence.datatypes import KVPair

logger = logging.getLogger(__name__)

# LayoutLMv3 token classification labels (FUNSD-style)
LABEL_MAP = {
    0: "other",
    1: "question",  # B-QUESTION
    2: "question",  # I-QUESTION
    3: "answer",    # B-ANSWER
    4: "answer",    # I-ANSWER
    5: "header",    # B-HEADER
    6: "header",    # I-HEADER
}


class LayoutLMv3Extractor:
    """Extracts key-value pairs and semantic labels using LayoutLMv3."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("layoutlmv3"):
            return False
        self._model = pool.load_model("layoutlmv3")
        self._processor = pool.load_processor("layoutlmv3")
        return self._model is not None and self._processor is not None

    def extract(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[Tuple[int, int, int, int]],
        page_number: int,
    ) -> List[KVPair]:
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("layoutlmv3"):
            return []

        if not words or not boxes:
            return []

        if not self._ensure_loaded():
            return []

        try:
            result = self._run_inference(image, words, boxes)
            token_labels = result.get("token_labels", [])
            return self._pair_question_answer(token_labels, page_number)
        except Exception as exc:
            logger.warning("LayoutLMv3 extraction failed on page %d: %s", page_number, exc)
            return []

    def _run_inference(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[Tuple[int, int, int, int]],
    ) -> Dict[str, Any]:
        import torch
        from src.visual_intelligence.model_pool import get_model_pool

        pool = get_model_pool()
        device = pool.get_device("layoutlmv3")

        # Normalize boxes to 0-1000 range (LayoutLMv3 convention)
        w, h = image.size
        normalized_boxes = []
        for box in boxes:
            x0, y0, x1, y1 = box
            normalized_boxes.append([
                int(x0 * 1000 / w), int(y0 * 1000 / h),
                int(x1 * 1000 / w), int(y1 * 1000 / h),
            ])

        encoding = self._processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().cpu().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]

        # Map predictions back to tokens
        token_labels = []
        word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else None

        if word_ids:
            prev_word_id = None
            for idx, word_id in enumerate(word_ids):
                if word_id is None or word_id == prev_word_id:
                    continue
                if idx < len(predictions):
                    label = LABEL_MAP.get(predictions[idx], "other")
                    token_labels.append({
                        "text": words[word_id] if word_id < len(words) else "",
                        "label": label,
                        "bbox": boxes[word_id] if word_id < len(boxes) else (0, 0, 0, 0),
                    })
                prev_word_id = word_id
        else:
            # Fallback: direct mapping
            for i, word in enumerate(words):
                if i < len(predictions):
                    label = LABEL_MAP.get(predictions[i], "other")
                    token_labels.append({
                        "text": word,
                        "label": label,
                        "bbox": boxes[i] if i < len(boxes) else (0, 0, 0, 0),
                    })

        return {"token_labels": token_labels}

    def _pair_question_answer(
        self, tokens: List[Dict[str, Any]], page: int
    ) -> List[KVPair]:
        pairs: List[KVPair] = []
        current_question: List[str] = []
        current_q_bbox = None
        current_answer: List[str] = []
        current_a_bbox = None

        def flush():
            if current_question and current_answer:
                pairs.append(KVPair(
                    key=" ".join(current_question),
                    value=" ".join(current_answer),
                    confidence=0.8,  # Default; could be refined with model scores
                    page=page,
                    bbox=current_q_bbox,
                    source="layoutlmv3",
                ))

        for token in tokens:
            label = token["label"]
            text = token["text"]
            bbox = token.get("bbox")

            if label == "question":
                if current_answer:
                    flush()
                    current_question = []
                    current_answer = []
                current_question.append(text)
                if not current_q_bbox:
                    current_q_bbox = bbox
            elif label == "answer":
                current_answer.append(text)
                if not current_a_bbox:
                    current_a_bbox = bbox
            elif label == "other" and current_answer:
                flush()
                current_question = []
                current_answer = []
                current_q_bbox = None
                current_a_bbox = None

        flush()
        return pairs
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_layoutlmv3.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/models/layoutlmv3.py tests/test_visual_intelligence_layoutlmv3.py
git commit -m "feat(visual-intelligence): add LayoutLMv3 wrapper for KIE and form understanding"
```

---

### Task 2: Wire LayoutLMv3 into Orchestrator

**Files:**
- Modify: `src/visual_intelligence/orchestrator.py`
- Test: `tests/test_visual_intelligence_orchestrator_p3.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_orchestrator_p3.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, KVPair,
)
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_orchestrator_runs_layoutlmv3_on_tier2():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.4)]
        extracted.sections = []
        extracted.full_text = "Invoice No 12345"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.4,
                          image_ratio=0.8, has_tables=False, has_forms=True),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(10, 20, 700, 900), label="text", confidence=0.9, page=1),
        ])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)

        # LayoutLMv3 extracts KV pairs
        orch.layoutlmv3_extractor = MagicMock()
        orch.layoutlmv3_extractor.extract = MagicMock(return_value=[
            KVPair(key="Invoice No", value="12345", confidence=0.85, page=1, source="layoutlmv3"),
        ])

        orch.kg_enricher = MagicMock()

        import fitz
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "t")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-p3", extracted, pdf_bytes,
                                    subscription_id="sub-1", profile_id="prof-1")
        orch.layoutlmv3_extractor.extract.assert_called_once()
        assert "layoutlmv3" in result.metrics.get("visual_intelligence_models", [])


@pytest.mark.asyncio
async def test_orchestrator_skips_layoutlmv3_on_tier1():
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
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.78,
                          image_ratio=0.2, has_tables=True, has_forms=False),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)
        orch.layoutlmv3_extractor = MagicMock()
        orch.kg_enricher = MagicMock()

        import fitz
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "t")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-skip-lm", extracted, pdf_bytes)
        # LayoutLMv3 should NOT be called on Tier 1
        orch.layoutlmv3_extractor.extract.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_orchestrator_p3.py -v`
Expected: FAIL (orchestrator doesn't have layoutlmv3_extractor)

**Step 3: Update orchestrator**

Add to `src/visual_intelligence/orchestrator.py`:

1. Lazy property for `layoutlmv3_extractor`
2. In `_process_page`, when `tier == Tier.FULL`: run LayoutLMv3 with page text and bounding boxes
3. Collect KV pairs from LayoutLMv3 into `VisualEnrichmentResult.kv_pairs`
4. Add "layoutlmv3" to `models_used` when run

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_orchestrator_p3.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/orchestrator.py tests/test_visual_intelligence_orchestrator_p3.py
git commit -m "feat(visual-intelligence): wire LayoutLMv3 into orchestrator for Tier 2 pages"
```

---

### Task 3: Batch Inference Optimization

**Files:**
- Modify: `src/visual_intelligence/orchestrator.py`
- Test: `tests/test_visual_intelligence_batch.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_batch.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import Tier, VisualRegion, PageComplexity
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_concurrent_page_processing():
    """Multiple pages should be processed concurrently up to max_concurrent_pages."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 4}
        extracted.tables = []
        extracted.figures = []
        extracted.sections = []
        extracted.full_text = "text"

        # All 4 pages are Tier 1
        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=i, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.2, has_tables=True, has_forms=False)
            for i in range(1, 5)
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=i, image=mock_img, width=800, height=1000, dpi=300)
            for i in range(1, 5)
        ])

        call_count = 0
        def mock_detect(img, page):
            nonlocal call_count
            call_count += 1
            return [VisualRegion(bbox=(10, 20, 300, 400), label="text",
                                confidence=0.9, page=page)]

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = mock_detect
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)
        orch.kg_enricher = MagicMock()

        import fitz
        doc = fitz.open()
        for _ in range(4):
            doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-batch", extracted, pdf_bytes)
        assert call_count == 4  # All 4 pages processed


def test_model_warmup():
    """Model pool should support pre-warming models."""
    from src.visual_intelligence.model_pool import ModelPool

    pool = ModelPool()
    with patch.object(pool, "load_model", return_value=MagicMock()) as mock_load:
        pool.warmup(["dit", "table_det"])
        assert mock_load.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_batch.py -v`
Expected: FAIL (warmup method doesn't exist)

**Step 3: Add warmup to ModelPool and ensure concurrent processing in orchestrator**

Add to `src/visual_intelligence/model_pool.py`:

```python
def warmup(self, model_keys: List[str]) -> None:
    """Pre-load models to avoid cold-start latency."""
    for key in model_keys:
        if self.is_available(key):
            self.load_model(key)
            logger.info("Warmed up model '%s'", key)
```

Ensure orchestrator processes pages concurrently using `asyncio.gather` with semaphore:

```python
# In enrich(), replace sequential page loop with:
sem = asyncio.Semaphore(max_concurrent_pages)
async def _process_with_sem(page_img, tier):
    async with sem:
        return await self._process_page(page_img, tier)

tasks = [_process_with_sem(page_map[pc.page], pc.tier) for pc in pages_to_process if pc.page in page_map]
page_results = await asyncio.gather(*tasks)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_batch.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/model_pool.py src/visual_intelligence/orchestrator.py tests/test_visual_intelligence_batch.py
git commit -m "feat(visual-intelligence): add batch processing and model warmup optimization"
```

---

### Task 4: Phase 3 End-to-End Smoke Test

**Files:**
- Create: `tests/test_visual_intelligence_e2e_p3.py`

**Step 1: Write the comprehensive smoke test**

```python
# tests/test_visual_intelligence_e2e_p3.py
"""End-to-end smoke test for Phase 3 (full visual intelligence pipeline)."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, StructuredTableResult, OCRPatch, KVPair,
)
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_full_pipeline_all_models():
    """Tier 2 page: DiT + Table Transformer + TrOCR + LayoutLMv3 + KG."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.sections = []
        extracted.tables = [MagicMock(page=1, csv="a,b\n1,2")]
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.3)]
        extracted.full_text = "Garblod Invoyse No 12345"
        extracted.metrics = {"total_pages": 1}

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.3,
                          image_ratio=0.8, has_tables=True, has_forms=True),
        ])

        mock_img = Image.new("RGB", (2550, 3300), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=2550, height=3300, dpi=300),
        ])

        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 50, 2400, 800), label="text", confidence=0.93, page=1),
            VisualRegion(bbox=(100, 850, 2400, 2500), label="table", confidence=0.91, page=1),
        ])

        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[
            StructuredTableResult(page=1, bbox=(100, 850, 2400, 2500),
                                 headers=["Item", "Amount"],
                                 rows=[["Service", "$500"], ["Tax", "$50"]],
                                 spans=[], confidence=0.87),
        ])

        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=OCRPatch(
            page=1, bbox=(100, 50, 2400, 800),
            original_text="Garblod Invoyse",
            enhanced_text="Garbled Invoice",
            original_confidence=0.3,
            enhanced_confidence=0.91,
            method="trocr_printed",
        ))

        orch.layoutlmv3_extractor = MagicMock()
        orch.layoutlmv3_extractor.extract = MagicMock(return_value=[
            KVPair(key="Invoice No", value="12345", confidence=0.88, page=1, source="layoutlmv3"),
        ])

        mock_kg = MagicMock()
        orch.kg_enricher = mock_kg

        import fitz
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "t")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich(
            "doc-full", extracted, pdf_bytes,
            subscription_id="sub-1", profile_id="prof-1",
        )

        # All models invoked
        orch.dit_detector.detect.assert_called_once()
        orch.table_detector.extract.assert_called_once()
        orch.trocr_enhancer.enhance_region.assert_called()
        orch.layoutlmv3_extractor.extract.assert_called_once()
        mock_kg.enqueue_enrichment.assert_called_once()

        # Provenance
        assert result.metrics.get("visual_intelligence_applied") is True
        models_used = result.metrics.get("visual_intelligence_models", [])
        assert "dit" in models_used
        assert "layoutlmv3" in models_used


@pytest.mark.asyncio
async def test_graceful_degradation_all_models_unavailable():
    """If all models fail to load, existing pipeline continues unchanged."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 1}
        extracted.tables = []
        extracted.figures = [MagicMock(page=1, ocr_confidence=0.3)]
        extracted.sections = []
        extracted.full_text = "original text"

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.FULL, ocr_confidence=0.3,
                          image_ratio=0.8, has_tables=True, has_forms=True),
        ])

        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        # All models return empty (simulating unavailable)
        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[])
        orch.table_detector = MagicMock()
        orch.table_detector.extract = MagicMock(return_value=[])
        orch.trocr_enhancer = MagicMock()
        orch.trocr_enhancer.enhance_region = MagicMock(return_value=None)
        orch.layoutlmv3_extractor = MagicMock()
        orch.layoutlmv3_extractor.extract = MagicMock(return_value=[])
        orch.kg_enricher = MagicMock()

        import fitz
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 72), "t")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-degrade", extracted, pdf_bytes)
        # Document should still be returned (not crash)
        assert result.full_text == "original text"
```

**Step 2: Run test**

Run: `pytest tests/test_visual_intelligence_e2e_p3.py -v`
Expected: PASS (2 tests)

**Step 3: Run ALL visual intelligence tests**

Run: `pytest tests/test_visual_intelligence_*.py -v`
Expected: All tests passing across Phase 1, 2, and 3

**Step 4: Commit**

```bash
git add tests/test_visual_intelligence_e2e_p3.py
git commit -m "test(visual-intelligence): add Phase 3 end-to-end and graceful degradation tests"
```

---

## Phase 3 Complete Checklist

After all 4 tasks:

- [ ] `src/visual_intelligence/models/layoutlmv3.py`
- [ ] `src/visual_intelligence/orchestrator.py` updated (LayoutLMv3, batch processing, warmup)
- [ ] `src/visual_intelligence/model_pool.py` updated (warmup method)
- [ ] 3 new test files passing

## Full Implementation Checklist (All Phases)

### Phase 1 Files
- [ ] `src/visual_intelligence/__init__.py`
- [ ] `src/visual_intelligence/datatypes.py`
- [ ] `src/visual_intelligence/complexity_scorer.py`
- [ ] `src/visual_intelligence/page_renderer.py`
- [ ] `src/visual_intelligence/model_pool.py`
- [ ] `src/visual_intelligence/models/__init__.py`
- [ ] `src/visual_intelligence/models/dit_layout.py`
- [ ] `src/visual_intelligence/enrichment_merger.py`
- [ ] `src/visual_intelligence/orchestrator.py`

### Phase 2 Files
- [ ] `src/visual_intelligence/models/table_transformer.py`
- [ ] `src/visual_intelligence/models/trocr_enhancer.py`
- [ ] `src/visual_intelligence/kg_enricher.py`

### Phase 3 Files
- [ ] `src/visual_intelligence/models/layoutlmv3.py`

### Modified Files
- [ ] `src/api/config.py` (VisualIntelligence config class)
- [ ] `src/api/extraction_service.py` (integration hook)
- [ ] `.env.example` (new env vars)
- [ ] `requirements.txt` (timm dependency)

### All Tests
- [ ] `tests/test_visual_intelligence_models.py`
- [ ] `tests/test_visual_intelligence_config.py`
- [ ] `tests/test_visual_intelligence_model_pool.py`
- [ ] `tests/test_visual_intelligence_renderer.py`
- [ ] `tests/test_visual_intelligence_scorer.py`
- [ ] `tests/test_visual_intelligence_dit.py`
- [ ] `tests/test_visual_intelligence_merger.py`
- [ ] `tests/test_visual_intelligence_orchestrator.py`
- [ ] `tests/test_visual_intelligence_integration.py`
- [ ] `tests/test_visual_intelligence_e2e.py`
- [ ] `tests/test_visual_intelligence_table.py`
- [ ] `tests/test_visual_intelligence_trocr.py`
- [ ] `tests/test_visual_intelligence_kg.py`
- [ ] `tests/test_visual_intelligence_orchestrator_p2.py`
- [ ] `tests/test_visual_intelligence_e2e_p2.py`
- [ ] `tests/test_visual_intelligence_layoutlmv3.py`
- [ ] `tests/test_visual_intelligence_orchestrator_p3.py`
- [ ] `tests/test_visual_intelligence_batch.py`
- [ ] `tests/test_visual_intelligence_e2e_p3.py`

**Final verification:** `pytest tests/test_visual_intelligence_*.py -v`
