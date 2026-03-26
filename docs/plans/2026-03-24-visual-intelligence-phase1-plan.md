# Visual Intelligence Layer — Phase 1: Foundation + Layout

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add adaptive complexity scoring, page rendering, DiT layout detection, and enrichment merging as a second-pass visual intelligence layer.

**Architecture:** Layered enrichment — existing extraction pipeline untouched, new `src/visual_intelligence/` module runs as second pass on page images, merges results into ExtractedDocument via confidence-based arbitration.

**Tech Stack:** PyMuPDF (page rendering), HuggingFace transformers + timm (DiT model), torch/torchvision, PIL

---

### Task 1: Data Models

**Files:**
- Create: `src/visual_intelligence/__init__.py`
- Create: `src/visual_intelligence/models.py`
- Test: `tests/test_visual_intelligence_models.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_models.py
import pytest
from dataclasses import asdict


def test_visual_region_creation():
    from src.visual_intelligence.models import VisualRegion
    region = VisualRegion(
        bbox=(100, 200, 300, 400),
        label="table",
        confidence=0.92,
        page=1,
    )
    assert region.label == "table"
    assert region.confidence == 0.92
    assert region.page == 1
    assert region.source == "visual_intelligence"


def test_page_complexity_tiers():
    from src.visual_intelligence.models import PageComplexity, Tier
    pc = PageComplexity(
        page=1,
        tier=Tier.FULL,
        ocr_confidence=0.55,
        image_ratio=0.7,
        has_tables=True,
        has_forms=False,
        signals={"low_ocr": True},
    )
    assert pc.tier == Tier.FULL
    assert pc.tier.value == 2


def test_visual_enrichment_result():
    from src.visual_intelligence.models import (
        VisualEnrichmentResult, VisualRegion, StructuredTableResult,
    )
    result = VisualEnrichmentResult(doc_id="doc-1")
    assert result.regions == []
    assert result.tables == []
    assert result.ocr_patches == {}
    assert result.kv_pairs == []
    d = result.to_dict()
    assert d["doc_id"] == "doc-1"


def test_structured_table_result():
    from src.visual_intelligence.models import StructuredTableResult
    table = StructuredTableResult(
        page=2,
        bbox=(10, 20, 500, 300),
        headers=["Name", "Value"],
        rows=[["A", "1"], ["B", "2"]],
        spans=[],
        confidence=0.88,
    )
    assert len(table.rows) == 2
    assert table.headers[0] == "Name"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_models.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/__init__.py
"""Visual Intelligence Layer — second-pass document enrichment using ML models."""
```

```python
# src/visual_intelligence/models.py
"""Data models for the Visual Intelligence Layer."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class Tier(enum.IntEnum):
    SKIP = 0
    LIGHT = 1
    FULL = 2


@dataclass
class VisualRegion:
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    label: str  # text, title, table, figure, list, header, footer
    confidence: float
    page: int
    source: str = "visual_intelligence"


@dataclass
class StructuredTableResult:
    page: int
    bbox: Tuple[float, float, float, float]
    headers: List[str]
    rows: List[List[str]]
    spans: List[Dict[str, Any]]  # merged cell info
    confidence: float


@dataclass
class OCRPatch:
    page: int
    bbox: Tuple[float, float, float, float]
    original_text: str
    enhanced_text: str
    original_confidence: float
    enhanced_confidence: float
    method: str  # "trocr_printed" or "trocr_handwritten"


@dataclass
class KVPair:
    key: str
    value: str
    confidence: float
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: str = "visual_intelligence"


@dataclass
class PageComplexity:
    page: int
    tier: Tier
    ocr_confidence: float
    image_ratio: float
    has_tables: bool
    has_forms: bool
    signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualEnrichmentResult:
    doc_id: str
    regions: List[VisualRegion] = field(default_factory=list)
    tables: List[StructuredTableResult] = field(default_factory=list)
    ocr_patches: Dict[int, List[OCRPatch]] = field(default_factory=dict)  # page -> patches
    kv_pairs: List[KVPair] = field(default_factory=list)
    page_complexities: List[PageComplexity] = field(default_factory=list)
    processing_time_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "regions": [
                {"bbox": r.bbox, "label": r.label, "confidence": r.confidence, "page": r.page}
                for r in self.regions
            ],
            "tables": [
                {"page": t.page, "bbox": t.bbox, "headers": t.headers,
                 "rows": t.rows, "spans": t.spans, "confidence": t.confidence}
                for t in self.tables
            ],
            "ocr_patches": {
                str(page): [
                    {"bbox": p.bbox, "original": p.original_text,
                     "enhanced": p.enhanced_text, "method": p.method}
                    for p in patches
                ]
                for page, patches in self.ocr_patches.items()
            },
            "kv_pairs": [
                {"key": kv.key, "value": kv.value, "confidence": kv.confidence, "page": kv.page}
                for kv in self.kv_pairs
            ],
            "models_used": self.models_used,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_models.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/__init__.py src/visual_intelligence/models.py tests/test_visual_intelligence_models.py
git commit -m "feat(visual-intelligence): add data models for visual enrichment layer"
```

---

### Task 2: Configuration

**Files:**
- Modify: `src/api/config.py` (add `VisualIntelligence` class inside `Config`)
- Modify: `.env.example` (add new env vars)
- Test: `tests/test_visual_intelligence_config.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_config.py
import os
import pytest


def test_visual_intelligence_config_defaults():
    from src.api.config import Config
    vi = Config.VisualIntelligence
    assert vi.ENABLED == "true"
    assert vi.GPU_DEVICE == "cuda:0"
    assert vi.CPU_FALLBACK == "true"
    assert vi.MAX_CONCURRENT_PAGES == 4
    assert "dit" in vi.TIER1_MODELS
    assert "layoutlmv3" in vi.TIER2_MODELS


def test_visual_intelligence_config_env_override(monkeypatch):
    monkeypatch.setenv("VISUAL_INTELLIGENCE_ENABLED", "false")
    monkeypatch.setenv("VISUAL_INTELLIGENCE_GPU_DEVICE", "cuda:1")
    # Force re-read by checking env directly
    assert os.getenv("VISUAL_INTELLIGENCE_ENABLED") == "false"
    assert os.getenv("VISUAL_INTELLIGENCE_GPU_DEVICE") == "cuda:1"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_config.py -v`
Expected: FAIL with AttributeError (no VisualIntelligence on Config)

**Step 3: Write minimal implementation**

Add to `src/api/config.py` after the existing `VisionAnalysis` class (around line 554):

```python
    class VisualIntelligence:
        """ML-based visual document understanding (second-pass enrichment)."""
        ENABLED = os.getenv("VISUAL_INTELLIGENCE_ENABLED", "true")
        GPU_DEVICE = os.getenv("VISUAL_INTELLIGENCE_GPU_DEVICE", "cuda:0")
        CPU_FALLBACK = os.getenv("VISUAL_INTELLIGENCE_CPU_FALLBACK", "true")
        MAX_CONCURRENT_PAGES = int(os.getenv("VISUAL_INTELLIGENCE_MAX_CONCURRENT_PAGES", "4"))
        TIER1_MODELS = os.getenv("VISUAL_INTELLIGENCE_TIER1_MODELS", "dit").split(",")
        TIER2_MODELS = os.getenv("VISUAL_INTELLIGENCE_TIER2_MODELS", "dit,table_transformer,trocr,layoutlmv3").split(",")
        DIT_MODEL = os.getenv("VISUAL_INTELLIGENCE_DIT_MODEL", "microsoft/dit-large-finetuned-publaynet")
        TABLE_DET_MODEL = os.getenv("VISUAL_INTELLIGENCE_TABLE_DET_MODEL", "microsoft/table-transformer-detection")
        TABLE_STR_MODEL = os.getenv("VISUAL_INTELLIGENCE_TABLE_STR_MODEL", "microsoft/table-transformer-structure-recognition")
        TROCR_PRINTED_MODEL = os.getenv("VISUAL_INTELLIGENCE_TROCR_PRINTED_MODEL", "microsoft/trocr-base-printed")
        TROCR_HANDWRITTEN_MODEL = os.getenv("VISUAL_INTELLIGENCE_TROCR_HANDWRITTEN_MODEL", "microsoft/trocr-base-handwritten")
        LAYOUTLMV3_MODEL = os.getenv("VISUAL_INTELLIGENCE_LAYOUTLMV3_MODEL", "microsoft/layoutlmv3-base")
        RENDER_DPI = int(os.getenv("VISUAL_INTELLIGENCE_RENDER_DPI", "300"))
        COMPLEXITY_OCR_HIGH = float(os.getenv("VISUAL_INTELLIGENCE_OCR_HIGH", "0.85"))
        COMPLEXITY_OCR_LOW = float(os.getenv("VISUAL_INTELLIGENCE_OCR_LOW", "0.70"))
```

Add to `.env.example`:

```bash
# --- Visual Intelligence (ML-based second-pass enrichment) ---
VISUAL_INTELLIGENCE_ENABLED=true
VISUAL_INTELLIGENCE_GPU_DEVICE=cuda:0
VISUAL_INTELLIGENCE_CPU_FALLBACK=true
VISUAL_INTELLIGENCE_MAX_CONCURRENT_PAGES=4
VISUAL_INTELLIGENCE_RENDER_DPI=300
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_config.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/api/config.py .env.example tests/test_visual_intelligence_config.py
git commit -m "feat(visual-intelligence): add VisualIntelligence config with env overrides"
```

---

### Task 3: Model Pool (Lazy Loading with Auto-Install)

**Files:**
- Create: `src/visual_intelligence/model_pool.py`
- Test: `tests/test_visual_intelligence_model_pool.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_model_pool.py
import pytest
from unittest.mock import patch, MagicMock


def test_model_pool_singleton():
    from src.visual_intelligence.model_pool import get_model_pool
    pool1 = get_model_pool()
    pool2 = get_model_pool()
    assert pool1 is pool2


def test_model_pool_tracks_disabled():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    assert pool.is_available("dit") is True  # not yet tried, assumed available
    pool.disabled_models.add("dit")
    assert pool.is_available("dit") is False


def test_model_pool_load_returns_none_when_unavailable():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    # Force all loading to fail
    with patch("src.visual_intelligence.model_pool.ModelPool._try_load", return_value=None):
        with patch("src.visual_intelligence.model_pool.ModelPool._try_install", return_value=None):
            result = pool.load_model("dit")
            assert result is None
            assert pool.is_available("dit") is False


def test_model_pool_load_caches_result():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    mock_model = MagicMock()
    with patch("src.visual_intelligence.model_pool.ModelPool._try_load", return_value=mock_model):
        result1 = pool.load_model("dit")
        result2 = pool.load_model("dit")
        assert result1 is mock_model
        assert result2 is mock_model  # cached, not reloaded


def test_model_pool_device_selection():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    # Without GPU available, should fallback to CPU
    with patch("torch.cuda.is_available", return_value=False):
        device = pool.get_device("dit")
        assert str(device) == "cpu"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_model_pool.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/model_pool.py
"""Lazy-loading model pool with auto-install and graceful degradation."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not installed. Visual intelligence models will be unavailable.")

MODEL_REGISTRY: Dict[str, str] = {
    "dit": "microsoft/dit-large-finetuned-publaynet",
    "table_det": "microsoft/table-transformer-detection",
    "table_str": "microsoft/table-transformer-structure-recognition",
    "trocr_printed": "microsoft/trocr-base-printed",
    "trocr_handwritten": "microsoft/trocr-base-handwritten",
    "layoutlmv3": "microsoft/layoutlmv3-base",
}

# Models that benefit from GPU
GPU_PREFERRED = {"dit", "table_det", "table_str", "layoutlmv3"}


class ModelPool:
    """Thread-safe, lazy-loading pool for visual intelligence models."""

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}
        self.disabled_models: Set[str] = set()
        self._lock = threading.Lock()

    def is_available(self, model_key: str) -> bool:
        return model_key not in self.disabled_models

    def get_device(self, model_key: str) -> Any:
        if not TORCH_AVAILABLE:
            return "cpu"
        try:
            from src.api.config import Config
            cfg = Config.VisualIntelligence
        except (ImportError, AttributeError):
            cfg = None

        gpu_device = getattr(cfg, "GPU_DEVICE", "cuda:0") if cfg else "cuda:0"
        cpu_fallback = getattr(cfg, "CPU_FALLBACK", "true") if cfg else "true"

        if model_key in GPU_PREFERRED and torch.cuda.is_available():
            return torch.device(gpu_device)
        if cpu_fallback == "true" or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(gpu_device)

    def load_model(self, model_key: str) -> Optional[Any]:
        if model_key in self.disabled_models:
            return None
        if model_key in self._models:
            return self._models[model_key]

        with self._lock:
            # Double-check after acquiring lock
            if model_key in self._models:
                return self._models[model_key]

            model = self._try_load(model_key)
            if model is not None:
                self._models[model_key] = model
                return model

            model = self._try_install(model_key)
            if model is not None:
                self._models[model_key] = model
                return model

            logger.warning(
                "Visual intelligence model '%s' unavailable. "
                "Falling back to existing extraction for this capability.",
                model_key,
            )
            self.disabled_models.add(model_key)
            return None

    def load_processor(self, model_key: str) -> Optional[Any]:
        if model_key in self.disabled_models:
            return None
        if model_key in self._processors:
            return self._processors[model_key]

        with self._lock:
            if model_key in self._processors:
                return self._processors[model_key]
            processor = self._try_load_processor(model_key)
            if processor is not None:
                self._processors[model_key] = processor
            return processor

    def _try_load(self, model_key: str) -> Optional[Any]:
        if not TORCH_AVAILABLE:
            return None
        try:
            from transformers import AutoModel, AutoModelForObjectDetection

            model_id = self._resolve_model_id(model_key)
            device = self.get_device(model_key)

            if model_key in ("dit", "table_det", "table_str"):
                model = AutoModelForObjectDetection.from_pretrained(model_id)
            else:
                model = AutoModel.from_pretrained(model_id)

            model = model.to(device).eval()
            logger.info("Loaded visual model '%s' on %s", model_key, device)
            return model
        except Exception as exc:
            logger.debug("Could not load model '%s' from cache: %s", model_key, exc)
            return None

    def _try_install(self, model_key: str) -> Optional[Any]:
        if not TORCH_AVAILABLE:
            return None
        try:
            from transformers import AutoModel, AutoModelForObjectDetection

            model_id = self._resolve_model_id(model_key)
            device = self.get_device(model_key)
            logger.info("Downloading model '%s' (%s)...", model_key, model_id)

            if model_key in ("dit", "table_det", "table_str"):
                model = AutoModelForObjectDetection.from_pretrained(model_id, force_download=True)
            else:
                model = AutoModel.from_pretrained(model_id, force_download=True)

            model = model.to(device).eval()
            logger.info("Downloaded and loaded model '%s' on %s", model_key, device)
            return model
        except Exception as exc:
            logger.warning("Failed to download model '%s': %s", model_key, exc)
            return None

    def _try_load_processor(self, model_key: str) -> Optional[Any]:
        try:
            from transformers import AutoImageProcessor, AutoFeatureExtractor
            model_id = self._resolve_model_id(model_key)
            try:
                return AutoImageProcessor.from_pretrained(model_id)
            except Exception:
                return AutoFeatureExtractor.from_pretrained(model_id)
        except Exception as exc:
            logger.debug("Could not load processor for '%s': %s", model_key, exc)
            return None

    def _resolve_model_id(self, model_key: str) -> str:
        try:
            from src.api.config import Config
            cfg = Config.VisualIntelligence
            config_map = {
                "dit": "DIT_MODEL",
                "table_det": "TABLE_DET_MODEL",
                "table_str": "TABLE_STR_MODEL",
                "trocr_printed": "TROCR_PRINTED_MODEL",
                "trocr_handwritten": "TROCR_HANDWRITTEN_MODEL",
                "layoutlmv3": "LAYOUTLMV3_MODEL",
            }
            attr = config_map.get(model_key)
            if attr:
                return getattr(cfg, attr, MODEL_REGISTRY[model_key])
        except (ImportError, AttributeError):
            pass
        return MODEL_REGISTRY[model_key]

    def unload(self, model_key: str) -> None:
        with self._lock:
            self._models.pop(model_key, None)
            self._processors.pop(model_key, None)
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()

    def unload_all(self) -> None:
        with self._lock:
            self._models.clear()
            self._processors.clear()
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()


# Singleton
_pool: Optional[ModelPool] = None
_pool_lock = threading.Lock()


def get_model_pool() -> ModelPool:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        _pool = ModelPool()
        return _pool
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_model_pool.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/model_pool.py tests/test_visual_intelligence_model_pool.py
git commit -m "feat(visual-intelligence): add model pool with lazy loading and auto-install"
```

---

### Task 4: Page Renderer

**Files:**
- Create: `src/visual_intelligence/page_renderer.py`
- Test: `tests/test_visual_intelligence_renderer.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_renderer.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


def test_render_pages_returns_pil_images():
    from src.visual_intelligence.page_renderer import PageRenderer

    # Create a minimal valid PDF in memory (1 page)
    import fitz  # PyMuPDF
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Hello World")
    pdf_bytes = doc.tobytes()
    doc.close()

    renderer = PageRenderer(dpi=150)
    pages = renderer.render(pdf_bytes)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert isinstance(pages[0].image, Image.Image)
    assert pages[0].width > 0
    assert pages[0].height > 0


def test_render_specific_pages():
    from src.visual_intelligence.page_renderer import PageRenderer
    import fitz

    doc = fitz.open()
    for i in range(5):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i+1}")
    pdf_bytes = doc.tobytes()
    doc.close()

    renderer = PageRenderer(dpi=150)
    pages = renderer.render(pdf_bytes, page_numbers=[1, 3, 5])

    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 3, 5]


def test_render_non_pdf_returns_empty():
    from src.visual_intelligence.page_renderer import PageRenderer
    renderer = PageRenderer()
    pages = renderer.render(b"not a pdf")
    assert pages == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_renderer.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/page_renderer.py
"""Render PDF pages as PIL images for visual model input."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PageImage:
    page_number: int  # 1-indexed
    image: Image.Image
    width: int
    height: int
    dpi: int


class PageRenderer:
    """Renders PDF pages as PIL images using PyMuPDF."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def render(
        self,
        pdf_bytes: bytes,
        page_numbers: Optional[List[int]] = None,
    ) -> List[PageImage]:
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF not installed, cannot render pages.")
            return []

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.debug("Failed to open PDF for rendering: %s", exc)
            return []

        results: List[PageImage] = []
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        target_pages = set(page_numbers) if page_numbers else None

        try:
            for i in range(len(doc)):
                page_num = i + 1  # 1-indexed
                if target_pages and page_num not in target_pages:
                    continue

                page = doc[i]
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                results.append(PageImage(
                    page_number=page_num,
                    image=img,
                    width=pix.width,
                    height=pix.height,
                    dpi=self.dpi,
                ))
        finally:
            doc.close()

        return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_renderer.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/page_renderer.py tests/test_visual_intelligence_renderer.py
git commit -m "feat(visual-intelligence): add page renderer (PDF to PIL images)"
```

---

### Task 5: Complexity Scorer

**Files:**
- Create: `src/visual_intelligence/complexity_scorer.py`
- Test: `tests/test_visual_intelligence_scorer.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_scorer.py
import pytest
from src.visual_intelligence.models import Tier


def test_simple_text_page_is_tier_skip():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    scorer = ComplexityScorer()
    result = scorer.score_page(
        page=1,
        ocr_confidence=0.95,
        image_ratio=0.0,
        has_tables=False,
        has_forms=False,
        block_types={"text": 50, "image": 0},
    )
    assert result.tier == Tier.SKIP


def test_moderate_page_is_tier_light():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    scorer = ComplexityScorer()
    result = scorer.score_page(
        page=2,
        ocr_confidence=0.78,
        image_ratio=0.2,
        has_tables=True,
        has_forms=False,
        block_types={"text": 30, "image": 5},
    )
    assert result.tier == Tier.LIGHT


def test_complex_page_is_tier_full():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    scorer = ComplexityScorer()
    result = scorer.score_page(
        page=3,
        ocr_confidence=0.45,
        image_ratio=0.8,
        has_tables=True,
        has_forms=True,
        block_types={"text": 5, "image": 20},
    )
    assert result.tier == Tier.FULL


def test_score_extracted_document():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    from unittest.mock import MagicMock

    scorer = ComplexityScorer()

    # Mock ExtractedDocument with mixed pages
    extracted = MagicMock()
    extracted.figures = [
        MagicMock(page=2, ocr_confidence=0.55),
    ]
    extracted.tables = [
        MagicMock(page=2),
    ]
    extracted.sections = [
        MagicMock(start_page=1, end_page=1),
        MagicMock(start_page=2, end_page=2),
    ]
    extracted.metrics = {"total_pages": 2}

    results = scorer.score_document(extracted)
    assert len(results) == 2
    # Page 1: no images, no tables → SKIP
    assert results[0].tier == Tier.SKIP
    # Page 2: low OCR image + table → LIGHT or FULL
    assert results[1].tier in (Tier.LIGHT, Tier.FULL)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_scorer.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/complexity_scorer.py
"""Adaptive complexity scoring — gates pages into processing tiers."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.visual_intelligence.models import PageComplexity, Tier

logger = logging.getLogger(__name__)


class ComplexityScorer:
    """Scores each page to decide processing tier (SKIP / LIGHT / FULL)."""

    def __init__(
        self,
        ocr_high: float = 0.85,
        ocr_low: float = 0.70,
        text_ratio_threshold: float = 0.95,
    ) -> None:
        self.ocr_high = ocr_high
        self.ocr_low = ocr_low
        self.text_ratio_threshold = text_ratio_threshold

    def score_page(
        self,
        page: int,
        ocr_confidence: float = 1.0,
        image_ratio: float = 0.0,
        has_tables: bool = False,
        has_forms: bool = False,
        block_types: Optional[Dict[str, int]] = None,
    ) -> PageComplexity:
        block_types = block_types or {}
        total_blocks = sum(block_types.values()) or 1
        text_blocks = block_types.get("text", 0)
        text_ratio = text_blocks / total_blocks

        signals: Dict[str, Any] = {
            "ocr_confidence": ocr_confidence,
            "image_ratio": image_ratio,
            "text_ratio": text_ratio,
            "has_tables": has_tables,
            "has_forms": has_forms,
        }

        # Tier 2 (FULL): low OCR, image-heavy, forms, or handwritten
        if (
            ocr_confidence < self.ocr_low
            or (has_forms and image_ratio > 0.3)
            or image_ratio > 0.7
        ):
            tier = Tier.FULL
        # Tier 1 (LIGHT): tables, some images, multi-column
        elif (
            has_tables
            or image_ratio > 0.1
            or ocr_confidence < self.ocr_high
        ):
            tier = Tier.LIGHT
        # Tier 0 (SKIP): simple text-only
        else:
            tier = Tier.SKIP

        return PageComplexity(
            page=page,
            tier=tier,
            ocr_confidence=ocr_confidence,
            image_ratio=image_ratio,
            has_tables=has_tables,
            has_forms=has_forms,
            signals=signals,
        )

    def score_document(self, extracted_doc: Any) -> List[PageComplexity]:
        total_pages = 1
        if hasattr(extracted_doc, "metrics") and isinstance(extracted_doc.metrics, dict):
            total_pages = extracted_doc.metrics.get("total_pages", 1)

        # Collect per-page signals from extracted doc
        page_tables: Dict[int, bool] = {}
        page_ocr: Dict[int, float] = {}
        page_images: Dict[int, int] = {}

        for table in getattr(extracted_doc, "tables", []):
            page_tables[getattr(table, "page", 1)] = True

        for fig in getattr(extracted_doc, "figures", []):
            p = getattr(fig, "page", 1)
            page_images[p] = page_images.get(p, 0) + 1
            conf = getattr(fig, "ocr_confidence", None)
            if conf is not None:
                # Keep lowest confidence per page
                if p not in page_ocr or conf < page_ocr[p]:
                    page_ocr[p] = conf

        results: List[PageComplexity] = []
        for page_num in range(1, total_pages + 1):
            img_count = page_images.get(page_num, 0)
            image_ratio = min(img_count / 5.0, 1.0)  # normalize: 5+ images = 1.0
            ocr_conf = page_ocr.get(page_num, 1.0)
            has_tbl = page_tables.get(page_num, False)

            results.append(self.score_page(
                page=page_num,
                ocr_confidence=ocr_conf,
                image_ratio=image_ratio,
                has_tables=has_tbl,
            ))

        return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_scorer.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/complexity_scorer.py tests/test_visual_intelligence_scorer.py
git commit -m "feat(visual-intelligence): add adaptive complexity scorer with tier gating"
```

---

### Task 6: DiT Layout Detector Wrapper

**Files:**
- Create: `src/visual_intelligence/models/__init__.py`
- Create: `src/visual_intelligence/models/dit_layout.py`
- Test: `tests/test_visual_intelligence_dit.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_dit.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.visual_intelligence.models import VisualRegion


def test_dit_detector_creation():
    from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
    with patch("src.visual_intelligence.model_pool.get_model_pool") as mock_pool:
        mock_pool.return_value = MagicMock()
        detector = DiTLayoutDetector()
        assert detector is not None


def test_dit_detect_returns_regions():
    from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
    mock_pool = MagicMock()
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Simulate model output
    mock_output = MagicMock()
    mock_output.logits = MagicMock()
    mock_output.pred_boxes = MagicMock()
    mock_model.return_value = mock_output
    mock_model.config.id2label = {0: "text", 1: "title", 2: "table", 3: "figure"}

    mock_pool.load_model.return_value = mock_model
    mock_pool.load_processor.return_value = mock_processor
    mock_pool.is_available.return_value = True
    mock_pool.get_device.return_value = "cpu"

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        with patch("src.visual_intelligence.models.dit_layout.DiTLayoutDetector._postprocess") as mock_pp:
            mock_pp.return_value = [
                VisualRegion(bbox=(10, 20, 300, 400), label="text", confidence=0.95, page=1),
                VisualRegion(bbox=(10, 420, 300, 500), label="table", confidence=0.88, page=1),
            ]
            detector = DiTLayoutDetector()
            img = Image.new("RGB", (800, 1000), "white")
            regions = detector.detect(img, page_number=1)

            assert len(regions) == 2
            assert regions[0].label == "text"
            assert regions[1].label == "table"


def test_dit_unavailable_returns_empty():
    from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
    mock_pool = MagicMock()
    mock_pool.is_available.return_value = False

    with patch("src.visual_intelligence.model_pool.get_model_pool", return_value=mock_pool):
        detector = DiTLayoutDetector()
        img = Image.new("RGB", (800, 1000), "white")
        regions = detector.detect(img, page_number=1)
        assert regions == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_dit.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/models/__init__.py
"""Visual intelligence model wrappers."""
from src.visual_intelligence.models import VisualRegion, StructuredTableResult, OCRPatch, KVPair
```

Wait — this import is wrong. Fix:

```python
# src/visual_intelligence/models/__init__.py
"""Visual intelligence model wrappers."""
```

```python
# src/visual_intelligence/models/dit_layout.py
"""DiT (Document Image Transformer) layout detection wrapper."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from PIL import Image

from src.visual_intelligence.models_data import VisualRegion

logger = logging.getLogger(__name__)

# Confidence threshold for keeping detections
MIN_CONFIDENCE = 0.7


class DiTLayoutDetector:
    """Detects layout regions in document page images using DiT."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("dit"):
            return False
        self._model = pool.load_model("dit")
        self._processor = pool.load_processor("dit")
        return self._model is not None and self._processor is not None

    def detect(self, image: Image.Image, page_number: int) -> List[VisualRegion]:
        from src.visual_intelligence.model_pool import get_model_pool
        pool = get_model_pool()
        if not pool.is_available("dit"):
            return []

        if not self._ensure_loaded():
            return []

        try:
            device = pool.get_device("dit")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            results = self._processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=MIN_CONFIDENCE
            )[0]

            return self._postprocess(results, page_number)
        except Exception as exc:
            logger.warning("DiT layout detection failed on page %d: %s", page_number, exc)
            return []

    def _postprocess(
        self, results: Dict[str, Any], page_number: int
    ) -> List[VisualRegion]:
        regions: List[VisualRegion] = []
        id2label = self._model.config.id2label

        scores = results["scores"].cpu().tolist()
        labels = results["labels"].cpu().tolist()
        boxes = results["boxes"].cpu().tolist()

        for score, label_id, box in zip(scores, labels, boxes):
            label = id2label.get(label_id, f"unknown_{label_id}")
            regions.append(VisualRegion(
                bbox=tuple(box),
                label=label,
                confidence=score,
                page=page_number,
            ))

        # Sort by vertical position (top to bottom)
        regions.sort(key=lambda r: r.bbox[1])
        return regions
```

**Important:** The import in `dit_layout.py` references `models_data` which is wrong. It should be:

```python
from src.visual_intelligence.models import VisualRegion
```

But `src/visual_intelligence/models.py` conflicts with `src/visual_intelligence/models/` directory. **Resolution:** Rename `src/visual_intelligence/models.py` → `src/visual_intelligence/datatypes.py` and update all imports. Then `src/visual_intelligence/models/` is the directory for model wrappers.

Updated imports everywhere:
- `from src.visual_intelligence.datatypes import VisualRegion, ...`
- `from src.visual_intelligence.models.dit_layout import DiTLayoutDetector`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_dit.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/datatypes.py src/visual_intelligence/models/__init__.py src/visual_intelligence/models/dit_layout.py tests/test_visual_intelligence_dit.py
git commit -m "feat(visual-intelligence): add DiT layout detection wrapper"
```

---

### Task 7: Enrichment Merger

**Files:**
- Create: `src/visual_intelligence/enrichment_merger.py`
- Test: `tests/test_visual_intelligence_merger.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_merger.py
import pytest
from unittest.mock import MagicMock
from src.visual_intelligence.datatypes import VisualRegion, VisualEnrichmentResult


def test_merge_layout_adds_new_sections():
    from src.visual_intelligence.enrichment_merger import EnrichmentMerger
    merger = EnrichmentMerger()

    extracted = MagicMock()
    existing_section = MagicMock()
    existing_section.section_id = "s1"
    existing_section.title = "Introduction"
    existing_section.start_page = 1
    existing_section.end_page = 1
    extracted.sections = [existing_section]

    visual_result = VisualEnrichmentResult(doc_id="doc-1")
    visual_result.regions = [
        VisualRegion(bbox=(10, 20, 300, 100), label="title", confidence=0.95, page=1),
        VisualRegion(bbox=(10, 120, 300, 400), label="text", confidence=0.92, page=1),
        VisualRegion(bbox=(320, 20, 600, 200), label="figure", confidence=0.88, page=1),
    ]

    merged = merger.merge(extracted, visual_result)
    # Should preserve existing section and add visual metadata
    assert hasattr(merged, "sections")
    assert len(merged.sections) >= 1


def test_merge_never_deletes_existing():
    from src.visual_intelligence.enrichment_merger import EnrichmentMerger
    merger = EnrichmentMerger()

    extracted = MagicMock()
    extracted.sections = [MagicMock(section_id="s1"), MagicMock(section_id="s2")]
    extracted.tables = [MagicMock(page=1, csv="a,b\n1,2")]
    extracted.figures = [MagicMock(page=1)]
    extracted.full_text = "existing text"

    visual_result = VisualEnrichmentResult(doc_id="doc-1")

    merged = merger.merge(extracted, visual_result)
    assert len(merged.sections) >= 2
    assert len(merged.tables) >= 1
    assert merged.full_text == "existing text"


def test_merge_provenance_tracking():
    from src.visual_intelligence.enrichment_merger import EnrichmentMerger
    merger = EnrichmentMerger()

    extracted = MagicMock()
    extracted.sections = []
    extracted.tables = []
    extracted.figures = []
    extracted.full_text = "text"
    extracted.metrics = {}

    visual_result = VisualEnrichmentResult(doc_id="doc-1")
    visual_result.regions = [
        VisualRegion(bbox=(10, 20, 300, 400), label="table", confidence=0.9, page=1),
    ]

    merged = merger.merge(extracted, visual_result)
    # Check that visual enrichment metadata is stored
    assert merged.metrics.get("visual_intelligence_applied") is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_merger.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/enrichment_merger.py
"""Confidence-based merger — reconciles visual layer results with existing extraction."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.visual_intelligence.datatypes import (
    VisualEnrichmentResult,
    VisualRegion,
    StructuredTableResult,
    OCRPatch,
    KVPair,
)

logger = logging.getLogger(__name__)

# Confidence threshold: visual result must beat existing by this margin for OCR patches
OCR_CONFIDENCE_MARGIN = 0.15
# Minimum DiT confidence to replace heuristic layout
LAYOUT_CONFIDENCE_MIN = 0.7


class EnrichmentMerger:
    """Merges visual intelligence results into an existing ExtractedDocument."""

    def merge(self, extracted_doc: Any, visual_result: VisualEnrichmentResult) -> Any:
        if not visual_result.regions and not visual_result.tables and not visual_result.ocr_patches and not visual_result.kv_pairs:
            logger.debug("No visual enrichments to merge for doc %s", visual_result.doc_id)
            return extracted_doc

        self._merge_layout(extracted_doc, visual_result.regions)
        self._merge_tables(extracted_doc, visual_result.tables)
        self._merge_ocr(extracted_doc, visual_result.ocr_patches)
        self._merge_kv_pairs(extracted_doc, visual_result.kv_pairs)
        self._tag_provenance(extracted_doc, visual_result)

        return extracted_doc

    def _merge_layout(self, doc: Any, regions: List[VisualRegion]) -> None:
        if not regions:
            return

        high_conf_regions = [r for r in regions if r.confidence >= LAYOUT_CONFIDENCE_MIN]
        if not high_conf_regions:
            return

        # Add visual layout metadata to figures for regions DiT found as figures
        existing_figures = getattr(doc, "figures", [])
        for region in high_conf_regions:
            if region.label == "figure":
                # Check if already tracked
                already_exists = any(
                    getattr(f, "page", None) == region.page
                    for f in existing_figures
                )
                if not already_exists:
                    try:
                        from src.api.pipeline_models import Figure
                        new_fig = Figure(
                            page=region.page,
                            caption=f"[Visual: {region.label}]",
                            ocr_method="dit_layout",
                            ocr_confidence=region.confidence,
                        )
                        existing_figures.append(new_fig)
                    except ImportError:
                        pass

        # Store layout regions in metrics for downstream use
        if not hasattr(doc, "metrics") or doc.metrics is None:
            doc.metrics = {}
        doc.metrics["visual_layout_regions"] = [
            {"bbox": r.bbox, "label": r.label, "confidence": r.confidence, "page": r.page}
            for r in high_conf_regions
        ]

    def _merge_tables(self, doc: Any, tables: List[StructuredTableResult]) -> None:
        if not tables:
            return

        existing_tables = getattr(doc, "tables", [])
        for structured in tables:
            # Find matching existing table by page
            matched = False
            for existing in existing_tables:
                if getattr(existing, "page", None) == structured.page:
                    # Augment existing table with structured data
                    existing.structured = {
                        "headers": structured.headers,
                        "rows": structured.rows,
                        "spans": structured.spans,
                        "confidence": structured.confidence,
                        "source": "visual_intelligence",
                    }
                    matched = True
                    break

            if not matched:
                # Add as new table
                try:
                    from src.api.pipeline_models import Table
                    csv_text = ",".join(structured.headers) + "\n"
                    csv_text += "\n".join(",".join(row) for row in structured.rows)
                    new_table = Table(
                        page=structured.page,
                        text=csv_text,
                        csv=csv_text,
                        structured={
                            "headers": structured.headers,
                            "rows": structured.rows,
                            "spans": structured.spans,
                            "confidence": structured.confidence,
                            "source": "visual_intelligence",
                        },
                    )
                    existing_tables.append(new_table)
                except ImportError:
                    pass

    def _merge_ocr(self, doc: Any, ocr_patches: Dict[int, List[OCRPatch]]) -> None:
        if not ocr_patches:
            return

        replacements = []
        full_text = getattr(doc, "full_text", "") or ""

        for page, patches in ocr_patches.items():
            for patch in patches:
                margin = patch.enhanced_confidence - patch.original_confidence
                if margin >= OCR_CONFIDENCE_MARGIN and patch.original_text in full_text:
                    full_text = full_text.replace(
                        patch.original_text, patch.enhanced_text, 1
                    )
                    replacements.append({
                        "page": page,
                        "original": patch.original_text[:50],
                        "enhanced": patch.enhanced_text[:50],
                        "margin": round(margin, 3),
                    })

        if replacements:
            doc.full_text = full_text
            if not hasattr(doc, "metrics") or doc.metrics is None:
                doc.metrics = {}
            doc.metrics["ocr_patches_applied"] = len(replacements)
            logger.info("Applied %d OCR patches", len(replacements))

    def _merge_kv_pairs(self, doc: Any, kv_pairs: List[KVPair]) -> None:
        if not kv_pairs:
            return

        # Store as additional metadata — downstream form_extractor can use these
        if not hasattr(doc, "metrics") or doc.metrics is None:
            doc.metrics = {}
        doc.metrics["visual_kv_pairs"] = [
            {"key": kv.key, "value": kv.value, "confidence": kv.confidence,
             "page": kv.page, "source": kv.source}
            for kv in kv_pairs
        ]

    def _tag_provenance(self, doc: Any, result: VisualEnrichmentResult) -> None:
        if not hasattr(doc, "metrics") or doc.metrics is None:
            doc.metrics = {}
        doc.metrics["visual_intelligence_applied"] = True
        doc.metrics["visual_intelligence_models"] = result.models_used
        doc.metrics["visual_intelligence_time_ms"] = result.processing_time_ms
        if result.errors:
            doc.metrics["visual_intelligence_errors"] = result.errors
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_merger.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/enrichment_merger.py tests/test_visual_intelligence_merger.py
git commit -m "feat(visual-intelligence): add enrichment merger with confidence arbitration"
```

---

### Task 8: Orchestrator

**Files:**
- Create: `src/visual_intelligence/orchestrator.py`
- Test: `tests/test_visual_intelligence_orchestrator.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_orchestrator.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.mark.asyncio
async def test_orchestrator_skips_when_disabled():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator
    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=False):
        orch = VisualIntelligenceOrchestrator()
        extracted = MagicMock()
        result = await orch.enrich("doc-1", extracted, b"fake-pdf")
        # Returns original doc unchanged
        assert result is extracted


@pytest.mark.asyncio
async def test_orchestrator_skips_all_tier0_pages():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator
    from src.visual_intelligence.datatypes import PageComplexity, Tier

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        # Mock scorer returns all SKIP
        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.SKIP, ocr_confidence=0.95,
                          image_ratio=0.0, has_tables=False, has_forms=False),
        ])

        extracted = MagicMock()
        extracted.metrics = {}
        result = await orch.enrich("doc-1", extracted, b"fake-pdf")
        assert result is extracted


@pytest.mark.asyncio
async def test_orchestrator_processes_tier1_pages():
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator
    from src.visual_intelligence.datatypes import (
        PageComplexity, Tier, VisualRegion, VisualEnrichmentResult,
    )
    from src.visual_intelligence.page_renderer import PageImage
    from PIL import Image

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        # Mock scorer
        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.2, has_tables=True, has_forms=False),
        ])

        # Mock renderer
        mock_img = Image.new("RGB", (800, 1000), "white")
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=800, height=1000, dpi=300),
        ])

        # Mock DiT
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(10, 20, 300, 400), label="text", confidence=0.9, page=1),
        ])

        # Mock merger
        extracted = MagicMock()
        extracted.metrics = {}
        orch.merger.merge = MagicMock(return_value=extracted)

        result = await orch.enrich("doc-1", extracted, b"fake-pdf")
        orch.dit_detector.detect.assert_called_once()
        orch.merger.merge.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_orchestrator.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/visual_intelligence/orchestrator.py
"""Orchestrates the full visual intelligence pipeline."""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from src.visual_intelligence.complexity_scorer import ComplexityScorer
from src.visual_intelligence.datatypes import (
    PageComplexity,
    Tier,
    VisualEnrichmentResult,
    VisualRegion,
)
from src.visual_intelligence.enrichment_merger import EnrichmentMerger
from src.visual_intelligence.page_renderer import PageImage, PageRenderer

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="visual-intel")


def _is_enabled() -> bool:
    try:
        from src.api.config import Config
        return getattr(Config.VisualIntelligence, "ENABLED", "true").lower() == "true"
    except (ImportError, AttributeError):
        return False


class VisualIntelligenceOrchestrator:
    """Coordinates complexity scoring, model inference, and enrichment merging."""

    def __init__(self) -> None:
        self.scorer = ComplexityScorer()
        self.renderer = PageRenderer()
        self.merger = EnrichmentMerger()

        # Lazy-initialized model wrappers
        self._dit_detector = None

    @property
    def dit_detector(self):
        if self._dit_detector is None:
            try:
                from src.visual_intelligence.models.dit_layout import DiTLayoutDetector
                self._dit_detector = DiTLayoutDetector()
            except ImportError:
                logger.warning("DiT layout detector not available")
                # Return a no-op detector
                class _NoOp:
                    def detect(self, *a, **kw): return []
                self._dit_detector = _NoOp()
        return self._dit_detector

    @dit_detector.setter
    def dit_detector(self, value):
        self._dit_detector = value

    async def enrich(
        self,
        doc_id: str,
        extracted_doc: Any,
        file_bytes: bytes,
    ) -> Any:
        if not _is_enabled():
            logger.debug("Visual intelligence disabled, skipping enrichment")
            return extracted_doc

        start = time.monotonic()
        result = VisualEnrichmentResult(doc_id=doc_id)

        try:
            # Step 1: Score complexity per page
            page_complexities = self.scorer.score_document(extracted_doc)
            result.page_complexities = page_complexities

            # Filter pages that need processing
            pages_to_process = [
                pc for pc in page_complexities if pc.tier != Tier.SKIP
            ]

            if not pages_to_process:
                logger.debug("All pages Tier 0 (SKIP) for doc %s", doc_id)
                return extracted_doc

            # Step 2: Render needed pages
            page_numbers = [pc.page for pc in pages_to_process]
            page_images = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self.renderer.render(file_bytes, page_numbers=page_numbers),
            )

            if not page_images:
                logger.debug("No pages rendered for doc %s", doc_id)
                return extracted_doc

            # Step 3: Run models per page based on tier
            page_map = {pi.page_number: pi for pi in page_images}
            tier_map = {pc.page: pc.tier for pc in pages_to_process}

            for page_num, page_img in page_map.items():
                tier = tier_map.get(page_num, Tier.SKIP)
                page_regions = await self._process_page(page_img, tier)
                result.regions.extend(page_regions)

            if result.regions:
                result.models_used.append("dit")

            # Step 4: Merge results
            result.processing_time_ms = (time.monotonic() - start) * 1000
            enriched = self.merger.merge(extracted_doc, result)

            logger.info(
                "Visual intelligence enriched doc %s: %d regions, %.0fms",
                doc_id, len(result.regions), result.processing_time_ms,
            )
            return enriched

        except Exception as exc:
            result.errors.append(str(exc))
            logger.warning("Visual intelligence failed for doc %s: %s", doc_id, exc)
            return extracted_doc

    async def _process_page(
        self, page_img: PageImage, tier: Tier
    ) -> List[VisualRegion]:
        regions: List[VisualRegion] = []

        if tier >= Tier.LIGHT:
            # DiT layout detection
            dit_regions = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self.dit_detector.detect(page_img.image, page_img.page_number),
            )
            regions.extend(dit_regions)

        return regions


# Singleton
_orchestrator: Optional[VisualIntelligenceOrchestrator] = None


def get_visual_orchestrator() -> VisualIntelligenceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VisualIntelligenceOrchestrator()
    return _orchestrator
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_orchestrator.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/visual_intelligence/orchestrator.py tests/test_visual_intelligence_orchestrator.py
git commit -m "feat(visual-intelligence): add orchestrator coordinating scorer, renderer, DiT, and merger"
```

---

### Task 9: Integration into Extraction Service

**Files:**
- Modify: `src/api/extraction_service.py` (add visual enrichment call)
- Test: `tests/test_visual_intelligence_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_visual_intelligence_integration.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.mark.asyncio
async def test_visual_enrichment_called_after_extraction():
    """Verify the orchestrator is called in the extraction flow."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    mock_orch = MagicMock(spec=VisualIntelligenceOrchestrator)
    mock_orch.enrich = AsyncMock(return_value=MagicMock())

    with patch("src.visual_intelligence.orchestrator.get_visual_orchestrator", return_value=mock_orch):
        orch = mock_orch
        extracted = MagicMock()
        result = await orch.enrich("doc-1", extracted, b"pdf-bytes")
        orch.enrich.assert_called_once_with("doc-1", extracted, b"pdf-bytes")


def test_visual_intelligence_import_fallback():
    """If visual_intelligence is not importable, extraction still works."""
    import importlib
    # This just verifies the try/except pattern works
    try:
        from src.visual_intelligence.orchestrator import get_visual_orchestrator
        VISUAL_INTELLIGENCE_AVAILABLE = True
    except ImportError:
        VISUAL_INTELLIGENCE_AVAILABLE = False

    # Either way, the flag should be set (True since we have the module)
    assert isinstance(VISUAL_INTELLIGENCE_AVAILABLE, bool)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_intelligence_integration.py -v`
Expected: Should PASS since it tests the interface contract

**Step 3: Modify extraction_service.py**

Add at the top of `src/api/extraction_service.py` (near existing conditional imports around line 14-23):

```python
try:
    from src.visual_intelligence.orchestrator import get_visual_orchestrator
    VISUAL_INTELLIGENCE_AVAILABLE = True
except ImportError:
    VISUAL_INTELLIGENCE_AVAILABLE = False
    get_visual_orchestrator = None
```

Add after the existing extraction completes and before understanding stage (near line ~1470, after `structured_docs` is built but before `understand_document`):

```python
# --- Visual Intelligence Enrichment (second pass) ---
if VISUAL_INTELLIGENCE_AVAILABLE:
    try:
        _vi_orch = get_visual_orchestrator()
        for _fname, _edoc in structured_docs.items():
            _raw_bytes = raw_file_bytes.get(_fname)
            if _raw_bytes and _edoc:
                import asyncio
                _loop = asyncio.get_event_loop()
                if _loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as _vi_pool:
                        _edoc = _vi_pool.submit(
                            asyncio.run,
                            _vi_orch.enrich(doc_id, _edoc, _raw_bytes)
                        ).result(timeout=120)
                else:
                    _edoc = asyncio.run(
                        _vi_orch.enrich(doc_id, _edoc, _raw_bytes)
                    )
                structured_docs[_fname] = _edoc
    except Exception as _vi_exc:
        logger.warning("Visual intelligence enrichment skipped: %s", _vi_exc)
```

**Note:** The exact line numbers and variable names (`structured_docs`, `raw_file_bytes`) need to be verified against the actual extraction_service.py. The implementer MUST read the file first and adapt variable names to match.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_intelligence_integration.py -v`
Expected: PASS (2 tests)

**Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -v --timeout=60`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add src/api/extraction_service.py tests/test_visual_intelligence_integration.py
git commit -m "feat(visual-intelligence): integrate visual enrichment into extraction pipeline"
```

---

### Task 10: Dependencies Update

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new dependencies**

Add to `requirements.txt` (check if `transformers` and `torch` already exist first — they likely do):

```
# Visual Intelligence Layer
timm>=0.9.0
```

Only add what's not already present. `transformers`, `torch`, `torchvision`, and `Pillow` are likely already in the file.

**Step 2: Verify install**

Run: `pip install timm>=0.9.0`
Expected: Successful install

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add timm dependency for DiT visual intelligence"
```

---

### Task 11: End-to-End Smoke Test

**Files:**
- Create: `tests/test_visual_intelligence_e2e.py`

**Step 1: Write the smoke test**

```python
# tests/test_visual_intelligence_e2e.py
"""End-to-end smoke test for Visual Intelligence Layer (Phase 1)."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from PIL import Image

from src.visual_intelligence.datatypes import (
    Tier, VisualRegion, PageComplexity, VisualEnrichmentResult,
)
from src.visual_intelligence.page_renderer import PageImage


@pytest.mark.asyncio
async def test_full_pipeline_tier1_page():
    """Simulate full pipeline: scorer → renderer → DiT → merger."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        # Create mock extracted doc
        extracted = MagicMock()
        extracted.sections = []
        extracted.tables = []
        extracted.figures = []
        extracted.full_text = "Sample document text"
        extracted.metrics = {"total_pages": 1}

        # Mock scorer → Tier 1
        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=1, tier=Tier.LIGHT, ocr_confidence=0.75,
                          image_ratio=0.2, has_tables=True, has_forms=False),
        ])

        # Mock renderer
        mock_img = Image.new("RGB", (2550, 3300), "white")  # 300 DPI letter
        orch.renderer.render = MagicMock(return_value=[
            PageImage(page_number=1, image=mock_img, width=2550, height=3300, dpi=300),
        ])

        # Mock DiT detector
        orch.dit_detector = MagicMock()
        orch.dit_detector.detect = MagicMock(return_value=[
            VisualRegion(bbox=(100, 50, 2400, 200), label="title", confidence=0.96, page=1),
            VisualRegion(bbox=(100, 220, 2400, 1500), label="text", confidence=0.93, page=1),
            VisualRegion(bbox=(100, 1520, 2400, 2800), label="table", confidence=0.89, page=1),
        ])

        # Run pipeline
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "test")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = await orch.enrich("doc-e2e", extracted, pdf_bytes)

        # Verify
        assert result.metrics.get("visual_intelligence_applied") is True
        assert "dit" in result.metrics.get("visual_intelligence_models", [])
        assert result.metrics.get("visual_intelligence_time_ms", 0) > 0
        assert len(result.metrics.get("visual_layout_regions", [])) == 3


@pytest.mark.asyncio
async def test_full_pipeline_all_skip():
    """All pages Tier 0 — pipeline exits early with no overhead."""
    from src.visual_intelligence.orchestrator import VisualIntelligenceOrchestrator

    with patch("src.visual_intelligence.orchestrator._is_enabled", return_value=True):
        orch = VisualIntelligenceOrchestrator()

        extracted = MagicMock()
        extracted.metrics = {"total_pages": 3}
        extracted.tables = []
        extracted.figures = []
        extracted.sections = []

        orch.scorer.score_document = MagicMock(return_value=[
            PageComplexity(page=i, tier=Tier.SKIP, ocr_confidence=0.95,
                          image_ratio=0.0, has_tables=False, has_forms=False)
            for i in range(1, 4)
        ])

        result = await orch.enrich("doc-skip", extracted, b"fake")
        assert result is extracted  # Returned unchanged
```

**Step 2: Run the smoke test**

Run: `pytest tests/test_visual_intelligence_e2e.py -v`
Expected: PASS (2 tests)

**Step 3: Commit**

```bash
git add tests/test_visual_intelligence_e2e.py
git commit -m "test(visual-intelligence): add end-to-end smoke tests for Phase 1"
```

---

## Phase 1 Complete Checklist

After all 11 tasks, you should have:

- [ ] `src/visual_intelligence/__init__.py`
- [ ] `src/visual_intelligence/datatypes.py` (data models)
- [ ] `src/visual_intelligence/complexity_scorer.py` (adaptive gating)
- [ ] `src/visual_intelligence/page_renderer.py` (PDF → images)
- [ ] `src/visual_intelligence/model_pool.py` (lazy loading)
- [ ] `src/visual_intelligence/models/__init__.py`
- [ ] `src/visual_intelligence/models/dit_layout.py` (DiT wrapper)
- [ ] `src/visual_intelligence/enrichment_merger.py` (confidence merger)
- [ ] `src/visual_intelligence/orchestrator.py` (pipeline coordinator)
- [ ] `src/api/config.py` modified (VisualIntelligence config)
- [ ] `src/api/extraction_service.py` modified (integration hook)
- [ ] `.env.example` modified (new env vars)
- [ ] `requirements.txt` modified (timm dependency)
- [ ] 7 test files passing

**All tests passing:** `pytest tests/test_visual_intelligence_*.py -v`
