"""Tests for visual intelligence data models."""

import pytest

from src.visual_intelligence.datatypes import (
    KVPair,
    OCRPatch,
    PageComplexity,
    StructuredTableResult,
    Tier,
    VisualEnrichmentResult,
    VisualRegion,
)


class TestTier:
    def test_tier_values(self):
        assert Tier.SKIP == 0
        assert Tier.LIGHT == 1
        assert Tier.FULL == 2

    def test_tier_is_int(self):
        assert isinstance(Tier.FULL, int)


class TestVisualRegion:
    def test_creation(self):
        region = VisualRegion(
            bbox=(10.0, 20.0, 100.0, 200.0),
            label="table",
            confidence=0.95,
            page=1,
        )
        assert region.bbox == (10.0, 20.0, 100.0, 200.0)
        assert region.label == "table"
        assert region.confidence == 0.95
        assert region.page == 1

    def test_default_source(self):
        region = VisualRegion(
            bbox=(0.0, 0.0, 1.0, 1.0),
            label="figure",
            confidence=0.8,
            page=0,
        )
        assert region.source == "visual_intelligence"

    def test_custom_source(self):
        region = VisualRegion(
            bbox=(0.0, 0.0, 1.0, 1.0),
            label="figure",
            confidence=0.8,
            page=0,
            source="custom_model",
        )
        assert region.source == "custom_model"


class TestStructuredTableResult:
    def test_creation(self):
        table = StructuredTableResult(
            page=2,
            bbox=(50.0, 100.0, 500.0, 400.0),
            headers=["Name", "Age", "City"],
            rows=[["Alice", "30", "NYC"], ["Bob", "25", "LA"]],
            spans=[{"row": 0, "col": 0, "rowspan": 1, "colspan": 2}],
            confidence=0.92,
        )
        assert table.page == 2
        assert len(table.headers) == 3
        assert len(table.rows) == 2
        assert table.confidence == 0.92
        assert len(table.spans) == 1

    def test_empty_table(self):
        table = StructuredTableResult(
            page=1,
            bbox=(0.0, 0.0, 1.0, 1.0),
            headers=[],
            rows=[],
            spans=[],
            confidence=0.5,
        )
        assert table.headers == []
        assert table.rows == []


class TestOCRPatch:
    def test_creation(self):
        patch = OCRPatch(
            page=1,
            bbox=(10.0, 20.0, 200.0, 40.0),
            original_text="helo wrld",
            enhanced_text="hello world",
            original_confidence=0.4,
            enhanced_confidence=0.95,
            method="trocr",
        )
        assert patch.original_text == "helo wrld"
        assert patch.enhanced_text == "hello world"
        assert patch.method == "trocr"


class TestKVPair:
    def test_creation_with_defaults(self):
        kv = KVPair(
            key="Invoice Number",
            value="INV-001",
            confidence=0.88,
            page=1,
        )
        assert kv.key == "Invoice Number"
        assert kv.value == "INV-001"
        assert kv.bbox is None
        assert kv.source == "visual_intelligence"

    def test_creation_with_bbox(self):
        kv = KVPair(
            key="Date",
            value="2026-03-24",
            confidence=0.9,
            page=1,
            bbox=(10.0, 50.0, 200.0, 70.0),
        )
        assert kv.bbox == (10.0, 50.0, 200.0, 70.0)


class TestPageComplexity:
    def test_creation(self):
        pc = PageComplexity(
            page=1,
            tier=Tier.FULL,
            ocr_confidence=0.6,
            image_ratio=0.4,
            has_tables=True,
            has_forms=False,
            signals={"low_ocr": True, "high_image_ratio": True},
        )
        assert pc.page == 1
        assert pc.tier == Tier.FULL
        assert pc.tier == 2
        assert pc.has_tables is True
        assert pc.has_forms is False
        assert "low_ocr" in pc.signals

    def test_skip_tier(self):
        pc = PageComplexity(
            page=3,
            tier=Tier.SKIP,
            ocr_confidence=0.99,
            image_ratio=0.0,
            has_tables=False,
            has_forms=False,
            signals={},
        )
        assert pc.tier == Tier.SKIP
        assert pc.tier == 0


class TestVisualEnrichmentResult:
    def test_defaults(self):
        result = VisualEnrichmentResult(doc_id="doc-123")
        assert result.doc_id == "doc-123"
        assert result.regions == []
        assert result.tables == []
        assert result.ocr_patches == {}
        assert result.kv_pairs == []
        assert result.page_complexities == []
        assert result.processing_time_ms == 0.0
        assert result.models_used == []
        assert result.errors == []

    def test_to_dict(self):
        region = VisualRegion(
            bbox=(0.0, 0.0, 1.0, 1.0),
            label="table",
            confidence=0.9,
            page=1,
        )
        result = VisualEnrichmentResult(
            doc_id="doc-456",
            regions=[region],
            processing_time_ms=150.5,
            models_used=["dit", "trocr"],
        )
        d = result.to_dict()
        assert d["doc_id"] == "doc-456"
        assert len(d["regions"]) == 1
        assert d["regions"][0]["label"] == "table"
        assert d["regions"][0]["source"] == "visual_intelligence"
        assert d["processing_time_ms"] == 150.5
        assert d["models_used"] == ["dit", "trocr"]
        assert d["errors"] == []

    def test_to_dict_with_all_fields(self):
        table = StructuredTableResult(
            page=1,
            bbox=(0.0, 0.0, 1.0, 1.0),
            headers=["A"],
            rows=[["1"]],
            spans=[],
            confidence=0.9,
        )
        patch = OCRPatch(
            page=1,
            bbox=(0.0, 0.0, 1.0, 1.0),
            original_text="x",
            enhanced_text="y",
            original_confidence=0.3,
            enhanced_confidence=0.9,
            method="trocr",
        )
        kv = KVPair(key="k", value="v", confidence=0.8, page=1)
        pc = PageComplexity(
            page=1,
            tier=Tier.LIGHT,
            ocr_confidence=0.7,
            image_ratio=0.2,
            has_tables=True,
            has_forms=False,
            signals={},
        )
        result = VisualEnrichmentResult(
            doc_id="doc-789",
            regions=[],
            tables=[table],
            ocr_patches={1: [patch]},
            kv_pairs=[kv],
            page_complexities=[pc],
            processing_time_ms=200.0,
            models_used=["table_transformer"],
            errors=["minor warning"],
        )
        d = result.to_dict()
        assert len(d["tables"]) == 1
        assert d["tables"][0]["headers"] == ["A"]
        assert 1 in d["ocr_patches"] or "1" in d["ocr_patches"]
        assert len(d["kv_pairs"]) == 1
        assert d["kv_pairs"][0]["source"] == "visual_intelligence"
        assert len(d["page_complexities"]) == 1
        assert d["page_complexities"][0]["tier"] == 1
        assert d["errors"] == ["minor warning"]
