import pytest
from src.visual_intelligence.datatypes import Tier


def test_simple_text_page_is_tier_skip():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    scorer = ComplexityScorer()
    result = scorer.score_page(
        page=1, ocr_confidence=0.95, image_ratio=0.0,
        has_tables=False, has_forms=False,
        block_types={"text": 50, "image": 0},
    )
    assert result.tier == Tier.SKIP


def test_moderate_page_is_tier_light():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    scorer = ComplexityScorer()
    result = scorer.score_page(
        page=2, ocr_confidence=0.78, image_ratio=0.2,
        has_tables=True, has_forms=False,
        block_types={"text": 30, "image": 5},
    )
    assert result.tier == Tier.LIGHT


def test_complex_page_is_tier_full():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    scorer = ComplexityScorer()
    result = scorer.score_page(
        page=3, ocr_confidence=0.45, image_ratio=0.8,
        has_tables=True, has_forms=True,
        block_types={"text": 5, "image": 20},
    )
    assert result.tier == Tier.FULL


def test_score_extracted_document():
    from src.visual_intelligence.complexity_scorer import ComplexityScorer
    from unittest.mock import MagicMock
    scorer = ComplexityScorer()
    extracted = MagicMock()
    extracted.figures = [MagicMock(page=2, ocr_confidence=0.55)]
    extracted.tables = [MagicMock(page=2)]
    extracted.sections = [MagicMock(start_page=1, end_page=1), MagicMock(start_page=2, end_page=2)]
    extracted.metrics = {"total_pages": 2}
    results = scorer.score_document(extracted)
    assert len(results) == 2
    assert results[0].tier == Tier.SKIP
    assert results[1].tier in (Tier.LIGHT, Tier.FULL)
