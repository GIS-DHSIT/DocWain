"""Tests for the filter_high_quality() fallback that keeps top-N chunks
even when all scores fall below the low threshold.

Root cause #1: vector-search scores often range 0.3-0.6, so the quality
pipeline used to drop everything when no chunk exceeded LOW_SCORE_THRESHOLD
(0.45).  The fix adds a ``min_keep`` parameter that guarantees at least N
chunks survive the filter.
"""

from src.rag_v3.retrieve import filter_high_quality
from src.rag_v3.types import Chunk, ChunkSource


# ── Helper ────────────────────────────────────────────────────────────────

def _make_chunk(text: str, score: float, *, doc_name: str = "doc.pdf") -> Chunk:
    return Chunk(
        id=f"chunk_{hash(text) % 10000}",
        text=text,
        score=score,
        source=ChunkSource(document_name=doc_name),
        meta={},
    )


# ── Tests ─────────────────────────────────────────────────────────────────

class TestFilterHighQualityFallback:
    """Verify the min_keep fallback in filter_high_quality()."""

    def test_all_below_threshold_returns_min_keep(self):
        """When all 5 chunks score 0.3 (below 0.45), the top 3 are kept."""
        chunks = [_make_chunk(f"low {i}", 0.3 - i * 0.01) for i in range(5)]
        result = filter_high_quality(chunks)
        assert len(result) == 3
        # Should be sorted descending by score
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_mixed_scores_prefers_high_and_medium(self):
        """When high (>=0.7) and medium (>=0.45) chunks exist, the fallback
        is NOT triggered -- high+medium are returned as before."""
        high = [_make_chunk("high", 0.8)]
        medium = [_make_chunk("med", 0.5)]
        low = [_make_chunk("low", 0.3)]
        result = filter_high_quality(high + medium + low)
        texts = {c.text for c in result}
        assert "high" in texts
        assert "med" in texts
        assert "low" not in texts

    def test_empty_returns_empty(self):
        """Empty input produces empty output."""
        assert filter_high_quality([]) == []

    def test_scroll_score_zero_returns_min_keep(self):
        """Profile-scan / scroll chunks carry score=0.0.  The fallback must
        still return min_keep chunks rather than dropping everything."""
        chunks = [_make_chunk(f"scroll {i}", 0.0) for i in range(6)]
        result = filter_high_quality(chunks)
        assert len(result) == 3

    def test_min_keep_parameter_respected(self):
        """A caller can override min_keep to get more (or fewer) chunks."""
        chunks = [_make_chunk(f"c {i}", 0.2) for i in range(10)]
        result = filter_high_quality(chunks, min_keep=5)
        assert len(result) == 5

    def test_min_keep_capped_by_input_size(self):
        """If fewer chunks than min_keep exist, all are returned."""
        chunks = [_make_chunk("only", 0.1), _make_chunk("two", 0.05)]
        result = filter_high_quality(chunks, min_keep=5)
        assert len(result) == 2

    def test_fallback_returns_highest_scored(self):
        """The fallback must return the highest-scored chunks, not arbitrary ones."""
        chunks = [
            _make_chunk("worst", 0.10),
            _make_chunk("best", 0.40),
            _make_chunk("mid", 0.25),
            _make_chunk("ok", 0.35),
            _make_chunk("poor", 0.15),
        ]
        result = filter_high_quality(chunks, min_keep=3)
        result_texts = [c.text for c in result]
        assert result_texts == ["best", "ok", "mid"]
