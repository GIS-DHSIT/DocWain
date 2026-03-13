"""Tests for src/generation/reasoner.py and src/generation/composer.py."""

import pytest
from dataclasses import fields
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# ReasonerResult dataclass
# ---------------------------------------------------------------------------


class TestReasonerResult:
    def test_structure(self):
        from src.generation.reasoner import ReasonerResult

        r = ReasonerResult(
            text="answer", sources=[{"s": 1}], grounded=True
        )
        assert r.text == "answer"
        assert r.sources == [{"s": 1}]
        assert r.grounded is True
        assert r.thinking is None
        assert r.usage == {}

    def test_defaults(self):
        from src.generation.reasoner import ReasonerResult

        r = ReasonerResult(text="x", sources=[], grounded=False)
        assert r.thinking is None
        assert isinstance(r.usage, dict)


# ---------------------------------------------------------------------------
# Reasoner class
# ---------------------------------------------------------------------------

def _make_evidence(n=2, texts=None):
    items = []
    for i in range(n):
        items.append({
            "source_name": f"doc{i}.pdf",
            "page": i + 1,
            "section": f"Section {i}",
            "chunk_id": f"chunk-{i}",
            "document_id": f"doc-{i}",
            "source_index": i + 1,
            "score": 0.9 - i * 0.1,
            "text": (texts[i] if texts else f"Evidence text {i} with value $100."),
        })
    return items


class TestReasonerInit:
    def test_init_stores_gateway(self):
        from src.generation.reasoner import Reasoner

        gw = MagicMock()
        r = Reasoner(gw)
        assert r._llm is gw


class TestReasonerReason:
    def test_reason_calls_llm_returns_result(self):
        from src.generation.reasoner import Reasoner, ReasonerResult

        gw = MagicMock()
        gw.generate_with_metadata.return_value = (
            "The answer is **$100**. [SOURCE-1]",
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )

        r = Reasoner(gw)
        evidence = _make_evidence(2)
        result = r.reason(
            query="What is the value?",
            task_type="lookup",
            output_format="prose",
            evidence=evidence,
            doc_context=None,
            conversation_history=None,
        )

        assert isinstance(result, ReasonerResult)
        assert result.text == "The answer is **$100**. [SOURCE-1]"
        assert result.grounded is True
        assert len(result.sources) == 2
        gw.generate_with_metadata.assert_called_once()
        call_kwargs = gw.generate_with_metadata.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.2

    def test_reason_with_thinking(self):
        from src.generation.reasoner import Reasoner

        gw = MagicMock()
        gw.generate_with_metadata.return_value = (
            "<think>Let me think...</think>The answer is $100.",
            {"thinking": "Let me think..."},
        )

        r = Reasoner(gw)
        result = r.reason(
            query="q", task_type="lookup", output_format="prose",
            evidence=_make_evidence(1), doc_context=None,
            conversation_history=None, use_thinking=True,
        )

        call_kwargs = gw.generate_with_metadata.call_args
        assert call_kwargs.kwargs.get("think") is True
        # Thinking text should be captured
        assert result.thinking == "Let me think..."

    def test_reason_no_evidence_not_grounded(self):
        from src.generation.reasoner import Reasoner

        gw = MagicMock()
        gw.generate_with_metadata.return_value = (
            "No evidence available.", {}
        )

        r = Reasoner(gw)
        result = r.reason(
            query="q", task_type="lookup", output_format="prose",
            evidence=[], doc_context=None, conversation_history=None,
        )

        assert result.grounded is False

    def test_reason_handles_llm_failure(self):
        from src.generation.reasoner import Reasoner

        gw = MagicMock()
        gw.generate_with_metadata.side_effect = Exception("LLM timeout")

        r = Reasoner(gw)
        result = r.reason(
            query="q", task_type="lookup", output_format="prose",
            evidence=_make_evidence(1), doc_context=None,
            conversation_history=None,
        )

        assert result.grounded is False
        assert "error" in result.text.lower() or "unable" in result.text.lower()


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_lookup_less_than_summarize(self):
        from src.generation.reasoner import Reasoner

        assert Reasoner._compute_token_budget("lookup", 3, False) < \
               Reasoner._compute_token_budget("summarize", 3, False)

    def test_thinking_multiplier(self):
        from src.generation.reasoner import Reasoner

        base = Reasoner._compute_token_budget("lookup", 3, False)
        thinking = Reasoner._compute_token_budget("lookup", 3, True)
        assert thinking > base

    def test_evidence_scaling(self):
        from src.generation.reasoner import Reasoner

        small = Reasoner._compute_token_budget("extract", 5, False)
        large = Reasoner._compute_token_budget("extract", 15, False)
        assert large > small

    def test_max_cap(self):
        from src.generation.reasoner import Reasoner

        budget = Reasoner._compute_token_budget("summarize", 20, True)
        assert budget <= 4096


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------


class TestGroundingCheck:
    def test_numbers_present_grounded(self):
        from src.generation.reasoner import Reasoner

        r = Reasoner(MagicMock())
        evidence = _make_evidence(1, texts=["The total is $100 and 42 items."])
        assert r._check_grounding("Total: $100 with 42 items.", evidence) is True

    def test_numbers_missing_not_grounded(self):
        from src.generation.reasoner import Reasoner

        r = Reasoner(MagicMock())
        evidence = _make_evidence(1, texts=["The total is $100."])
        # Answer has numbers not in evidence
        assert r._check_grounding(
            "The total is $999 and $888 and $777.", evidence
        ) is False

    def test_no_evidence_not_grounded(self):
        from src.generation.reasoner import Reasoner

        r = Reasoner(MagicMock())
        assert r._check_grounding("Some answer with $100.", []) is False


# ---------------------------------------------------------------------------
# Composer: compose_response
# ---------------------------------------------------------------------------


class TestComposeResponse:
    def test_basic_structure(self):
        from src.generation.composer import compose_response

        result = compose_response(
            text="The answer.",
            evidence=_make_evidence(1),
            grounded=True,
            task_type="lookup",
            metadata={},
        )

        assert "response" in result
        assert "sources" in result
        assert "grounded" in result
        assert "context_found" in result
        assert "metadata" in result
        assert result["grounded"] is True
        assert result["context_found"] is True

    def test_no_evidence_context_found_false(self):
        from src.generation.composer import compose_response

        result = compose_response(
            text="No info.", evidence=[], grounded=False,
            task_type="lookup", metadata={},
        )
        assert result["context_found"] is False

    def test_cleans_preambles(self):
        from src.generation.composer import compose_response

        result = compose_response(
            text="Based on my analysis, the value is $100.",
            evidence=_make_evidence(1),
            grounded=True, task_type="lookup", metadata={},
        )
        assert not result["response"].startswith("Based on my analysis")

    def test_merges_adjacent_citations(self):
        from src.generation.composer import compose_response

        result = compose_response(
            text="The value is $100 [SOURCE-1][SOURCE-2] confirmed.",
            evidence=_make_evidence(2),
            grounded=True, task_type="lookup", metadata={},
        )
        assert "[SOURCE-1, SOURCE-2]" in result["response"]

    def test_source_deduplication(self):
        from src.generation.composer import compose_response

        # Two evidence items from same doc
        evidence = [
            {"source_name": "doc.pdf", "document_id": "d1", "page": 1,
             "section": "A", "chunk_id": "c1", "source_index": 1,
             "score": 0.9, "text": "text1"},
            {"source_name": "doc.pdf", "document_id": "d1", "page": 2,
             "section": "B", "chunk_id": "c2", "source_index": 2,
             "score": 0.8, "text": "text2"},
        ]

        result = compose_response(
            text="Answer.", evidence=evidence,
            grounded=True, task_type="lookup", metadata={},
        )
        # Should be deduplicated to one source entry
        assert len(result["sources"]) == 1
        # Should aggregate sections
        src = result["sources"][0]
        assert "A" in src.get("sections", []) and "B" in src.get("sections", [])

    def test_metadata_includes_engine(self):
        from src.generation.composer import compose_response

        result = compose_response(
            text="Answer.", evidence=_make_evidence(2),
            grounded=True, task_type="extract", metadata={"extra": "val"},
        )
        assert result["metadata"]["engine"] == "docwain_core_agent"
        assert result["metadata"]["task_type"] == "extract"
        assert result["metadata"]["evidence_count"] == 2
