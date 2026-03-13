"""Tests for src/generation/prompts.py — unified prompt templates."""

import pytest

from src.generation.prompts import (
    TASK_FORMATS,
    build_reason_prompt,
    build_subagent_prompt,
    build_system_prompt,
    build_understand_prompt,
)


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_returns_non_empty_string(self):
        result = build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_grounding_rules(self):
        prompt = build_system_prompt()
        # Must enforce exact values, cite sources, no preamble
        assert "exact" in prompt.lower() or "verbatim" in prompt.lower()
        assert "source" in prompt.lower()
        assert "preamble" in prompt.lower()

    def test_contains_conflict_reporting(self):
        prompt = build_system_prompt()
        assert "conflict" in prompt.lower()

    def test_contains_completeness_rule(self):
        prompt = build_system_prompt()
        assert "complete" in prompt.lower()


# ---------------------------------------------------------------------------
# TASK_FORMATS
# ---------------------------------------------------------------------------

class TestTaskFormats:
    REQUIRED_KEYS = [
        "extract",
        "compare",
        "summarize",
        "investigate",
        "lookup",
        "aggregate",
        "list",
    ]

    def test_has_all_required_keys(self):
        for key in self.REQUIRED_KEYS:
            assert key in TASK_FORMATS, f"Missing task format key: {key}"

    def test_values_are_non_empty_strings(self):
        for key in self.REQUIRED_KEYS:
            assert isinstance(TASK_FORMATS[key], str)
            assert len(TASK_FORMATS[key]) > 0


# ---------------------------------------------------------------------------
# build_understand_prompt
# ---------------------------------------------------------------------------

class TestBuildUnderstandPrompt:
    def test_includes_query(self):
        prompt = build_understand_prompt("What is the revenue?", None, None)
        assert "What is the revenue?" in prompt

    def test_includes_doc_intelligence(self):
        doc_intel = {"documents": ["report.pdf"], "topics": ["finance"]}
        prompt = build_understand_prompt("query", doc_intel, None)
        assert "report.pdf" in prompt
        assert "finance" in prompt

    def test_handles_empty_doc_intelligence(self):
        # Should not raise with None or empty dict
        prompt = build_understand_prompt("query", None, None)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        prompt2 = build_understand_prompt("query", {}, None)
        assert isinstance(prompt2, str)
        assert len(prompt2) > 0

    def test_includes_conversation_history(self):
        history = [
            {"role": "user", "content": "Tell me about revenue"},
            {"role": "assistant", "content": "Revenue was $5M"},
        ]
        prompt = build_understand_prompt("More details?", None, history)
        assert "Tell me about revenue" in prompt
        assert "Revenue was $5M" in prompt

    def test_limits_conversation_history_to_5_turns(self):
        history = [
            {"role": "user", "content": f"turn-{i}"}
            for i in range(10)
        ]
        prompt = build_understand_prompt("query", None, history)
        # Last 5 turns should be present
        assert "turn-9" in prompt
        assert "turn-5" in prompt
        # Earlier turns should NOT be present
        assert "turn-0" not in prompt


# ---------------------------------------------------------------------------
# build_reason_prompt
# ---------------------------------------------------------------------------

class TestBuildReasonPrompt:
    SAMPLE_EVIDENCE = [
        {
            "text": "Revenue was $5M in Q3.",
            "source_name": "annual_report.pdf",
            "section": "Financials",
            "page": 12,
            "relevance_score": 0.95,
        },
        {
            "text": "Expenses totaled $3M.",
            "source_name": "annual_report.pdf",
            "section": "Financials",
            "page": 13,
            "relevance_score": 0.88,
        },
    ]

    def test_includes_evidence_numbered_as_source_n(self):
        prompt = build_reason_prompt(
            query="What is the revenue?",
            task_type="extract",
            output_format="prose",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=None,
            conversation_history=None,
        )
        assert "[SOURCE-1]" in prompt
        assert "[SOURCE-2]" in prompt
        assert "Revenue was $5M" in prompt

    def test_includes_evidence_metadata(self):
        prompt = build_reason_prompt(
            query="q",
            task_type="extract",
            output_format="prose",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=None,
            conversation_history=None,
        )
        assert "annual_report.pdf" in prompt
        assert "Financials" in prompt

    def test_includes_task_format_instructions(self):
        prompt = build_reason_prompt(
            query="Compare A and B",
            task_type="compare",
            output_format="table",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=None,
            conversation_history=None,
        )
        # Should contain the compare task instructions
        assert TASK_FORMATS["compare"] in prompt or "compare" in prompt.lower()

    def test_includes_output_format_instructions(self):
        prompt = build_reason_prompt(
            query="q",
            task_type="extract",
            output_format="table",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=None,
            conversation_history=None,
        )
        assert "table" in prompt.lower()

    def test_doc_context_before_evidence(self):
        doc_ctx = {"profile": "Financial Analysis", "doc_types": ["annual report"]}
        prompt = build_reason_prompt(
            query="q",
            task_type="extract",
            output_format="prose",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=doc_ctx,
            conversation_history=None,
        )
        # doc context should appear before evidence
        ctx_pos = prompt.find("Financial Analysis")
        ev_pos = prompt.find("[SOURCE-1]")
        assert ctx_pos < ev_pos, "Document context must appear before evidence"

    def test_includes_conversation_history(self):
        history = [
            {"role": "user", "content": "prior-question"},
            {"role": "assistant", "content": "prior-answer"},
        ]
        prompt = build_reason_prompt(
            query="follow-up?",
            task_type="lookup",
            output_format="prose",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=None,
            conversation_history=history,
        )
        assert "prior-question" in prompt

    def test_limits_conversation_history_to_3_turns(self):
        history = [
            {"role": "user", "content": f"reason-turn-{i}"}
            for i in range(8)
        ]
        prompt = build_reason_prompt(
            query="q",
            task_type="extract",
            output_format="prose",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=None,
            conversation_history=history,
        )
        assert "reason-turn-7" in prompt
        assert "reason-turn-5" in prompt
        assert "reason-turn-0" not in prompt


# ---------------------------------------------------------------------------
# build_subagent_prompt
# ---------------------------------------------------------------------------

class TestBuildSubagentPrompt:
    def test_includes_role(self):
        prompt = build_subagent_prompt(
            role="financial analyst",
            evidence=[{"text": "Revenue was $5M", "source_name": "report.pdf"}],
            doc_context=None,
        )
        assert "financial analyst" in prompt.lower()

    def test_includes_evidence(self):
        evidence = [
            {"text": "Clause 3.2 states liability is capped.", "source_name": "contract.pdf"},
        ]
        prompt = build_subagent_prompt(
            role="legal reviewer",
            evidence=evidence,
            doc_context=None,
        )
        assert "Clause 3.2" in prompt

    def test_includes_doc_context(self):
        doc_ctx = {"profile": "Legal", "doc_types": ["contract"]}
        prompt = build_subagent_prompt(
            role="reviewer",
            evidence=[],
            doc_context=doc_ctx,
        )
        assert "Legal" in prompt
