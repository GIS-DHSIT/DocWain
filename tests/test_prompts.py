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
        assert "all parts" in prompt.lower() or "all 3" in prompt.lower()

    def test_contains_bolding_rule(self):
        prompt = build_system_prompt()
        assert "bold" in prompt.lower()
        assert "**value**" in prompt


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
            assert len(TASK_FORMATS[key]) > 20, f"TASK_FORMATS['{key}'] is too short"

    def test_extract_has_table_guidance(self):
        assert "table" in TASK_FORMATS["extract"].lower()

    def test_compare_has_table_guidance(self):
        assert "table" in TASK_FORMATS["compare"].lower()

    def test_summarize_has_bullet_guidance(self):
        assert "bullet" in TASK_FORMATS["summarize"].lower()

    def test_list_has_count_guidance(self):
        assert "count" in TASK_FORMATS["list"].lower()


# ---------------------------------------------------------------------------
# build_understand_prompt
# ---------------------------------------------------------------------------

class TestBuildUnderstandPrompt:
    def test_includes_query(self):
        prompt = build_understand_prompt("What is the revenue?", [], None)
        assert "What is the revenue?" in prompt

    def test_includes_doc_intelligence_list(self):
        doc_intel = [
            {
                "document_id": "doc_1",
                "profile_id": "prof_hr",
                "profile_name": "HR",
                "summary": "Employment contract for John Smith",
                "entities": ["John Smith", "Acme Corp"],
                "answerable_topics": ["salary", "compensation"],
            }
        ]
        prompt = build_understand_prompt("query", doc_intel, None)
        assert "[doc_1]" in prompt
        assert "HR" in prompt
        assert "Employment contract for John Smith" in prompt
        assert "John Smith" in prompt
        assert "Acme Corp" in prompt
        assert "salary" in prompt
        assert "compensation" in prompt

    def test_handles_empty_doc_intelligence(self):
        # Should not raise with empty list or None-like
        prompt = build_understand_prompt("query", [], None)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_conversation_history_query_response_keys(self):
        history = [
            {"query": "Tell me about revenue", "response": "Revenue was $5M"},
            {"query": "More details?", "response": "Q3 revenue was $2M"},
        ]
        prompt = build_understand_prompt("Breakdown?", [], history)
        assert "Tell me about revenue" in prompt
        assert "Revenue was $5M" in prompt

    def test_limits_conversation_history_to_5_turns(self):
        history = [
            {"query": f"turn-{i}", "response": f"resp-{i}"}
            for i in range(10)
        ]
        prompt = build_understand_prompt("query", [], history)
        # Last 5 turns should be present
        assert "turn-9" in prompt
        assert "turn-5" in prompt
        # Earlier turns should NOT be present
        assert "turn-0" not in prompt

    def test_output_contains_json_schema_instruction(self):
        prompt = build_understand_prompt("What is the salary?", [], None)
        assert "Respond ONLY with JSON" in prompt
        assert '"task_type"' in prompt
        assert '"complexity"' in prompt
        assert '"resolved_query"' in prompt
        assert '"output_format"' in prompt
        assert '"relevant_documents"' in prompt
        assert '"cross_profile"' in prompt
        assert '"sub_tasks"' in prompt
        assert '"entities"' in prompt
        assert '"needs_clarification"' in prompt
        assert '"clarification_question"' in prompt

    def test_multiple_documents_formatted(self):
        doc_intel = [
            {
                "document_id": "doc_1",
                "profile_id": "prof_hr",
                "profile_name": "HR",
                "summary": "Employment contract",
                "entities": ["Alice"],
                "answerable_topics": ["salary"],
            },
            {
                "document_id": "doc_2",
                "profile_id": "prof_legal",
                "profile_name": "Legal",
                "summary": "NDA agreement",
                "entities": ["Bob"],
                "answerable_topics": ["confidentiality"],
            },
        ]
        prompt = build_understand_prompt("query", doc_intel, None)
        assert "[doc_1]" in prompt
        assert "[doc_2]" in prompt
        assert "HR" in prompt
        assert "Legal" in prompt


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
            "score": 0.95,
            "source_index": 1,
        },
        {
            "text": "Expenses totaled $3M.",
            "source_name": "annual_report.pdf",
            "section": "Financials",
            "page": 13,
            "score": 0.88,
            "source_index": 2,
        },
    ]

    def test_includes_evidence_with_source_index(self):
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

    def test_uses_score_not_relevance_score(self):
        prompt = build_reason_prompt(
            query="q",
            task_type="extract",
            output_format="prose",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=None,
            conversation_history=None,
        )
        assert "0.95" in prompt
        assert "0.88" in prompt

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
        assert TASK_FORMATS["compare"] in prompt

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
        doc_ctx = {"summary": "Financial Analysis Report"}
        prompt = build_reason_prompt(
            query="q",
            task_type="extract",
            output_format="prose",
            evidence=self.SAMPLE_EVIDENCE,
            doc_context=doc_ctx,
            conversation_history=None,
        )
        # doc context should appear before evidence
        ctx_pos = prompt.find("Financial Analysis Report")
        ev_pos = prompt.find("[SOURCE-1]")
        assert ctx_pos < ev_pos, "Document context must appear before evidence"

    def test_includes_conversation_history_query_response_keys(self):
        history = [
            {"query": "prior-question", "response": "prior-answer"},
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
        assert "prior-answer" in prompt

    def test_limits_conversation_history_to_3_turns(self):
        history = [
            {"query": f"reason-turn-{i}", "response": f"resp-{i}"}
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

    def test_source_index_from_evidence_dict(self):
        """source_index should come from the evidence dict, not sequential enumeration."""
        evidence = [
            {
                "text": "Chunk A",
                "source_name": "a.pdf",
                "score": 0.9,
                "source_index": 5,
            },
            {
                "text": "Chunk B",
                "source_name": "b.pdf",
                "score": 0.8,
                "source_index": 12,
            },
        ]
        prompt = build_reason_prompt(
            query="q",
            task_type="extract",
            output_format="prose",
            evidence=evidence,
        )
        assert "[SOURCE-5]" in prompt
        assert "[SOURCE-12]" in prompt


# ---------------------------------------------------------------------------
# build_subagent_prompt
# ---------------------------------------------------------------------------

class TestBuildSubagentPrompt:
    def test_includes_role(self):
        prompt = build_subagent_prompt(
            role="financial analyst",
            evidence=[{"text": "Revenue was $5M", "source_name": "report.pdf", "score": 0.9, "source_index": 1}],
            doc_context=None,
        )
        assert "financial analyst" in prompt.lower()

    def test_includes_evidence(self):
        evidence = [
            {"text": "Clause 3.2 states liability is capped.", "source_name": "contract.pdf", "score": 0.85, "source_index": 1},
        ]
        prompt = build_subagent_prompt(
            role="legal reviewer",
            evidence=evidence,
            doc_context=None,
        )
        assert "Clause 3.2" in prompt

    def test_includes_doc_context(self):
        doc_ctx = {"summary": "Legal contract review"}
        prompt = build_subagent_prompt(
            role="reviewer",
            evidence=[],
            doc_context=doc_ctx,
        )
        assert "Legal contract review" in prompt
