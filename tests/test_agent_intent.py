"""Tests for src.agent.intent — IntentAnalyzer and QueryUnderstanding."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.intent import IntentAnalyzer, QueryUnderstanding


# ---------------------------------------------------------------------------
# QueryUnderstanding dataclass
# ---------------------------------------------------------------------------


class TestQueryUnderstanding:
    """Test QueryUnderstanding dataclass defaults and properties."""

    def test_defaults(self):
        qu = QueryUnderstanding(
            task_type="lookup",
            complexity="simple",
            resolved_query="test",
            output_format="prose",
            relevant_documents=[],
            cross_profile=False,
        )
        assert qu.sub_tasks is None
        assert qu.entities == []
        assert qu.needs_clarification is False
        assert qu.clarification_question is None

    def test_is_conversational_true(self):
        qu = QueryUnderstanding(
            task_type="conversational",
            complexity="simple",
            resolved_query="hello",
            output_format="prose",
            relevant_documents=[],
            cross_profile=False,
        )
        assert qu.is_conversational is True

    def test_is_conversational_false(self):
        qu = QueryUnderstanding(
            task_type="extract",
            complexity="simple",
            resolved_query="get the total",
            output_format="prose",
            relevant_documents=[],
            cross_profile=False,
        )
        assert qu.is_conversational is False

    def test_is_complex_true(self):
        qu = QueryUnderstanding(
            task_type="investigate",
            complexity="complex",
            resolved_query="compare all contracts",
            output_format="table",
            relevant_documents=[],
            cross_profile=False,
            sub_tasks=["task1", "task2"],
        )
        assert qu.is_complex is True

    def test_is_complex_false_simple_complexity(self):
        qu = QueryUnderstanding(
            task_type="lookup",
            complexity="simple",
            resolved_query="test",
            output_format="prose",
            relevant_documents=[],
            cross_profile=False,
            sub_tasks=["task1"],
        )
        assert qu.is_complex is False

    def test_is_complex_false_no_subtasks(self):
        qu = QueryUnderstanding(
            task_type="lookup",
            complexity="complex",
            resolved_query="test",
            output_format="prose",
            relevant_documents=[],
            cross_profile=False,
            sub_tasks=None,
        )
        assert qu.is_complex is False


# ---------------------------------------------------------------------------
# IntentAnalyzer
# ---------------------------------------------------------------------------


class TestIntentAnalyzerInit:
    def test_init_stores_gateway(self):
        gateway = MagicMock()
        analyzer = IntentAnalyzer(gateway)
        assert analyzer._llm is gateway


class TestIntentAnalyzerAnalyze:
    """Test analyze() with mock LLM."""

    def _make_analyzer(self, llm_response: str) -> IntentAnalyzer:
        gateway = MagicMock()
        gateway.generate.return_value = llm_response
        return IntentAnalyzer(gateway)

    def test_valid_json_response(self):
        response = json.dumps({
            "task_type": "extract",
            "complexity": "simple",
            "resolved_query": "What is the total revenue?",
            "output_format": "prose",
            "relevant_documents": [
                {"document_id": "doc1", "profile_id": "p1", "reason": "has financials"}
            ],
            "cross_profile": False,
            "sub_tasks": None,
            "entities": ["revenue"],
            "needs_clarification": False,
            "clarification_question": None,
        })
        analyzer = self._make_analyzer(response)
        result = analyzer.analyze(
            query="What is the total revenue?",
            subscription_id="sub1",
            profile_id="p1",
            doc_intelligence=[],
            conversation_history=None,
        )
        assert isinstance(result, QueryUnderstanding)
        assert result.task_type == "extract"
        assert result.complexity == "simple"
        assert result.resolved_query == "What is the total revenue?"
        assert result.entities == ["revenue"]

    def test_malformed_response_returns_safe_defaults(self):
        analyzer = self._make_analyzer("this is not json at all")
        result = analyzer.analyze(
            query="What is the total?",
            subscription_id="sub1",
            profile_id="p1",
            doc_intelligence=[],
            conversation_history=None,
        )
        assert result.task_type == "lookup"
        assert result.complexity == "simple"
        assert result.resolved_query == "What is the total?"

    def test_greeting_fast_path_no_llm_call(self):
        gateway = MagicMock()
        analyzer = IntentAnalyzer(gateway)
        result = analyzer.analyze(
            query="Hello!",
            subscription_id="sub1",
            profile_id="p1",
            doc_intelligence=[],
            conversation_history=None,
        )
        assert result.task_type == "conversational"
        gateway.generate.assert_not_called()

    def test_farewell_fast_path(self):
        gateway = MagicMock()
        analyzer = IntentAnalyzer(gateway)
        result = analyzer.analyze(
            query="Thank you!",
            subscription_id="sub1",
            profile_id="p1",
            doc_intelligence=[],
            conversation_history=None,
        )
        assert result.task_type == "conversational"
        gateway.generate.assert_not_called()

    def test_meta_question_fast_path(self):
        gateway = MagicMock()
        analyzer = IntentAnalyzer(gateway)
        result = analyzer.analyze(
            query="Who are you?",
            subscription_id="sub1",
            profile_id="p1",
            doc_intelligence=[],
            conversation_history=None,
        )
        assert result.task_type == "conversational"
        gateway.generate.assert_not_called()

    def test_various_greeting_patterns(self):
        gateway = MagicMock()
        analyzer = IntentAnalyzer(gateway)
        greetings = ["hi", "Hey there", "good morning", "howdy", "Greetings", "yo"]
        for greeting in greetings:
            result = analyzer.analyze(
                query=greeting,
                subscription_id="sub1",
                profile_id="p1",
                doc_intelligence=[],
                conversation_history=None,
            )
            assert result.task_type == "conversational", f"Failed for: {greeting}"
            gateway.generate.assert_not_called()

    def test_doc_intelligence_passed_to_prompt(self):
        response = json.dumps({
            "task_type": "extract",
            "complexity": "simple",
            "resolved_query": "test",
            "output_format": "prose",
            "relevant_documents": [],
            "cross_profile": False,
        })
        gateway = MagicMock()
        gateway.generate.return_value = response
        analyzer = IntentAnalyzer(gateway)

        doc_intel = [{"document_id": "doc123", "summary": "A contract document"}]

        with patch("src.agent.intent.build_understand_prompt") as mock_prompt:
            mock_prompt.return_value = "mocked prompt"
            analyzer.analyze(
                query="What is the contract value?",
                subscription_id="sub1",
                profile_id="p1",
                doc_intelligence=doc_intel,
                conversation_history=None,
            )
            mock_prompt.assert_called_once_with(
                "What is the contract value?", doc_intel, None
            )
            # Verify LLM was called with the built prompt
            gateway.generate.assert_called_once()
            call_args = gateway.generate.call_args
            assert call_args[0][0] == "mocked prompt"

    def test_invalid_task_type_falls_back_to_lookup(self):
        response = json.dumps({
            "task_type": "invalid_type",
            "complexity": "simple",
            "resolved_query": "test",
            "output_format": "prose",
            "relevant_documents": [],
            "cross_profile": False,
        })
        analyzer = self._make_analyzer(response)
        result = analyzer.analyze(
            query="test query",
            subscription_id="sub1",
            profile_id="p1",
            doc_intelligence=[],
            conversation_history=None,
        )
        assert result.task_type == "lookup"

    def test_markdown_fenced_json_parsed(self):
        raw = '```json\n{"task_type": "summarize", "complexity": "simple", "resolved_query": "summarize", "output_format": "sections", "relevant_documents": [], "cross_profile": false}\n```'
        analyzer = self._make_analyzer(raw)
        result = analyzer.analyze(
            query="summarize the document",
            subscription_id="sub1",
            profile_id="p1",
            doc_intelligence=[],
            conversation_history=None,
        )
        assert result.task_type == "summarize"
        assert result.output_format == "sections"
