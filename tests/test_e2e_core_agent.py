"""End-to-end tests for the Core Agent pipeline (UNDERSTAND -> RETRIEVE -> REASON -> COMPOSE).

Tests exercise the real CoreAgent class with mock LLM, Qdrant, embedder, and MongoDB
dependencies, verifying the full pipeline produces correct AnswerPayload structures.
"""

import json
import pytest
from unittest.mock import MagicMock, PropertyMock

from src.agent.core_agent import CoreAgent


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_intent_json(
    task_type: str = "extract",
    complexity: str = "simple",
    resolved_query: str = "What is John Smith's salary?",
    output_format: str = "prose",
) -> str:
    """Return valid UNDERSTAND JSON that IntentAnalyzer._parse_response accepts."""
    return json.dumps({
        "task_type": task_type,
        "complexity": complexity,
        "resolved_query": resolved_query,
        "output_format": output_format,
        "relevant_documents": [],
        "cross_profile": False,
        "entities": ["John Smith", "salary"],
    })


def _make_mock_llm(answer_text: str = "John Smith's salary is $125,000 per year.") -> MagicMock:
    """Mock LLM gateway.

    - .generate() returns UNDERSTAND JSON when system prompt contains "JSON",
      otherwise returns plain text (sub-agent calls).
    - .generate_with_metadata() returns (answer_text, metadata) for REASON calls.
    """
    llm = MagicMock()

    def _generate(prompt, system="", **kwargs):
        if "JSON" in (system or ""):
            return _make_intent_json()
        return answer_text

    llm.generate.side_effect = _generate
    llm.generate_with_metadata.return_value = (
        answer_text,
        {"usage": {"prompt_tokens": 200, "completion_tokens": 50}, "backend": "vllm"},
    )
    return llm


def _make_mock_point(
    text: str = "John Smith receives an annual salary of $125,000.",
    doc_id: str = "doc_001",
    profile_id: str = "hr",
    score: float = 0.94,
) -> MagicMock:
    """Create a single mock Qdrant point with canonical payload schema."""
    point = MagicMock()
    point.score = score
    point.id = "point-uuid-001"
    point.payload = {
        "canonical_text": text,
        "document_id": doc_id,
        "profile_id": profile_id,
        "subscription_id": "sub_1",
        "source_file": "employees.pdf",
        "chunk": {
            "id": "chunk-abc-001",
            "index": 0,
            "type": "text",
            "hash": "deadbeef",
        },
        "section": {
            "id": "sec-1",
            "title": "Compensation",
            "path": ["HR", "Compensation"],
        },
        "provenance": {
            "page_start": 3,
            "page_end": 3,
            "source_file": "employees.pdf",
        },
    }
    return point


def _make_mock_qdrant(points=None) -> MagicMock:
    """Mock QdrantClient.

    .query_points() returns a result object with .points list.
    .scroll() returns ([], None) for keyword fallback path.
    """
    qdrant = MagicMock()
    if points is None:
        points = [_make_mock_point(), _make_mock_point(
            text="Employee benefits include health insurance and 401k matching.",
            doc_id="doc_001",
            score=0.88,
        )]

    query_result = MagicMock()
    query_result.points = points
    qdrant.query_points.return_value = query_result

    # Keyword fallback returns empty
    qdrant.scroll.return_value = ([], None)
    return qdrant


def _make_mock_embedder() -> MagicMock:
    """Mock embedder where .encode([text]) returns [[0.1] * 1024]."""
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1] * 1024]
    return embedder


def _make_mock_mongodb() -> MagicMock:
    """Mock MongoDB collection.

    .find() returns a list of document intelligence dicts.
    """
    mongodb = MagicMock()
    mongodb.find.return_value = [
        {
            "document_id": "doc_001",
            "profile_id": "hr",
            "profile_name": "Human Resources",
            "intelligence": {
                "summary": "Employee compensation and benefits records.",
                "entities": ["John Smith", "salary", "$125,000"],
                "answerable_topics": ["salary", "benefits", "compensation"],
                "key_facts": ["John Smith earns $125,000 annually"],
                "document_type": "HR record",
            },
        }
    ]
    return mongodb


def _build_agent(**overrides) -> CoreAgent:
    """Construct a CoreAgent with all-mock dependencies."""
    kwargs = {
        "llm_gateway": _make_mock_llm(),
        "qdrant_client": _make_mock_qdrant(),
        "embedder": _make_mock_embedder(),
        "mongodb": _make_mock_mongodb(),
    }
    kwargs.update(overrides)
    return CoreAgent(**kwargs)


def _default_handle_kwargs() -> dict:
    """Common keyword arguments for CoreAgent.handle()."""
    return {
        "query": "What is John Smith's salary?",
        "subscription_id": "sub_1",
        "profile_id": "hr",
        "user_id": "user@test.com",
        "session_id": "sess_1",
        "conversation_history": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimpleExtractionQuery:
    """Full pipeline: extraction query with mocked evidence containing $125,000."""

    def test_response_contains_salary(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert "$125,000" in result["response"]

    def test_grounded_is_true(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert result["grounded"] is True

    def test_context_found_is_true(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert result["context_found"] is True

    def test_sources_not_empty(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert len(result["sources"]) >= 1

    def test_metadata_has_task_type_and_timing(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        meta = result["metadata"]
        assert "task_type" in meta
        assert "timing" in meta
        timing = meta["timing"]
        assert "understand_ms" in timing
        assert "retrieve_ms" in timing
        assert "reason_ms" in timing


class TestConversationalGreeting:
    """Greeting query should short-circuit without an LLM REASON call."""

    def test_response_mentions_docwain(self):
        agent = _build_agent()
        result = agent.handle(
            query="Hello!",
            subscription_id="sub_1",
            profile_id="hr",
            user_id="user@test.com",
            session_id="sess_1",
            conversation_history=None,
        )
        assert "DocWain" in result["response"]

    def test_task_type_is_conversational(self):
        agent = _build_agent()
        result = agent.handle(
            query="Hello!",
            subscription_id="sub_1",
            profile_id="hr",
            user_id="user@test.com",
            session_id="sess_1",
            conversation_history=None,
        )
        assert result["metadata"]["task_type"] == "conversational"

    def test_generate_with_metadata_not_called(self):
        llm = _make_mock_llm()
        agent = _build_agent(llm_gateway=llm)
        agent.handle(
            query="Hello!",
            subscription_id="sub_1",
            profile_id="hr",
            user_id="user@test.com",
            session_id="sess_1",
            conversation_history=None,
        )
        llm.generate_with_metadata.assert_not_called()


class TestMissingIdentifiers:
    """Empty subscription_id or profile_id must raise ValueError."""

    def test_missing_subscription_raises(self):
        agent = _build_agent()
        with pytest.raises(ValueError, match="subscription_id"):
            agent.handle(
                query="Any question",
                subscription_id="",
                profile_id="hr",
                user_id="user@test.com",
                session_id="sess_1",
                conversation_history=None,
            )

    def test_missing_profile_raises(self):
        agent = _build_agent()
        with pytest.raises(ValueError, match="profile_id"):
            agent.handle(
                query="Any question",
                subscription_id="sub_1",
                profile_id="",
                user_id="user@test.com",
                session_id="sess_1",
                conversation_history=None,
            )


class TestResponseStructure:
    """Verify the result dict has all AnswerPayload keys with correct types."""

    REQUIRED_KEYS = {"response", "sources", "grounded", "context_found", "metadata"}

    def test_all_keys_present(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert self.REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - result.keys()}"
        )

    def test_response_is_string(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert isinstance(result["response"], str)

    def test_sources_is_list(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert isinstance(result["sources"], list)

    def test_grounded_is_bool(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert isinstance(result["grounded"], bool)

    def test_context_found_is_bool(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert isinstance(result["context_found"], bool)

    def test_metadata_is_dict(self):
        agent = _build_agent()
        result = agent.handle(**_default_handle_kwargs())
        assert isinstance(result["metadata"], dict)


class TestNoEvidenceReturnsContextFoundFalse:
    """When Qdrant returns no points, context_found should be False."""

    def test_context_found_false_with_empty_results(self):
        qdrant = _make_mock_qdrant(points=[])
        agent = _build_agent(qdrant_client=qdrant)
        result = agent.handle(**_default_handle_kwargs())
        assert result["context_found"] is False
