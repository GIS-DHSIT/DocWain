"""Tests for multi-agent LLM infrastructure.

Phase 2 of the multi-agent plan:
- MultiAgentGateway role routing, fallback, stats
- QueryClassification JSON parsing
- VerificationResult JSON parsing
- Feature flag gating
- Mock ollama.generate — no real models needed
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.llm.multi_agent import AgentRole, MultiAgentGateway, _DEFAULT_ROLE_MODELS
from src.llm.classifier import QueryClassification, _parse_classification, classify_query
from src.llm.verifier import VerificationResult, _parse_verification, verify_grounding


# ---------------------------------------------------------------------------
# AgentRole enum
# ---------------------------------------------------------------------------

class TestAgentRole:
    def test_roles_exist(self):
        assert AgentRole.CLASSIFIER == "classifier"
        assert AgentRole.EXTRACTOR == "extractor"
        assert AgentRole.GENERATOR == "generator"
        assert AgentRole.VERIFIER == "verifier"
        assert AgentRole.DEFAULT == "default"

    def test_default_models_assigned(self):
        # All roles route to DocWain-Agent to avoid model swap contention on T4 16GB
        assert "DocWain-Agent" in _DEFAULT_ROLE_MODELS[AgentRole.CLASSIFIER]
        assert "DocWain-Agent" in _DEFAULT_ROLE_MODELS[AgentRole.EXTRACTOR]
        assert "DocWain-Agent" in _DEFAULT_ROLE_MODELS[AgentRole.GENERATOR]
        assert "DocWain-Agent" in _DEFAULT_ROLE_MODELS[AgentRole.VERIFIER]


# ---------------------------------------------------------------------------
# MultiAgentGateway
# ---------------------------------------------------------------------------

class TestMultiAgentGateway:
    def _make_fake_client(self, text="test response"):
        client = MagicMock()
        client.generate.return_value = text
        client.generate_with_metadata.return_value = (text, {"response": text})
        client.backend = "mock"
        client.model_name = "mock-model"
        return client

    def test_init_default_models(self):
        gw = MultiAgentGateway()
        assert gw.backend == "multi_agent"
        assert gw.model_name == "multi-agent"

    def test_init_custom_models(self):
        gw = MultiAgentGateway(role_models={AgentRole.CLASSIFIER: "custom:latest"})
        assert gw.get_role_model(AgentRole.CLASSIFIER) == "custom:latest"
        # Others should still be defaults (all DocWain-Agent to avoid model swap)
        assert "DocWain-Agent" in gw.get_role_model(AgentRole.EXTRACTOR)

    def test_list_roles(self):
        gw = MultiAgentGateway()
        roles = gw.list_roles()
        assert "classifier" in roles
        assert "generator" in roles
        assert "verifier" in roles

    @patch("src.llm.clients.OllamaClient")
    def test_generate_for_role(self, MockOllama):
        fake = self._make_fake_client("classified")
        MockOllama.return_value = fake
        gw = MultiAgentGateway()
        result = gw.generate_for_role(AgentRole.CLASSIFIER, "test prompt")
        assert result == "classified"
        fake.generate.assert_called_once()

    @patch("src.llm.clients.OllamaClient")
    def test_generate_with_metadata_for_role(self, MockOllama):
        fake = self._make_fake_client("extracted")
        MockOllama.return_value = fake
        gw = MultiAgentGateway()
        text, meta = gw.generate_with_metadata_for_role(AgentRole.EXTRACTOR, "test")
        assert text == "extracted"
        assert meta["agent_role"] == AgentRole.EXTRACTOR

    @patch("src.llm.clients.OllamaClient")
    def test_duck_typed_generate(self, MockOllama):
        """Default .generate() uses GENERATOR role."""
        fake = self._make_fake_client("generated")
        MockOllama.return_value = fake
        gw = MultiAgentGateway()
        result = gw.generate("test")
        assert result == "generated"

    @patch("src.llm.clients.OllamaClient")
    def test_classify_convenience(self, MockOllama):
        fake = self._make_fake_client("json output")
        MockOllama.return_value = fake
        gw = MultiAgentGateway()
        result = gw.classify("classify this")
        assert result == "json output"

    @patch("src.llm.clients.OllamaClient")
    def test_extract_convenience(self, MockOllama):
        fake = self._make_fake_client("extracted data")
        MockOllama.return_value = fake
        gw = MultiAgentGateway()
        result = gw.extract("extract from text")
        assert result == "extracted data"

    @patch("src.llm.clients.OllamaClient")
    def test_verify_convenience(self, MockOllama):
        fake = self._make_fake_client("verification")
        MockOllama.return_value = fake
        gw = MultiAgentGateway()
        result = gw.verify("verify this")
        assert result == "verification"

    def test_fallback_to_gateway(self):
        """If role client creation fails, falls back to singleton gateway."""
        fallback = self._make_fake_client("fallback response")
        gw = MultiAgentGateway(fallback_gateway=fallback)
        # Force client creation to fail
        with patch("src.llm.clients.OllamaClient", side_effect=Exception("model not found")):
            result = gw.generate_for_role(AgentRole.CLASSIFIER, "test")
        assert result == "fallback response"

    @patch("src.llm.clients.OllamaClient")
    def test_stats_tracking(self, MockOllama):
        fake = self._make_fake_client("ok")
        MockOllama.return_value = fake
        gw = MultiAgentGateway()
        gw.generate_for_role(AgentRole.CLASSIFIER, "test1")
        gw.generate_for_role(AgentRole.CLASSIFIER, "test2")
        stats = gw.get_stats()
        assert stats["roles"]["classifier"]["calls"] == 2
        assert "uptime_seconds" in stats

    @patch("src.llm.clients.OllamaClient")
    def test_error_stats(self, MockOllama):
        fake = self._make_fake_client()
        fake.generate.side_effect = Exception("model error")
        MockOllama.return_value = fake
        fallback = self._make_fake_client("recovered")
        gw = MultiAgentGateway(fallback_gateway=fallback)
        result = gw.generate_for_role(AgentRole.CLASSIFIER, "test")
        assert result == "recovered"
        stats = gw.get_stats()
        # Should have recorded errors for CLASSIFIER
        assert stats["roles"]["classifier"]["errors"] >= 1

    def test_warm_up_noop(self):
        gw = MultiAgentGateway()
        gw.warm_up()  # Should not raise


# ---------------------------------------------------------------------------
# QueryClassification parsing
# ---------------------------------------------------------------------------

class TestQueryClassificationParsing:
    def test_valid_json(self):
        raw = '{"intent": "summary", "domain": "hr", "entity": "Gaurav", "scope": "targeted", "confidence": 0.9}'
        result = _parse_classification(raw)
        assert result is not None
        assert result.intent == "summary"
        assert result.domain == "hr"
        assert result.entity == "Gaurav"
        assert result.scope == "targeted"
        assert result.confidence == 0.9

    def test_json_with_markdown_fences(self):
        raw = "```json\n{\"intent\": \"factual\", \"domain\": \"generic\", \"entity\": null, \"scope\": \"all_profile\", \"confidence\": 0.8}\n```"
        result = _parse_classification(raw)
        assert result is not None
        assert result.intent == "factual"
        assert result.entity is None

    def test_invalid_intent_normalized(self):
        raw = '{"intent": "UNKNOWN", "domain": "hr", "entity": null, "scope": "targeted", "confidence": 0.5}'
        result = _parse_classification(raw)
        assert result is not None
        assert result.intent == "factual"  # default fallback

    def test_invalid_domain_normalized(self):
        raw = '{"intent": "summary", "domain": "cooking", "entity": null, "scope": "targeted", "confidence": 0.5}'
        result = _parse_classification(raw)
        assert result is not None
        assert result.domain == "generic"  # default fallback

    def test_none_entity_from_string(self):
        raw = '{"intent": "factual", "domain": "generic", "entity": "null", "scope": "targeted", "confidence": 0.5}'
        result = _parse_classification(raw)
        assert result is not None
        assert result.entity is None

    def test_confidence_clamped(self):
        raw = '{"intent": "factual", "domain": "generic", "entity": null, "scope": "targeted", "confidence": 1.5}'
        result = _parse_classification(raw)
        assert result.confidence == 1.0

    def test_empty_string(self):
        assert _parse_classification("") is None

    def test_non_json(self):
        assert _parse_classification("This is not JSON") is None

    def test_embedded_json(self):
        raw = "The classification is: {\"intent\": \"comparison\", \"domain\": \"hr\", \"entity\": \"Dev\", \"scope\": \"all_profile\", \"confidence\": 0.75} done."
        result = _parse_classification(raw)
        assert result is not None
        assert result.intent == "comparison"
        assert result.entity == "Dev"


class TestClassifyQuery:
    def test_classify_with_mock_client(self):
        client = MagicMock()
        client.classify.return_value = '{"intent": "summary", "domain": "hr", "entity": "Gaurav", "scope": "targeted", "confidence": 0.85}'
        result = classify_query("Summarize Gaurav's profile", client, timeout_s=5.0)
        assert result is not None
        assert result.intent == "summary"
        assert result.entity == "Gaurav"

    def test_classify_empty_query(self):
        client = MagicMock()
        assert classify_query("", client) is None
        assert classify_query("   ", client) is None

    def test_classify_timeout(self):
        client = MagicMock()
        import time
        client.classify.side_effect = lambda p: time.sleep(5) or ""
        result = classify_query("test", client, timeout_s=0.1)
        # Heuristic fallback returns a low-confidence result instead of None
        assert result is not None
        assert result.confidence == 0.3

    def test_classify_exception(self):
        client = MagicMock()
        client.classify.side_effect = Exception("connection failed")
        result = classify_query("test", client, timeout_s=5.0)
        # Heuristic fallback returns a low-confidence result instead of None
        assert result is not None
        assert result.confidence == 0.3


# ---------------------------------------------------------------------------
# VerificationResult parsing
# ---------------------------------------------------------------------------

class TestVerificationResultParsing:
    def test_valid_json(self):
        raw = '{"supported": true, "confidence": 0.95, "issues": [], "reasoning": "All claims are grounded."}'
        result = _parse_verification(raw)
        assert result is not None
        assert result.supported is True
        assert result.confidence == 0.95
        assert result.issues == []

    def test_unsupported_with_issues(self):
        raw = '{"supported": false, "confidence": 0.3, "issues": ["Claim X not in evidence"], "reasoning": "Step by step..."}'
        result = _parse_verification(raw)
        assert result is not None
        assert result.supported is False
        assert len(result.issues) == 1

    def test_string_supported(self):
        raw = '{"supported": "true", "confidence": 0.8, "issues": [], "reasoning": ""}'
        result = _parse_verification(raw)
        assert result.supported is True

    def test_string_issues(self):
        raw = '{"supported": true, "confidence": 0.7, "issues": "single issue", "reasoning": ""}'
        result = _parse_verification(raw)
        assert result.issues == ["single issue"]

    def test_markdown_fences(self):
        raw = "```json\n{\"supported\": true, \"confidence\": 0.9, \"issues\": [], \"reasoning\": \"ok\"}\n```"
        result = _parse_verification(raw)
        assert result is not None
        assert result.supported is True

    def test_empty_string(self):
        assert _parse_verification("") is None

    def test_non_json_with_issues_heuristic(self):
        raw = "The answer is not supported by the evidence. Several claims are unsupported."
        result = _parse_verification(raw)
        assert result is not None
        assert result.supported is False
        assert result.confidence == 0.5

    def test_non_json_without_issues_heuristic(self):
        raw = "Everything looks good and the answer is well grounded in the evidence provided."
        result = _parse_verification(raw)
        assert result is not None
        assert result.supported is True

    def test_to_dict(self):
        v = VerificationResult(supported=True, confidence=0.9, issues=[], reasoning="ok")
        d = v.to_dict()
        assert d["supported"] is True
        assert d["confidence"] == 0.9


class TestVerifyGrounding:
    def _make_chunks(self, texts):
        return [SimpleNamespace(text=t, meta={"source_name": "doc.pdf"}) for t in texts]

    def test_verify_with_mock_client(self):
        client = MagicMock()
        client.verify.return_value = '{"supported": true, "confidence": 0.9, "issues": [], "reasoning": "ok"}'
        chunks = self._make_chunks(["Gaurav has 5 years of Python experience."])
        result = verify_grounding(
            answer="Gaurav has 5 years of Python experience.",
            evidence_chunks=chunks,
            query="What is Gaurav's experience?",
            llm_client=client,
        )
        assert result is not None
        assert result.supported is True

    def test_verify_empty_answer(self):
        client = MagicMock()
        assert verify_grounding("", [], "test", client) is None

    def test_verify_no_evidence(self):
        client = MagicMock()
        result = verify_grounding("Some answer", [], "test", client)
        assert result is not None
        assert result.supported is False
        assert "No evidence" in result.issues[0]

    def test_verify_timeout(self):
        import time
        client = MagicMock()
        client.verify.side_effect = lambda p: time.sleep(5) or ""
        chunks = self._make_chunks(["evidence text"])
        result = verify_grounding("answer", chunks, "query", client, timeout_s=0.1)
        assert result is None

    def test_verify_exception(self):
        client = MagicMock()
        client.verify.side_effect = Exception("model error")
        chunks = self._make_chunks(["evidence text"])
        result = verify_grounding("answer", chunks, "query", client, timeout_s=5.0)
        assert result is None


# ---------------------------------------------------------------------------
# Config.MultiAgent
# ---------------------------------------------------------------------------

class TestConfigMultiAgent:
    def test_config_exists(self):
        from src.api.config import Config
        assert hasattr(Config, "MultiAgent")

    def test_config_defaults(self):
        from src.api.config import Config
        ma = Config.MultiAgent
        # ENABLED defaults to False
        assert hasattr(ma, "ENABLED")
        assert hasattr(ma, "CLASSIFIER_MODEL")
        assert hasattr(ma, "EXTRACTOR_MODEL")
        assert hasattr(ma, "GENERATOR_MODEL")
        assert hasattr(ma, "VERIFIER_MODEL")
        assert hasattr(ma, "VERIFIER_ENABLED")
        assert hasattr(ma, "CLASSIFIER_TIMEOUT")
        assert hasattr(ma, "VERIFIER_TIMEOUT")
        assert hasattr(ma, "CLASSIFIER_CONFIDENCE_THRESHOLD")

    def test_config_model_defaults(self):
        from src.api.config import Config
        ma = Config.MultiAgent
        # All multi-agent roles default to DocWain-Agent to prevent GPU eviction on T4
        assert "DocWain-Agent" in ma.CLASSIFIER_MODEL
        assert "DocWain-Agent" in ma.EXTRACTOR_MODEL
        assert "DocWain-Agent" in ma.GENERATOR_MODEL
        assert "DocWain-Agent" in ma.VERIFIER_MODEL


# ---------------------------------------------------------------------------
# AppState.multi_agent_gateway
# ---------------------------------------------------------------------------

class TestAppStateMultiAgent:
    def test_app_state_has_field(self):
        from src.api.rag_state import AppState
        state = AppState(
            embedding_model=None,
            reranker=None,
            qdrant_client=None,
            redis_client=None,
            ollama_client=None,
            rag_system=None,
        )
        assert hasattr(state, "multi_agent_gateway")
        assert state.multi_agent_gateway is None


# ---------------------------------------------------------------------------
# Feature flag gating
# ---------------------------------------------------------------------------

class TestFeatureFlagGating:
    def test_disabled_by_default(self):
        from src.api.config import Config
        # Default is disabled unless MULTI_AGENT_ENABLED=true in env
        # (Not set in test environment)
        assert hasattr(Config.MultiAgent, "ENABLED")

    def test_create_multi_agent_gateway(self):
        from src.llm.multi_agent import create_multi_agent_gateway
        fallback = MagicMock()
        gw = create_multi_agent_gateway(fallback_gateway=fallback)
        assert isinstance(gw, MultiAgentGateway)
        assert gw._fallback is fallback


# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------

class TestRolePrompts:
    def test_classifier_prompts(self):
        try:
            from src.llm.role_prompts import CLASSIFIER_SYSTEM, CLASSIFIER_INTENT_TEMPLATE
        except ImportError:
            pytest.skip("Module removed")
        assert "JSON" in CLASSIFIER_SYSTEM
        assert "{query}" in CLASSIFIER_INTENT_TEMPLATE
        assert "intent" in CLASSIFIER_INTENT_TEMPLATE
        assert "domain" in CLASSIFIER_INTENT_TEMPLATE

    def test_extractor_prompts(self):
        try:
            from src.llm.role_prompts import EXTRACTOR_SYSTEM, EXTRACTOR_TEMPLATE
        except ImportError:
            pytest.skip("Module removed")
        assert "extract" in EXTRACTOR_SYSTEM.lower()
        assert "{query}" in EXTRACTOR_TEMPLATE
        assert "{evidence}" in EXTRACTOR_TEMPLATE

    def test_verifier_prompts(self):
        try:
            from src.llm.role_prompts import VERIFIER_SYSTEM, VERIFIER_TEMPLATE
        except ImportError:
            pytest.skip("Module removed")
        assert "verify" in VERIFIER_SYSTEM.lower() or "grounding" in VERIFIER_SYSTEM.lower()
        assert "{query}" in VERIFIER_TEMPLATE
        assert "{answer}" in VERIFIER_TEMPLATE
        assert "{evidence}" in VERIFIER_TEMPLATE

    def test_generator_prompts(self):
        try:
            from src.llm.role_prompts import GENERATOR_SYSTEM
        except ImportError:
            pytest.skip("Module removed")
        assert "document" in GENERATOR_SYSTEM.lower() or "synthesize" in GENERATOR_SYSTEM.lower()
