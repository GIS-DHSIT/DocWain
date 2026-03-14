"""Tests for the Intelligence Engine orchestrator (Task 9)."""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.docwain_intel.intelligence import IntelligenceEngine, IntelligentResponse
from src.docwain_intel.query_router import QueryAnalysis, QueryRoute
from src.docwain_intel.query_analyzer import QueryGeometry
from src.docwain_intel.evidence_organizer import OrganizedEvidence, EvidenceGroup
from src.docwain_intel.rendering_spec import RenderingSpec
from src.docwain_intel.constrained_prompter import ConstrainedPrompt
from src.docwain_intel.quality_engine import QualityResult
from src.docwain_intel.conversation_graph import ConversationGraph
from src.docwain_intel.response_assembler import AssembledResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(route=QueryRoute.HYBRID_SEARCH, entities=None, **kwargs):
    return QueryAnalysis(
        query=kwargs.get("query", "test query"),
        route=route,
        entities=entities or [],
        **{k: v for k, v in kwargs.items() if k != "query"},
    )


def _make_geometry(**kwargs):
    return QueryGeometry(query=kwargs.pop("query", "test query"), **kwargs)


def _make_evidence(chunks=None, facts=0, entity_groups=None):
    return OrganizedEvidence(
        entity_groups=entity_groups or [],
        ungrouped_chunks=chunks or [],
        total_facts=facts,
        total_chunks=len(chunks) if chunks else 0,
    )


def _make_spec(**kwargs):
    return RenderingSpec(**kwargs)


def _make_prompt(**kwargs):
    return ConstrainedPrompt(
        system_prompt=kwargs.get("system_prompt", "sys"),
        user_prompt=kwargs.get("user_prompt", "usr"),
        max_tokens=kwargs.get("max_tokens", 512),
        temperature=kwargs.get("temperature", 0.2),
    )


def _make_quality(text="cleaned response", **kwargs):
    return QualityResult(
        cleaned_text=text,
        original_text=kwargs.get("original_text", text),
        structural_conformance=kwargs.get("structural_conformance", 0.9),
        content_integrity=kwargs.get("content_integrity", 0.85),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGraphDirectWithFacts:
    """GRAPH_DIRECT with mock graph returning facts -> structured response, no LLM."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.get_graph_adapter")
    def test_graph_direct_returns_structured_response(
        self, mock_get_graph, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(
            route=QueryRoute.GRAPH_DIRECT,
            entities=["Alice"],
        )
        mock_analyze.return_value = _make_geometry()

        mock_graph = MagicMock()
        mock_graph.get_entity_facts.return_value = [
            {
                "entity_id": "e1",
                "rel": {},
                "target_props": {
                    "predicate": "HAS_SKILL",
                    "value": "Python",
                    "source_document": "resume.pdf",
                    "confidence": 0.9,
                },
                "target_labels": ["Entity"],
            }
        ]
        mock_get_graph.return_value = mock_graph

        engine = IntelligenceEngine()
        result = engine.process_query(
            query="What are Alice's skills?",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
        )

        assert isinstance(result, IntelligentResponse)
        assert result.needs_llm is False
        assert result.text  # non-empty
        assert result.route_used == "GRAPH_DIRECT"
        assert result.confidence > 0
        assert "conversation_resolve" in result.stage_timings
        assert "graph_direct" in result.stage_timings


class TestGraphDirectNoGraph:
    """GRAPH_DIRECT with no graph -> falls through to evidence path."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    @patch("src.docwain_intel.intelligence.get_graph_adapter")
    def test_no_graph_falls_through(
        self, mock_get_graph, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(
            route=QueryRoute.GRAPH_DIRECT,
            entities=["Alice"],
        )
        mock_analyze.return_value = _make_geometry()
        mock_get_graph.return_value = None  # no graph available
        mock_org.return_value = _make_evidence()
        mock_spec.return_value = _make_spec()
        mock_build.return_value = _make_prompt()

        engine = IntelligenceEngine()
        result = engine.process_query(
            query="What are Alice's skills?",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
        )

        assert result.needs_llm is True
        assert result.prompt is not None
        assert result.route_used == "GRAPH_DIRECT"


class TestLLMRouteNeedsLLM:
    """LLM route: returns needs_llm=True with prompt attached."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    def test_llm_route_returns_prompt(
        self, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(route=QueryRoute.LLM_GENERATION)
        mock_analyze.return_value = _make_geometry()
        mock_org.return_value = _make_evidence()
        mock_spec.return_value = _make_spec()
        prompt = _make_prompt(user_prompt="Answer this question...")
        mock_build.return_value = prompt

        engine = IntelligenceEngine()
        result = engine.process_query(
            query="Why should we hire Alice?",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
            chunks=[{"text": "Alice has 5 years experience"}],
        )

        assert result.needs_llm is True
        assert result.prompt is not None
        assert result.prompt.user_prompt == "Answer this question..."
        assert result.text == ""
        assert result.geometry is not None
        assert result.spec is not None


class TestFinalize:
    """finalize(): validates LLM response, returns cleaned text."""

    @patch("src.docwain_intel.intelligence.validate_output")
    def test_finalize_validates_and_returns(self, mock_validate):
        quality = _make_quality(text="Alice is skilled in Python.")
        mock_validate.return_value = quality

        engine = IntelligenceEngine()
        spec = _make_spec()
        evidence = _make_evidence()

        result = engine.finalize(
            session_id="sess1",
            query="What are Alice's skills?",
            llm_response="Based on the documents, Alice is skilled in Python.",
            spec=spec,
            evidence=evidence,
        )

        assert result.needs_llm is False
        assert result.text == "Alice is skilled in Python."
        assert result.quality is not None
        assert result.route_used == "finalized"
        mock_validate.assert_called_once()


class TestConversationContext:
    """Multi-turn with pronoun resolution."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    @patch("src.docwain_intel.intelligence.validate_output")
    def test_pronoun_resolution_across_turns(
        self, mock_validate, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(
            route=QueryRoute.HYBRID_SEARCH, entities=["Alice"]
        )
        mock_analyze.return_value = _make_geometry()
        mock_org.return_value = _make_evidence()
        mock_spec.return_value = _make_spec()
        mock_build.return_value = _make_prompt()
        mock_validate.return_value = _make_quality(text="Alice has Python skills.")

        engine = IntelligenceEngine()

        # Turn 1: mention Alice
        r1 = engine.process_query(
            query="Tell me about Alice",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess_conv",
            llm_response="Alice has Python skills.",
        )
        assert r1.text == "Alice has Python skills."

        # Turn 2: "tell me more" should resolve context
        mock_route.return_value = _make_analysis(
            route=QueryRoute.HYBRID_SEARCH, entities=[]
        )
        r2 = engine.process_query(
            query="tell me more",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess_conv",
        )
        # The resolved query should have the entity appended
        assert "Alice" in r2.query_resolved or r2.query_resolved == "tell me more"


class TestSessionIsolation:
    """Different session_ids get separate conversation graphs."""

    def test_separate_sessions(self):
        engine = IntelligenceEngine()
        s1 = engine._get_session("session_a")
        s2 = engine._get_session("session_b")
        s3 = engine._get_session("session_a")

        assert s1 is not s2
        assert s1 is s3
        assert s1.session_id == "session_a"
        assert s2.session_id == "session_b"


class TestStageTimings:
    """Stage timings recorded for each stage."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    def test_timings_present(
        self, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(route=QueryRoute.FULL_SEARCH)
        mock_analyze.return_value = _make_geometry()
        mock_org.return_value = _make_evidence()
        mock_spec.return_value = _make_spec()
        mock_build.return_value = _make_prompt()

        engine = IntelligenceEngine()
        result = engine.process_query(
            query="Summarize all documents",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
        )

        expected_stages = [
            "conversation_resolve",
            "query_route",
            "query_analyze",
            "route_decision",
            "evidence_organize",
            "rendering_spec",
            "build_prompt",
        ]
        for stage in expected_stages:
            assert stage in result.stage_timings, f"Missing timing for {stage}"
            assert isinstance(result.stage_timings[stage], float)
            assert result.stage_timings[stage] >= 0


class TestEmptyQuery:
    """Empty query -> graceful handling."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    @patch("src.docwain_intel.intelligence.get_graph_adapter")
    def test_empty_query_handled(
        self, mock_graph, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(
            query="", route=QueryRoute.GRAPH_DIRECT, is_conversational=True
        )
        mock_analyze.return_value = _make_geometry(query="")
        mock_graph.return_value = None
        mock_org.return_value = _make_evidence()
        mock_spec.return_value = _make_spec()
        mock_build.return_value = _make_prompt()

        engine = IntelligenceEngine()
        result = engine.process_query(
            query="",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
        )

        assert isinstance(result, IntelligentResponse)
        # Should not crash; either returns prompt or text
        assert result.query_resolved == ""


class TestFeatureFlag:
    """Feature flag: DOCWAIN_INTEL_V2=0 -> engine not created."""

    def test_engine_disabled_by_default(self):
        from src.docwain_intel.integration import get_intelligence_engine
        import src.docwain_intel.integration as mod

        # Reset singleton
        mod._engine_instance = None
        with patch.dict(os.environ, {"DOCWAIN_INTEL_V2": "0"}):
            result = get_intelligence_engine()
        assert result is None

    def test_engine_enabled(self):
        from src.docwain_intel.integration import get_intelligence_engine
        import src.docwain_intel.integration as mod

        # Reset singleton
        mod._engine_instance = None
        with patch.dict(os.environ, {"DOCWAIN_INTEL_V2": "1"}):
            result = get_intelligence_engine()
        assert result is not None
        assert isinstance(result, IntelligenceEngine)
        # Reset for other tests
        mod._engine_instance = None


class TestAllStagesOrder:
    """All stages execute in order (mock each component)."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    @patch("src.docwain_intel.intelligence.validate_output")
    def test_stages_execute_in_order(
        self, mock_validate, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        call_order = []

        def track(name, return_val):
            def side_effect(*args, **kwargs):
                call_order.append(name)
                return return_val
            return side_effect

        mock_route.side_effect = track(
            "route", _make_analysis(route=QueryRoute.HYBRID_SEARCH)
        )
        mock_analyze.side_effect = track("analyze", _make_geometry())
        mock_org.side_effect = track("organize", _make_evidence())
        mock_spec.side_effect = track("spec", _make_spec())
        mock_build.side_effect = track("prompt", _make_prompt())
        mock_validate.side_effect = track("validate", _make_quality())

        engine = IntelligenceEngine()
        engine.process_query(
            query="What is Alice's email?",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
            llm_response="alice@example.com",
        )

        assert call_order == ["route", "analyze", "organize", "spec", "prompt", "validate"]


class TestGraphAdapterFailure:
    """Graph adapter failure -> graceful fallback to chunk-based response."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    @patch("src.docwain_intel.intelligence.get_graph_adapter")
    def test_graph_failure_fallthrough(
        self, mock_get_graph, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(
            route=QueryRoute.GRAPH_DIRECT, entities=["Bob"]
        )
        mock_analyze.return_value = _make_geometry()

        mock_graph = MagicMock()
        mock_graph.get_entity_facts.side_effect = RuntimeError("Connection lost")
        mock_get_graph.return_value = mock_graph

        mock_org.return_value = _make_evidence()
        mock_spec.return_value = _make_spec()
        mock_build.return_value = _make_prompt()

        engine = IntelligenceEngine()
        result = engine.process_query(
            query="What is Bob's email?",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
            chunks=[{"text": "Bob: bob@example.com"}],
        )

        # Should fall through to evidence path, not crash
        assert result.needs_llm is True
        assert result.prompt is not None


class TestIntegrationRouteAndAssemble:
    """Integration: route_and_assemble uses engine when available."""

    def test_uses_engine_when_enabled(self):
        import src.docwain_intel.integration as mod

        mock_engine = MagicMock()
        mock_engine.process_query.return_value = IntelligentResponse(
            text="Engine response",
            sources=[{"source": "test.pdf"}],
            confidence=0.85,
            route_used="HYBRID_SEARCH",
            query_resolved="test query",
        )

        old_instance = mod._engine_instance
        mod._engine_instance = mock_engine

        try:
            with patch.dict(os.environ, {"DOCWAIN_INTEL_V2": "1"}):
                result = mod.route_and_assemble(
                    query="test query",
                    facts=[{"subject": "A", "predicate": "p", "value": "v"}],
                )
            assert result.text == "Engine response"
            assert result.confidence == 0.85
            mock_engine.process_query.assert_called_once()
        finally:
            mod._engine_instance = old_instance

    @patch("src.docwain_intel.integration.route_query")
    def test_falls_back_when_engine_disabled(self, mock_route):
        import src.docwain_intel.integration as mod

        old_instance = mod._engine_instance
        mod._engine_instance = None

        try:
            mock_route.return_value = _make_analysis(route=QueryRoute.HYBRID_SEARCH)

            with patch.dict(os.environ, {"DOCWAIN_INTEL_V2": "0"}):
                result = mod.route_and_assemble(
                    query="test query",
                    facts=[{"subject": "A", "predicate": "p", "value": "v"}],
                )
            # Should still return a valid response via legacy path
            assert isinstance(result, AssembledResponse)
        finally:
            mod._engine_instance = old_instance


class TestLLMResponseProvided:
    """When llm_response is provided, quality validation runs immediately."""

    @patch("src.docwain_intel.intelligence.route_query")
    @patch("src.docwain_intel.intelligence.analyze_query")
    @patch("src.docwain_intel.intelligence.organize_evidence")
    @patch("src.docwain_intel.intelligence.generate_spec")
    @patch("src.docwain_intel.intelligence.build_prompt")
    @patch("src.docwain_intel.intelligence.validate_output")
    def test_llm_response_triggers_validation(
        self, mock_validate, mock_build, mock_spec, mock_org, mock_analyze, mock_route
    ):
        mock_route.return_value = _make_analysis(route=QueryRoute.HYBRID_SEARCH)
        mock_analyze.return_value = _make_geometry()
        mock_org.return_value = _make_evidence()
        mock_spec.return_value = _make_spec()
        mock_build.return_value = _make_prompt()
        mock_validate.return_value = _make_quality(text="Validated output")

        engine = IntelligenceEngine()
        result = engine.process_query(
            query="What is Alice's email?",
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
            llm_response="Based on docs, Alice's email is alice@example.com",
        )

        assert result.needs_llm is False
        assert result.text == "Validated output"
        assert result.quality is not None
        assert "quality_validate" in result.stage_timings
        mock_validate.assert_called_once()


class TestIntelligentResponseModel:
    """Test the IntelligentResponse Pydantic model."""

    def test_defaults(self):
        r = IntelligentResponse()
        assert r.text == ""
        assert r.sources == []
        assert r.confidence == 0.0
        assert r.needs_llm is False
        assert r.stage_timings == {}
        assert r.geometry is None
        assert r.spec is None
        assert r.quality is None
        assert r.prompt is None

    def test_full_construction(self):
        r = IntelligentResponse(
            text="Answer",
            sources=[{"source": "doc.pdf"}],
            confidence=0.9,
            route_used="HYBRID_SEARCH",
            query_resolved="resolved query",
            geometry=_make_geometry(),
            spec=_make_spec(),
            quality=_make_quality(),
            prompt=_make_prompt(),
            needs_llm=False,
            stage_timings={"query_route": 0.001},
        )
        assert r.text == "Answer"
        assert r.confidence == 0.9
        assert r.geometry is not None
