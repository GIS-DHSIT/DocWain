"""Intelligence Engine orchestrator — routes queries through the full intelligence pipeline.

This is the central orchestrator (Task 9) that ties together all components:
  1. Conversation Graph (pronoun resolution, context entities)
  2. Query Router (route classification)
  3. Query Analyzer (geometry derivation)
  4. Graph Adapter (direct entity lookups)
  5. Evidence Organizer (chunk + fact restructuring)
  6. Rendering Spec Generator (output format derivation)
  7. Constrained Prompter (LLM prompt construction)
  8. Quality Engine (output validation)
  9. Response Assembler (structured response assembly)
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .query_router import route_query, QueryAnalysis, QueryRoute
from .query_analyzer import analyze_query, QueryGeometry
from .evidence_organizer import organize_evidence, OrganizedEvidence
from .rendering_spec import generate_spec, RenderingSpec
from .constrained_prompter import build_prompt, ConstrainedPrompt
from .quality_engine import validate_output, QualityResult
from .conversation_graph import ConversationGraph
from .graph_adapter import get_graph_adapter, CypherGraphAdapter
from .response_assembler import assemble_response, AssembledResponse

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class IntelligentResponse(BaseModel):
    """Final response from the intelligence pipeline."""

    text: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    route_used: str = ""
    query_resolved: str = ""
    geometry: Optional[QueryGeometry] = None
    spec: Optional[RenderingSpec] = None
    quality: Optional[QualityResult] = None
    prompt: Optional[ConstrainedPrompt] = None
    needs_llm: bool = False
    stage_timings: Dict[str, float] = Field(default_factory=dict)

# ---------------------------------------------------------------------------
# IntelligenceEngine
# ---------------------------------------------------------------------------

class IntelligenceEngine:
    """Central orchestrator for the full intelligence pipeline."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationGraph] = {}
        self._sessions_lock = threading.Lock()
        self._graph_adapter: Optional[CypherGraphAdapter] = None
        self._graph_adapter_loaded = False
        self._graph_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str) -> ConversationGraph:
        """Get or create a conversation graph for the session. Thread-safe."""
        with self._sessions_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationGraph(session_id=session_id)
            return self._sessions[session_id]

    # ------------------------------------------------------------------
    # Graph adapter (lazy)
    # ------------------------------------------------------------------

    def _get_graph(self) -> Optional[CypherGraphAdapter]:
        """Lazy-load graph adapter. Returns None if unavailable."""
        if self._graph_adapter_loaded:
            return self._graph_adapter
        with self._graph_lock:
            if self._graph_adapter_loaded:
                return self._graph_adapter
            try:
                self._graph_adapter = get_graph_adapter()
            except Exception:
                logger.debug("Graph adapter unavailable", exc_info=True)
                self._graph_adapter = None
            self._graph_adapter_loaded = True
            return self._graph_adapter

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        session_id: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
        llm_response: Optional[str] = None,
    ) -> IntelligentResponse:
        """Process a query through the full intelligence pipeline.

        Parameters
        ----------
        query : str
            The user's raw query.
        subscription_id, profile_id : str
            Tenant scoping identifiers.
        session_id : str
            Conversation session identifier.
        chunks : list, optional
            Pre-retrieved vector search chunks.
        facts : list, optional
            Pre-retrieved knowledge graph facts.
        llm_response : str, optional
            If the caller already ran the LLM, pass the raw response here
            so the quality engine can validate it.

        Returns
        -------
        IntelligentResponse
            Contains the final text (or a prompt for the caller to send to an LLM
            when ``needs_llm`` is True).
        """
        chunks = chunks or []
        facts = facts or []
        timings: Dict[str, float] = {}

        # --- Stage 1: Conversation context + pronoun resolution ---
        t0 = time.monotonic()
        session = self._get_session(session_id)
        resolved_query = session.resolve_query(query)
        context_entities = session.get_context_entities()
        timings["conversation_resolve"] = round(time.monotonic() - t0, 4)

        # --- Stage 2: Query routing ---
        t0 = time.monotonic()
        analysis: QueryAnalysis = route_query(resolved_query)
        timings["query_route"] = round(time.monotonic() - t0, 4)

        # --- Stage 3: Query geometry ---
        t0 = time.monotonic()
        geometry: QueryGeometry = analyze_query(resolved_query, analysis)
        timings["query_analyze"] = round(time.monotonic() - t0, 4)

        # --- Stage 4: Route decision ---
        t0 = time.monotonic()
        if analysis.route == QueryRoute.GRAPH_DIRECT:
            graph = self._get_graph()
            if graph is not None:
                try:
                    graph_facts = self._query_graph_for_entities(
                        graph, analysis.entities, subscription_id, profile_id,
                    )
                    if graph_facts:
                        # Assemble structured response directly — no LLM needed
                        assembled = assemble_response(
                            query=resolved_query,
                            route=analysis.route,
                            facts=graph_facts,
                            chunks=chunks,
                            is_comparison=analysis.is_comparison,
                            is_aggregation=analysis.is_aggregation,
                        )
                        timings["graph_direct"] = round(time.monotonic() - t0, 4)

                        # Record turn
                        self._record_turn(
                            session, resolved_query, analysis.entities,
                            graph_facts, assembled.text,
                        )

                        return IntelligentResponse(
                            text=assembled.text,
                            sources=assembled.sources,
                            confidence=assembled.confidence,
                            route_used=analysis.route.value,
                            query_resolved=resolved_query,
                            geometry=geometry,
                            spec=None,
                            quality=None,
                            prompt=None,
                            needs_llm=False,
                            stage_timings=timings,
                        )
                except Exception:
                    logger.debug("Graph query failed, falling through", exc_info=True)
            # Fall through to evidence-based path
        timings["route_decision"] = round(time.monotonic() - t0, 4)

        # --- Stage 5: Evidence organization ---
        t0 = time.monotonic()
        evidence: OrganizedEvidence = organize_evidence(
            chunks=chunks,
            facts=facts,
            query_entities=analysis.entities,
        )
        timings["evidence_organize"] = round(time.monotonic() - t0, 4)

        # --- Stage 6: Rendering spec ---
        t0 = time.monotonic()
        spec: RenderingSpec = generate_spec(geometry, evidence)
        timings["rendering_spec"] = round(time.monotonic() - t0, 4)

        # --- Stage 7: Constrained prompt ---
        t0 = time.monotonic()
        prompt: ConstrainedPrompt = build_prompt(spec, evidence, resolved_query)
        timings["build_prompt"] = round(time.monotonic() - t0, 4)

        # --- Stage 8: LLM response handling ---
        if llm_response is not None:
            t0 = time.monotonic()
            quality: QualityResult = validate_output(llm_response, spec, evidence)
            timings["quality_validate"] = round(time.monotonic() - t0, 4)

            # Record turn
            self._record_turn(
                session, resolved_query, analysis.entities,
                facts, quality.cleaned_text,
            )

            return IntelligentResponse(
                text=quality.cleaned_text,
                sources=self._extract_sources(evidence),
                confidence=self._compute_confidence(quality, evidence),
                route_used=analysis.route.value,
                query_resolved=resolved_query,
                geometry=geometry,
                spec=spec,
                quality=quality,
                prompt=prompt,
                needs_llm=False,
                stage_timings=timings,
            )

        # No LLM response yet — caller must invoke LLM
        return IntelligentResponse(
            text="",
            sources=[],
            confidence=0.0,
            route_used=analysis.route.value,
            query_resolved=resolved_query,
            geometry=geometry,
            spec=spec,
            quality=None,
            prompt=prompt,
            needs_llm=True,
            stage_timings=timings,
        )

    # ------------------------------------------------------------------
    # Finalize (after caller runs LLM)
    # ------------------------------------------------------------------

    def finalize(
        self,
        session_id: str,
        query: str,
        llm_response: str,
        spec: RenderingSpec,
        evidence: OrganizedEvidence,
    ) -> IntelligentResponse:
        """Finalize the response after the caller invokes the LLM.

        Parameters
        ----------
        session_id : str
            Session for recording the turn.
        query : str
            The resolved query (as returned in the initial IntelligentResponse).
        llm_response : str
            The raw LLM output.
        spec : RenderingSpec
            The spec from the initial response.
        evidence : OrganizedEvidence
            The organized evidence from the initial response.

        Returns
        -------
        IntelligentResponse
            Final validated response.
        """
        timings: Dict[str, float] = {}

        # Validate
        t0 = time.monotonic()
        quality = validate_output(llm_response, spec, evidence)
        timings["quality_validate"] = round(time.monotonic() - t0, 4)

        # Record turn
        t0 = time.monotonic()
        session = self._get_session(session_id)
        # Extract entity names from evidence groups
        entities = [
            grp.entity_text
            for grp in evidence.entity_groups
            if grp.entity_text
        ]
        self._record_turn(session, query, entities, [], quality.cleaned_text)
        timings["record_turn"] = round(time.monotonic() - t0, 4)

        return IntelligentResponse(
            text=quality.cleaned_text,
            sources=self._extract_sources(evidence),
            confidence=self._compute_confidence(quality, evidence),
            route_used="finalized",
            query_resolved=query,
            geometry=None,
            spec=spec,
            quality=quality,
            prompt=None,
            needs_llm=False,
            stage_timings=timings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_graph_for_entities(
        graph: CypherGraphAdapter,
        entities: List[str],
        subscription_id: str,
        profile_id: str,
    ) -> List[Dict[str, Any]]:
        """Query the graph adapter for facts about the given entities."""
        all_facts: List[Dict[str, Any]] = []
        for entity_text in entities:
            rows = graph.get_entity_facts(
                entity_text=entity_text,
                subscription_id=subscription_id,
                profile_id=profile_id,
            )
            for row in rows:
                target_props = row.get("target_props", {})
                all_facts.append({
                    "subject": entity_text,
                    "predicate": target_props.get("predicate", "related_to"),
                    "value": target_props.get("value", str(target_props)),
                    "source": target_props.get("source_document", "graph"),
                    "confidence": target_props.get("confidence", 0.8),
                })
        return all_facts

    @staticmethod
    def _record_turn(
        session: ConversationGraph,
        query: str,
        entities: List[str],
        facts: List[Any],
        response_text: str,
    ) -> None:
        """Record a conversation turn in the session graph."""
        fact_strs = []
        for f in facts:
            if isinstance(f, dict):
                subj = f.get("subject", "")
                pred = f.get("predicate", "")
                val = f.get("value", "")
                fact_strs.append(f"{subj}:{pred}={val}")
            else:
                fact_strs.append(str(f))

        session.add_turn(
            query=query,
            entities=entities,
            entity_labels={},
            facts_disclosed=fact_strs,
            response_text=response_text,
        )

    @staticmethod
    def _extract_sources(evidence: OrganizedEvidence) -> List[Dict[str, Any]]:
        """Extract source citations from evidence provenance."""
        sources: List[Dict[str, Any]] = []
        seen: set = set()
        for prov in evidence.provenance:
            key = (prov.source_document, prov.page)
            if key not in seen:
                seen.add(key)
                entry: Dict[str, Any] = {"source": prov.source_document}
                if prov.page is not None:
                    entry["page"] = prov.page
                sources.append(entry)
        return sources

    @staticmethod
    def _compute_confidence(
        quality: QualityResult,
        evidence: OrganizedEvidence,
    ) -> float:
        """Compute overall confidence from quality metrics and evidence coverage."""
        # Weighted combination of structural conformance and content integrity
        base = 0.5 * quality.structural_conformance + 0.5 * quality.content_integrity

        # Slight boost if we have evidence
        if evidence.total_chunks > 0 or evidence.total_facts > 0:
            base = min(1.0, base + 0.05)

        # Penalize if there are gaps
        if evidence.gaps:
            base = max(0.0, base - 0.1 * len(evidence.gaps))

        return round(max(0.0, min(1.0, base)), 4)

__all__ = [
    "IntelligenceEngine",
    "IntelligentResponse",
]
