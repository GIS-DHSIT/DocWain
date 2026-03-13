"""Domain Dispatcher — detects domain-specific tasks and delegates to
specialized agents within the CoreAgent pipeline.

Sits between UNDERSTAND and RETRIEVE in the pipeline:
    UNDERSTAND → [domain dispatch?]
        YES → DomainAgent.execute(task, context) → format response
        NO  → continue to RETRIEVE → REASON → COMPOSE
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agentic.domain_agents import detect_agent_task, get_domain_agent

logger = logging.getLogger(__name__)


class DomainDispatcher:
    """Detects domain tasks and delegates to specialized agents.

    Returns a formatted response dict on success, or None to signal
    the caller should continue with the standard RAG pipeline.
    """

    def __init__(self, llm_gateway: Any) -> None:
        self._llm = llm_gateway

    def try_handle(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        evidence: List[Dict[str, Any]],
        doc_context: Dict[str, Any],
        *,
        agent_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt domain agent dispatch.

        Returns a response dict (same shape as compose_response) if a domain
        agent handled the query successfully, or None if the query should
        fall through to the standard RAG pipeline.
        """
        detection = self._detect(query, agent_name)
        if detection is None:
            return None

        domain = detection["domain"]
        task_type = detection["task_type"]

        agent = get_domain_agent(domain, llm_client=self._llm)
        if agent is None:
            logger.debug("No agent found for domain=%s", domain)
            return None

        context = self._build_agent_context(query, evidence, doc_context)

        try:
            result = agent.execute(task_type, context)
        except Exception:
            logger.exception("Domain agent %s failed on task %s", domain, task_type)
            return None

        if not result.success or not result.output:
            logger.debug("Domain agent %s returned failure for task %s", domain, task_type)
            return None

        response: Dict[str, Any] = {
            "response": result.output,
            "sources": result.sources or [],
            "grounded": True,
            "context_found": bool(evidence),
            "metadata": {
                "agent": domain,
                "agent_task": task_type,
                "agent_handled": True,
                "engine": "docwain_core_agent",
            },
        }

        if (result.structured_data
                and isinstance(result.structured_data.get("media"), list)):
            response["media"] = result.structured_data["media"]

        return response

    @staticmethod
    def _detect(query: str, agent_name: Optional[str]) -> Optional[Dict[str, str]]:
        """Detect domain and task type."""
        if agent_name:
            detection = detect_agent_task(query, domain=agent_name)
            if detection:
                return detection
            agent = get_domain_agent(agent_name)
            if agent:
                caps = agent.get_capabilities()
                return {"domain": agent_name, "task_type": caps[0] if caps else "default"}
            return None

        return detect_agent_task(query)

    @staticmethod
    def _build_agent_context(
        query: str,
        evidence: List[Dict[str, Any]],
        doc_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build context dict for the domain agent from retrieved evidence."""
        context: Dict[str, Any] = {"query": query}

        if evidence:
            text_parts = [e.get("text", "") for e in evidence if e.get("text")]
            if text_parts:
                context["text"] = "\n\n".join(text_parts)

        return context
