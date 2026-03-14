"""ReAct-style agentic reasoning loop.

Implements: Thought -> Action -> Observation -> ... -> Final Answer

The agent uses an LLM to decide which tool to call next (or to produce a final
answer), executes the tool, feeds the observation back, and repeats until
either a final answer is produced or the step limit is reached.

Feature-flag gated: only activates when ``Config.Execution.ALLOW_AGENT_MODE``
is True AND the request specifies ``agent_mode=True``.
"""
from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

@dataclass
class AgentThought:
    """A single reasoning step."""
    reasoning: str = ""
    action: Optional[str] = None        # Tool name to call, or None if final answer
    action_input: Any = None            # Input to the tool
    final_answer: Optional[str] = None  # Set when agent decides to answer

@dataclass
class AgentStep:
    """One complete step in the reasoning loop."""
    step_number: int
    thought: AgentThought
    observation: Optional[str] = None
    elapsed_ms: float = 0.0

@dataclass
class AgentResult:
    """Final result of the agent loop."""
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    tool_calls_made: int = 0
    total_elapsed_ms: float = 0.0
    completed: bool = True

_AGENT_SYSTEM_PROMPT = """You are a reasoning agent that solves queries step by step.

Available tools:
{tool_descriptions}

For each step, respond in this exact JSON format:
{{
    "reasoning": "Your chain of thought about what to do next",
    "action": "tool_name_or_null",
    "action_input": "input for the tool, or null",
    "final_answer": "Your final answer if you're done, or null"
}}

Rules:
- Set "action" to a tool name to use a tool, OR set "final_answer" to provide the answer
- Never set both "action" and "final_answer" at the same time
- Use the information gathered from tools to build your final answer
- If no tool is needed, provide final_answer directly
- Be concise and factual in your final answer
"""

class AgentLoop:
    """ReAct-style reasoning loop with tool chaining."""

    def __init__(
        self,
        llm_client: Any,
        tool_registry: Any,
        *,
        max_steps: int = 5,
        timeout: float = 30.0,
        thinking_client: Any = None,
    ):
        self._llm = llm_client
        self._thinking = thinking_client  # lfm2.5-thinking for fast reasoning steps
        self._tools = tool_registry
        self._max_steps = min(max_steps, 10)  # Hard cap at 10
        self._timeout = timeout

    async def run(
        self,
        query: str,
        context: str = "",
        available_tools: Optional[List[str]] = None,
    ) -> AgentResult:
        """Execute the reasoning loop.

        Args:
            query: The user's question
            context: Retrieved document context (if any)
            available_tools: Subset of tool names to make available (None = all)

        Returns:
            AgentResult with the final answer and step history
        """
        start_time = time.time()
        tool_descriptions = self._tools.get_tool_descriptions()
        self._tools.reset_chain()

        system = _AGENT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
        history: List[AgentStep] = []

        for step_num in range(1, self._max_steps + 1):
            elapsed = (time.time() - start_time) * 1000
            if elapsed > self._timeout * 1000:
                logger.warning("Agent loop timeout after %.1fms", elapsed)
                return self._synthesize(history, start_time, completed=False)

            # Build prompt with history
            prompt = self._build_prompt(system, query, context, history)

            # Think
            step_start = time.time()
            thought = self._think(prompt)
            step_elapsed = (time.time() - step_start) * 1000

            step = AgentStep(
                step_number=step_num,
                thought=thought,
                elapsed_ms=step_elapsed,
            )

            # Check for final answer
            if thought.final_answer:
                history.append(step)
                return AgentResult(
                    answer=thought.final_answer,
                    steps=history,
                    tool_calls_made=sum(1 for s in history if s.thought.action),
                    total_elapsed_ms=(time.time() - start_time) * 1000,
                    completed=True,
                )

            # Execute tool
            if thought.action:
                if available_tools and thought.action not in available_tools:
                    step.observation = f"Tool '{thought.action}' is not available."
                else:
                    result = await self._tools.invoke(thought.action, thought.action_input)
                    if result.success:
                        obs = str(result.output) if result.output else "Tool executed successfully."
                        step.observation = obs[:2000]  # Cap observation length
                    else:
                        step.observation = f"Tool error: {result.error}"

            history.append(step)

        # Max steps reached — synthesize from history
        return self._synthesize(history, start_time, completed=False)

    def _think(self, prompt: str) -> AgentThought:
        """Ask the LLM to produce a reasoning step.

        Uses lfm2.5-thinking (fast, 731MB) when available for MoE routing,
        falls back to DocWain-Agent (the base model) if unavailable.
        """
        client = self._thinking if self._thinking is not None else self._llm
        try:
            from src.llm.task_router import TaskType, task_scope
            with task_scope(TaskType.AGENT_REASONING):
                text, _ = client.generate_with_metadata(
                    prompt,
                    options={"temperature": 0.05, "num_predict": 512, "num_ctx": 8192},
                )
            return self._parse_thought(text)
        except Exception as exc:
            # Fallback to base model if thinking client failed
            if client is not self._llm:
                logger.debug("Thinking model failed, falling back to base model: %s", exc)
                try:
                    text, _ = self._llm.generate_with_metadata(
                        prompt,
                        options={"temperature": 0.1, "num_predict": 512, "num_ctx": 8192},
                    )
                    return self._parse_thought(text)
                except Exception as exc2:
                    exc = exc2
            logger.warning("Agent think step failed: %s", exc)
            return AgentThought(
                reasoning=f"Error during reasoning: {exc}",
                final_answer="I encountered an error while reasoning. Please try again.",
            )

    def _parse_thought(self, text: str) -> AgentThought:
        """Parse LLM output into an AgentThought."""
        # Try JSON parse first
        try:
            # Find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return AgentThought(
                    reasoning=data.get("reasoning", ""),
                    action=data.get("action") if data.get("action") else None,
                    action_input=data.get("action_input"),
                    final_answer=data.get("final_answer") if data.get("final_answer") else None,
                )
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: treat entire response as final answer
        return AgentThought(reasoning="", final_answer=text.strip() or "Unable to generate a response.")

    def _build_prompt(
        self,
        system: str,
        query: str,
        context: str,
        history: List[AgentStep],
    ) -> str:
        """Build the full prompt including system, context, query, and history."""
        parts = [system]
        if context:
            parts.append(f"\nDocument context:\n{context[:4000]}")
        parts.append(f"\nUser query: {query}")

        if history:
            parts.append("\nPrevious steps:")
            for step in history:
                parts.append(f"\nStep {step.step_number}:")
                parts.append(f"  Reasoning: {step.thought.reasoning}")
                if step.thought.action:
                    parts.append(f"  Action: {step.thought.action}({step.thought.action_input})")
                if step.observation:
                    parts.append(f"  Observation: {step.observation[:500]}")

        parts.append("\nProvide your next step as JSON:")
        return "\n".join(parts)

    def _synthesize(self, history: List[AgentStep], start_time: float, completed: bool) -> AgentResult:
        """Synthesize a final answer from the step history when max steps reached."""
        observations = [s.observation for s in history if s.observation]
        if observations:
            answer = "Based on my analysis:\n" + "\n".join(f"- {obs[:200]}" for obs in observations[-3:])
        else:
            answer = "I was unable to find enough information to answer your question."

        return AgentResult(
            answer=answer,
            steps=history,
            tool_calls_made=sum(1 for s in history if s.thought.action),
            total_elapsed_ms=(time.time() - start_time) * 1000,
            completed=completed,
        )

__all__ = ["AgentLoop", "AgentResult", "AgentStep", "AgentThought"]
