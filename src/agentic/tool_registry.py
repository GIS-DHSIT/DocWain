"""Agent tool registry — generates tool schemas and manages tool invocation for the agentic loop.

Wraps the existing ``ToolRegistry`` from ``src/tools/base.py`` to provide:
- Tool schema generation in function-calling format
- Tool result formatting for agent context
- Tool chaining state management
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = get_logger(__name__)

@dataclass
class ToolSchema:
    """Describes a tool available to the agent."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResult:
    """Result of a tool invocation."""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0

class AgentToolRegistry:
    """Registry of tools available to the agentic reasoning loop.

    Wraps existing registered tool handlers and provides schema generation
    for LLM native tool calling.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}
        self._handlers: Dict[str, Callable] = {}
        self._chain_state: List[ToolResult] = []

    def register(self, name: str, description: str, handler: Callable, parameters: Optional[Dict] = None) -> None:
        """Register a tool with its handler and schema."""
        self._tools[name] = ToolSchema(
            name=name,
            description=description,
            parameters=parameters or {},
        )
        self._handlers[name] = handler

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Generate tool schemas in Ollama function-calling format."""
        schemas = []
        for schema in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters or {"type": "object", "properties": {}},
                },
            })
        return schemas

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """Return a formatted string of all tool descriptions for prompt injection."""
        lines = []
        for schema in self._tools.values():
            lines.append(f"- {schema.name}: {schema.description}")
        return "\n".join(lines)

    async def invoke(self, tool_name: str, tool_input: Any) -> ToolResult:
        """Invoke a registered tool and return the result."""
        handler = self._handlers.get(tool_name)
        if not handler:
            result = ToolResult(tool_name=tool_name, success=False, error=f"Unknown tool: {tool_name}")
            self._chain_state.append(result)
            return result

        start = time.time()
        try:
            import asyncio
            if asyncio.iscoroutinefunction(handler):
                output = await handler(tool_input)
            else:
                output = handler(tool_input)
            elapsed = (time.time() - start) * 1000
            result = ToolResult(tool_name=tool_name, success=True, output=output, elapsed_ms=elapsed)
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            logger.warning("Tool %s failed: %s", tool_name, exc)
            result = ToolResult(tool_name=tool_name, success=False, error=str(exc), elapsed_ms=elapsed)

        self._chain_state.append(result)
        return result

    def get_chain_state(self) -> List[ToolResult]:
        """Return the history of tool invocations in this chain."""
        return list(self._chain_state)

    def reset_chain(self) -> None:
        """Clear the chain state for a new reasoning session."""
        self._chain_state.clear()

    @classmethod
    def from_existing_registry(cls) -> "AgentToolRegistry":
        """Build an AgentToolRegistry from the existing ToolRegistry in src/tools/base.py.

        Discovers all registered tool handlers and wraps them.
        """
        registry = cls()
        try:
            from src.tools.base import ToolRegistry as BaseRegistry
            base = BaseRegistry()
            for name, handler in base._handlers.items():
                doc = getattr(handler, "__doc__", "") or f"Execute the {name} tool"
                registry.register(name, doc.strip().split("\n")[0], handler)
        except Exception as exc:
            logger.warning("Could not import existing ToolRegistry: %s", exc)
        return registry

__all__ = ["AgentToolRegistry", "ToolSchema", "ToolResult"]
