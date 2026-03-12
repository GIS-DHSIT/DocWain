import asyncio
from src.utils.logging_utils import get_logger
import time
import uuid
from typing import Any, Callable, Dict, Optional

logger = get_logger(__name__)

def generate_correlation_id(value: Optional[str] = None) -> str:
    """Use provided correlation id when available, otherwise create a new UUID."""
    if value:
        return value
    return str(uuid.uuid4())

def standard_response(
    tool_name: str,
    *,
    status: str = "success",
    grounded: bool = True,
    context_found: bool = True,
    result: Optional[Dict[str, Any]] = None,
    sources: Optional[list[Dict[str, Any]]] = None,
    warnings: Optional[list[str]] = None,
    error: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical tool/agent response shape."""
    return {
        "status": status,
        "tool_name": tool_name,
        "agent_name": tool_name,
        "correlation_id": generate_correlation_id(correlation_id),
        "grounded": bool(grounded),
        "context_found": bool(context_found),
        "sources": sources or [],
        "result": result or {},
        "warnings": warnings or [],
        "error": error,
    }

class ToolError(Exception):
    """Structured error used by tool/agent implementations for predictable responses."""

    def __init__(self, message: str, code: str = "tool_error", status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code

    def as_dict(self) -> Dict[str, Any]:
        return {"message": str(self), "code": self.code}

# Agent-primary alias
AgentError = ToolError

async def _maybe_await(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

class ToolRegistry:
    """
    Registry for callable agent/tool backends. Keeps wiring isolated from FastAPI
    routes so agents can also be invoked programmatically (e.g., inside RAG flow).
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}
        self.default_timeout = 45.0

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        if not name:
            raise ValueError("Agent name is required")
        if not callable(handler):
            raise ValueError("Agent handler must be callable")
        self._registry[name] = handler
        logger.info("Registered agent handler: %s", name)

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._registry:
            raise ToolError(f"Agent '{name}' is not registered", code="agent_not_found", status_code=404)
        return self._registry[name]

    async def invoke(
        self,
        name: str,
        payload: Dict[str, Any],
        *,
        correlation_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        cid = generate_correlation_id(correlation_id)
        handler = self.get(name)
        try:
            result = await asyncio.wait_for(
                _maybe_await(handler, payload, correlation_id=cid),
                timeout=timeout or self.default_timeout,
            )
            response = standard_response(
                name,
                status="success",
                result=result.get("result") if isinstance(result, dict) else result,
                sources=result.get("sources") if isinstance(result, dict) else [],
                warnings=result.get("warnings") if isinstance(result, dict) else [],
                grounded=result.get("grounded", True) if isinstance(result, dict) else True,
                context_found=result.get("context_found", True) if isinstance(result, dict) else True,
                correlation_id=cid,
            )
        except asyncio.TimeoutError:
            logger.error("Agent timed out | agent=%s correlation_id=%s", name, cid)
            raise ToolError("Agent execution timed out", code="timeout", status_code=504)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent invocation failed | agent=%s correlation_id=%s", name, cid, exc_info=True)
            raise ToolError(str(exc), code="internal_error", status_code=500) from exc

        logger.info(
            "Agent completed | agent=%s correlation_id=%s latency_ms=%d",
            name,
            cid,
            int((time.time() - start) * 1000),
        )
        return response

# Agent-primary alias
AgentRegistry = ToolRegistry

registry = ToolRegistry()

def register_tool(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to simplify registration of agent handlers."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        registry.register(name, func)
        return func

    return _decorator

# Agent-primary alias
register_agent = register_tool
