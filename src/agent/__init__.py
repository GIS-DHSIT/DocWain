"""DocWain Core Agent — intelligent document analysis orchestration."""


def __getattr__(name: str):
    """Lazy imports to avoid heavy import chains at module load time."""
    if name == "AgentOrchestrator":
        from src.agent.orchestrator import AgentOrchestrator
        return AgentOrchestrator
    if name == "CoreAgent":
        from src.agent.core_agent import CoreAgent
        return CoreAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentOrchestrator", "CoreAgent"]
