"""Centralized LLM gateway — all LLM calls route through this package.

Usage:
    from src.llm import create_llm_gateway, LLMGateway
    gateway = create_llm_gateway()
    text = gateway.generate("What is 2+2?")

    # Multi-agent mode (specialized models per pipeline role):
    from src.llm.multi_agent import MultiAgentGateway, AgentRole, create_multi_agent_gateway
    ma = create_multi_agent_gateway(fallback_gateway=gateway)
    ma.classify("What is 2+2?")         # llama3.2 (fast)
    ma.extract("Extract skills from...")  # mistral (precise)
    ma.generate("Synthesize answer...")   # DocWain-Agent (powerful)
    ma.verify("Check grounding...")       # deepseek-r1 (CoT)
"""
from src.llm.gateway import LLMGateway, create_llm_gateway

__all__ = ["LLMGateway", "create_llm_gateway"]
