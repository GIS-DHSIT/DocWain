from __future__ import annotations

from typing import Optional

from src.prompting.persona import build_persona_block, get_docwain_persona


def inject_persona_prompt(
    base_prompt: str,
    persona: Optional[str],
    profile_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    redis_client: Optional[object] = None,
    include_docwain_persona: bool = True,
) -> str:
    persona_memory = get_docwain_persona(profile_id, subscription_id, redis_client) if include_docwain_persona else ""
    persona_block = build_persona_block(persona, persona_memory)
    if not persona_block:
        return base_prompt
    return f"{persona_block}\n{base_prompt}".strip()
