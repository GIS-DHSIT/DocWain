"""Chat-format converters for DocWain V2 SFT training.

Converts raw dataset rows into the multi-turn chat format expected by the
V2 training pipeline (vision SFT, tool-call SFT, plain SFT).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# System prompt constant
# ---------------------------------------------------------------------------

DOCWAIN_V2_SYSTEM: str = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You can analyse document images, extract tables, charts, and text, "
    "and answer questions grounded in visual evidence. "
    "When specialised tools are available you may invoke them via "
    "tool-call blocks; otherwise answer directly from the document content."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_message(tools_json: str | None = None) -> Dict[str, Any]:
    """Build the system message, optionally embedding available tools."""
    content = DOCWAIN_V2_SYSTEM
    if tools_json:
        content += f"\n\nAvailable tools:\n{tools_json}"
    return {"role": "system", "content": content}


def _tool_call_block(call: Dict[str, Any]) -> str:
    """Serialise a single tool call into a ``<tool_call>`` block."""
    return (
        "<tool_call>\n"
        + json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, indent=2)
        + "\n</tool_call>"
    )


def _tool_response_block(result: Any) -> str:
    """Serialise a tool result into a ``<tool_response>`` block."""
    return (
        "<tool_response>\n"
        + json.dumps(result, indent=2, default=str)
        + "\n</tool_response>"
    )


# ---------------------------------------------------------------------------
# Public format functions
# ---------------------------------------------------------------------------


def format_vision_sft(
    image_path: str,
    question: str,
    answer: str,
    tools_json: str | None = None,
) -> Dict[str, Any]:
    """Create a 3-message vision SFT training pair.

    The user turn includes an ``<image>`` token so the vision encoder can
    inject the image embedding at that position during training.

    Returns
    -------
    dict with a ``messages`` list of ``[system, user, assistant]``.
    """
    return {
        "messages": [
            _system_message(tools_json),
            {
                "role": "user",
                "content": f"<image>{image_path}</image>\n{question}",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
    }


def format_tool_call_sft(
    query: str,
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Any],
    final_answer: str,
    tools_json: str | None = None,
) -> Dict[str, Any]:
    """Create a multi-turn tool-call SFT training pair.

    Supports parallel tool calls — each call gets its own ``<tool_call>``
    block inside the assistant turn, and each result its own
    ``<tool_response>`` block inside the subsequent tool turn.

    Returns
    -------
    dict with a ``messages`` list:
      [system, user, assistant(tool_calls), tool(results), assistant(final)]
    """
    # Build the assistant turn with one or more tool-call blocks
    call_blocks = "\n".join(_tool_call_block(tc) for tc in tool_calls)

    # Build the tool-response turn
    response_blocks = "\n".join(
        _tool_response_block(tr) for tr in tool_results
    )

    return {
        "messages": [
            _system_message(tools_json),
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "assistant",
                "content": call_blocks,
            },
            {
                "role": "tool",
                "content": response_blocks,
            },
            {
                "role": "assistant",
                "content": final_answer,
            },
        ],
    }


def format_no_tool_sft(
    query: str,
    answer: str,
    tools_json: str | None = None,
) -> Dict[str, Any]:
    """Create a simple 3-message SFT pair (no tool use).

    Returns
    -------
    dict with a ``messages`` list of ``[system, user, assistant]``.
    """
    return {
        "messages": [
            _system_message(tools_json),
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
    }
