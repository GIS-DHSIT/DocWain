"""Shared LLM infrastructure for tool handlers.

Provides domain-aware generation, structured JSON extraction, and
a self-assessed quality scoring framework (Tool IQ).

Each tool calls ``tool_generate()`` or ``tool_generate_structured()``
with a domain key that maps to tuned generation parameters.
"""
from __future__ import annotations

import concurrent.futures
import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# ── Per-Domain Generation Parameters ────────────────────────────────

_DOMAIN_OPTIONS: Dict[str, Dict[str, Any]] = {
    "hr":          {"temperature": 0.1,  "num_ctx": 4096, "num_predict": 2048, "top_p": 0.85, "top_k": 30},
    "medical":     {"temperature": 0.05, "num_ctx": 4096, "num_predict": 2048, "top_p": 0.80, "top_k": 20},
    "legal":       {"temperature": 0.1,  "num_ctx": 6144, "num_predict": 2048, "top_p": 0.85, "top_k": 30},
    "general":     {"temperature": 0.3,  "num_ctx": 4096, "num_predict": 2048, "top_p": 0.85, "top_k": 40},
    "creative":    {"temperature": 0.7,  "num_ctx": 4096, "num_predict": 2048, "top_p": 0.90, "top_k": 50},
    "translation": {"temperature": 0.2,  "num_ctx": 4096, "num_predict": 2048, "top_p": 0.85, "top_k": 30},
}

# ── Source confidence base scores ───────────────────────────────────

_SOURCE_CONFIDENCE: Dict[str, float] = {
    "llm": 0.75,
    "regex": 0.45,
    "template": 0.30,
    "hybrid": 0.60,
    "argos": 0.55,
}

# ── Markdown fence pattern ──────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)

# ── Gateway access ──────────────────────────────────────────────────

def get_tool_llm_client() -> Optional[Any]:
    """Return the singleton LLMGateway if available and enabled.

    Returns ``None`` when:
    - ``Config.Tools.LLM_ENABLED`` is False
    - The gateway is in cooldown
    - Import or access fails
    """
    try:
        from src.api.config import Config
        if not getattr(Config.Tools, "LLM_ENABLED", True):
            return None
    except Exception:
        return None

    try:
        from src.llm.gateway import get_llm_gateway
        client = get_llm_gateway()
        if client is None:
            return None
        if hasattr(client, "in_cooldown") and client.in_cooldown():
            return None
        return client
    except Exception:
        return None

# ── Core generation ─────────────────────────────────────────────────

def tool_generate(
    prompt: str,
    domain: str = "general",
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Generate text using the LLM gateway with domain-tuned parameters.

    Returns the generated text, or ``None`` on any failure.
    """
    client = get_tool_llm_client()
    if client is None:
        return None

    from src.api.config import Config
    if timeout is None:
        timeout = getattr(Config.Tools, "LLM_TIMEOUT", 12.0)

    options = dict(_DOMAIN_OPTIONS.get(domain, _DOMAIN_OPTIONS["general"]))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_generate, client, prompt, options)
            text = future.result(timeout=timeout)
        if text and len(text.strip()) >= 5:
            return text.strip()
        return None
    except concurrent.futures.TimeoutError:
        logger.warning("tool_generate timed out after %.1fs (domain=%s)", timeout, domain)
        return None
    except Exception as exc:
        logger.debug("tool_generate failed (domain=%s): %s", domain, exc)
        return None

def _call_generate(client: Any, prompt: str, options: Dict[str, Any]) -> str:
    """Invoke the LLM client's generate method."""
    try:
        from src.llm.task_router import task_scope, TaskType
        _ctx = task_scope(TaskType.TOOL_EXECUTION)
    except ImportError:
        from contextlib import nullcontext
        _ctx = nullcontext()
    with _ctx:
        if hasattr(client, "generate_with_metadata"):
            result = client.generate_with_metadata(prompt, options=options)
            if isinstance(result, tuple):
                return result[0] or ""
            return str(result) if result else ""
        if hasattr(client, "generate"):
            return client.generate(prompt, options=options) or ""
        return ""

def tool_generate_structured(
    prompt: str,
    domain: str = "general",
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Generate structured JSON output from the LLM.

    Appends a JSON-only instruction to the prompt, strips markdown
    fences, and parses the response.  Returns ``None`` on failure.
    """
    full_prompt = prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no commentary."
    raw = tool_generate(full_prompt, domain=domain, timeout=timeout)
    if not raw:
        return None
    return _parse_json_response(raw)

def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()

    # Strip markdown fences
    fence_match = _FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start:brace_end + 1])
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None

# ── Tool IQ Score ───────────────────────────────────────────────────

@dataclass
class ToolIQScore:
    """Self-assessed quality score for a tool response."""

    overall: float = 0.0
    completeness: float = 0.0
    confidence: float = 0.0
    source: str = "unknown"
    domain: str = "general"
    field_coverage: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 3),
            "completeness": round(self.completeness, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "domain": self.domain,
            "field_coverage": self.field_coverage,
        }

def score_tool_response(
    result: Dict[str, Any],
    domain: str,
    expected_fields: List[str],
    source: str,
) -> ToolIQScore:
    """Compute a Tool IQ score for a tool response.

    - **completeness** = filled / expected (non-empty values)
    - **confidence** = source base adjusted by completeness
    - **overall** = 0.6 * completeness + 0.4 * confidence
    """
    field_coverage: Dict[str, bool] = {}
    filled = 0
    for f in expected_fields:
        val = result.get(f)
        present = _is_field_present(val)
        field_coverage[f] = present
        if present:
            filled += 1

    completeness = filled / max(len(expected_fields), 1)
    base_confidence = _SOURCE_CONFIDENCE.get(source, 0.40)
    confidence = min(1.0, base_confidence + 0.15 * completeness)
    overall = 0.6 * completeness + 0.4 * confidence

    return ToolIQScore(
        overall=overall,
        completeness=completeness,
        confidence=confidence,
        source=source,
        domain=domain,
        field_coverage=field_coverage,
    )

def _is_field_present(val: Any) -> bool:
    """Check if a field value is meaningfully present."""
    if val is None:
        return False
    if isinstance(val, str):
        return len(val.strip()) > 0
    if isinstance(val, (list, dict)):
        return len(val) > 0
    return True

# ── Prompt building ─────────────────────────────────────────────────

def build_extraction_prompt(
    tool_name: str,
    text: str,
    query: str,
    json_schema: str,
) -> str:
    """Build a domain-expert extraction prompt using ToolProfile.

    Pulls system_prompt and extraction_focus from intelligence.py,
    caps text at Config.Tools.LLM_MAX_INPUT_CHARS.
    """
    try:
        from src.api.config import Config
        max_chars = getattr(Config.Tools, "LLM_MAX_INPUT_CHARS", 3500)
    except Exception:
        max_chars = 3500

    system_prompt = ""
    extraction_focus = ""
    try:
        from src.tools.intelligence import get_tool_profile
        profile = get_tool_profile(tool_name)
        if profile:
            system_prompt = profile.system_prompt or ""
            extraction_focus = profile.extraction_focus or ""
    except Exception:
        pass

    parts = []
    if system_prompt:
        parts.append(system_prompt)

    if extraction_focus:
        parts.append(f"\nEXTRACTION FOCUS:\n{extraction_focus}")

    parts.append(f"\nDOCUMENT TEXT:\n{text[:max_chars]}")

    if query:
        parts.append(f"\nUSER QUERY: {query}")

    parts.append(f"\nExtract the requested information as JSON matching this schema:\n{json_schema}")

    return "\n".join(parts)

def build_generation_prompt(
    tool_name: str,
    instructions: str,
    reference: str,
) -> str:
    """Build a generation prompt (for creator/email tools) using ToolProfile."""
    try:
        from src.api.config import Config
        max_chars = getattr(Config.Tools, "LLM_MAX_INPUT_CHARS", 3500)
    except Exception:
        max_chars = 3500

    system_prompt = ""
    try:
        from src.tools.intelligence import get_tool_profile
        profile = get_tool_profile(tool_name)
        if profile:
            system_prompt = profile.system_prompt or ""
    except Exception:
        pass

    parts = []
    if system_prompt:
        parts.append(system_prompt)
    parts.append(f"\n{instructions}")
    if reference:
        parts.append(f"\nREFERENCE MATERIAL:\n{reference[:max_chars]}")
    return "\n".join(parts)
