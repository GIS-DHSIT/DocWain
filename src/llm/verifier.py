"""Chain-of-thought grounding verifier using deepseek-r1.

Verifies whether a generated answer is fully supported by the evidence.
Uses CoT reasoning to identify unsupported claims. Advisory only —
never blocks the pipeline response.
"""
from __future__ import annotations

import concurrent.futures
import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .role_prompts import VERIFIER_SYSTEM, VERIFIER_TEMPLATE

logger = get_logger(__name__)

@dataclass
class VerificationResult:
    """Result of grounding verification."""
    supported: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported": self.supported,
            "confidence": self.confidence,
            "issues": self.issues,
            "reasoning": self.reasoning[:500] if self.reasoning else "",
        }

def _parse_verification(raw: str) -> Optional[VerificationResult]:
    """Parse JSON from verifier output into a VerificationResult."""
    if not raw:
        return None

    cleaned = raw.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try to extract JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
    if not match:
        # No JSON found — treat non-empty text as reasoning with heuristic
        if len(cleaned) > 20:
            has_issues = any(word in cleaned.lower() for word in ("not supported", "unsupported", "contradicted", "no evidence"))
            return VerificationResult(
                supported=not has_issues,
                confidence=0.5,
                issues=["Could not parse structured response"],
                reasoning=cleaned[:500],
            )
        return None

    try:
        data = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None

    supported = data.get("supported", True)
    if isinstance(supported, str):
        supported = supported.lower() in ("true", "yes", "1")

    try:
        confidence = float(data.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5

    issues = data.get("issues", [])
    if isinstance(issues, str):
        issues = [issues] if issues.strip() else []
    elif not isinstance(issues, list):
        issues = []
    issues = [str(i) for i in issues if i]

    reasoning = str(data.get("reasoning", ""))

    return VerificationResult(
        supported=bool(supported),
        confidence=confidence,
        issues=issues,
        reasoning=reasoning,
    )

def _build_evidence_text(chunks: List[Any], max_chars: int = 4000) -> str:
    """Build a compact evidence string from chunks."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        text = getattr(chunk, "text", "") or ""
        if not text:
            continue
        source = ""
        meta = getattr(chunk, "meta", None) or {}
        if meta.get("source_name"):
            source = f" [{meta['source_name']}]"
        elif meta.get("filename"):
            source = f" [{meta['filename']}]"
        entry = f"[{i + 1}]{source}: {text.strip()}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                parts.append(entry[:remaining] + "...")
            break
        parts.append(entry)
        total += len(entry) + 1
    return "\n".join(parts)

def verify_grounding(
    answer: str,
    evidence_chunks: List[Any],
    query: str,
    llm_client: Any,
    timeout_s: float = 30.0,
) -> Optional[VerificationResult]:
    """Verify whether the answer is grounded in the evidence.

    Parameters
    ----------
    answer : str
        The generated answer to verify.
    evidence_chunks : list
        Chunks used as evidence for the answer.
    query : str
        The original user query.
    llm_client : Any
        An LLM client. If it has a `verify()` method, that will be used.
    timeout_s : float
        Maximum seconds to wait for verification.

    Returns
    -------
    VerificationResult or None if verification fails.
    """
    if not answer or not answer.strip():
        return None
    if not evidence_chunks:
        return VerificationResult(
            supported=False,
            confidence=0.0,
            issues=["No evidence provided"],
            reasoning="Cannot verify without evidence.",
        )

    evidence_text = _build_evidence_text(evidence_chunks)
    if not evidence_text:
        return None

    prompt = (
        f"{VERIFIER_SYSTEM}\n\n"
        f"{VERIFIER_TEMPLATE.format(query=query, answer=answer, evidence=evidence_text)}"
    )

    def _call() -> str:
        try:
            from src.llm.task_router import task_scope, TaskType
            _ctx = task_scope(TaskType.GROUNDING_VERIFY)
        except ImportError:
            from contextlib import nullcontext
            _ctx = nullcontext()
        with _ctx:
            if hasattr(llm_client, "verify"):
                return llm_client.verify(prompt)
            if hasattr(llm_client, "generate_for_role"):
                from .multi_agent import AgentRole
                return llm_client.generate_for_role(AgentRole.VERIFIER, prompt, max_retries=1, backoff=0.5)
            return llm_client.generate(prompt, max_retries=1, backoff=0.5)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        raw = future.result(timeout=timeout_s)
        result = _parse_verification(raw)
        if result:
            logger.info(
                "Verification: supported=%s confidence=%.2f issues=%d",
                result.supported, result.confidence, len(result.issues),
            )
        return result
    except concurrent.futures.TimeoutError:
        logger.warning("Grounding verification timed out after %.1fs", timeout_s)
        return None
    except Exception as exc:
        logger.warning("Grounding verification failed: %s", exc)
        return None
    finally:
        executor.shutdown(wait=False)
