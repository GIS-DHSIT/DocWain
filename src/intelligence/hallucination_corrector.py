"""
Hallucination self-correction for DocWain.

When grounding gate detects low confidence, automatically re-generates with
tighter constraints. Uses sentence-level Jaccard grounding (reuses fast_grounding
pattern) and either LLM correction or deterministic removal of unsupported sentences.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_CORRECTION_PROMPT = (
    "Rewrite the following sentence using ONLY the facts from the evidence below. "
    "If the sentence cannot be supported by the evidence, respond with exactly: REMOVE\n\n"
    "Sentence: {sentence}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Rewritten sentence:"
)

_BATCH_CORRECTION_PROMPT = (
    "For each numbered sentence below, rewrite it using ONLY facts from the evidence. "
    "If a sentence cannot be supported, respond with REMOVE for that sentence.\n"
    "Return one line per sentence in the format: NUMBER: rewritten sentence (or REMOVE)\n\n"
    "SENTENCES:\n{sentences}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "CORRECTIONS:"
)


@dataclass
class CorrectionResult:
    """Result of hallucination correction pass."""

    original: str
    corrected: str
    corrections_made: int
    removed_sentences: List[str]
    corrected_sentences: List[Tuple[str, str]]  # (original, corrected)
    was_modified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corrections_made": self.corrections_made,
            "removed_count": len(self.removed_sentences),
            "corrected_count": len(self.corrected_sentences),
            "was_modified": self.was_modified,
        }


def _tokenize(text: str) -> Set[str]:
    """Extract lowercase tokens from text."""
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s and len(s.strip()) > 5]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / max(len(union), 1)


def _score_sentence(sentence: str, chunk_texts: List[str]) -> float:
    """Score a single sentence against all evidence chunks using Jaccard overlap."""
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 1.0  # Empty/trivial sentences pass

    all_evidence_tokens: Set[str] = set()
    for chunk in chunk_texts:
        all_evidence_tokens.update(_tokenize(chunk))

    return _jaccard(sent_tokens, all_evidence_tokens)


def _is_structural(sentence: str) -> bool:
    """Check if sentence is structural (heading, list intro) and should be kept."""
    stripped = sentence.strip().rstrip(":")
    if len(stripped) < 20:
        return True
    if stripped.startswith(("**", "##", "- ", "* ", "• ")):
        return True
    if stripped.endswith(":"):
        return True
    return False


def _call_llm_correction(
    llm_client: Any, sentence: str, evidence: str, timeout: float = 5.0
) -> Optional[str]:
    """Call LLM for single sentence correction with timeout."""
    if llm_client is None:
        return None

    prompt = _CORRECTION_PROMPT.format(sentence=sentence, evidence=evidence[:2000])

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_llm, llm_client, prompt)
            result = future.result(timeout=timeout)

        if not result:
            return None

        result = result.strip()
        if result.upper() == "REMOVE":
            return "REMOVE"
        if len(result) < 5:
            return None
        return result

    except (FuturesTimeout, Exception) as exc:
        logger.debug("LLM correction failed for sentence: %s", exc)
        return None


def _call_llm(llm_client: Any, prompt: str) -> str:
    """Call LLM client."""
    if hasattr(llm_client, "generate"):
        result = llm_client.generate(prompt)
        if isinstance(result, tuple):
            return result[0] if result[0] else ""
        return result or ""
    return ""


def _call_llm_batch_correction(
    llm_client: Any,
    sentences: List[Tuple[int, str]],
    evidence: str,
    timeout: float = 10.0,
) -> Dict[int, str]:
    """Batch-correct multiple sentences in a single LLM call.

    Args:
        llm_client: LLM client with generate() method
        sentences: List of (index, sentence) tuples to correct
        evidence: Evidence text for grounding
        timeout: Timeout for the single LLM call

    Returns:
        Dict mapping sentence index to correction result ("REMOVE" or rewritten text)
    """
    if llm_client is None or not sentences:
        return {}

    numbered = "\n".join(f"{idx}: {sent}" for idx, sent in sentences)
    prompt = _BATCH_CORRECTION_PROMPT.format(
        sentences=numbered, evidence=evidence[:2000],
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_llm, llm_client, prompt)
            result = future.result(timeout=timeout)
    except (FuturesTimeout, Exception) as exc:
        logger.debug("Batch LLM correction failed: %s", exc)
        return {}

    if not result:
        return {}

    corrections: Dict[int, str] = {}
    for line in result.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Parse "NUMBER: text" format
        colon_pos = line.find(":")
        if colon_pos < 1:
            continue
        num_part = line[:colon_pos].strip()
        try:
            idx = int(num_part)
        except ValueError:
            continue
        text = line[colon_pos + 1:].strip()
        if text.upper() == "REMOVE":
            corrections[idx] = "REMOVE"
        elif len(text) >= 5:
            corrections[idx] = text
    return corrections


def correct_hallucinations(
    response: str,
    chunk_texts: List[str],
    llm_client: Any = None,
    score_threshold: float = 0.5,
    max_corrections: int = 3,
    per_correction_timeout: float = 5.0,
) -> CorrectionResult:
    """
    Attempt to correct hallucinated sentences in the response.

    For each sentence below the score_threshold:
      1. Try LLM correction with strict evidence prompt
      2. If LLM returns REMOVE or fails, deterministically remove the sentence

    Args:
        response: The generated response text
        chunk_texts: Evidence chunks to ground against
        llm_client: Optional LLM client for re-generation
        score_threshold: Jaccard score below which a sentence is unsupported
        max_corrections: Maximum corrections to attempt
        per_correction_timeout: Timeout per LLM correction call

    Returns:
        CorrectionResult with corrected text and metadata
    """
    if not response or not chunk_texts:
        return CorrectionResult(
            original=response,
            corrected=response,
            corrections_made=0,
            removed_sentences=[],
            corrected_sentences=[],
            was_modified=False,
        )

    sentences = _split_sentences(response)
    if not sentences:
        return CorrectionResult(
            original=response,
            corrected=response,
            corrections_made=0,
            removed_sentences=[],
            corrected_sentences=[],
            was_modified=False,
        )

    evidence_text = "\n".join(chunk_texts[:10])
    corrected_parts: List[str] = []
    removed: List[str] = []
    corrected_pairs: List[Tuple[str, str]] = []

    # Score all sentences and collect those needing correction
    unsupported: List[Tuple[int, str, float]] = []  # (index, sentence, score)
    sentence_results: List[Optional[str]] = [None] * len(sentences)  # None = keep original

    for i, sentence in enumerate(sentences):
        if _is_structural(sentence):
            sentence_results[i] = sentence
            continue

        score = _score_sentence(sentence, chunk_texts)
        if score >= score_threshold:
            sentence_results[i] = sentence
            continue

        if len(unsupported) < max_corrections:
            unsupported.append((i, sentence, score))
        else:
            sentence_results[i] = sentence  # Over budget, keep as-is

    # Batch-correct all unsupported sentences in a single LLM call
    if unsupported and llm_client is not None:
        batch_input = [(idx, sent) for idx, sent, _score in unsupported]
        batch_results = _call_llm_batch_correction(
            llm_client, batch_input, evidence_text, timeout=per_correction_timeout * 2,
        )

        for idx, sentence, score in unsupported:
            llm_result = batch_results.get(idx)
            if llm_result == "REMOVE":
                removed.append(sentence)
                logger.debug("Removed unsupported sentence: %s", sentence[:80])
            elif llm_result and llm_result != sentence:
                new_score = _score_sentence(llm_result, chunk_texts)
                if new_score > score:
                    sentence_results[idx] = llm_result
                    corrected_pairs.append((sentence, llm_result))
                    logger.debug(
                        "Corrected sentence (%.2f->%.2f): %s",
                        score, new_score, sentence[:80],
                    )
                else:
                    removed.append(sentence)
                    logger.debug(
                        "Correction didn't improve grounding, removing: %s",
                        sentence[:80],
                    )
            else:
                removed.append(sentence)
                logger.debug(
                    "No LLM correction available, removing unsupported: %s",
                    sentence[:80],
                )
    elif unsupported:
        # No LLM client — remove unsupported sentences
        for idx, sentence, _score in unsupported:
            removed.append(sentence)

    # Build final corrected parts from sentence_results
    for result in sentence_results:
        if result is not None:
            corrected_parts.append(result)

    was_modified = bool(removed) or bool(corrected_pairs)
    corrected_text = " ".join(corrected_parts).strip() if corrected_parts else response

    if corrected_text and len(corrected_text) < 20 and len(response) > 50:
        corrected_text = response
        was_modified = False

    return CorrectionResult(
        original=response,
        corrected=corrected_text,
        corrections_made=len(removed) + len(corrected_pairs),
        removed_sentences=removed,
        corrected_sentences=corrected_pairs,
        was_modified=was_modified,
    )
