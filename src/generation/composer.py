"""Response Composer — the COMPOSE step of the DocWain Core Agent pipeline.

Cleans, formats, and packages the LLM answer into the standard
AnswerPayload structure.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Preamble patterns to strip
# ---------------------------------------------------------------------------

_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"Based on (?:my |the )?analysis[,.]?\s*"
    r"|After reviewing (?:the )?(?:documents?|evidence)[,.]?\s*"
    r"|According to the documents?[,.]?\s*"
    r"|Upon (?:reviewing|examining) (?:the )?(?:documents?|evidence)[,.]?\s*"
    r"|From (?:my |the )?review[,.]?\s*"
    r"|Having (?:reviewed|analyzed) (?:the )?(?:documents?|evidence)[,.]?\s*"
    r")",
    re.IGNORECASE,
)

# Merge adjacent citations: [SOURCE-1][SOURCE-2] → [SOURCE-1, SOURCE-2]
_ADJACENT_CITE_RE = re.compile(
    r"\[SOURCE-(\d+)\](?:\s*\[SOURCE-(\d+)\])+"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compose_response(
    text: str,
    evidence: List[Dict[str, Any]],
    grounded: bool,
    task_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Package the answer into the standard AnswerPayload dict.

    Returns:
        Dict with keys: response, sources, grounded, context_found, metadata.
    """
    cleaned = _clean_response(text)
    sources = _build_sources(evidence)

    merged_metadata: Dict[str, Any] = dict(metadata) if metadata else {}
    merged_metadata.update({
        "task_type": task_type,
        "evidence_count": len(evidence),
        "engine": "docwain_core_agent",
    })

    return {
        "response": cleaned,
        "sources": sources,
        "grounded": grounded,
        "context_found": len(evidence) > 0,
        "metadata": merged_metadata,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_response(text: str) -> str:
    """Remove common preambles and merge adjacent citations."""
    cleaned = _PREAMBLE_RE.sub("", text).lstrip()

    # Capitalise after preamble removal if needed
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Merge adjacent citations
    def _merge_cites(m: re.Match) -> str:
        nums = re.findall(r"SOURCE-(\d+)", m.group(0))
        labels = ", ".join(f"SOURCE-{n}" for n in nums)
        return f"[{labels}]"

    cleaned = _ADJACENT_CITE_RE.sub(_merge_cites, cleaned)
    return cleaned


def _build_sources(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicated source list from evidence, aggregating sections."""
    seen: Dict[str, Dict[str, Any]] = {}  # key = (source_name, document_id)

    for item in evidence:
        source_name = item.get("source_name", "unknown")
        document_id = item.get("document_id", "unknown")
        key = f"{source_name}::{document_id}"

        if key not in seen:
            seen[key] = {
                "source_name": source_name,
                "document_id": document_id,
                "pages": [],
                "sections": [],
                "score": item.get("score", 0),
            }

        page = item.get("page")
        if page is not None and page not in seen[key]["pages"]:
            seen[key]["pages"].append(page)

        section = item.get("section")
        if section and section not in seen[key]["sections"]:
            seen[key]["sections"].append(section)

        # Keep highest score
        score = item.get("score", 0)
        if score and score > seen[key]["score"]:
            seen[key]["score"] = score

    return list(seen.values())
