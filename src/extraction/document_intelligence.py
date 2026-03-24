"""Universal document intelligence extractor.

Makes a single LLM call to extract structured metadata from any document type
(contracts, invoices, resumes, reports, etc.).

Public API:
    extract_document_intelligence(full_text, filename, llm_gateway?) -> dict
    build_doc_index_text(filename, intelligence) -> str
    build_doc_intelligence_text(filename, intelligence) -> str
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a document analyst. Extract ALL structured information "
    "from this document. Output valid JSON only."
)

_USER_PROMPT_TEMPLATE = """\
--- DOCUMENT ---
{text}
--- END DOCUMENT ---

Extract everything that is factually present in this document:

1. document_type: what kind of document this is (e.g., contract, invoice, resume, report, policy, statement, presentation, letter, form, other)
2. parties: all named people, organizations, or entities with their roles as described in the document
3. key_dates: all dates mentioned with their context
4. key_values: all monetary amounts, quantities, percentages, durations, limits, scores, or measurable values with context
5. key_sections: list each section/heading with a one-line summary of what it contains
6. key_facts: important statements, terms, conditions, findings, or conclusions — whatever the document asserts
7. one_line_summary: single sentence describing this document

Rules:
- Extract ONLY what is explicitly stated. Never infer or fabricate.
- Adapt to the document — a resume has different structure than an invoice.
- If a field has no data in the document, omit it entirely.
- Return valid JSON only."""

# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    text = raw.strip()

    # Try stripping markdown code blocks first
    m = _CODE_BLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response ({len(raw)} chars)")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_document_intelligence(
    full_text: str,
    filename: str,
    llm_gateway: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract structured intelligence from a document via a single LLM call.

    Args:
        full_text: The full document text.
        filename: Original filename (used for logging).
        llm_gateway: Optional pre-created LLM gateway instance.

    Returns:
        Dict with keys such as ``document_type``, ``parties``, ``key_dates``,
        ``key_values``, ``key_sections``, ``key_facts``, ``one_line_summary``.
        On failure, returns a minimal dict with ``one_line_summary`` only.
    """
    logger.info(
        "[DOC_INTELLIGENCE] Extracting intelligence for %s (%d chars)",
        filename,
        len(full_text),
    )

    if llm_gateway is None:
        from src.llm.gateway import get_llm_gateway
        llm_gateway = get_llm_gateway()

    prompt = _USER_PROMPT_TEMPLATE.format(text=full_text[:15000])
    t0 = time.perf_counter()

    try:
        text, _metadata = llm_gateway.generate_with_metadata(
            prompt,
            system=_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=4096,
        )
        result = _parse_json_response(text)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "[DOC_INTELLIGENCE] Extraction failed for %s: %s (after %dms)",
            filename,
            exc,
            elapsed_ms,
        )
        snippet = full_text[:200].replace("\n", " ").strip()
        return {"one_line_summary": snippet}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    doc_type = result.get("document_type", "unknown")
    parties = result.get("parties", [])
    facts = result.get("key_facts", [])
    logger.info(
        "[DOC_INTELLIGENCE] Extracted: type=%s parties=%d facts=%d in %dms",
        doc_type,
        len(parties) if isinstance(parties, list) else 0,
        len(facts) if isinstance(facts, list) else 0,
        elapsed_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------


def build_doc_index_text(filename: str, intelligence: Dict[str, Any]) -> str:
    """Build a compact one-line summary (~50 tokens) for document indexing.

    Dynamically assembles whichever fields are present.  Example output:
        ``US_Healthcare_Contract_5.pdf | Healthcare Equipment Agreement |
        Leica Biosystems (Supplier) <-> Prime Diagnostics (Customer) | Jan 2026 | Net 30 days``
    """
    parts: list[str] = [filename]

    doc_type = intelligence.get("document_type")
    if doc_type:
        parts.append(str(doc_type))

    summary = intelligence.get("one_line_summary")
    if summary and not doc_type:
        # Use summary only when doc_type is absent to avoid redundancy
        parts.append(str(summary))

    # Parties — compact representation
    parties = intelligence.get("parties")
    if parties and isinstance(parties, list):
        party_strs = []
        for p in parties[:4]:  # cap to avoid overly long lines
            if isinstance(p, dict):
                name = p.get("name", p.get("entity", ""))
                role = p.get("role", "")
                party_strs.append(f"{name} ({role})" if role else str(name))
            else:
                party_strs.append(str(p))
        if party_strs:
            parts.append(" \u2194 ".join(party_strs))

    # First key date
    dates = intelligence.get("key_dates")
    if dates and isinstance(dates, list):
        first = dates[0]
        if isinstance(first, dict):
            parts.append(str(first.get("date", first.get("value", ""))))
        else:
            parts.append(str(first))

    # First key value
    values = intelligence.get("key_values")
    if values and isinstance(values, list):
        first = values[0]
        if isinstance(first, dict):
            parts.append(str(first.get("value", first.get("amount", ""))))
        else:
            parts.append(str(first))

    return " | ".join(p for p in parts if p)


def build_doc_intelligence_text(
    filename: str, intelligence: Dict[str, Any]
) -> str:
    """Build a narrative summary (~200-500 tokens) for Qdrant canonical_text.

    Assembles all extracted fields into readable prose/structured text.
    """
    lines: list[str] = []
    lines.append(f"Document: {filename}")

    doc_type = intelligence.get("document_type")
    if doc_type:
        lines.append(f"Type: {doc_type}")

    summary = intelligence.get("one_line_summary")
    if summary:
        lines.append(f"Summary: {summary}")

    # Parties
    parties = intelligence.get("parties")
    if parties and isinstance(parties, list):
        lines.append("Parties:")
        for p in parties:
            if isinstance(p, dict):
                name = p.get("name", p.get("entity", ""))
                role = p.get("role", "")
                lines.append(f"  - {name} ({role})" if role else f"  - {name}")
            else:
                lines.append(f"  - {p}")

    # Key dates
    dates = intelligence.get("key_dates")
    if dates and isinstance(dates, list):
        lines.append("Key Dates:")
        for d in dates:
            if isinstance(d, dict):
                date_val = d.get("date", d.get("value", ""))
                ctx = d.get("context", d.get("description", ""))
                lines.append(f"  - {date_val}: {ctx}" if ctx else f"  - {date_val}")
            else:
                lines.append(f"  - {d}")

    # Key values
    values = intelligence.get("key_values")
    if values and isinstance(values, list):
        lines.append("Key Values:")
        for v in values:
            if isinstance(v, dict):
                val = v.get("value", v.get("amount", ""))
                ctx = v.get("context", v.get("description", ""))
                lines.append(f"  - {val}: {ctx}" if ctx else f"  - {val}")
            else:
                lines.append(f"  - {v}")

    # Key sections
    sections = intelligence.get("key_sections")
    if sections and isinstance(sections, list):
        lines.append("Sections:")
        for s in sections:
            if isinstance(s, dict):
                title = s.get("title", s.get("heading", ""))
                desc = s.get("summary", s.get("description", ""))
                lines.append(f"  - {title}: {desc}" if desc else f"  - {title}")
            else:
                lines.append(f"  - {s}")

    # Key facts
    facts = intelligence.get("key_facts")
    if facts and isinstance(facts, list):
        lines.append("Key Facts:")
        for f in facts:
            if isinstance(f, str):
                lines.append(f"  - {f}")
            elif isinstance(f, dict):
                lines.append(f"  - {f.get('fact', f.get('statement', str(f)))}")

    return "\n".join(lines)
