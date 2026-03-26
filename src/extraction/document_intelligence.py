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
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a document analyst. Extract ALL structured information "
    "from this document. Output valid JSON only. "
    "Do NOT include any explanation, thinking, or commentary — ONLY the JSON object."
)

_USER_PROMPT_TEMPLATE = """\
--- DOCUMENT ---
{text}
--- END DOCUMENT ---

Extract everything that is factually present in this document:

1. document_type: what kind of document this is (e.g., contract, invoice, resume, report, policy, statement, presentation, letter, form, other)
2. parties: all named people, organizations, or entities with their roles as described in the document
3. start_date: the effective date, commencement date, or "entered into on" date
4. end_date: the expiration date. If not explicitly stated but a term duration IS given (e.g. "initial term of 3 years" from start_date "January 1, 2026"), then CALCULATE: end_date = "December 31, 2028". Always provide this calculated value.
5. term: the contract duration (e.g. "3 years", "12 months", "60 days"). If start and end dates are given, calculate the difference.
6. key_dates: all other dates mentioned with their context
7. key_values: ALL monetary amounts, quantities, percentages, durations, limits, scores, notice periods, uptime percentages, liability caps, and any other measurable values with context. Be exhaustive — capture EVERY number and percentage in the document.
8. key_sections: list each section/heading with a one-line summary that INCLUDES the specific values mentioned in that section (e.g., "Payment: Net 30 days" not just "Defines payment terms")
9. key_facts: important statements, terms, conditions, findings, or conclusions — capture EVERY clause, condition, and obligation
10. one_line_summary: single sentence describing this document

Rules:
- Extract ONLY what is explicitly stated. Never infer or fabricate.
- Adapt to the document — a resume has different structure than an invoice.
- If a field has no data in the document, omit it entirely.
- Return valid JSON only. No markdown, no explanation, no commentary."""

_RETRY_PROMPT_TEMPLATE = """\
The previous attempt to extract document intelligence failed to produce valid JSON.
Please try again with this document. Return ONLY a valid JSON object, nothing else.

--- DOCUMENT ---
{text}
--- END DOCUMENT ---

Return a JSON object with these fields (include only fields with data):
{{"document_type": "...", "parties": [{{"name": "...", "role": "..."}}], "start_date": "...", "end_date": "...", "key_dates": [{{"date": "...", "context": "..."}}], "key_values": [{{"value": "...", "context": "..."}}], "key_sections": [{{"heading": "...", "summary": "include specific values like amounts, percentages, durations"}}], "key_facts": ["..."], "one_line_summary": "..."}}"""

# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response with multiple fallback strategies."""
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response")

    text = raw.strip()

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code blocks
    m = _CODE_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost { ... } braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Strategy 4: Try fixing common JSON issues
            fixed = _fix_json_issues(candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # Strategy 5: Extract key-value pairs with regex as last resort
    fallback = _regex_extract_fields(text)
    if fallback:
        return fallback

    raise ValueError(f"Could not parse JSON from LLM response ({len(raw)} chars)")


def _fix_json_issues(text: str) -> str:
    """Fix common JSON formatting issues from LLM output."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Fix single quotes to double quotes (careful with contractions)
    # Only do this if there are no double quotes at all
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return text


def _regex_extract_fields(text: str) -> Optional[Dict[str, Any]]:
    """Last-resort extraction of key fields using regex patterns."""
    result: Dict[str, Any] = {}

    # Extract document_type
    m = re.search(r'"?document_type"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        result["document_type"] = m.group(1)

    # Extract one_line_summary
    m = re.search(r'"?one_line_summary"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        result["one_line_summary"] = m.group(1)

    # Extract party names
    parties = []
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)".*?"role"\s*:\s*"([^"]+)"', text):
        parties.append({"name": m.group(1), "role": m.group(2)})
    if parties:
        result["parties"] = parties

    # Extract dates
    dates = []
    for m in re.finditer(r'"date"\s*:\s*"([^"]+)"(?:.*?"context"\s*:\s*"([^"]+)")?', text):
        entry = {"date": m.group(1)}
        if m.group(2):
            entry["context"] = m.group(2)
        dates.append(entry)
    if dates:
        result["key_dates"] = dates

    return result if result else None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3


def extract_document_intelligence(
    full_text: str,
    filename: str,
    llm_gateway: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract structured intelligence from a document via LLM call.

    Uses up to 3 attempts with different strategies:
    1. Standard prompt with low temperature
    2. Retry with simplified prompt and explicit JSON example
    3. Final attempt with no-thinking mode and minimal prompt

    On complete failure, returns a dict built from the document text directly.
    """
    logger.info(
        "[DOC_INTELLIGENCE] Extracting intelligence for %s (%d chars)",
        filename, len(full_text),
    )

    if llm_gateway is None:
        from src.llm.gateway import get_llm_gateway
        llm_gateway = get_llm_gateway()

    truncated_text = full_text[:15000]
    t0 = time.perf_counter()

    # Attempt 1: Standard extraction
    result = _attempt_extraction(
        llm_gateway, truncated_text, filename,
        prompt_template=_USER_PROMPT_TEMPLATE,
        system=_SYSTEM_PROMPT,
        temperature=0.1,
        think=False,
        attempt=1,
    )
    if result is not None:
        _log_success(filename, result, t0)
        return result

    # Attempt 2: Retry with simplified prompt and JSON example
    result = _attempt_extraction(
        llm_gateway, truncated_text, filename,
        prompt_template=_RETRY_PROMPT_TEMPLATE,
        system="You are a JSON extraction tool. Return ONLY valid JSON. No text, no explanation.",
        temperature=0.0,
        think=False,
        attempt=2,
    )
    if result is not None:
        _log_success(filename, result, t0)
        return result

    # Attempt 3: Minimal prompt, force short response
    minimal_prompt = (
        f"Extract from this document and return JSON only:\n\n"
        f"{truncated_text[:5000]}\n\n"
        f'Return: {{"document_type":"...","parties":[{{"name":"...","role":"..."}}],'
        f'"one_line_summary":"...","key_facts":["..."]}}'
    )
    result = _attempt_extraction(
        llm_gateway, truncated_text, filename,
        prompt_text=minimal_prompt,
        system="Return valid JSON only.",
        temperature=0.0,
        think=False,
        attempt=3,
        max_tokens=2048,
    )
    if result is not None:
        _log_success(filename, result, t0)
        return result

    # All attempts failed — build intelligence from text directly
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.warning(
        "[DOC_INTELLIGENCE] All %d attempts failed for %s (after %dms). "
        "Building intelligence from text directly.",
        _MAX_RETRIES, filename, elapsed_ms,
    )
    return _build_fallback_intelligence(full_text, filename)


def _attempt_extraction(
    llm_gateway: Any,
    text: str,
    filename: str,
    *,
    prompt_template: Optional[str] = None,
    prompt_text: Optional[str] = None,
    system: str = "",
    temperature: float = 0.1,
    think: bool = False,
    attempt: int = 1,
    max_tokens: int = 4096,
) -> Optional[Dict[str, Any]]:
    """Single extraction attempt. Returns parsed dict or None on failure."""
    try:
        prompt = prompt_text or prompt_template.format(text=text[:15000])
        raw_response, metadata = llm_gateway.generate_with_metadata(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
        )

        # Handle thinking mode — response might be in metadata
        if not raw_response and metadata and metadata.get("thinking"):
            logger.debug("[DOC_INTELLIGENCE] Attempt %d: response was in thinking block", attempt)
            raw_response = metadata.get("thinking", "")

        if not raw_response or not raw_response.strip():
            logger.warning(
                "[DOC_INTELLIGENCE] Attempt %d for %s: empty response",
                attempt, filename,
            )
            return None

        result = _parse_json_response(raw_response)

        # Validate minimum fields
        if not isinstance(result, dict):
            logger.warning("[DOC_INTELLIGENCE] Attempt %d: response is not a dict", attempt)
            return None

        # Ensure at least one meaningful field exists
        meaningful = any(result.get(k) for k in (
            "document_type", "parties", "key_facts", "one_line_summary",
        ))
        if not meaningful:
            logger.warning("[DOC_INTELLIGENCE] Attempt %d: no meaningful fields extracted", attempt)
            return None

        return result

    except Exception as exc:
        logger.warning(
            "[DOC_INTELLIGENCE] Attempt %d for %s failed: %s",
            attempt, filename, exc,
        )
        return None


def _log_success(filename: str, result: Dict, t0: float) -> None:
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


def _build_fallback_intelligence(full_text: str, filename: str) -> Dict[str, Any]:
    """Build basic intelligence from raw text when LLM extraction fails completely.

    Uses regex patterns to extract entities, dates, and amounts directly from text.
    This ensures EVERY document gets at least basic intelligence — zero failures.
    """
    result: Dict[str, Any] = {}

    # Document type from filename
    fname_lower = filename.lower()
    if "contract" in fname_lower or "agreement" in fname_lower:
        result["document_type"] = "contract"
    elif "invoice" in fname_lower:
        result["document_type"] = "invoice"
    elif "resume" in fname_lower or "cv" in fname_lower:
        result["document_type"] = "resume"
    elif "report" in fname_lower:
        result["document_type"] = "report"
    else:
        result["document_type"] = "document"

    # Extract entities with common patterns
    parties: List[Dict[str, str]] = []

    # "between X and Y" pattern
    m = re.search(
        r'between\s+(.+?)\s*\("([^"]+)"\)\s*and\s+(.+?)\s*\("([^"]+)"\)',
        full_text, re.IGNORECASE,
    )
    if m:
        parties.append({"name": m.group(1).strip(), "role": m.group(2)})
        parties.append({"name": m.group(3).strip(), "role": m.group(4)})

    # "Company Name Inc./LLC/Ltd" patterns
    if not parties:
        for m in re.finditer(
            r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc|LLC|Ltd|Corp|Co|LLP|Pvt)\.?)\b',
            full_text,
        ):
            name = m.group(1).strip()
            if name not in [p["name"] for p in parties]:
                parties.append({"name": name, "role": ""})
            if len(parties) >= 4:
                break

    if parties:
        result["parties"] = parties

    # Extract dates
    dates: List[Dict[str, str]] = []
    for m in re.finditer(
        r'\b((?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},?\s+\d{4})\b',
        full_text,
    ):
        if m.group(1) not in [d.get("date") for d in dates]:
            dates.append({"date": m.group(1), "context": ""})
        if len(dates) >= 5:
            break

    if dates:
        result["key_dates"] = dates

    # Extract monetary values
    values: List[Dict[str, str]] = []
    for m in re.finditer(r'\$[\d,]+(?:\.\d{2})?', full_text):
        if m.group(0) not in [v.get("value") for v in values]:
            values.append({"value": m.group(0), "context": ""})
        if len(values) >= 5:
            break

    # Extract percentages
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*%', full_text):
        val = f"{m.group(1)}%"
        if val not in [v.get("value") for v in values]:
            values.append({"value": val, "context": ""})
        if len(values) >= 8:
            break

    if values:
        result["key_values"] = values

    # Key facts from first few sentences
    sentences = re.split(r'[.!?]\s+', full_text[:2000])
    key_facts = [s.strip() + "." for s in sentences[:5] if len(s.strip()) > 30]
    if key_facts:
        result["key_facts"] = key_facts

    # Summary
    result["one_line_summary"] = full_text[:200].replace("\n", " ").strip()

    logger.info(
        "[DOC_INTELLIGENCE] Fallback extraction for %s: type=%s parties=%d facts=%d",
        filename, result.get("document_type"), len(parties), len(key_facts),
    )

    return result


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------


def build_doc_index_text(filename: str, intelligence: Dict[str, Any]) -> str:
    """Build a compact one-line summary (~50 tokens) for document indexing."""
    parts: list[str] = [filename]

    doc_type = intelligence.get("document_type")
    if doc_type:
        parts.append(str(doc_type))

    summary = intelligence.get("one_line_summary")
    if summary and not doc_type:
        parts.append(str(summary))

    # Parties
    parties = intelligence.get("parties")
    if parties and isinstance(parties, list):
        party_strs = []
        for p in parties[:4]:
            if isinstance(p, dict):
                name = p.get("name", p.get("entity", ""))
                role = p.get("role", "")
                party_strs.append(f"{name} ({role})" if role else str(name))
            else:
                party_strs.append(str(p))
        if party_strs:
            parts.append(" \u2194 ".join(party_strs))

    # Start/end dates
    start_date = intelligence.get("start_date")
    if start_date:
        parts.append(f"Start: {start_date}")
    end_date = intelligence.get("end_date")
    if end_date:
        parts.append(f"End: {end_date}")

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
    """Build a narrative summary (~200-500 tokens) for Qdrant canonical_text."""
    lines: list[str] = []
    lines.append(f"Document: {filename}")

    doc_type = intelligence.get("document_type")
    if doc_type:
        lines.append(f"Type: {doc_type}")

    summary = intelligence.get("one_line_summary")
    if summary:
        lines.append(f"Summary: {summary}")

    start_date = intelligence.get("start_date")
    if start_date:
        lines.append(f"Start Date: {start_date}")
    end_date = intelligence.get("end_date")
    if end_date:
        lines.append(f"End Date: {end_date}")

    # Term — from explicit field, or calculated from dates, or from key_values
    term = intelligence.get("term")
    if term:
        lines.append(f"Term: {term}")
    elif start_date and end_date:
        lines.append(f"Term: {start_date} to {end_date}")
    else:
        for v in (intelligence.get("key_values") or []):
            if isinstance(v, dict):
                ctx = str(v.get("context", "")).lower()
                val = str(v.get("value", ""))
                if "initial term" in ctx or "contract term" in ctx or "term duration" in ctx or "term" == ctx:
                    lines.append(f"Term: {val}")
                    break

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

    facts = intelligence.get("key_facts")
    if facts and isinstance(facts, list):
        lines.append("Key Facts:")
        for f in facts:
            if isinstance(f, str):
                lines.append(f"  - {f}")
            elif isinstance(f, dict):
                lines.append(f"  - {f.get('fact', f.get('statement', str(f)))}")

    return "\n".join(lines)
