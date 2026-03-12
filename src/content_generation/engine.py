"""Content generation engine — 6-step pipeline for document-grounded content.

Pipeline steps:
  1. DETECT  — Identify content type from query
  2. EXTRACT — Pull relevant facts from chunks
  3. PLAN    — Structure content outline
  4. GENERATE — Use LLM with domain-specific prompt
  5. VERIFY  — Check grounding against source chunks
  6. FORMAT  — Final formatting and metadata
"""
from __future__ import annotations

import concurrent.futures
import json
from src.utils.logging_utils import get_logger
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from .prompts import ContentPromptBuilder
from .registry import (
    ContentType,
    detect_content_type,
    detect_content_type_with_domain,
    get_content_type,
)
from .verifier import ContentVerifier, VerificationResult

logger = get_logger(__name__)

# Tunables
_LLM_TIMEOUT_S = 60.0
_LLM_MAX_OUTPUT_TOKENS = 2048
_MAX_EVIDENCE_CHARS = 6144
_MAX_CHUNKS = 10

# ---------------------------------------------------------------------------
# Fact extraction helpers (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def _get_chunk_text(chunk: Any) -> str:
    """Extract text from a chunk (supports dict and object forms)."""
    if isinstance(chunk, dict):
        return (
            chunk.get("canonical_text")
            or chunk.get("text")
            or chunk.get("content")
            or ""
        )
    return getattr(chunk, "text", "") or getattr(chunk, "canonical_text", "") or ""

def _get_chunk_meta(chunk: Any) -> Dict[str, Any]:
    """Extract metadata from a chunk."""
    if isinstance(chunk, dict):
        return chunk.get("metadata") or chunk.get("payload") or chunk
    return getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}

def _extract_person_names(text: str) -> List[str]:
    """Extract likely person names from text."""
    # 2-3 capitalized words at start of line or after common labels
    patterns = [
        re.compile(r"(?:^|Name\s*:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", re.M),
        re.compile(r"(?:Candidate|Applicant|Patient)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", re.I),
    ]
    names: List[str] = []
    for pat in patterns:
        names.extend(pat.findall(text))
    return list(dict.fromkeys(names))  # dedupe preserving order

def _extract_skills(text: str) -> List[str]:
    """Extract skills from text."""
    skills: List[str] = []
    # Look for skill-related sections
    skill_section = re.search(
        r"(?:skills?|technologies|competencies|expertise)\s*:?\s*\n?(.*?)(?:\n\n|\Z)",
        text, re.I | re.S,
    )
    if skill_section:
        raw = skill_section.group(1)
        # Split on commas, bullets, pipes, newlines
        for item in re.split(r"[,|\n•\-]", raw):
            item = item.strip().strip("- •")
            if 2 < len(item) < 50:
                skills.append(item)
    return list(dict.fromkeys(skills))[:20]

def _extract_amounts(text: str) -> List[str]:
    """Extract monetary amounts from text (currencies, percentages, large numbers with suffixes)."""
    amounts = re.findall(
        r"(?:[$€£¥₹])\s?[\d,]+(?:\.\d{1,2})?(?:\s?[kKmMbB])?"    # $5,000 or $150K
        r"|\b[\d,]+(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|INR|AED)\b"     # 5000 USD
        r"|\b[\d,]+(?:\.\d{1,2})?\s?%"                              # 15.5%
        r"|\b[\d,]+(?:\.\d{1,2})?\s?(?:per\s+(?:month|year|annum|hour|day))\b",  # 5000 per month
        text, re.I,
    )
    return list(dict.fromkeys(amounts))

def _extract_dates(text: str) -> List[str]:
    """Extract dates from text (numeric, month-name, quarter, and year-only)."""
    dates = re.findall(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"                          # 12/31/2024, 2024-01-15
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b|"                             # ISO dates
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|"                             # March 15, 2024
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\.?\s+\d{4}\b|"                                          # March 2024
        r"\b[QqFf][1-4]\s+\d{4}\b|"                                      # Q1 2024, FY 2023
        r"\b\d{4}\s*[-–]\s*\d{4}\b|"                                     # 2020-2024 (year range)
        r"\b(?:present|current)\b",                                       # "present" in date ranges
        text, re.I,
    )
    return list(dict.fromkeys(dates))

def _extract_organizations(text: str) -> List[str]:
    """Extract organization names from text."""
    orgs: List[str] = []
    patterns = [
        re.compile(r"(?:company|employer|organization|firm|agency|hospital|university|institute)\s*:\s*(.+?)(?:\n|$)", re.I),
        re.compile(r"(?:at|with|for)\s+([A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+){0,4})\b"),
        # Match "Org Inc.", "Org Corp.", "Org Ltd." patterns
        re.compile(r"\b([A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+){0,3}\s+(?:Inc|Corp|Ltd|LLC|LLP|Pvt|Pte|GmbH|AG)\.?)\b"),
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            org = m.group(1).strip()
            if 2 < len(org) < 80:
                orgs.append(org)
    return list(dict.fromkeys(orgs))[:10]

def _extract_facts_from_chunks(
    chunks: Sequence[Any],
    content_type: ContentType,
) -> Dict[str, Any]:
    """Deterministic fact extraction from chunks based on content type fields."""
    all_text = "\n\n".join(_get_chunk_text(c) for c in chunks)
    facts: Dict[str, Any] = {}

    all_fields = set(content_type.required_fields) | set(content_type.optional_fields)
    # Add default extractions for all types
    all_fields |= {"key_information"}

    if "person_name" in all_fields or content_type.domain == "hr":
        names = _extract_person_names(all_text)
        # Also check filenames
        for chunk in chunks:
            meta = _get_chunk_meta(chunk)
            fname = meta.get("source_name") or meta.get("filename") or ""
            if fname:
                # Strip extension and clean
                name_part = re.sub(r"\.[a-z]{2,4}$", "", fname, flags=re.I)
                name_part = re.sub(r"[_\-]", " ", name_part).strip()
                if name_part and name_part not in names:
                    names.append(name_part)
        if names:
            facts["person_name"] = names[0] if len(names) == 1 else names

    if "skills" in all_fields:
        skills = _extract_skills(all_text)
        if skills:
            facts["skills"] = skills

    if "amounts" in all_fields or content_type.domain == "invoice":
        amounts = _extract_amounts(all_text)
        if amounts:
            facts["amounts"] = amounts

    if any(f in all_fields for f in ("dates", "date")):
        dates = _extract_dates(all_text)
        if dates:
            facts["dates"] = dates

    if "organizations" in all_fields:
        orgs = _extract_organizations(all_text)
        if orgs:
            facts["organizations"] = orgs

    if "experience_years" in all_fields:
        years = re.findall(r"\b(\d{1,2})\s+years?\b", all_text.lower())
        if years:
            facts["experience_years"] = years

    if "certifications" in all_fields:
        certs = re.findall(
            r"(?:certified|certification|certificate)\s*(?:in|:)?\s*([^\n,]{3,40})",
            all_text, re.I,
        )
        if certs:
            facts["certifications"] = [c.strip() for c in certs][:10]

    if "education" in all_fields:
        edu = re.findall(
            r"\b(?:B\.?S\.?|M\.?S\.?|Ph\.?D\.?|MBA|B\.?A\.?|M\.?A\.?|Bachelor|Master|Doctorate)"
            r"\s*(?:of|in)?\s*([^\n,]{3,50})",
            all_text, re.I,
        )
        if edu:
            facts["education"] = [e.strip() for e in edu][:5]

    return facts

# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

def _build_evidence_text(chunks: Sequence[Any], max_chars: int = _MAX_EVIDENCE_CHARS) -> str:
    """Build evidence text from chunks, grouped by document."""
    doc_groups: Dict[str, List[str]] = {}
    for chunk in chunks:
        meta = _get_chunk_meta(chunk)
        doc_id = str(meta.get("document_id") or meta.get("doc_id") or "unknown")
        source_name = meta.get("source_name") or meta.get("filename") or doc_id
        text = _get_chunk_text(chunk)
        if text:
            doc_groups.setdefault(source_name, []).append(text)

    parts: List[str] = []
    for source, texts in doc_groups.items():
        parts.append(f"--- Source: {source} ---")
        parts.extend(texts)
        parts.append("")

    combined = "\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[Evidence truncated]"
    return combined

# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    timeout: float = _LLM_TIMEOUT_S,
) -> Optional[str]:
    """Call LLM with system + user prompt. Returns text or None on failure."""
    if llm_client is None:
        return None

    # Build combined prompt (most local LLMs don't support system/user separation)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    options = {
        "num_predict": _LLM_MAX_OUTPUT_TOKENS,
        "max_output_tokens": _LLM_MAX_OUTPUT_TOKENS,
        "num_ctx": 4096,
        "stop": [],
    }

    def _call() -> str:
        try:
            from src.llm.task_router import task_scope, TaskType
            _ctx = task_scope(TaskType.CONTENT_GENERATION)
        except ImportError:
            from contextlib import nullcontext
            _ctx = nullcontext()
        with _ctx:
            if hasattr(llm_client, "generate_with_metadata"):
                text, _meta = llm_client.generate_with_metadata(
                    full_prompt, options=options, max_retries=1, backoff=0.4,
                )
                return text or ""
            return llm_client.generate(full_prompt, max_retries=1, backoff=0.4) or ""

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        result = future.result(timeout=timeout)
        return result if result and result.strip() else None
    except concurrent.futures.TimeoutError:
        logger.warning("Content generation LLM call timed out after %.0fs", timeout)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Content generation LLM call failed: %s", exc)
        return None
    finally:
        executor.shutdown(wait=False)

# ---------------------------------------------------------------------------
# Deterministic fallback generators
# ---------------------------------------------------------------------------

def _deterministic_cover_letter(facts: Dict[str, Any], evidence: str) -> str:
    """Generate a cover letter without LLM using extracted facts."""
    name = facts.get("person_name", "Candidate")
    if isinstance(name, list):
        name = name[0] if name else "Candidate"

    skills = facts.get("skills", [])[:4]
    orgs = facts.get("organizations", [])[:2]
    years = facts.get("experience_years", [])

    parts = ["Dear Hiring Manager,", "", "I am writing to express interest in the role."]
    if years:
        parts.append(f"The candidate brings {years[0]} years of experience relevant to this position.")
    if skills:
        parts.append(f"Key skills evidenced in the documents include {', '.join(skills)}.")
    if orgs:
        parts.append(f"Experience includes work with {', '.join(orgs)}.")
    parts.extend([
        "",
        "I would welcome the opportunity to discuss how these strengths align with your needs.",
        "",
        f"Sincerely,",
        name,
    ])
    return "\n".join(parts)

def _deterministic_summary(facts: Dict[str, Any], evidence: str, content_type: ContentType) -> str:
    """Generate a deterministic summary from facts."""
    parts: List[str] = []
    parts.append(f"## {content_type.name}")
    parts.append("")

    for key, value in facts.items():
        if isinstance(value, list) and value:
            parts.append(f"**{key.replace('_', ' ').title()}**: {', '.join(str(v) for v in value)}")
        elif value:
            parts.append(f"**{key.replace('_', ' ').title()}**: {value}")

    if not facts:
        # Fallback: extract first few meaningful sentences from evidence
        sentences = re.split(r"(?<=[.!?])\s+", evidence.strip())
        meaningful = [s for s in sentences if len(s) > 20][:5]
        if meaningful:
            parts.extend(meaningful)
        else:
            parts.append("Insufficient information to generate content.")

    return "\n".join(parts)

def _deterministic_key_points(facts: Dict[str, Any], evidence: str) -> str:
    """Extract key points as bullet list."""
    points: List[str] = []
    for key, value in facts.items():
        if isinstance(value, list):
            for item in value[:5]:
                points.append(f"- {key.replace('_', ' ').title()}: {item}")
        elif value:
            points.append(f"- {key.replace('_', ' ').title()}: {value}")

    if not points:
        # Extract from evidence
        sentences = re.split(r"(?<=[.!?])\s+", evidence.strip())
        for s in sentences[:8]:
            if len(s) > 20:
                points.append(f"- {s.strip()}")

    return "## Key Points\n\n" + "\n".join(points) if points else "No key points found."

_DETERMINISTIC_GENERATORS = {
    "cover_letter": _deterministic_cover_letter,
    "key_points": _deterministic_key_points,
}

# ---------------------------------------------------------------------------
# Content generation engine
# ---------------------------------------------------------------------------

class ContentGenerationEngine:
    """6-step pipeline for document-grounded content generation."""

    def __init__(self, llm_client: Any = None):
        self._llm_client = llm_client
        self._prompt_builder = ContentPromptBuilder()
        self._verifier = ContentVerifier()

    def generate(
        self,
        query: str,
        chunks: Sequence[Any],
        *,
        content_type_id: Optional[str] = None,
        chunk_domain: Optional[str] = None,
        llm_client: Optional[Any] = None,
        extra_instructions: str = "",
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full 6-step content generation pipeline.

        Args:
            query: User query / content request.
            chunks: Source document chunks.
            content_type_id: Explicit content type ID (overrides detection).
            chunk_domain: Domain hint from chunk metadata.
            llm_client: LLM client (overrides instance default).
            extra_instructions: Additional user instructions.
            correlation_id: Request correlation ID.

        Returns:
            Dict with: response, sources, grounded, context_found, metadata.
        """
        start = time.time()
        client = llm_client or self._llm_client
        metadata: Dict[str, Any] = {
            "task": "content_generation",
            "query": query,
            "correlation_id": correlation_id,
        }

        # ---- Step 0: Validate input ----
        if not chunks:
            return self._empty_response(
                "Insufficient document evidence to generate content.",
                metadata=metadata,
            )

        # ---- Step 1: DETECT — identify content type ----
        content_type = self._detect(query, content_type_id, chunk_domain)
        if content_type is None:
            return self._empty_response(
                "Could not determine what type of content to generate. "
                "Try specifying the content type explicitly (e.g., 'generate a cover letter').",
                metadata=metadata,
            )
        metadata["content_type"] = content_type.id
        metadata["domain"] = content_type.domain

        # Check multi-doc requirement
        if content_type.supports_multi_doc and content_type.min_chunks > len(chunks):
            return self._empty_response(
                f"'{content_type.name}' requires at least {content_type.min_chunks} "
                f"document chunks but only {len(chunks)} were provided.",
                metadata=metadata,
            )

        # ---- Step 2: EXTRACT — pull facts from chunks ----
        top_chunks = self._select_top_chunks(chunks)
        facts = _extract_facts_from_chunks(top_chunks, content_type)
        metadata["facts_extracted"] = len(facts)

        # ---- Step 3: PLAN — build evidence text ----
        evidence_text = _build_evidence_text(top_chunks)

        # ---- Step 4: GENERATE — LLM or deterministic ----
        generated_text, used_llm = self._generate(
            content_type, facts, evidence_text, query,
            extra_instructions, client,
        )
        if not generated_text:
            return self._empty_response(
                "Content generation failed. The LLM did not produce a response.",
                metadata=metadata,
            )
        metadata["generation_method"] = "llm" if used_llm else "deterministic"

        # ---- Step 5: VERIFY — grounding check ----
        verification = self._verifier.verify(generated_text, top_chunks, facts)
        metadata["verification"] = verification.to_dict()

        # ---- Step 6: FORMAT — build final response ----
        sources = self._build_sources(top_chunks)
        duration_ms = int((time.time() - start) * 1000)
        metadata["duration_ms"] = duration_ms

        return {
            "response": generated_text,
            "sources": sources,
            "grounded": verification.grounded,
            "context_found": True,
            "metadata": metadata,
            "warnings": verification.warnings,
        }

    # -- Step 1: DETECT ---------------------------------------------------

    def _detect(
        self,
        query: str,
        explicit_type_id: Optional[str],
        chunk_domain: Optional[str],
    ) -> Optional[ContentType]:
        """Detect content type from query and/or explicit ID."""
        if explicit_type_id:
            ct = get_content_type(explicit_type_id)
            if ct:
                return ct

        return detect_content_type_with_domain(query, chunk_domain)

    # -- Step 2: EXTRACT helpers ------------------------------------------

    @staticmethod
    def _select_top_chunks(chunks: Sequence[Any]) -> List[Any]:
        """Select top chunks by score."""
        scored = []
        for c in chunks:
            score = getattr(c, "score", 0.0) or 0.0
            if isinstance(c, dict):
                score = c.get("score", 0.0) or 0.0
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:_MAX_CHUNKS]]

    # -- Step 4: GENERATE -------------------------------------------------

    def _generate(
        self,
        content_type: ContentType,
        facts: Dict[str, Any],
        evidence_text: str,
        query: str,
        extra_instructions: str,
        llm_client: Any,
    ) -> tuple[Optional[str], bool]:
        """Generate content via LLM or deterministic fallback.

        Returns:
            Tuple of (generated_text, used_llm).
        """
        # Try LLM first
        if llm_client is not None:
            system_prompt = self._prompt_builder.build_system_prompt(content_type)
            user_prompt = self._prompt_builder.build_generation_prompt(
                content_type, facts, evidence_text, query, extra_instructions,
            )
            result = _call_llm(llm_client, system_prompt, user_prompt)
            if result:
                return result.strip(), True

        # Deterministic fallback
        generator = _DETERMINISTIC_GENERATORS.get(content_type.id)
        if generator:
            return generator(facts, evidence_text), False

        # Generic deterministic fallback
        return _deterministic_summary(facts, evidence_text, content_type), False

    # -- Step 6: FORMAT helpers -------------------------------------------

    @staticmethod
    def _build_sources(chunks: Sequence[Any]) -> List[Dict[str, Any]]:
        """Build source citations from chunks."""
        sources: List[Dict[str, Any]] = []
        seen: set = set()
        for chunk in chunks:
            meta = _get_chunk_meta(chunk)
            source_name = meta.get("source_name") or meta.get("filename") or ""
            page = meta.get("page") or meta.get("page_start") or meta.get("page_number")
            key = (source_name, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "source_name": source_name,
                "doc_domain": meta.get("doc_domain"),
                "page": page,
            })
        return sources[:12]

    @staticmethod
    def _empty_response(
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "response": message,
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": metadata or {},
            "warnings": [],
        }
