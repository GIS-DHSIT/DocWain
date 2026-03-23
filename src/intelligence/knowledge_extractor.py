"""LLM-powered deep knowledge extraction for document ingestion.

Extracts entities, facts, relationships, key claims, and document structure
from text using the LLM gateway. Every extraction carries a verbatim evidence
span from the source text for hallucination prevention.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    name: str
    type: str
    context: str
    evidence: str
    location: Dict[str, Any]
    confidence: float = 0.0


@dataclass
class ExtractedFact:
    statement: str
    evidence: str
    confidence: float = 0.0
    location: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    subject: str
    object: str
    relation: str
    evidence: str
    confidence: float = 0.0


@dataclass
class ExtractedClaim:
    claim: str
    evidence: str
    confidence: float = 0.0


@dataclass
class SectionStructure:
    section_title: str
    covers: str
    depends_on: List[str] = field(default_factory=list)


@dataclass
class KnowledgeExtractionResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    facts: List[ExtractedFact] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    claims: List[ExtractedClaim] = field(default_factory=list)
    structure: Optional[SectionStructure] = None
    extraction_time_ms: float = 0.0
    rejected_count: int = 0


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are a precise knowledge extraction engine. Your task is to extract "
    "structured knowledge from document text.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Extract ONLY what is explicitly stated in the text.\n"
    "2. For EVERY extraction, include the exact quote from the text that supports it "
    "in the 'evidence' field.\n"
    "3. Do NOT infer, assume, generalize, or add information not present in the text.\n"
    "4. If you cannot quote supporting text for an extraction, do NOT include it.\n"
    "5. Assign confidence between 0.0 and 1.0 based on how clearly the text states "
    "the information.\n"
    "6. Entity types should be descriptive and contextual (e.g., 'chemical_compound', "
    "'temperature_specification', 'organization', 'person', 'date', 'measurement') — "
    "do NOT use a fixed type list.\n\n"
    "Respond with valid JSON only. No markdown, no explanation."
)

_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract structured knowledge from this document section.\n\n"
    "TEXT (page {page}, section: {section}):\n"
    "---\n"
    "{text}\n"
    "---\n\n"
    "Return a JSON object with these fields:\n"
    '{{\n'
    '  "entities": [\n'
    '    {{"name": "...", "type": "...", "context": "brief context of what this entity represents", '
    '"evidence": "exact quote from text", "confidence": 0.0-1.0}}\n'
    '  ],\n'
    '  "facts": [\n'
    '    {{"statement": "declarative fact the document states", '
    '"evidence": "exact quote from text", "confidence": 0.0-1.0}}\n'
    '  ],\n'
    '  "relationships": [\n'
    '    {{"subject": "entity1", "object": "entity2", "relation": "how they relate", '
    '"evidence": "exact quote from text", "confidence": 0.0-1.0}}\n'
    '  ],\n'
    '  "key_claims": [\n'
    '    {{"claim": "main assertion or conclusion", '
    '"evidence": "exact quote from text", "confidence": 0.0-1.0}}\n'
    '  ],\n'
    '  "structure": {{\n'
    '    "section_title": "title of this section",\n'
    '    "covers": "brief description of what this section is about",\n'
    '    "depends_on": ["other sections this references"]\n'
    '  }}\n'
    '}}'
)

# ---------------------------------------------------------------------------
# Document summary prompt
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "You are a precise document analyst. Produce a compact knowledge summary "
    "that captures the essential information a subject matter expert would need "
    "to answer questions about this document. Be factual and concise."
)

_SUMMARY_PROMPT_TEMPLATE = (
    "Create a compact knowledge summary of this document.\n\n"
    "DOCUMENT CONTENT:\n---\n{text}\n---\n\n"
    "Return a JSON object:\n"
    '{{\n'
    '  "summary": "2-4 sentence expert summary of what this document contains and its key points",\n'
    '  "domain": "the primary domain/field this document belongs to (e.g., scientific_regulatory, '
    'technical, legal, financial, medical, hr, general)",\n'
    '  "key_topics": ["list", "of", "main", "topics"],\n'
    '  "document_type": "what kind of document this is (e.g., protocol, manual, contract, report)"\n'
    '}}'
)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class KnowledgeExtractor:
    """LLM-powered deep knowledge extraction from document text.

    Uses the LLM gateway to extract entities, facts, relationships, claims,
    and structure from document sections. Every extraction is evidence-anchored.
    """

    def __init__(
        self,
        llm_gateway=None,
        confidence_threshold: float = 0.7,
        max_tokens: int = 4096,
        timeout_per_chunk: float = 30.0,
    ):
        self._llm = llm_gateway
        self._confidence_threshold = confidence_threshold
        self._max_tokens = max_tokens
        self._timeout = timeout_per_chunk

    def _get_llm(self):
        """Lazy-load LLM gateway."""
        if self._llm is None:
            from src.llm.gateway import get_llm_gateway
            self._llm = get_llm_gateway()
        return self._llm

    def extract_section(
        self,
        text: str,
        page: int = 1,
        section: str = "unknown",
        confidence_threshold: Optional[float] = None,
    ) -> KnowledgeExtractionResult:
        """Extract structured knowledge from a single document section.

        Args:
            text: The section text to analyze.
            page: Page number for location tracking.
            section: Section title/identifier.
            confidence_threshold: Override default confidence gate.

        Returns:
            KnowledgeExtractionResult with all extractions.
        """
        threshold = confidence_threshold or self._confidence_threshold
        start = time.monotonic()

        if not text or len(text.strip()) < 50:
            return KnowledgeExtractionResult()

        # Truncate very long sections to avoid token overflow
        sample = text.strip()[:3000]

        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(
            page=page, section=section, text=sample
        )

        try:
            llm = self._get_llm()
            raw_text, _meta = llm.generate_with_metadata(
                prompt,
                system=_EXTRACTION_SYSTEM,
                temperature=0.1,
                max_tokens=self._max_tokens,
            )
        except Exception as e:
            logger.warning(
                "[KnowledgeExtractor] LLM call failed for page=%s section=%s: %s",
                page, section, e,
            )
            return KnowledgeExtractionResult(
                extraction_time_ms=(time.monotonic() - start) * 1000
            )

        # Parse JSON response — try answer first, then thinking block
        parsed = self._parse_json(raw_text)
        if not parsed:
            # Thinking model may have put JSON in thinking block
            thinking = (_meta or {}).get("thinking", "")
            if thinking:
                parsed = self._parse_json(thinking)
                if parsed:
                    logger.info(
                        "[KnowledgeExtractor] Recovered JSON from thinking block for page=%s section=%s",
                        page, section,
                    )
        if not parsed and raw_text and len(raw_text.strip()) >= 10:
            # Only retry if LLM produced content that failed to parse as JSON.
            # Empty responses won't improve with a retry.
            try:
                retry_prompt = (
                    "You MUST respond with ONLY a valid JSON object. No text before or after.\n\n"
                    + prompt
                )
                raw_text_retry, _meta_retry = llm.generate_with_metadata(
                    retry_prompt,
                    system=_EXTRACTION_SYSTEM + "\n\nCRITICAL: Output ONLY the JSON object. No markdown fences, no explanation.",
                    temperature=0.05,
                    max_tokens=self._max_tokens,
                )
                parsed = self._parse_json(raw_text_retry)
                if not parsed:
                    thinking_retry = (_meta_retry or {}).get("thinking", "")
                    if thinking_retry:
                        parsed = self._parse_json(thinking_retry)
                if parsed:
                    logger.info(
                        "[KnowledgeExtractor] Recovered via retry for page=%s section=%s",
                        page, section,
                    )
            except Exception as retry_err:
                logger.debug("[KnowledgeExtractor] Retry failed for page=%s section=%s: %s", page, section, retry_err)
        if not parsed:
            if not raw_text or len(raw_text.strip()) < 10:
                logger.debug(
                    "[KnowledgeExtractor] Empty LLM response for page=%s section=%s (skipping)",
                    page, section,
                )
            else:
                snippet = (raw_text or "")[:200].replace("\n", " ")
                logger.warning(
                    "[KnowledgeExtractor] Failed to parse JSON for page=%s section=%s raw_preview='%s'",
                    page, section, snippet,
                )
            return KnowledgeExtractionResult(
                extraction_time_ms=(time.monotonic() - start) * 1000
            )

        # Build result with confidence gating
        result = self._build_result(parsed, page, section, threshold)
        result.extraction_time_ms = (time.monotonic() - start) * 1000

        logger.info(
            "[KnowledgeExtractor] page=%s section=%s entities=%d facts=%d "
            "relationships=%d claims=%d rejected=%d time=%.0fms",
            page, section,
            len(result.entities), len(result.facts),
            len(result.relationships), len(result.claims),
            result.rejected_count, result.extraction_time_ms,
        )

        return result

    def extract_document(
        self,
        sections: List[Dict[str, Any]],
        confidence_threshold: Optional[float] = None,
    ) -> List[KnowledgeExtractionResult]:
        """Extract knowledge from all sections of a document.

        Args:
            sections: List of dicts with keys: text, page (optional),
                      section_title (optional).
            confidence_threshold: Override default confidence gate.

        Returns:
            List of KnowledgeExtractionResult, one per section.
        """
        import concurrent.futures

        if len(sections) <= 1:
            # No benefit from parallelization
            results = []
            for sec in sections:
                text = sec.get("text", "")
                page = sec.get("page", sec.get("start_page", 1))
                section_title = sec.get("section_title", sec.get("title", "unknown"))
                result = self.extract_section(
                    text=text,
                    page=page,
                    section=section_title,
                    confidence_threshold=confidence_threshold,
                )
                results.append(result)
            return results

        max_workers = min(3, len(sections))

        def _extract_one(sec):
            text = sec.get("text", "")
            page = sec.get("page", sec.get("start_page", 1))
            section_title = sec.get("section_title", sec.get("title", "unknown"))
            return self.extract_section(
                text=text,
                page=page,
                section=section_title,
                confidence_threshold=confidence_threshold,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_extract_one, sec): i for i, sec in enumerate(sections)}
            results = [None] * len(sections)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=120.0)
                except Exception as exc:
                    logger.warning("[KnowledgeExtractor] Section %d failed: %s", idx, exc)
                    results[idx] = KnowledgeExtractionResult()

        return results

    def generate_document_summary(
        self, full_text: str
    ) -> Dict[str, Any]:
        """Generate a compact knowledge summary of the entire document.

        Args:
            full_text: The complete document text.

        Returns:
            Dict with summary, domain, key_topics, document_type.
        """
        # Use first 4000 chars for summary (enough to understand the document)
        sample = full_text.strip()[:4000]
        if len(sample) < 50:
            return {
                "summary": "",
                "domain": "general",
                "key_topics": [],
                "document_type": "unknown",
            }

        prompt = _SUMMARY_PROMPT_TEMPLATE.format(text=sample)

        try:
            llm = self._get_llm()
            raw_text, _meta = llm.generate_with_metadata(
                prompt,
                system=_SUMMARY_SYSTEM,
                temperature=0.1,
                max_tokens=500,
            )
            parsed = self._parse_json(raw_text)
            if parsed:
                return {
                    "summary": parsed.get("summary", ""),
                    "domain": parsed.get("domain", "general"),
                    "key_topics": parsed.get("key_topics", []),
                    "document_type": parsed.get("document_type", "unknown"),
                }
        except Exception as e:
            logger.warning("[KnowledgeExtractor] Summary generation failed: %s", e)

        return {
            "summary": "",
            "domain": "general",
            "key_topics": [],
            "document_type": "unknown",
        }

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM output, handling common formatting issues."""
        if not raw:
            return None

        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # Try fixing common LLM JSON issues: trailing commas, single quotes
        import re
        cleaned = text[start:end] if (start >= 0 and end > start) else text
        # Remove trailing commas before } or ]
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        # Replace single quotes with double quotes (crude but effective for simple cases)
        if "'" in cleaned and '"' not in cleaned:
            cleaned = cleaned.replace("'", '"')
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _build_result(
        self,
        parsed: Dict[str, Any],
        page: int,
        section: str,
        threshold: float,
    ) -> KnowledgeExtractionResult:
        """Build KnowledgeExtractionResult from parsed JSON with confidence gating."""
        result = KnowledgeExtractionResult()
        rejected = 0
        location = {"page": page, "section": section}

        # Entities
        for ent in parsed.get("entities", []):
            conf = float(ent.get("confidence", 0.0))
            if conf < threshold:
                rejected += 1
                continue
            if not ent.get("evidence"):
                rejected += 1
                continue
            result.entities.append(ExtractedEntity(
                name=str(ent.get("name", "")),
                type=str(ent.get("type", "unknown")),
                context=str(ent.get("context", "")),
                evidence=str(ent.get("evidence", "")),
                location=location,
                confidence=conf,
            ))

        # Facts
        for fact in parsed.get("facts", []):
            conf = float(fact.get("confidence", 0.0))
            if conf < threshold:
                rejected += 1
                continue
            if not fact.get("evidence"):
                rejected += 1
                continue
            result.facts.append(ExtractedFact(
                statement=str(fact.get("statement", "")),
                evidence=str(fact.get("evidence", "")),
                confidence=conf,
                location=location,
            ))

        # Relationships
        for rel in parsed.get("relationships", []):
            conf = float(rel.get("confidence", 0.0))
            if conf < threshold:
                rejected += 1
                continue
            if not rel.get("evidence"):
                rejected += 1
                continue
            result.relationships.append(ExtractedRelationship(
                subject=str(rel.get("subject", "")),
                object=str(rel.get("object", "")),
                relation=str(rel.get("relation", "")),
                evidence=str(rel.get("evidence", "")),
                confidence=conf,
            ))

        # Key claims
        for claim in parsed.get("key_claims", []):
            conf = float(claim.get("confidence", 0.0))
            if conf < threshold:
                rejected += 1
                continue
            if not claim.get("evidence"):
                rejected += 1
                continue
            result.claims.append(ExtractedClaim(
                claim=str(claim.get("claim", "")),
                evidence=str(claim.get("evidence", "")),
                confidence=conf,
            ))

        # Structure
        struct = parsed.get("structure")
        if struct and isinstance(struct, dict):
            result.structure = SectionStructure(
                section_title=str(struct.get("section_title", section)),
                covers=str(struct.get("covers", "")),
                depends_on=struct.get("depends_on", []),
            )

        result.rejected_count = rejected
        return result


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_extractor: Optional[KnowledgeExtractor] = None


def get_knowledge_extractor(
    confidence_threshold: float = 0.7,
) -> KnowledgeExtractor:
    """Get or create the singleton KnowledgeExtractor."""
    global _extractor
    if _extractor is None:
        _extractor = KnowledgeExtractor(
            confidence_threshold=confidence_threshold,
        )
    return _extractor


__all__ = [
    "KnowledgeExtractor",
    "KnowledgeExtractionResult",
    "ExtractedEntity",
    "ExtractedFact",
    "ExtractedRelationship",
    "ExtractedClaim",
    "SectionStructure",
    "get_knowledge_extractor",
]
