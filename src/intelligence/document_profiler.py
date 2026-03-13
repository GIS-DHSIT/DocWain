"""
DocumentProfiler — one-shot document profiling at ingestion time.

Runs a single LLM call over the first few chunks of a document to determine
its domain, terminology, field types, and structure.  The resulting
DocumentProfile is stored in Qdrant payload metadata alongside chunks so
downstream retrieval and generation can adapt to the document's nature.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentProfile:
    """Compact profile describing a document's nature and content."""

    domain: str = "general"
    document_type: str = "document"
    key_terminology: List[str] = field(default_factory=list)
    field_types: List[str] = field(default_factory=list)
    structure_pattern: str = "unstructured"
    language_register: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "document_type": self.document_type,
            "key_terminology": self.key_terminology,
            "field_types": self.field_types,
            "structure_pattern": self.structure_pattern,
            "language_register": self.language_register,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentProfile":
        return cls(
            domain=data.get("domain", "general"),
            document_type=data.get("document_type", "document"),
            key_terminology=data.get("key_terminology", []),
            field_types=data.get("field_types", []),
            structure_pattern=data.get("structure_pattern", "unstructured"),
            language_register=data.get("language_register", "neutral"),
        )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROFILE_PROMPT = """\
You are a document analyst. Given the filename and opening text of a document, \
produce a JSON profile with exactly these fields:

- "domain": the subject area (e.g. "finance", "legal", "medical", "engineering", "hr", "general")
- "document_type": what kind of document this is (e.g. "invoice", "contract", "report", "memo", "policy", "resume", "datasheet")
- "key_terminology": list of 5-15 domain-specific terms or phrases found in the text
- "field_types": list of structured field types present (e.g. "date", "currency", "percentage", "name", "address", "id_number", "table")
- "structure_pattern": one of "form", "narrative", "tabular", "mixed", "list", "hierarchical", "unstructured"
- "language_register": one of "formal", "technical", "casual", "legal", "neutral"

Respond with ONLY the JSON object, no other text.

Filename: {filename}

Document text:
{text}
"""


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class DocumentProfiler:
    """Profiles documents via a single LLM call at ingestion time."""

    def __init__(self, llm_gateway: Any) -> None:
        self._llm = llm_gateway

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, chunks: List[str], filename: str) -> DocumentProfile:
        """Profile a single document from its first chunks and filename.

        Args:
            chunks: First 3-5 text chunks of the document.
            filename: Original filename (helps the LLM infer document type).

        Returns:
            A DocumentProfile with the inferred metadata.
        """
        sample_chunks = chunks[:5]
        text = "\n\n---\n\n".join(sample_chunks)

        # Cap the text to avoid blowing context limits
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated]"

        prompt = _PROFILE_PROMPT.format(filename=filename, text=text)

        try:
            response = self._llm.generate(prompt, temperature=0.0, max_tokens=512)
            return self._parse_profile(response)
        except Exception:
            logger.warning(
                "LLM call failed during document profiling for %s, returning default profile",
                filename,
                exc_info=True,
            )
            return DocumentProfile()

    def profile_batch(self, documents: List[Dict[str, Any]]) -> List[DocumentProfile]:
        """Profile multiple documents.

        Args:
            documents: List of dicts, each with keys ``chunks`` (list[str])
                       and ``filename`` (str).

        Returns:
            List of DocumentProfile in the same order as input.
        """
        profiles: List[DocumentProfile] = []
        for doc in documents:
            chunks = doc.get("chunks", [])
            filename = doc.get("filename", "unknown")
            profiles.append(self.profile(chunks, filename))
        return profiles

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_profile(self, raw: str) -> DocumentProfile:
        """Best-effort JSON extraction from an LLM response."""
        data = self._try_parse_json(raw)
        if data is None:
            logger.warning("Could not parse profile JSON, using defaults")
            return DocumentProfile()
        return self._build_profile(data)

    @staticmethod
    def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """Try several strategies to pull a JSON object from the response."""
        text = raw.strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Markdown code block (```json ... ``` or ``` ... ```)
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. First { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _build_profile(data: Dict[str, Any]) -> DocumentProfile:
        """Safely build a DocumentProfile from parsed JSON, falling back for
        any missing or malformed fields."""
        def as_list(val: Any) -> List[str]:
            if isinstance(val, list):
                return [str(v) for v in val]
            return []

        return DocumentProfile(
            domain=str(data.get("domain", "general")),
            document_type=str(data.get("document_type", "document")),
            key_terminology=as_list(data.get("key_terminology")),
            field_types=as_list(data.get("field_types")),
            structure_pattern=str(data.get("structure_pattern", "unstructured")),
            language_register=str(data.get("language_register", "neutral")),
        )
