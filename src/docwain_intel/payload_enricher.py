"""Payload enricher for Qdrant chunk payloads.

Adds entity IDs, fingerprint tags, and unit type metadata to chunk
payloads so vector search returns structured intelligence alongside text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    DocumentFingerprint,
    ExtractionResult,
    StructuredDocument,
)


def _word_boundary_match(entity_text: str, chunk_text: str) -> bool:
    """Check if entity_text appears in chunk_text with word boundaries.

    Case-insensitive. "John" matches "John Smith" but not "Johnson".
    """
    pattern = r"(?<![a-zA-Z0-9])" + re.escape(entity_text) + r"(?![a-zA-Z0-9])"
    return bool(re.search(pattern, chunk_text, re.IGNORECASE))


def _find_unit_type(
    chunk_text: str,
    structured_doc: Optional[StructuredDocument],
) -> Optional[str]:
    """Find the unit type for a chunk by text overlap with semantic units."""
    if not structured_doc or not structured_doc.units:
        return None

    chunk_lower = chunk_text.lower().strip()
    if not chunk_lower:
        return None

    best_overlap = 0.0
    best_type: Optional[str] = None

    for unit in structured_doc.units:
        if not unit.text:
            continue
        unit_lower = unit.text.lower().strip()
        if not unit_lower:
            continue

        # Check bidirectional containment
        if chunk_lower in unit_lower or unit_lower in chunk_lower:
            shorter = min(len(chunk_lower), len(unit_lower))
            longer = max(len(chunk_lower), len(unit_lower))
            overlap = shorter / longer if longer > 0 else 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_type = unit.unit_type.value

    # Require at least 30% overlap
    if best_overlap >= 0.3:
        return best_type
    return None


def enrich_payload(
    chunk_payload: Dict[str, Any],
    chunk_text: str,
    extraction: ExtractionResult,
    fingerprint: Optional[DocumentFingerprint] = None,
    structured_doc: Optional[StructuredDocument] = None,
) -> Dict[str, Any]:
    """Enrich a Qdrant chunk payload with intelligence metadata.

    Adds keys prefixed with ``intel_`` without overwriting any existing
    payload fields.  Mutates *chunk_payload* in place **and** returns it.
    """
    # --- Entity matching ---
    matched_ids: List[str] = []
    matched_labels: set[str] = set()

    for entity in extraction.entities:
        if not entity.text:
            continue
        if _word_boundary_match(entity.text, chunk_text):
            matched_ids.append(entity.entity_id)
            matched_labels.add(entity.label)

    # --- Fingerprint tags ---
    tags: List[str] = []
    if fingerprint and fingerprint.auto_tags:
        tags = list(fingerprint.auto_tags)

    # --- Unit type ---
    unit_type = _find_unit_type(chunk_text, structured_doc)

    # --- Apply enrichment (never overwrite existing keys) ---
    enrichments: Dict[str, Any] = {
        "intel_entity_ids": matched_ids,
        "intel_entity_labels": sorted(matched_labels),
        "intel_fingerprint_tags": tags,
        "intel_unit_type": unit_type,
    }

    for key, value in enrichments.items():
        if key not in chunk_payload:
            chunk_payload[key] = value

    return chunk_payload


def batch_enrich(
    payloads: List[Tuple[Dict[str, Any], str]],
    extraction: ExtractionResult,
    fingerprint: Optional[DocumentFingerprint] = None,
    structured_doc: Optional[StructuredDocument] = None,
) -> List[Dict[str, Any]]:
    """Apply :func:`enrich_payload` to a batch of (payload, chunk_text) pairs.

    Returns a list of enriched payload dicts.
    """
    return [
        enrich_payload(payload, text, extraction, fingerprint, structured_doc)
        for payload, text in payloads
    ]
