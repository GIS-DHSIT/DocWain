"""Universal entity and fact extraction engine.

Uses spaCy NER + regex patterns + textacy SVO triples + structure-aware
extraction from KV groups and tables.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.docwain_intel.models import (
    EntitySpan,
    ExtractionResult,
    FactTriple,
    SemanticUnit,
    StructuredDocument,
    UnitType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_spacy_nlp = None
_gliner_model = None


def _get_spacy():
    """Load spaCy model (lazy singleton). Prefers en_core_web_lg."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    import spacy

    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            _spacy_nlp = spacy.load(model_name)
            logger.info("Loaded spaCy model: %s", model_name)
            return _spacy_nlp
        except OSError:
            continue
    raise RuntimeError("No spaCy English model found. Install en_core_web_lg or en_core_web_sm.")


def _get_gliner():
    """Load GLiNER model (lazy singleton, optional)."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model
    try:
        from gliner import GLiNER

        _gliner_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        logger.info("Loaded GLiNER model")
        return _gliner_model
    except Exception as exc:  # noqa: BLE001
        logger.warning("GLiNER not available: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Regex patterns for structured entity extraction
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# Phone: require at least 7 digit characters to avoid false positives
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"  # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"  # optional area code
    r"\d{3,4}[\s\-.]?\d{3,4}"  # main number
)
_URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")
_DURATION_RE = re.compile(
    r"\b\d+\s*(?:years?|months?|weeks?|days?|hours?|hrs?|minutes?|mins?)\b",
    re.IGNORECASE,
)


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# spaCy NER extraction
# ---------------------------------------------------------------------------
def _extract_spacy_entities(text: str, unit_id: str) -> List[EntitySpan]:
    """Extract named entities using spaCy NER."""
    nlp = _get_spacy()
    doc = nlp(text)
    entities: List[EntitySpan] = []
    for ent in doc.ents:
        entities.append(
            EntitySpan(
                entity_id=f"ent_{_uid()}",
                text=ent.text,
                normalized=ent.text.lower().strip(),
                label=ent.label_,
                unit_id=unit_id,
                char_start=ent.start_char,
                char_end=ent.end_char,
                confidence=0.85,
                source="spacy",
            )
        )
    return entities


# ---------------------------------------------------------------------------
# Regex pattern extraction
# ---------------------------------------------------------------------------
def _extract_pattern_entities(text: str, unit_id: str) -> List[EntitySpan]:
    """Extract EMAIL, PHONE, URL, DURATION via regex."""
    entities: List[EntitySpan] = []
    patterns: List[Tuple[re.Pattern, str, int]] = [
        (_EMAIL_RE, "EMAIL", 0),
        (_URL_RE, "URL", 0),
        (_DURATION_RE, "DURATION", 0),
    ]
    for pat, label, _ in patterns:
        for m in pat.finditer(text):
            entities.append(
                EntitySpan(
                    entity_id=f"ent_{_uid()}",
                    text=m.group(),
                    normalized=m.group().lower().strip(),
                    label=label,
                    unit_id=unit_id,
                    char_start=m.start(),
                    char_end=m.end(),
                    confidence=0.95,
                    source="pattern",
                )
            )

    # Phone — extra digit-count validation
    for m in _PHONE_RE.finditer(text):
        digit_count = sum(c.isdigit() for c in m.group())
        if digit_count >= 7:
            entities.append(
                EntitySpan(
                    entity_id=f"ent_{_uid()}",
                    text=m.group(),
                    normalized=m.group().strip(),
                    label="PHONE",
                    unit_id=unit_id,
                    char_start=m.start(),
                    char_end=m.end(),
                    confidence=0.90,
                    source="pattern",
                )
            )

    return entities


# ---------------------------------------------------------------------------
# SVO fact extraction (textacy with fallback)
# ---------------------------------------------------------------------------
def _extract_svo_facts(
    text: str, unit_id: str, entities: List[EntitySpan]
) -> List[FactTriple]:
    """Extract subject-verb-object triples using textacy or fallback dep parse."""
    nlp = _get_spacy()
    doc = nlp(text)
    facts: List[FactTriple] = []

    # Build entity lookup by char range for subject/object ID resolution
    def _find_entity_id(span_text: str) -> Optional[str]:
        span_lower = span_text.lower().strip()
        for e in entities:
            if e.normalized == span_lower or span_lower in e.normalized:
                return e.entity_id
        return None

    # Try textacy SVO extraction
    try:
        import textacy.extract

        triples = list(textacy.extract.subject_verb_object_triples(doc))
        for triple in triples:
            subj_text = " ".join([t.text for t in triple.subject])
            verb_text = " ".join([t.text for t in triple.verb])
            obj_text = " ".join([t.text for t in triple.object])

            subj_id = _find_entity_id(subj_text)
            obj_id = _find_entity_id(obj_text)

            facts.append(
                FactTriple(
                    fact_id=f"fact_{_uid()}",
                    subject_id=subj_id or f"anon_{_uid()}",
                    predicate=verb_text,
                    object_id=obj_id,
                    object_value=obj_text,
                    unit_id=unit_id,
                    raw_text=text,
                    confidence=0.75,
                    extraction_method="textacy_svo",
                )
            )
    except Exception:  # noqa: BLE001
        # Fallback: manual dependency parse for nsubj -> ROOT -> dobj
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subj = None
                obj = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = " ".join([t.text for t in child.subtree])
                    if child.dep_ in ("dobj", "attr", "oprd"):
                        obj = " ".join([t.text for t in child.subtree])
                if subj:
                    subj_id = _find_entity_id(subj)
                    obj_id = _find_entity_id(obj) if obj else None
                    facts.append(
                        FactTriple(
                            fact_id=f"fact_{_uid()}",
                            subject_id=subj_id or f"anon_{_uid()}",
                            predicate=token.lemma_,
                            object_id=obj_id,
                            object_value=obj,
                            unit_id=unit_id,
                            raw_text=text,
                            confidence=0.60,
                            extraction_method="dep_parse",
                        )
                    )
    return facts


# ---------------------------------------------------------------------------
# KV group extraction
# ---------------------------------------------------------------------------
def _extract_kv_facts(
    unit: SemanticUnit,
) -> Tuple[List[EntitySpan], List[FactTriple], List[Dict[str, Any]]]:
    """Extract entities, facts, and kv_pairs from a KV_GROUP unit."""
    entities: List[EntitySpan] = []
    facts: List[FactTriple] = []
    kv_pairs: List[Dict[str, Any]] = []

    if not unit.kv_pairs:
        return entities, facts, kv_pairs

    for key, value in unit.kv_pairs.items():
        kv_pairs.append({"key": key, "value": value, "unit_id": unit.unit_id})

        # Create a fact for each KV pair
        facts.append(
            FactTriple(
                fact_id=f"fact_{_uid()}",
                subject_id=f"kv_{_uid()}",
                predicate=key,
                object_value=value,
                unit_id=unit.unit_id,
                raw_text=f"{key}: {value}",
                confidence=0.95,
                extraction_method="kv_structured",
            )
        )

        # Try to extract entities from values
        val_entities = _extract_spacy_entities(value, unit.unit_id)
        entities.extend(val_entities)

        pat_entities = _extract_pattern_entities(value, unit.unit_id)
        entities.extend(pat_entities)

    return entities, facts, kv_pairs


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------
def _extract_table_facts(
    unit: SemanticUnit,
) -> Tuple[List[EntitySpan], List[FactTriple], List[Dict[str, Any]]]:
    """Extract entities, facts, and structured table data from a TABLE unit."""
    entities: List[EntitySpan] = []
    facts: List[FactTriple] = []
    tables_structured: List[Dict[str, Any]] = []

    if not unit.table_rows:
        return entities, facts, tables_structured

    headers = unit.table_headers or []

    # Store structured table representation
    tables_structured.append(
        {
            "unit_id": unit.unit_id,
            "headers": headers,
            "rows": unit.table_rows,
            "row_count": len(unit.table_rows),
        }
    )

    # Create per-row-per-column facts
    for row_idx, row in enumerate(unit.table_rows):
        row_subject_id = f"row_{unit.unit_id}_{row_idx}"
        for col_key, col_val in row.items():
            if not col_val:
                continue
            raw = f"{col_key}: {col_val}"
            facts.append(
                FactTriple(
                    fact_id=f"fact_{_uid()}",
                    subject_id=row_subject_id,
                    predicate=col_key,
                    object_value=str(col_val),
                    unit_id=unit.unit_id,
                    raw_text=raw,
                    confidence=0.90,
                    extraction_method="table_structured",
                )
            )

        # Extract entities from row cell values
        row_text = " ".join(str(v) for v in row.values() if v)
        if row_text.strip():
            row_ents = _extract_spacy_entities(row_text, unit.unit_id)
            entities.extend(row_ents)

    return entities, facts, tables_structured


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def _deduplicate_entities(entities: List[EntitySpan]) -> List[EntitySpan]:
    """Deduplicate entities by (label, normalized) keeping highest confidence."""
    seen: Dict[str, EntitySpan] = {}
    for ent in entities:
        key = f"{ent.label}|{ent.normalized}"
        if key not in seen or ent.confidence > seen[key].confidence:
            seen[key] = ent
    return list(seen.values())


# ---------------------------------------------------------------------------
# Validation: no hallucinated entities
# ---------------------------------------------------------------------------
def _validate_entities(
    entities: List[EntitySpan], units: List[SemanticUnit]
) -> List[EntitySpan]:
    """Remove entities whose text does not appear in the source unit text."""
    unit_text_map: Dict[str, str] = {u.unit_id: u.text.lower() for u in units}
    valid: List[EntitySpan] = []
    for ent in entities:
        source_text = unit_text_map.get(ent.unit_id, "")
        if ent.text.lower() in source_text:
            valid.append(ent)
        else:
            logger.debug(
                "Dropping hallucinated entity '%s' (not in unit %s)",
                ent.text,
                ent.unit_id,
            )
    return valid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def extract_entities_and_facts(doc: StructuredDocument) -> ExtractionResult:
    """Extract all entities and facts from a StructuredDocument.

    Processes each SemanticUnit using:
    - spaCy NER for named entities
    - Regex patterns for EMAIL, PHONE, URL, DURATION
    - textacy SVO for fact triples (paragraph/list text)
    - Structured extraction for KV_GROUP and TABLE units

    Returns an ExtractionResult with deduplicated, validated entities and
    provenance-carrying facts.
    """
    all_entities: List[EntitySpan] = []
    all_facts: List[FactTriple] = []
    all_kv_pairs: List[Dict[str, Any]] = []
    all_tables: List[Dict[str, Any]] = []

    if not doc.units:
        return ExtractionResult(document_id=doc.document_id)

    for unit in doc.units:
        text = unit.text or ""
        if not text.strip():
            continue

        # --- Structure-aware extraction ---
        if unit.unit_type == UnitType.KV_GROUP and unit.kv_pairs:
            kv_ents, kv_facts, kv_pairs = _extract_kv_facts(unit)
            all_entities.extend(kv_ents)
            all_facts.extend(kv_facts)
            all_kv_pairs.extend(kv_pairs)
            # Also run spaCy on the full text for additional entities
            all_entities.extend(_extract_spacy_entities(text, unit.unit_id))
            all_entities.extend(_extract_pattern_entities(text, unit.unit_id))
            continue

        if unit.unit_type == UnitType.TABLE and unit.table_rows:
            tbl_ents, tbl_facts, tbl_structured = _extract_table_facts(unit)
            all_entities.extend(tbl_ents)
            all_facts.extend(tbl_facts)
            all_tables.extend(tbl_structured)
            # Also run spaCy on the full text
            all_entities.extend(_extract_spacy_entities(text, unit.unit_id))
            continue

        # --- General text extraction ---
        spacy_ents = _extract_spacy_entities(text, unit.unit_id)
        all_entities.extend(spacy_ents)

        pattern_ents = _extract_pattern_entities(text, unit.unit_id)
        all_entities.extend(pattern_ents)

        # SVO facts for prose units
        if unit.unit_type in (
            UnitType.PARAGRAPH,
            UnitType.LIST,
            UnitType.FRAGMENT,
        ):
            unit_entities = spacy_ents + pattern_ents
            svo_facts = _extract_svo_facts(text, unit.unit_id, unit_entities)
            all_facts.extend(svo_facts)

    # Deduplicate and validate
    all_entities = _deduplicate_entities(all_entities)
    all_entities = _validate_entities(all_entities, doc.units)

    return ExtractionResult(
        document_id=doc.document_id,
        entities=all_entities,
        facts=all_facts,
        tables_structured=all_tables,
        kv_pairs=all_kv_pairs,
    )
