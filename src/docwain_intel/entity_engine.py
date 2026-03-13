"""Universal entity and fact extraction engine.

Uses spaCy NER + regex patterns + textacy SVO triples + structure-aware
extraction from KV groups and tables.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import threading
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

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_spacy_lock = threading.Lock()
_spacy_nlp = None

def _get_spacy():
    """Load spaCy model (lazy singleton, thread-safe). Prefers en_core_web_lg."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    with _spacy_lock:
        if _spacy_nlp is not None:  # double-check after acquiring lock
            return _spacy_nlp
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_lg")
        except OSError:
            logger.debug("en_core_web_lg not found, falling back to en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

# ---------------------------------------------------------------------------
# Lazy-loaded GLiNER singleton
# ---------------------------------------------------------------------------
_gliner_lock = threading.Lock()
_gliner_model = None
_gliner_available = True  # set to False on first ImportError

_GLINER_LABELS = [
    "skill",
    "qualification",
    "product",
    "service",
    "medical_condition",
    "legal_clause",
    "policy_term",
    "financial_metric",
    "job_title",
    "certification",
]

def _get_gliner():
    """Load GLiNER model (lazy singleton, thread-safe). Returns None if unavailable."""
    global _gliner_model, _gliner_available
    if not _gliner_available:
        return None
    if _gliner_model is not None:
        return _gliner_model
    with _gliner_lock:
        if not _gliner_available:
            return None
        if _gliner_model is not None:
            return _gliner_model
        try:
            from gliner import GLiNER  # noqa: F401

            _gliner_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
            logger.info("GLiNER model loaded successfully")
        except ImportError:
            logger.debug(
                "GLiNER not installed — zero-shot entity extraction disabled. "
                "Install with: pip install gliner"
            )
            _gliner_available = False
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("GLiNER model failed to load: %s", exc)
            _gliner_available = False
            return None
    return _gliner_model

# ---------------------------------------------------------------------------
# GLiNER zero-shot entity extraction
# ---------------------------------------------------------------------------
def _extract_gliner_entities(text: str, unit_id: str) -> List[EntitySpan]:
    """Extract domain-specific entities using GLiNER zero-shot NER.

    Finds entities that spaCy misses: SKILL, QUALIFICATION, PRODUCT,
    MEDICAL_CONDITION, JOB_TITLE, CERTIFICATION, etc.
    """
    model = _get_gliner()
    if model is None:
        return []

    # Truncate to GLiNER's context window
    truncated = text[:MAX_GLINER_TEXT_CHARS]
    if not truncated.strip():
        return []

    entities: List[EntitySpan] = []
    try:
        predictions = model.predict_entities(truncated, _GLINER_LABELS, threshold=0.5)
        for pred in predictions:
            label_upper = pred["label"].upper()
            ent_text = pred["text"]
            entities.append(
                EntitySpan(
                    entity_id=f"ent_{_uid()}",
                    text=ent_text,
                    normalized=ent_text.lower().strip(),
                    label=label_upper,
                    unit_id=unit_id,
                    char_start=pred.get("start", 0),
                    char_end=pred.get("end", 0),
                    confidence=0.80,
                    source="gliner",
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("GLiNER extraction failed: %s", exc)

    return entities

# ---------------------------------------------------------------------------
# Regex patterns for structured entity extraction
# ---------------------------------------------------------------------------
MAX_GLINER_TEXT_CHARS = 4096

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# Phone: require at least 7 digit characters to avoid false positives
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"  # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"  # optional area code
    r"\d{3,4}[\s\-.]?\d{3,4}"  # main number
)
_URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")
MAX_UNIT_TEXT_CHARS = 50_000
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
    except ImportError:
        logger.debug("textacy not available, using manual SVO extraction")
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("SVO extraction failed: %s", exc)
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
    """Remove entities whose text does not appear in the source unit text or heading paths."""
    unit_text_map: Dict[str, str] = {}
    for u in units:
        # Include heading paths in searchable text so heading-derived entities validate
        heading_text = " ".join(u.heading_path or []).lower()
        unit_text_map[u.unit_id] = f"{u.text.lower()} {heading_text}"
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

    # Extract entities from heading paths — headings aren't standalone units
    # but carry names/entities that must be discoverable (e.g., "Gokul Ramanathan")
    _seen_headings: set = set()
    _COMMON_SECTION_TITLES = frozenset({
        "summary", "experience", "education", "skills", "certifications",
        "work experience", "professional summary", "references", "projects",
        "objective", "contact", "abstract", "introduction", "conclusion",
        "appendix", "table of contents", "invoice", "agreement", "contract",
        "terms", "scope", "services", "compensation", "confidentiality",
    })
    for unit in doc.units:
        for heading in (unit.heading_path or []):
            if heading and heading not in _seen_headings:
                _seen_headings.add(heading)
                # Try spaCy with context injection
                contextualized = f"This document is about {heading}."
                heading_ents = _extract_spacy_entities(contextualized, unit.unit_id)
                heading_lower = heading.lower()
                heading_ents = [e for e in heading_ents if e.text.lower() in heading_lower]
                all_entities.extend(heading_ents)

                # Name heuristic: if heading looks like a person name (2-3 capitalized
                # words, not a common section title, no digits), add as PERSON entity.
                # This handles names spaCy doesn't recognize (e.g., uncommon names).
                words = heading.strip().split()
                is_name_like = (
                    2 <= len(words) <= 4
                    and all(w[0].isupper() and w.isalpha() for w in words)
                    and heading_lower not in _COMMON_SECTION_TITLES
                    and not any(e.label == "PERSON" and e.text.lower() == heading_lower for e in heading_ents)
                )
                if is_name_like:
                    all_entities.append(EntitySpan(
                        entity_id=f"ent_{_uid()}",
                        text=heading,
                        normalized=heading_lower,
                        label="PERSON",
                        unit_id=unit.unit_id,
                        confidence=0.75,
                        source="heading_heuristic",
                    ))

    for unit in doc.units:
        text = (unit.text or "")[:MAX_UNIT_TEXT_CHARS]
        if not text.strip():
            continue

        # --- Structure-aware extraction ---
        if unit.unit_type == UnitType.KV_GROUP and unit.kv_pairs:
            kv_ents, kv_facts, kv_pairs = _extract_kv_facts(unit)
            all_entities.extend(kv_ents)
            all_facts.extend(kv_facts)
            all_kv_pairs.extend(kv_pairs)
            # Pattern entities only — spaCy NER already runs on individual values
            # inside _extract_kv_facts(); skip redundant full-text spaCy pass
            all_entities.extend(_extract_pattern_entities(text, unit.unit_id))
            continue

        if unit.unit_type == UnitType.TABLE and unit.table_rows:
            tbl_ents, tbl_facts, tbl_structured = _extract_table_facts(unit)
            all_entities.extend(tbl_ents)
            all_facts.extend(tbl_facts)
            all_tables.extend(tbl_structured)
            # spaCy NER already runs on individual row texts inside
            # _extract_table_facts(); skip redundant full-text spaCy pass
            continue

        # --- General text extraction ---
        spacy_ents = _extract_spacy_entities(text, unit.unit_id)
        all_entities.extend(spacy_ents)

        pattern_ents = _extract_pattern_entities(text, unit.unit_id)
        all_entities.extend(pattern_ents)

        # GLiNER zero-shot extraction for prose units (after spaCy)
        if unit.unit_type in (
            UnitType.PARAGRAPH,
            UnitType.LIST,
            UnitType.FRAGMENT,
        ):
            gliner_ents = _extract_gliner_entities(text, unit.unit_id)
            all_entities.extend(gliner_ents)

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
