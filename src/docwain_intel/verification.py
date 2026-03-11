"""Extraction verification and conflict detection.

Implements 5 self-verification checks on ExtractionResult objects:
1. Entity type consistency
2. Numeric conflict detection
3. Temporal logic validation
4. Duplicate fact detection
5. Provenance validation
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    ConflictRecord,
    ExtractionResult,
    FactTriple,
    StructuredDocument,
    VerificationResult,
)

_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")
_DATE_RE = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
)


def _extract_numbers(text: str) -> List[float]:
    """Extract numeric values from a string."""
    nums = []
    for m in _NUMBER_RE.finditer(text):
        try:
            nums.append(float(m.group().replace(",", "")))
        except ValueError:
            continue
    return nums


def _check_entity_type_consistency(
    extraction: ExtractionResult,
) -> List[ConflictRecord]:
    """Check 1: Flag entities with same normalized text but different labels."""
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for ent in extraction.entities:
        groups[ent.normalized].append((ent.entity_id, ent.label))

    conflicts = []
    for normalized, entries in groups.items():
        labels = {label for _, label in entries}
        if len(labels) > 1:
            ids = [eid for eid, _ in entries]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    label_i = entries[i][1]
                    label_j = entries[j][1]
                    if label_i != label_j:
                        conflicts.append(
                            ConflictRecord(
                                fact_id_1=ids[i],
                                fact_id_2=ids[j],
                                conflict_type="entity_type_inconsistency",
                                description=(
                                    f"Entity '{normalized}' labeled as "
                                    f"'{label_i}' and '{label_j}'"
                                ),
                            )
                        )
    return conflicts


def _check_numeric_conflicts(
    extraction: ExtractionResult,
) -> List[ConflictRecord]:
    """Check 2: Flag facts with same subject+predicate but different numeric values."""
    groups: Dict[Tuple[str, str], List[FactTriple]] = defaultdict(list)
    for fact in extraction.facts:
        key = (fact.subject_id, fact.predicate.upper())
        groups[key].append(fact)

    conflicts = []
    for (subj, pred), facts in groups.items():
        if len(facts) < 2:
            continue
        for i in range(len(facts)):
            nums_i = _extract_numbers(facts[i].object_value or "")
            if not nums_i:
                continue
            for j in range(i + 1, len(facts)):
                nums_j = _extract_numbers(facts[j].object_value or "")
                if not nums_j:
                    continue
                if nums_i != nums_j:
                    conflicts.append(
                        ConflictRecord(
                            fact_id_1=facts[i].fact_id,
                            fact_id_2=facts[j].fact_id,
                            conflict_type="numeric_mismatch",
                            description=(
                                f"Numeric conflict for {subj}.{pred}: "
                                f"'{facts[i].object_value}' vs "
                                f"'{facts[j].object_value}'"
                            ),
                        )
                    )
    return conflicts


def _check_temporal_logic(
    extraction: ExtractionResult,
) -> List[str]:
    """Check 3: Warn if date-type facts have logical issues (e.g. end < start)."""
    warnings = []
    date_facts: Dict[str, Dict[str, str]] = defaultdict(dict)

    for fact in extraction.facts:
        pred_upper = fact.predicate.upper()
        if "DATE" in pred_upper or "START" in pred_upper or "END" in pred_upper:
            date_facts[fact.subject_id][pred_upper] = fact.object_value or ""

    for subj, dates in date_facts.items():
        start_keys = [k for k in dates if "START" in k]
        end_keys = [k for k in dates if "END" in k]
        for sk in start_keys:
            for ek in end_keys:
                start_dates = _DATE_RE.findall(dates[sk])
                end_dates = _DATE_RE.findall(dates[ek])
                if start_dates and end_dates:
                    try:
                        s = start_dates[0].replace("/", "-")
                        e = end_dates[0].replace("/", "-")
                        if s > e:
                            warnings.append(
                                f"Temporal issue for {subj}: start date "
                                f"'{dates[sk]}' appears after end date "
                                f"'{dates[ek]}'"
                            )
                    except (ValueError, IndexError):
                        pass
    return warnings


def _check_duplicate_facts(
    extraction: ExtractionResult,
) -> List[ConflictRecord]:
    """Check 4: Flag same subject+predicate with different object values."""
    groups: Dict[Tuple[str, str], List[FactTriple]] = defaultdict(list)
    for fact in extraction.facts:
        key = (fact.subject_id, fact.predicate.upper())
        groups[key].append(fact)

    conflicts = []
    for (subj, pred), facts in groups.items():
        if len(facts) < 2:
            continue
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                val_i = (facts[i].object_value or "").strip().lower()
                val_j = (facts[j].object_value or "").strip().lower()
                if val_i != val_j:
                    conflicts.append(
                        ConflictRecord(
                            fact_id_1=facts[i].fact_id,
                            fact_id_2=facts[j].fact_id,
                            conflict_type="duplicate_predicate",
                            description=(
                                f"Duplicate predicate '{pred}' for subject "
                                f"'{subj}': '{facts[i].object_value}' vs "
                                f"'{facts[j].object_value}'"
                            ),
                        )
                    )
    return conflicts


def _check_provenance(
    extraction: ExtractionResult,
    doc: StructuredDocument,
) -> List[str]:
    """Check 5: Verify fact raw_text keywords appear in source unit text."""
    unit_map: Dict[str, str] = {u.unit_id: u.text.lower() for u in doc.units}
    warnings = []

    for fact in extraction.facts:
        source_text = unit_map.get(fact.unit_id, "")
        if not source_text:
            warnings.append(
                f"Fact '{fact.fact_id}': source unit '{fact.unit_id}' not found"
            )
            continue

        raw_words = set(re.findall(r"\w+", fact.raw_text.lower()))
        # Filter out very short words (stopwords-like)
        keywords = {w for w in raw_words if len(w) > 2}
        if not keywords:
            continue

        source_words = set(re.findall(r"\w+", source_text))
        overlap = keywords & source_words
        coverage = len(overlap) / len(keywords) if keywords else 1.0

        if coverage < 0.3:
            warnings.append(
                f"Fact '{fact.fact_id}': raw_text has low provenance match "
                f"({coverage:.0%}) with source unit '{fact.unit_id}'"
            )

    return warnings


def verify_extraction(
    extraction: ExtractionResult,
    doc: StructuredDocument,
) -> VerificationResult:
    """Run all verification checks on an extraction result.

    Parameters
    ----------
    extraction:
        The extraction result to verify.
    doc:
        The structured document the extraction was derived from.

    Returns
    -------
    VerificationResult with conflicts, warnings, quality_score, and is_valid.
    """
    conflicts: List[ConflictRecord] = []
    warnings: List[str] = []

    # 1. Entity type consistency
    conflicts.extend(_check_entity_type_consistency(extraction))

    # 2. Numeric conflict detection
    conflicts.extend(_check_numeric_conflicts(extraction))

    # 3. Temporal logic
    warnings.extend(_check_temporal_logic(extraction))

    # 4. Duplicate fact detection
    conflicts.extend(_check_duplicate_facts(extraction))

    # 5. Provenance validation
    warnings.extend(_check_provenance(extraction, doc))

    # Deduplicate conflicts by (fact_id_1, fact_id_2, conflict_type)
    seen: Set[Tuple[str, str, str]] = set()
    unique_conflicts = []
    for c in conflicts:
        key = (c.fact_id_1, c.fact_id_2, c.conflict_type)
        if key not in seen:
            seen.add(key)
            unique_conflicts.append(c)

    # Quality score
    score = 1.0
    score -= 0.1 * len(unique_conflicts)
    score -= 0.02 * len(warnings)
    score = max(score, 0.0)

    return VerificationResult(
        is_valid=score > 0.3,
        conflicts=unique_conflicts,
        warnings=warnings,
        quality_score=round(score, 4),
    )
